"""Clawd-native reasoning budgets (brief 027).

One portable ``reasoning{enabled, effort, budget_tokens}`` setting that
resolves per model call, plus phase/role-aware budget policies -- with a
Clawd-native fallback (forced reflection turns) where the provider exposes
no thinking control.

Layers:
  - *Schema*: :class:`ReasoningBudget` -- ``enabled``, ``effort``
    (low|medium|high|max), ``budget_tokens`` (optional override).
  - *Policy*: :func:`resolve_budget` -- phase/role/agent resolution from the
    ``reasoning`` config section. Defaults: planning > execution,
    investigator/reviewer high, classifiers/summarizers/titles low or off,
    compaction medium. All user-overridable.
  - *Adapter*: :mod:`src.providers.reasoning` -- maps the portable budget to
    native provider options (or nothing, never a 400).
  - *Fallback*: :func:`run_fallback_reflection` -- where no provider knob
    exists, effort becomes behavior: high = +N forced reflection turns
    before acting, low = none. Depends on brief 023's think tool
    DEFENSIVELY (duck-typing + import guard); works standalone.

User-facing: the ``/effort`` command implementation lives here
(:func:`effort_command_call`); registration in
``src/command_system/builtins.py`` is PENDING (014 owns the registry this
wave) -- see the exact lines in the builder's handoff report.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

EFFORT_LOW = "low"
EFFORT_MEDIUM = "medium"
EFFORT_HIGH = "high"
EFFORT_MAX = "max"

EFFORTS = (EFFORT_LOW, EFFORT_MEDIUM, EFFORT_HIGH, EFFORT_MAX)

#: Effort rank so "planning > executing" comparisons are arithmetic.
_EFFORT_RANK = {EFFORT_LOW: 1, EFFORT_MEDIUM: 2, EFFORT_HIGH: 3, EFFORT_MAX: 4}

#: Spec strings that disable reasoning outright.
_DISABLE_SPECS = frozenset({"off", "none", "disabled", "false", "no", "0"})

#: Config key under which reasoning-budget settings live.
CONFIG_KEY = "reasoning"

#: Phases recognized by the default policy.
PHASE_PLANNING = "planning"
PHASE_EXECUTION = "execution"
PHASE_COMPACTION = "compaction"

#: Roles recognized by the default policy.
ROLE_INVESTIGATOR = "investigator"
ROLE_REVIEWER = "reviewer"
ROLE_CLASSIFIER = "classifier"
ROLE_SUMMARIZER = "summarizer"
ROLE_SESSION_TITLE = "session_title"
ROLE_COMPACTION = "compaction"

#: Default phase/role policies (user-overridable via config).
DEFAULT_PHASE_EFFORTS: dict[str, str] = {
    PHASE_PLANNING: EFFORT_HIGH,
    PHASE_EXECUTION: EFFORT_MEDIUM,
    PHASE_COMPACTION: EFFORT_MEDIUM,
}

DEFAULT_ROLE_EFFORTS: dict[str, str] = {
    ROLE_INVESTIGATOR: EFFORT_HIGH,
    ROLE_REVIEWER: EFFORT_HIGH,
    ROLE_CLASSIFIER: EFFORT_LOW,
    ROLE_SUMMARIZER: EFFORT_LOW,
    ROLE_SESSION_TITLE: "off",
    ROLE_COMPACTION: EFFORT_MEDIUM,
}

#: Default config values for the ``reasoning`` section.
DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "default_effort": EFFORT_MEDIUM,
    "budget_tokens": None,
    # Prefer 023's think tool for the fallback when it exists.
    "use_think_tool": True,
    "phases": dict(DEFAULT_PHASE_EFFORTS),
    "roles": dict(DEFAULT_ROLE_EFFORTS),
    # Per-agent overrides: {"my-agent": "low"} / {"my-agent": "off"}.
    "agents": {},
}


@dataclass
class ReasoningBudget:
    """Portable reasoning budget for one model call."""

    enabled: bool = True
    effort: str = EFFORT_MEDIUM
    budget_tokens: Optional[int] = None
    #: Where the budget came from: default|phase|role|agent.
    source: str = "default"

    def rank(self) -> int:
        """Numeric effort rank (for planning > executing comparisons)."""
        return _EFFORT_RANK.get(self.effort, 2)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _merged_section(config: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge the ``reasoning`` config section over DEFAULTS. Never raises."""
    section: dict[str, Any] = {}
    try:
        raw = (config or {}).get(CONFIG_KEY)
        if isinstance(raw, Mapping):
            section = dict(raw)
    except Exception:
        section = {}
    merged = dict(DEFAULTS)
    merged.update(section)
    for key in ("phases", "roles", "agents"):
        base = dict(DEFAULTS[key])
        override = section.get(key)
        if isinstance(override, Mapping):
            base.update({str(k): v for k, v in override.items()})
        merged[key] = base
    return merged


def _parse_effort_spec(value: Any) -> tuple[bool, str]:
    """Parse an effort spec into (enabled, effort).

    "off"/"none"/"disabled" -> (False, "medium"); unknown strings fall
    back to "medium"; valid efforts pass through. Never raises.
    """
    try:
        text = str(value or "").strip().lower()
    except Exception:
        return True, EFFORT_MEDIUM
    if text in _DISABLE_SPECS:
        return False, EFFORT_MEDIUM
    if text in _EFFORT_RANK:
        return True, text
    return True, EFFORT_MEDIUM


def resolve_budget(
    config: Optional[Mapping[str, Any]] = None,
    *,
    phase: Optional[str] = None,
    role: Optional[str] = None,
    agent: Optional[str] = None,
) -> ReasoningBudget:
    """Resolve the portable reasoning budget for one call site.

    Precedence: per-agent override > role > phase > default effort.
    A disable spec ("off") at any level wins. Never raises.
    """
    try:
        section = _merged_section(config)
        if not section.get("enabled", True):
            return ReasoningBudget(enabled=False, source="default")

        effort = str(section.get("default_effort") or EFFORT_MEDIUM).strip().lower()
        if effort not in _EFFORT_RANK:
            effort = EFFORT_MEDIUM
        source = "default"

        if phase:
            spec = section["phases"].get(str(phase).strip().lower())
            if spec is not None:
                on, parsed = _parse_effort_spec(spec)
                if not on:
                    return ReasoningBudget(enabled=False, source="phase")
                effort, source = parsed, "phase"

        if role:
            spec = section["roles"].get(str(role).strip().lower())
            if spec is not None:
                on, parsed = _parse_effort_spec(spec)
                if not on:
                    return ReasoningBudget(enabled=False, source="role")
                effort, source = parsed, "role"

        if agent:
            spec = section["agents"].get(str(agent))
            if spec is not None:
                on, parsed = _parse_effort_spec(spec)
                if not on:
                    return ReasoningBudget(enabled=False, source="agent")
                effort, source = parsed, "agent"

        tokens = section.get("budget_tokens")
        try:
            tokens = int(tokens) if tokens is not None else None
        except (TypeError, ValueError):
            tokens = None
        return ReasoningBudget(
            enabled=True, effort=effort, budget_tokens=tokens, source=source
        )
    except Exception:
        return ReasoningBudget()


# ---------------------------------------------------------------------------
# Clawd-native fallback: forced reflection turns (brief 023, defensive)
# ---------------------------------------------------------------------------

#: Prompt used for Clawd-native forced reflection turns.
REFLECTION_PROMPT = (
    "Before you act, think step by step: state your plan, the key risks, "
    "and what you will check after acting. Keep it brief."
)

#: Forced reflection turns per effort level. High = +N think turns before
#: acting; low = none. Effort becomes behavior, not just a parameter.
FALLBACK_TURNS: dict[str, int] = {
    EFFORT_LOW: 0,
    EFFORT_MEDIUM: 1,
    EFFORT_HIGH: 2,
    EFFORT_MAX: 3,
}

LOG_PREFIX = "[reasoning_budget]"

#: Candidate locations for brief 023's think tool. Duck-typed: any object
#: found here is tried defensively; on any failure the fallback degrades
#: to prompt-based reflection turns. 023 is in progress this wave, so this
#: list is a hook, not a requirement.
_THINK_TOOL_CANDIDATES = (
    ("src.tool_system.tools.think", "ThinkTool"),
    ("src.tools.think", "ThinkTool"),
    ("src.tool_system.tools.think", "think"),
)


def _think_tool_callable() -> Optional[Any]:
    """Locate 023's think tool if it exists. Import-guarded; None when 023
    has not landed or is unimportable. Never raises."""
    for module_name, attr in _THINK_TOOL_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        try:
            found = getattr(module, attr, None)
        except Exception:
            continue
        if found is not None:
            return found
    return None


def fallback_turn_count(budget: ReasoningBudget) -> int:
    """Forced reflection turns for an effort level (0 when disabled)."""
    try:
        if not budget.enabled:
            return 0
        return FALLBACK_TURNS.get(str(budget.effort).strip().lower(), 0)
    except Exception:
        return 0


def should_emulate(
    provider_name: Optional[str],
    model: Optional[str],
    budget: ReasoningBudget,
) -> bool:
    """Whether the Clawd-native fallback applies: the budget is enabled,
    the effort calls for reflection turns, and the model advertises no
    native reasoning control. Never raises."""
    try:
        from src.providers.reasoning import may_reason

        return (
            bool(getattr(budget, "enabled", False))
            and fallback_turn_count(budget) > 0
            and not may_reason(model, provider_name)
        )
    except Exception:
        return False


def _message_dicts(messages: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            out.append(dict(msg))
        elif hasattr(msg, "to_dict"):
            try:
                out.append(dict(msg.to_dict()))
                continue
            except Exception:
                pass
        out.append(
            {
                "role": getattr(msg, "role", "user"),
                "content": getattr(msg, "content", ""),
            }
        )
    return out


def run_fallback_reflection(
    provider: Any,
    messages: Any,
    budget: ReasoningBudget,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    call_kwargs: Optional[dict[str, Any]] = None,
    on_event: Optional[Callable[[str], None]] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Clawd-native reasoning emulation for no-knob providers.

    Issues ``fallback_turn_count(budget)`` forced reflection turns against
    ``provider`` and returns the transcript entries to splice into the
    conversation. Prefers 023's think tool when it is importable and
    cooperative (duck-typed, import-guarded); otherwise falls back to
    prompt-based reflection turns (``REFLECTION_PROMPT``).

    Returns:
        (transcript_entries, log_lines). ``transcript_entries`` alternates
        user reflection prompts and the model's assistant replies. Log
        lines carry a ``forced think turn`` marker for the agent loop and
        UI. Never raises -- on provider failure the turns already made are
        returned with a log line explaining the stop.
    """
    log: list[str] = []

    def emit(line: str) -> None:
        log.append(line)
        if on_event is not None:
            try:
                on_event(line)
            except Exception:
                pass

    turns = fallback_turn_count(budget)
    if turns <= 0:
        return [], log

    # 023 hook: try the think tool defensively.
    section = _merged_section(config)
    if section.get("use_think_tool", True):
        think = _think_tool_callable()
        if think is not None:
            try:
                entries = _run_via_think_tool(
                    think, provider, messages, budget, turns,
                    tools=tools, call_kwargs=call_kwargs,
                )
                if entries:
                    emit(
                        f"{LOG_PREFIX} {turns} forced think turn(s) via "
                        f"023 think tool (effort={budget.effort})"
                    )
                    return entries, log
            except Exception as exc:
                emit(
                    f"{LOG_PREFIX} think-tool path failed ({exc}); "
                    "using prompt-based reflection"
                )

    entries: list[dict[str, Any]] = []
    history = _message_dicts(messages)
    for i in range(1, turns + 1):
        prompt = {"role": "user", "content": REFLECTION_PROMPT}
        try:
            response = provider.chat(
                history + [prompt], tools=tools, **(call_kwargs or {})
            )
        except Exception as exc:
            emit(f"{LOG_PREFIX} forced think turn {i}/{turns} failed: {exc}")
            break
        content = getattr(response, "content", "") or ""
        reply = {"role": "assistant", "content": content}
        entries.append(prompt)
        entries.append(reply)
        history = history + [prompt, reply]
        emit(
            f"{LOG_PREFIX} forced think turn {i}/{turns} "
            f"(effort={budget.effort}, +{len(content)} chars)"
        )
    return entries, log


def _run_via_think_tool(
    think: Any,
    provider: Any,
    messages: Any,
    budget: ReasoningBudget,
    turns: int,
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    call_kwargs: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Best-effort 023 think-tool path. Duck-typed: a ThinkTool class is
    instantiated with no args and ``run``/``__call__``'d per turn; a plain
    function is called per turn. Any signature mismatch raises and the
    caller degrades to prompt-based reflection."""
    entries: list[dict[str, Any]] = []
    history = _message_dicts(messages)
    for _ in range(turns):
        prompt = {"role": "user", "content": REFLECTION_PROMPT}
        if isinstance(think, type):
            tool = think()
            runner = getattr(tool, "run", None) or tool
            content = runner(prompt["content"])
        else:
            content = think(prompt["content"])
        if not isinstance(content, str):
            content = str(content)
        reply = {"role": "assistant", "content": content}
        entries.append(prompt)
        entries.append(reply)
        history = history + [prompt, reply]
    return entries


# ---------------------------------------------------------------------------
# /effort command implementation
# ---------------------------------------------------------------------------
# Registration in src/command_system/builtins.py is PENDING (014 owns the
# registry this wave). Exact lines are in the builder's handoff report; this
# module provides the handler so the integrator only wires it up.


def _persist_reasoning_update(mutator: Callable[[dict[str, Any]], str]) -> str:
    """Apply ``mutator`` to the live reasoning config section and save.

    ``mutator`` receives the section dict and returns a status message.
    Never raises -- errors are returned as text (the command surface is
    user-facing).
    """
    try:
        from src.config import load_config, save_config

        config = load_config()
        section = config.get(CONFIG_KEY)
        if not isinstance(section, dict):
            section = {}
            config[CONFIG_KEY] = section
        message = mutator(section)
        save_config(config)
        return message
    except Exception as exc:
        return f"Could not update reasoning config: {exc}"


def _status_text(config: Optional[Mapping[str, Any]] = None) -> str:
    section = _merged_section(config)
    lines = ["Reasoning budgets:", ""]
    lines.append(f"  enabled: {section.get('enabled', True)}")
    lines.append(f"  default effort: {section.get('default_effort', EFFORT_MEDIUM)}")
    tokens = section.get("budget_tokens")
    lines.append(f"  budget_tokens: {tokens if tokens is not None else '(per-effort default)'}")
    lines.append("")
    lines.append("  phases:")
    for phase, spec in section["phases"].items():
        lines.append(f"    {phase}: {spec}")
    lines.append("  roles:")
    for role, spec in section["roles"].items():
        lines.append(f"    {role}: {spec}")
    agents = section.get("agents") or {}
    lines.append("  per-agent overrides:")
    if agents:
        for name, spec in agents.items():
            lines.append(f"    {name}: {spec}")
    else:
        lines.append("    (none)")
    return "\n".join(lines)


def effort_command_call(args: str, context: Any) -> Any:
    """Handle /effort -- inspect and set the portable reasoning budget.

    Usage:
        /effort                    show current budgets
        /effort low|medium|high|max
                                   set the default effort
        /effort off|on             disable/enable reasoning budgets
        /effort phase <name> <effort|off>
        /effort role <name> <effort|off>
        /effort agent <name> <effort|off>
        /effort agent <name> clear

    Returns a LocalCommandResult (imported lazily to keep this module
    import-light for tests).
    """
    from src.command_system.types import LocalCommandResult

    parts = str(args or "").strip().split()

    def show() -> Any:
        try:
            live = context.config if context is not None else None
        except Exception:
            live = None
        if not live:
            try:
                from src.config import load_config

                live = load_config()
            except Exception:
                live = None
        return LocalCommandResult(type="text", value=_status_text(live))

    if not parts or parts[0].lower() in ("show", "status"):
        return show()

    head = parts[0].lower()

    if head in _EFFORT_RANK:
        effort = head

        def set_default(section: dict[str, Any]) -> str:
            section["enabled"] = True
            section["default_effort"] = effort
            return f"Default reasoning effort set to {effort}."

        return LocalCommandResult(
            type="text", value=_persist_reasoning_update(set_default)
        )

    if head == "off":

        def disable(section: dict[str, Any]) -> str:
            section["enabled"] = False
            return "Reasoning budgets disabled (no native options sent, no fallback turns)."

        return LocalCommandResult(
            type="text", value=_persist_reasoning_update(disable)
        )

    if head == "on":

        def enable(section: dict[str, Any]) -> str:
            section["enabled"] = True
            return "Reasoning budgets enabled."

        return LocalCommandResult(
            type="text", value=_persist_reasoning_update(enable)
        )

    if head in ("phase", "role", "agent") and len(parts) >= 3:
        scope, name, spec = head, parts[1], parts[2].lower()

        def set_override(section: dict[str, Any]) -> str:
            if scope == "agent" and spec == "clear":
                agents = section.setdefault("agents", {})
                agents.pop(name, None)
                return f"Per-agent reasoning override for {name!r} cleared."
            on, parsed = _parse_effort_spec(spec)
            value = parsed if on else "off"
            bucket = {"phase": "phases", "role": "roles", "agent": "agents"}[scope]
            section.setdefault(bucket, {})[name] = value
            return f"Reasoning {scope} override: {name} -> {value}."

        return LocalCommandResult(
            type="text", value=_persist_reasoning_update(set_override)
        )

    return LocalCommandResult(
        type="text",
        value=(
            "Usage: /effort [low|medium|high|max|off|on] | "
            "/effort phase <name> <effort|off> | "
            "/effort role <name> <effort|off> | "
            "/effort agent <name> <effort|off|clear>"
        ),
    )


__all__ = [
    "CONFIG_KEY",
    "DEFAULTS",
    "DEFAULT_PHASE_EFFORTS",
    "DEFAULT_ROLE_EFFORTS",
    "EFFORTS",
    "EFFORT_HIGH",
    "EFFORT_LOW",
    "EFFORT_MAX",
    "EFFORT_MEDIUM",
    "FALLBACK_TURNS",
    "LOG_PREFIX",
    "PHASE_COMPACTION",
    "PHASE_EXECUTION",
    "PHASE_PLANNING",
    "REFLECTION_PROMPT",
    "ROLE_CLASSIFIER",
    "ROLE_COMPACTION",
    "ROLE_INVESTIGATOR",
    "ROLE_REVIEWER",
    "ROLE_SESSION_TITLE",
    "ROLE_SUMMARIZER",
    "ReasoningBudget",
    "_merged_section",
    "effort_command_call",
    "fallback_turn_count",
    "resolve_budget",
    "run_fallback_reflection",
    "should_emulate",
]
