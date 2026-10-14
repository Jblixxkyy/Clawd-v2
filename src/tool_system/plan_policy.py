"""Plan-mode state machine policy (brief 016).

Declarative, mode-scoped tool policy in the gemini-cli ``plan.toml`` spirit:
in plan mode, deny all tools by default; re-allow read-only tools; allow
writes only to ``*.md`` under the plans directory. This module is the single
source of truth for the policy; the tool dispatcher (``registry.py``) consults
it, and ``register_plan_mode_hooks()`` wires the BeforeToolSelection
filter (031's reference example) so the model never even sees tools it may
not call.

Also holds:
- OpenHands-style plan schema validation (PLAN_STRUCTURE sections).
- plans-dir resolution helpers.
- compaction-safe reminder / approved-plan context builders (the opencode
  ``session.hook("context")`` self-healing-reminder pattern, as data the
  loop injects post-compaction).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Plans directory
# ---------------------------------------------------------------------------

#: Default plans subdir, relative to the workspace root.
PLANS_SUBDIR = Path(".clawd") / "plans"

#: Marker delimiters bracketing machine-generated plans (crush pattern).
#: The UI opens an approve/edit dialog when it detects these markers.
PLAN_MARKER_START = "<!-- PLAN_START -->"
PLAN_MARKER_END = "<!-- PLAN_END -->"


def resolve_plans_dir(context: Any, override: str | Path | None = None) -> Path:
    """Return the plans directory for this session, creating it if needed."""
    if override is not None:
        plans = Path(override).expanduser()
    else:
        stored = getattr(context, "plan_dir", None)
        if stored:
            plans = Path(stored)
        else:
            plans = Path(context.workspace_root) / PLANS_SUBDIR
    if not plans.is_absolute():
        plans = (Path(context.workspace_root) / plans)
    plans.mkdir(parents=True, exist_ok=True)
    return plans.resolve()


def is_plan_path(path: str | Path, plans_dir: str | Path) -> bool:
    """True when *path* is a ``*.md`` file inside *plans_dir*."""
    try:
        p = Path(path).expanduser()
        base = Path(plans_dir).expanduser()
        if not p.is_absolute():
            # Caller passes cwd-relative paths; resolve against cwd when
            # available, else the plans dir itself.
            p = (base / p)
        p = p.resolve()
        base = base.resolve()
        return p.suffix.lower() == ".md" and (p == base or base in p.parents)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Plan schema validation (OpenHands PLAN_STRUCTURE)
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = (
    "OBJECTIVE",
    "CONTEXT SUMMARY",
    "APPROACH OVERVIEW",
    "IMPLEMENTATION STEPS",
    "TESTING AND VALIDATION",
)

_STEP_TUPLE_PARTS = ("goal", "method", "reference")

_SECTION_RE = re.compile(r"(?im)^(?:#{{1,4}}\s*)?({names})\s*:?\s*$".format(
    names="|".join(re.escape(s) for s in REQUIRED_SECTIONS)))


def _section_spans(text: str) -> dict[str, tuple[int, int]]:
    """Map section name -> (content_start, content_end) offsets."""
    matches = list(_SECTION_RE.finditer(text))
    spans: dict[str, tuple[int, int]] = {}
    for i, m in enumerate(matches):
        name = m.group(1).upper()
        if name in spans:
            continue  # first occurrence wins
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        spans[name] = (start, end)
    return spans


def validate_plan_schema(text: str) -> list[str]:
    """Validate *text* against the OpenHands PLAN_STRUCTURE.

    Returns a list of human-readable errors; empty means the plan is valid.
    """
    errors: list[str] = []
    if not isinstance(text, str) or not text.strip():
        return ["plan is empty"]
    spans = _section_spans(text)
    for section in REQUIRED_SECTIONS:
        if section not in spans:
            errors.append(f"missing required section: {section}")
            continue
        start, end = spans[section]
        body = text[start:end].strip()
        # Strip a leading marker line if the planner wrapped the plan.
        if not body or body in (PLAN_MARKER_START, PLAN_MARKER_END):
            errors.append(f"section '{section}' is present but empty")
    if "IMPLEMENTATION STEPS" in spans:
        start, end = spans["IMPLEMENTATION STEPS"]
        steps_body = text[start:end]
        lowered = steps_body.lower()
        missing_parts = [p for p in _STEP_TUPLE_PARTS if p not in lowered]
        if missing_parts:
            errors.append(
                "IMPLEMENTATION STEPS must be goal+method+reference tuples; "
                f"missing: {', '.join(missing_parts)}"
            )
        if not re.search(r"(?m)^\s*(?:\d+[.)]|[-*])\s+\S", steps_body):
            errors.append("IMPLEMENTATION STEPS must list at least one step")
    return errors


def read_plan_file(path: str | Path) -> tuple[str | None, str | None]:
    """Read a plan file. Returns (text, error); error None on success."""
    p = Path(path)
    if not p.exists():
        return None, f"plan file does not exist: {p}"
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        return None, f"cannot read plan file {p}: {exc}"
    if not text.strip():
        return None, f"plan file is empty: {p}"
    return text, None


# ---------------------------------------------------------------------------
# Mode-scoped tool policy (gemini plan.toml pattern)
# ---------------------------------------------------------------------------

#: Tools that always terminate/enter the plan-mode state, regardless of
#: read-only flags. Old PascalCase names kept for back-compat.
WAY_OUT_TOOL_NAMES = frozenset({
    "EnterPlanMode", "ExitPlanMode",      # legacy
    "enter_plan_mode", "exit_plan_mode",  # 016 state-machine tools
})

#: Tool input keys that may carry the target file path for write tools.
_PATH_INPUT_KEYS = ("file_path", "path", "filePath", "filepath")


def _write_target_path(tool_name: str, tool_input: dict[str, Any]) -> str | None:
    if not isinstance(tool_input, dict):
        return None
    for key in _PATH_INPUT_KEYS:
        value = tool_input.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def tool_allowed_in_plan_mode(
    spec_name: str,
    is_read_only: bool,
    tool_input: dict[str, Any] | None,
    context: Any,
) -> tuple[bool, str | None]:
    """Decide whether a tool call is allowed while plan mode is active.

    Priority tiers (gemini ``plan.toml`` style):
    1. read-only tools -> allow.
    2. plan-mode state tools (enter/exit) -> allow (the way out must work).
    3. Write/Edit whose target is ``*.md`` under the plans dir -> allow.
    4. everything else -> deny.

    Returns (allowed, reason). ``reason`` is a denial message when denied.
    """
    if is_read_only:
        return True, None
    if spec_name in WAY_OUT_TOOL_NAMES:
        return True, None
    if spec_name in ("Write", "Edit"):
        target = _write_target_path(spec_name, tool_input or {})
        if target is not None:
            try:
                plans_dir = resolve_plans_dir(context)
            except Exception:
                plans_dir = None
            if plans_dir is not None and is_plan_path(target, plans_dir):
                return True, None
        return False, (
            f"plan mode is on: '{spec_name}' may only write *.md files "
            "inside the plans directory. Finish exploring, then call "
            "exit_plan_mode with the plan for approval."
        )
    return False, (
        "plan mode is on: this tool is disabled while planning. "
        "Only read-only tools and plan-file writes (*.md under the plans "
        "directory) are allowed. Finish exploring, then call exit_plan_mode "
        "with the plan for approval."
    )


# ---------------------------------------------------------------------------
# Compaction-safe reminders + approved-plan context injection
# ---------------------------------------------------------------------------

def build_plan_reminder(context: Any) -> str:
    """Short plan-mode reminder re-injected when context is rebuilt.

    Compaction-safe: derived entirely from session state (``plan_mode``,
    ``plan_file_path``), never from transcript history, so it survives
    compaction cycles verbatim.
    """
    plan_file = getattr(context, "plan_file_path", None)
    plans_dir = getattr(context, "plan_dir", None) or str(PLANS_SUBDIR)
    lines = [
        "[plan-mode reminder] You are in PLAN MODE (read-only exploration).",
        f"- Draft the plan as markdown under {plans_dir}/ (implementation_plan.md).",
        "- Only *.md writes inside the plans directory are permitted;",
        "  all other mutation tools are blocked by policy, not just discouraged.",
        "- When the plan is complete and matches the plan schema, call",
        "  exit_plan_mode with planFilePath for validation and user approval.",
    ]
    if plan_file:
        lines.append(f"- Current plan draft: {plan_file}")
    return "\n".join(lines)


def build_approved_plan_context(plan_text: str, plan_file_path: str) -> str:
    """The approved plan, injected as the session's single source of truth.

    (gemini-cli ``snippets.ts`` pattern: "Treat this file as your single
    source of truth".)
    """
    return (
        "[approved plan] The user approved the following plan. "
        "Treat this plan as your single source of truth for the "
        f"implementation. Plan file: {plan_file_path}\n\n{plan_text.strip()}"
    )


def plan_context_snippet(context: Any) -> str | None:
    """Context snippet the loop should inject post-compaction.

    Returns the plan-mode reminder while planning, or the approved-plan
    injection after approval — whichever keeps the plan state alive across
    compaction. Returns None when no plan state exists.
    """
    if getattr(context, "plan_mode", False):
        return build_plan_reminder(context)
    approved = getattr(context, "approved_plan", None)
    if isinstance(approved, str) and approved.strip():
        path = getattr(context, "plan_file_path", None) or "(inline plan)"
        return build_approved_plan_context(approved, str(path))
    return None


# ---------------------------------------------------------------------------
# Hook registration entry point
# ---------------------------------------------------------------------------

#: 031's reference BeforeToolSelection implementation of this policy.
PLAN_MODE_FILTER_SCRIPT = (
    Path(__file__).resolve().parent
    / "hooks" / "examples" / "plan_mode_tool_filter.py"
)


def register_plan_mode_hooks(hooks_json: str | Path | None = None) -> dict[str, Any]:
    """Wire the plan-mode BeforeToolSelection filter into hooks config.

    Appends ``plan_mode_tool_filter.py`` (031's reference implementation,
    which filters the candidate tool list down to read-only tools +
    ExitPlanMode/exit_plan_mode while ``plan_mode`` is true) to the
    ``BeforeToolSelection`` list in ``~/.clawd/hooks.json``. Idempotent:
    re-running never duplicates the entry.

    NOTE (pending wiring): hooks only fire when registered. This entry
    point is for the integrator / startup path to call so the policy is
    live by default; until then the dispatcher gate in ``registry.py``
    remains the live enforcement point.
    """
    target = Path(hooks_json).expanduser() if hooks_json else (
        Path(os.path.expanduser("~")) / ".clawd" / "hooks.json"
    )
    script = str(PLAN_MODE_FILTER_SCRIPT)
    config: dict[str, Any] = {}
    if target.exists():
        try:
            config = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
    entries = config.get("BeforeToolSelection")
    if not isinstance(entries, list):
        entries = []
    if not any(
        isinstance(e, dict) and script in str(e.get("command", ""))
        for e in entries
    ):
        entries.append({"command": f"python3 {script}"})
    config["BeforeToolSelection"] = entries
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {
        "hooks_json": str(target),
        "event": "BeforeToolSelection",
        "script": script,
        "registered": True,
    }
