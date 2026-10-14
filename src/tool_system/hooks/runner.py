"""Hook runner: executes configured shell commands around agent lifecycle events.

Fail-open by design: any hook problem (missing config, timeout, non-zero
exit, invalid JSON) is logged as a dim warning and the agent proceeds.

Events (brief 008; gemini-cli parity):
    SessionStart, SessionEnd, BeforeAgent, AfterAgent, PreCompress,
    BeforeModel, AfterModel, PreToolUse, PostToolUse

Config: ~/.clawd/hooks.json — each event maps to a per-event command list:
    {"BeforeAgent": [{"command": "python3 /path/to/hook.py"}],
     "PreToolUse": "python3 /path/to/hook.py"}
  A bare string command per event is also accepted.

Stdin: each hook command runs via ``subprocess`` with ``shell=True`` and
receives event-specific JSON on stdin. Tool events:
    {"event": "PreToolUse", "tool": "Bash", "input": {...}}
  (PostToolUse adds an "output" key with the tool's result.)
  Lifecycle events receive {"event": <name>, ...event-specific keys...}:
    SessionStart: {"session_id", "provider", "model", "created_at", "resumed"}
    SessionEnd:   {"session_id", "reason"}
    BeforeAgent:  {"session_id", "model", "max_turns"}
    AfterAgent:   {"session_id", "response_text", "num_turns", "usage",
                   "cancelled"} (or {"error"} when the run raised)
    PreCompress:  {"session_id", "model", "estimated_tokens", "trigger"}
    BeforeModel:  {"session_id", "turn", "model", "message_count", "messages"}
    AfterModel:   {"session_id", "turn", "model", "content", "tool_uses",
                   "usage"}

Stdout decisions: a hook replies with JSON. New style:
    {"decision": "block", "message": "reason..."}
  or the legacy PreToolUse style:
    {"allow": false, "message": "reason..."}   (means "block")
  Supported decisions: "allow" (proceed), "block" (halt with the message),
  "ask" (currently enforced like "block": no interactive approval UI is
  wired to hooks yet, so asking degrades to halting rather than silently
  allowing).

  Only blocking-capable events honor block/ask:
    PreToolUse  -> blocks the tool call (existing behavior).
    BeforeAgent -> halts the run before the first model call; the hook's
                   message becomes the run's response text.
    BeforeModel -> halts the turn; the hook's message becomes the response.
    PreCompress -> vetoes this compaction; the loop continues uncompacted.
  Every other event is informational: hook output is ignored and the agent
  proceeds (same as PostToolUse today). Malformed hook output always fails
  open, for every event.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Optional

try:  # optional: dim warnings only when rich is available
    from rich.console import Console as _RichConsole
except Exception:  # pragma: no cover - rich is a hard dep here, but stay safe
    _RichConsole = None  # type: ignore


#: All hook events the runner understands.
TOOL_EVENTS = ("PreToolUse", "PostToolUse")
LIFECYCLE_EVENTS = (
    "SessionStart",
    "SessionEnd",
    "BeforeAgent",
    "AfterAgent",
    "PreCompress",
    "BeforeModel",
    "AfterModel",
)
HOOK_EVENTS: tuple[str, ...] = LIFECYCLE_EVENTS + TOOL_EVENTS

#: Events whose hooks may halt what they wrap (block/ask decisions honored).
#: All other events are informational; hook output is ignored (fail open).
BLOCKING_EVENTS = frozenset({"PreToolUse", "BeforeAgent", "BeforeModel", "PreCompress"})

#: Decisions a hook may return on stdout.
HOOK_DECISIONS = ("allow", "block", "ask")


def _default_config_path() -> Path:
    return Path.home() / ".clawd" / "hooks.json"


def _warn(message: str) -> None:
    if _RichConsole is None:
        return
    try:
        _RichConsole().print(f"[dim]{message}[/dim]", highlight=False)
    except Exception:
        pass


def _normalize_commands(raw: Any) -> list[str]:
    """Accept a bare string command or a list of {"command": ...} entries."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    commands: list[str] = []
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, str):
            commands.append(entry)
        elif isinstance(entry, dict) and entry.get("command"):
            commands.append(str(entry["command"]))
    return commands


def _parse_decision(raw: Any) -> tuple[str, Optional[str]]:
    """Parse a hook's stdout JSON into (decision, message).

    Accepts the new ``{"decision": "allow"|"block"|"ask", "message": ...}``
    style and the legacy ``{"allow": false, "message": ...}`` style (which
    means "block"). Anything else — non-dict, missing/unknown decision,
    ``{"allow": true}`` — is "allow". Never raises.
    """
    if not isinstance(raw, dict):
        return "allow", None
    decision = raw.get("decision")
    if isinstance(decision, str) and decision.lower() in HOOK_DECISIONS:
        name = decision.lower()
    elif raw.get("allow") is False:
        name = "block"  # legacy PreToolUse style
    else:
        name = "allow"
    message = raw.get("message")
    return name, (str(message) if message is not None else None)


def run_hooks(
    event: str,
    tool_name: Optional[str] = None,
    tool_input: Optional[dict] = None,
    tool_output: Optional[dict] = None,
    config_path: Optional[Path] = None,
    timeout: float = 10.0,
    extra: Optional[dict] = None,
) -> dict:
    """Run all hooks configured for *event*.

    ``tool_name``/``tool_input``/``tool_output`` describe tool events
    (PreToolUse/PostToolUse); ``extra`` carries event-specific keys for
    lifecycle events. Both may be combined; the payload is always
    ``{"event": event, ...extra..., "tool": ..., "input": ...}``.

    Returns ``{"blocked": bool, "decision": str, "message": str | None}``.
    ``blocked`` is True only when a hook returned block/ask on a
    blocking-capable event. ``decision`` is the parsed decision
    ("allow"/"block"/"ask"; informational events always report "allow").
    Always fails open: malformed output, timeouts, and non-zero exits never
    block, for any event.
    """
    cfg = config_path or _default_config_path()
    try:
        config = json.loads(cfg.read_text())
    except FileNotFoundError:
        return {"blocked": False, "decision": "allow", "message": None}
    except Exception as exc:
        _warn(f"[hooks] could not read {cfg}: {exc}; skipping hooks")
        return {"blocked": False, "decision": "allow", "message": None}

    if not isinstance(config, dict):
        return {"blocked": False, "decision": "allow", "message": None}

    commands = _normalize_commands(config.get(event))
    if not commands:
        # Fast path: no hooks for this event. Skips payload serialization,
        # which matters for events carrying large payloads (BeforeModel
        # passes the full message list).
        return {"blocked": False, "decision": "allow", "message": None}
    payload: dict[str, Any] = {"event": event}
    if extra:
        for key, value in extra.items():
            if key != "event":
                payload[key] = value
    if tool_name is not None:
        payload["tool"] = tool_name
        payload["input"] = tool_input if tool_input is not None else {}
        if tool_output is not None:
            payload["output"] = tool_output
    try:
        stdin_data = json.dumps(payload, default=str)
    except Exception as exc:
        _warn(f"[hooks] could not serialize {event} payload: {exc}; skipping hooks")
        return {"blocked": False, "decision": "allow", "message": None}

    blocked = False
    decision_name = "allow"
    message: Optional[str] = None

    for command in commands:
        try:
            proc = subprocess.run(
                command,
                input=stdin_data,
                capture_output=True,
                text=True,
                shell=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            _warn(f"[hooks] hook timed out after {timeout}s, skipped: {command}")
            continue
        except Exception as exc:  # e.g. command not launchable
            _warn(f"[hooks] hook failed to run, skipped: {exc}")
            continue

        if proc.returncode != 0:
            _warn(
                f"[hooks] hook exited with code {proc.returncode}, skipped: {command}"
            )
            continue

        out = proc.stdout.strip()
        if not out:
            continue  # empty output = allow, nothing to say
        try:
            raw_decision = json.loads(out)
        except Exception:
            _warn(f"[hooks] hook returned invalid JSON, skipped: {command}")
            continue

        name, msg = _parse_decision(raw_decision)
        if name == "allow":
            continue
        if event not in BLOCKING_EVENTS:
            # Informational event (PostToolUse, AfterAgent, ...): hook
            # output is ignored; always fail open.
            continue
        blocked = True
        decision_name = name
        message = msg
        break  # first blocking hook wins

    return {"blocked": blocked, "decision": decision_name, "message": message}
