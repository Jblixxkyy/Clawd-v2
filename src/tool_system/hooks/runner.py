"""Hook runner: executes configured shell commands around agent lifecycle events.

Fail-open by design: any hook problem (missing config, timeout, non-zero
exit, invalid JSON) is logged as a dim warning and the agent proceeds.

Events (briefs 008 + 031 + 015; gemini-cli/opencode parity):
    SessionStart, SessionEnd, BeforeAgent, AfterAgent, PreCompress,
    BeforeModel, AfterModel, PreToolUse, PostToolUse,
    BeforeToolSelection, ToolExecuteBefore, SessionRetry, Stop

Ordering guarantees (brief 031):
    BeforeToolSelection  ->  ToolExecuteBefore  ->  PreToolUse
        ->  execute  ->  PostToolUse
  i.e. selection (which tools the model may choose from) happens before
  any tool input is finalized; ToolExecuteBefore is the last mutation seam
  before PreToolUse's policy gate and the tool's own execution. BeforeModel
  wraps the outgoing model call (it fires before SessionRetry matters for
  that call: a vetoed retry re-raises instead of looping).

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
    BeforeToolSelection: {"session_id", "turn", "model", "plan_mode",
                          "read_only_tools", "tools"} — "tools" is the
                          candidate schema list the model is about to see.
    ToolExecuteBefore:  {"event", "tool", "input"} (same shape as PreToolUse)
    SessionRetry: {"attempt", "max_attempts", "error", "error_type"} —
                          fired before backing off for another attempt of a
                          transient-failing provider call.
    Stop: {"session_id", "turn", "model", "tool", "outcome", "summary",
           "reason", "artifacts", "blocks_so_far", "workspace_root"} —
           fired when the agent loop is about to terminate via a terminal
           tool (complete_task / ReportFailure / Exit): the hook is the
           done-gate and may veto completion (brief 015).

Stdout decisions: a hook replies with JSON. New style:
    {"decision": "block", "message": "reason..."}
  or the legacy PreToolUse style:
    {"allow": false, "message": "reason..."}   (means "block")
  Supported decisions: "allow" (proceed), "block" (halt with the message),
  "ask" (currently enforced like "block": no interactive approval UI is
  wired to hooks yet, so asking degrades to halting rather than silently
  allowing), plus the brief-031 control decisions:
    "inject"  (BeforeModel only): append a message to the outgoing prompt.
              {"decision": "inject", "role": "user", "content": "..."}
              ("content" may be given as "message"; role defaults to
              "user".)
    "rewrite" (BeforeModel / BeforeToolSelection / ToolExecuteBefore):
              replace what the hook wrapped —
              BeforeModel:          {"decision": "rewrite", "messages": [...]}
              BeforeToolSelection:  {"decision": "rewrite", "tools": [...]}
              ToolExecuteBefore:    {"decision": "rewrite", "input": {...}}
    "fail"    (ToolExecuteBefore): fail the tool call pre-execution with the
              message (same effect as "block" there; no partial effects —
              permissions, checkpoints, and the tool itself never run).

  Blocking-capable events (block/ask/fail honored):
    PreToolUse  -> blocks the tool call (existing behavior).
    BeforeAgent -> halts the run before the first model call; the hook's
                   message becomes the run's response text.
    BeforeModel -> halts the turn; the hook's message becomes the response.
    PreCompress -> vetoes this compaction; the loop continues uncompacted.
    ToolExecuteBefore -> fails the tool call pre-execution with the
                   hook's message; no partial effects.
    SessionRetry -> vetoes the pending provider-call retry; the original
                   error is re-raised immediately instead of backing off.
    Stop -> vetoes the terminal-tool return (brief 015): a block decision
                   feeds the hook's message back as a continuation prompt
                   and the loop resumes instead of ending; consecutive
                   blocks are capped and then escalated to the user.
  Rewrite-capable events (rewrite/inject honored; block/ask/fail ignored):
    BeforeModel, BeforeToolSelection, ToolExecuteBefore.
  Every other event is informational: hook output is ignored and the agent
  proceeds (same as PostToolUse today). Malformed hook output always fails
  open, for every event.

Guardrails (brief 031):
  - Timeouts: every hook command is bounded by ``timeout`` (default 10s).
    A hanging hook is killed, a warning is logged, and the loop continues.

Isolation (brief 033 — crash isolation, not a security boundary):
  - Every hook command runs inside an isolated execution context
    (:mod:`isolation`): a dedicated daemon worker thread, never the
    agent loop's main thread. The hook itself is a child process, so a
    segfaulting or non-zero-exiting hook dies in its own process; worker
    exceptions (even SystemExit) are captured, never propagated. The loop
    is provably immune: a crashing hook cannot poison loop state. Per-hook
    timeouts are fail-open: the hook is skipped, the turn continues, and
    the failure is recorded in the audit log as "hook-timeout"/"hook-failed"
    (the plugin is marked failed). This is liveness protection only —
    hook code still runs with the harness's privileges; do not mistake it
    for a sandbox.
  - Loud load failures: a plugin that cannot load (missing bootstrap
    script/interpreter, bad config) raises a named error
    (``PluginBootstrapMissingError``/``PluginConfigError``) naming the
    exact missing piece, surfaced in /doctor output and session-start
    warnings. Hooks are never skipped quietly.
  - Recursion protection: hooks cannot trigger themselves. While a hook
    for event E is in flight, a nested ``run_hooks(E)`` (e.g. from a hook
    script that shells back into the harness) returns ``allow``
    immediately with ``recursion_blocked: True`` instead of spawning
    hooks again. The in-flight set is tracked thread-locally AND passed
    to hook subprocesses via the ``CLAWD_HOOK_EVENTS`` environment
    variable, so a hook script in a child process is guarded too.
    Per-event tracking catches multi-event cycles as well (any cycle
    repeats an event).
  - Audit logging: every block/rewrite/inject decision is appended as a
    JSON line to ~/.clawd/hooks-audit.log (timestamp, event, decision,
    message, command). Audit failures never affect the hook outcome.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .isolation import PluginConfigError, run_command_isolated

try:  # optional: dim warnings only when rich is available
    from rich.console import Console as _RichConsole
except Exception:  # pragma: no cover - rich is a hard dep here, but stay safe
    _RichConsole = None  # type: ignore


#: All hook events the runner understands.
TOOL_EVENTS = ("PreToolUse", "PostToolUse", "ToolExecuteBefore")
LIFECYCLE_EVENTS = (
    "SessionStart",
    "SessionEnd",
    "BeforeAgent",
    "AfterAgent",
    "PreCompress",
    "BeforeModel",
    "AfterModel",
    "BeforeToolSelection",
    "SessionRetry",
    "Stop",
)
HOOK_EVENTS: tuple[str, ...] = LIFECYCLE_EVENTS + TOOL_EVENTS

#: Events whose hooks may halt what they wrap (block/ask/fail honored).
#: All other events are informational for blocking; hook output is ignored
#: (fail open).
BLOCKING_EVENTS = frozenset({
    "PreToolUse",
    "BeforeAgent",
    "BeforeModel",
    "PreCompress",
    "ToolExecuteBefore",
    "SessionRetry",
    "Stop",
})

#: Events whose hooks may rewrite what they wrap. "inject" is honored on
#: BeforeModel only; "rewrite" on BeforeModel/BeforeToolSelection/
#: ToolExecuteBefore.
REWRITE_EVENTS = frozenset({"BeforeModel", "BeforeToolSelection", "ToolExecuteBefore"})

#: Decisions a hook may return on stdout.
HOOK_DECISIONS = ("allow", "block", "ask", "inject", "rewrite", "fail")

#: Per-event expectations for a "rewrite" decision: which stdout key carries
#: the replacement payload, and what shape it must have ("message" dict,
#: "messages" list, "tools" list, or "input" dict). "inject" on BeforeModel
#: accepts "content" or "message" for the message text.
_REWRITE_SPEC: dict[str, dict[str, str]] = {
    ("BeforeModel", "inject"): {"key": "content", "shape": "message"},
    ("BeforeModel", "rewrite"): {"key": "messages", "shape": "messages"},
    ("BeforeToolSelection", "rewrite"): {"key": "tools", "shape": "tools"},
    ("ToolExecuteBefore", "rewrite"): {"key": "input", "shape": "input"},
}


def _default_config_path() -> Path:
    return Path.home() / ".clawd" / "hooks.json"


def _audit_log_path() -> Path:
    return Path.home() / ".clawd" / "hooks-audit.log"


def _warn(message: str) -> None:
    if _RichConsole is None:
        return
    try:
        _RichConsole().print(f"[dim]{message}[/dim]", highlight=False)
    except Exception:
        pass


def _audit(event: str, decision: str, message: Optional[str], command: Optional[str]) -> None:
    """Append one audit line for a block/rewrite/inject decision. Never raises."""
    try:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "decision": decision,
            "message": message,
            "command": command,
        }
        path = _audit_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass


_thread_state = threading.local()


def _in_flight_events() -> set[str]:
    """Thread-local set of hook events currently executing hooks.

    Recursion protection: a hook script that re-enters ``run_hooks`` for an
    event already in flight gets an immediate fail-open "allow" instead of
    spawning hooks again (see ``run_hooks``). The set is seeded from the
    ``CLAWD_HOOK_EVENTS`` environment variable, which ``run_hooks`` sets on
    every hook subprocess, so hook scripts running in child processes are
    guarded too.
    """
    events = getattr(_thread_state, "hook_events", None)
    if events is None:
        events = set()
        for name in os.environ.get("CLAWD_HOOK_EVENTS", "").split(","):
            name = name.strip()
            if name:
                events.add(name)
        _thread_state.hook_events = events
    return events


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

    Accepts the new ``{"decision": ...}`` style (allow/block/ask/inject/
    rewrite/fail) and the legacy ``{"allow": false, "message": ...}`` style
    (which means "block"). Anything else — non-dict, missing/unknown
    decision, ``{"allow": true}`` — is "allow". Never raises.
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


def _extract_rewrite(event: str, decision: str, raw: Any) -> Any:
    """Extract the replacement payload for an inject/rewrite decision.

    Returns the payload in the shape the caller consumes, or None when the
    decision is not a well-formed rewrite for *event* (fail open). Never
    raises.
    """
    if not isinstance(raw, dict):
        return None
    spec = _REWRITE_SPEC.get((event, decision))
    if spec is None:
        return None
    shape = spec["shape"]
    if shape == "message":
        content = raw.get("content", raw.get("message"))
        if not isinstance(content, str):
            return None
        role = raw.get("role", "user")
        if not isinstance(role, str) or not role:
            role = "user"
        return {"role": role, "content": content}
    payload = raw.get(spec["key"])
    if shape == "messages":
        if not isinstance(payload, list) or not payload:
            return None
        if not all(
            isinstance(m, dict) and isinstance(m.get("role"), str)
            and isinstance(m.get("content"), (str, list, dict))
            for m in payload
        ):
            return None
        return payload
    if shape == "tools":
        if not isinstance(payload, list):
            return None
        if not all(isinstance(t, dict) and isinstance(t.get("name"), str) for t in payload):
            return None
        return payload
    if shape == "input":
        if not isinstance(payload, dict):
            return None
        return payload
    return None


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
    (PreToolUse/PostToolUse/ToolExecuteBefore); ``extra`` carries
    event-specific keys for lifecycle events. Both may be combined; the
    payload is always ``{"event": event, ...extra..., "tool": ..., "input": ...}``.

    Returns ``{"blocked": bool, "decision": str, "message": str | None,
    "rewrite": Any | None}``. ``blocked`` is True only when a hook returned
    block/ask/fail on a blocking-capable event. ``decision`` is the parsed
    decision ("allow"/"block"/"ask"/"inject"/"rewrite"/"fail"; informational
    events always report "allow"). ``rewrite`` carries the replacement
    payload for inject/rewrite decisions (message dict, message list, tool
    list, or input dict depending on the event), else None.

    The first control decision (block/ask/fail/rewrite/inject) wins; hooks
    listed after it are not run. Every such decision is audit-logged.

    Always fails open: malformed output, timeouts, and non-zero exits never
    block, for any event. A hanging hook is killed after ``timeout`` seconds
    and skipped with a logged warning. Every hook command runs inside the
    isolated execution context (brief 033): a dedicated worker thread, never
    the caller's thread — a crashing hook cannot poison loop state. Hook
    timeouts and crashes are recorded in the audit log ("hook-timeout" /
    "hook-failed") and the plugin is treated as failed for this event.
    """
    cfg = config_path or _default_config_path()
    try:
        config = json.loads(cfg.read_text())
    except FileNotFoundError:
        return {"blocked": False, "decision": "allow", "message": None, "rewrite": None}
    except Exception as exc:
        # Loud, named load failure: a present-but-broken hooks.json is a
        # plugin config problem, not a silent skip. Fail open, but name it.
        named = PluginConfigError(
            plugin=str(cfg),
            missing=f"valid JSON ({exc})",
            fix=f"fix the JSON syntax in {cfg}, or delete it to disable hooks",
        )
        _warn(f"[hooks] {named}; skipping hooks")
        return {"blocked": False, "decision": "allow", "message": None, "rewrite": None}

    if not isinstance(config, dict):
        return {"blocked": False, "decision": "allow", "message": None, "rewrite": None}

    commands = _normalize_commands(config.get(event))
    if not commands:
        # Fast path: no hooks for this event. Skips payload serialization,
        # which matters for events carrying large payloads (BeforeModel
        # passes the full message list).
        return {"blocked": False, "decision": "allow", "message": None, "rewrite": None}

    # Recursion protection: a hook script that re-enters run_hooks for an
    # event already in flight (hooks cannot trigger themselves) gets an
    # immediate fail-open allow instead of spawning hooks again.
    in_flight = _in_flight_events()
    if event in in_flight:
        _warn(f"[hooks] recursive hook invocation for {event} blocked")
        _audit(event, "recursion-blocked", None, None)
        return {
            "blocked": False,
            "decision": "allow",
            "message": None,
            "rewrite": None,
            "recursion_blocked": True,
        }

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
        return {"blocked": False, "decision": "allow", "message": None, "rewrite": None}

    blocked = False
    decision_name = "allow"
    message: Optional[str] = None
    rewrite: Any = None

    in_flight.add(event)
    # Propagate the in-flight set to hook subprocesses so a hook script
    # that shells back into the harness cannot trigger itself there.
    hook_env = {**os.environ, "CLAWD_HOOK_EVENTS": ",".join(sorted(in_flight))}
    try:
        for command in commands:
            # Brief 033: hook code never runs on the loop's thread. The
            # isolated context runs the command on a dedicated daemon
            # worker thread with a per-plugin timeout; crashes, timeouts,
            # and launch failures are captured, audit-logged, and skipped
            # fail-open. The turn always continues.
            result = run_command_isolated(
                command, stdin_data, timeout=timeout, env=hook_env
            )
            if result.status == "timeout":
                _warn(f"[hooks] hook timed out after {timeout}s, skipped: {command}")
                _audit(
                    event,
                    "hook-timeout",
                    f"timed out after {timeout}s: {result.error}",
                    command,
                )
                continue
            if result.status == "crashed":
                _warn(
                    f"[hooks] hook crashed (exit {result.returncode}), "
                    f"skipped: {command}"
                )
                _audit(event, "hook-failed", result.error, command)
                continue
            if result.status == "error":
                _warn(f"[hooks] hook failed to run, skipped: {command} ({result.error})")
                _audit(event, "hook-failed", result.error, command)
                continue

            out = result.stdout.strip()
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
            if name in ("inject", "rewrite"):
                # Control-but-not-blocking decisions: honored only on
                # rewrite-capable events, and only with a well-formed
                # payload; anything else fails open.
                if event not in REWRITE_EVENTS:
                    continue
                payload_rewrite = _extract_rewrite(event, name, raw_decision)
                if payload_rewrite is None:
                    _warn(f"[hooks] hook returned malformed {name} payload, skipped: {command}")
                    continue
                decision_name = name
                rewrite = payload_rewrite
                _audit(event, name, msg, command)
                break  # first control decision wins
            if event not in BLOCKING_EVENTS:
                # Informational event (PostToolUse, AfterAgent, ...): hook
                # output is ignored; always fail open.
                continue
            blocked = True
            decision_name = name
            message = msg
            _audit(event, name, msg, command)
            break  # first blocking hook wins
    finally:
        in_flight.discard(event)

    return {"blocked": blocked, "decision": decision_name, "message": message, "rewrite": rewrite}
