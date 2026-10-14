"""Harness self-awareness guards (briefs 036, 037, 039).

These mechanisms compensate for small-model weaknesses observed in the wild
(Oct 14, 2026): a 30B model burned 34 turns calling Bash on a machine with no
bash and `grep` inside PowerShell, never adapting to the tool errors.

- 036: environment capability card — the harness introduces itself at session start.
- 037: error-class circuit breaker — same tool + same error 3x gets a redirect,
  then short-circuits instead of re-executing.
- 039: progress nudge — N tool turns with no file modifications triggers a
  planning nudge.

All fail-open: guard code never raises into the agent loop.
"""
from __future__ import annotations

import json
import platform
import re
import shutil
from typing import Any


# ---------------------------------------------------------------------------
# 036 — Environment capability card
# ---------------------------------------------------------------------------

def _which(cmd: str) -> bool:
    try:
        return shutil.which(cmd) is not None
    except Exception:
        return False


def detect_environment() -> dict[str, Any]:
    """Detect the machine's capabilities. Never raises."""
    try:
        os_name = platform.system() or "unknown"
    except Exception:
        os_name = "unknown"
    return {
        "os": os_name,
        "is_windows": os_name.lower() == "windows",
        "bash_available": _which("bash"),
        "powershell_available": _which("powershell") or _which("pwsh"),
    }


def build_env_card(tool_context: Any = None) -> str:
    """Build a compact environment card for the turn-0 system message.

    Tells the model what exists on THIS machine so it doesn't discover it
    through failed tool calls. Returns "" when detection fails.
    """
    try:
        env = detect_environment()
    except Exception:
        return ""
    lines = ["[Environment]"]
    lines.append(f"OS: {env['os']}.")
    if env["is_windows"]:
        if env["bash_available"]:
            lines.append("Bash IS installed — the Bash tool works (bash syntax only).")
        else:
            lines.append(
                "Bash is NOT installed — the Bash tool WILL FAIL "
                "('bash executable not found'). Do not call it; use PowerShell."
            )
        if env["powershell_available"]:
            lines.append(
                "PowerShell is available — PowerShell syntax ONLY (never bash). "
                "PowerShell has no `grep`/`ls`/`cat`: use Select-String, "
                "Get-ChildItem, Get-Content — or the dedicated Grep/Read tools, "
                "which are preferred for searching and reading files."
            )
    else:
        if env["bash_available"]:
            lines.append("Bash IS installed — the Bash tool works (bash syntax only).")
        else:
            lines.append("Bash is NOT installed — the Bash tool WILL FAIL. Do not call it.")
    try:
        ws = getattr(tool_context, "workspace_root", None) or getattr(tool_context, "cwd", None)
        if ws:
            lines.append(f"Workspace: {ws}. `cd` persists across Bash/PowerShell calls.")
    except Exception:
        pass
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 037 — Error-class circuit breaker
# ---------------------------------------------------------------------------

def normalize_error_signature(output: Any) -> str:
    """Normalize a tool error into a stable signature for counting repeats.

    First line, lowercased, with paths and numbers stripped so trivial
    output differences don't defeat the matching.
    """
    try:
        if isinstance(output, dict):
            text = output.get("error") or output.get("message") or json.dumps(output)[:200]
        else:
            text = str(output)
        first = text.strip().splitlines()[0] if text.strip() else ""
        sig = first.lower()
        sig = re.sub(r"[a-z]:\\[^\s]*", "<path>", sig)  # Windows paths
        sig = re.sub(r"/[^\s]*", "<path>", sig)  # posix paths
        sig = re.sub(r"\d+", "<n>", sig)
        return sig[:160]
    except Exception:
        return "<unparseable>"


class CircuitBreaker:
    """Trips when the same tool fails with the same error signature 3x.

    After tripping, the harness injects a plain-English redirect and
    short-circuits further calls to that tool (returns the last error
    without executing) until the model moves on.
    """

    TRIP_THRESHOLD = 3
    COOLDOWN_TURNS = 5

    def __init__(self) -> None:
        # (tool_name_lower, error_sig) -> consecutive count
        self._counts: dict[tuple[str, str], int] = {}
        # tool_name_lower -> {"error": str, "quiet_turns": int}
        self._open: dict[str, dict[str, Any]] = {}

    def _key(self, tool_name: str, output: Any) -> tuple[str, str]:
        return (tool_name.lower(), normalize_error_signature(output))

    def record(self, tool_name: str, output: Any, is_error: bool) -> str | None:
        """Record one tool outcome. Returns a redirect message when the
        breaker trips on this call, else None. Never raises."""
        try:
            name = tool_name.lower()
            if not is_error:
                # Any success clears this tool's counts and starts cooldown credit.
                self._counts = {k: v for k, v in self._counts.items() if k[0] != name}
                return None
            key = self._key(tool_name, output)
            self._counts[key] = self._counts.get(key, 0) + 1
            # Drop counts for other signatures of the same tool — we count
            # *consecutive same-signature* failures.
            for other in [k for k in self._counts if k[0] == name and k != key]:
                del self._counts[other]
            if self._counts[key] >= self.TRIP_THRESHOLD and name not in self._open:
                err_text = self._last_error_text(output)
                self._open[name] = {"error": err_text, "quiet_turns": 0}
                return self._redirect_text(tool_name, err_text)
            return None
        except Exception:
            return None

    def _last_error_text(self, output: Any) -> str:
        try:
            if isinstance(output, dict):
                return str(output.get("error") or output.get("message") or "")[:300]
            return str(output)[:300]
        except Exception:
            return ""

    def _redirect_text(self, tool_name: str, err_text: str) -> str:
        hint = ""
        if tool_name.lower() == "bash" and "bash executable not found" in err_text.lower():
            hint = " Use the PowerShell tool instead (PowerShell syntax only)."
        elif "not recognized as the name of a cmdlet" in err_text.lower():
            hint = " That command does not exist in PowerShell — use Select-String (not grep), Get-ChildItem (not ls), Get-Content (not cat), or the dedicated Grep/Read tools."
        return (
            f"[Harness] The {tool_name} tool has failed {self.TRIP_THRESHOLD}x with the "
            f"same error ('{err_text[:120]}'). It will keep failing — do not call it "
            f"again for this purpose.{hint}"
        )

    def should_short_circuit(self, tool_name: str) -> str | None:
        """If the circuit is open for this tool, return the stored error text
        (caller synthesizes the result without executing). Else None."""
        try:
            state = self._open.get(tool_name.lower())
            return state["error"] if state else None
        except Exception:
            return None

    def note_turn_without_tool(self, tool_name: str) -> None:
        """Age open circuits: close after COOLDOWN_TURNS turns of not calling it."""
        self.note_turn_end({tool_name} if tool_name else ())

    def note_turn_end(self, tools_called: Any) -> None:
        """Age open circuits at the end of a turn.

        An open circuit whose tool was called this turn resets its quiet
        counter; every other open circuit ages by one turn and closes at
        COOLDOWN_TURNS. Never raises (fail-open).
        """
        try:
            called = {str(t).lower() for t in (tools_called or ())}
            for open_name in list(self._open):
                if open_name in called:
                    self._open[open_name]["quiet_turns"] = 0
                else:
                    self._open[open_name]["quiet_turns"] += 1
                    if self._open[open_name]["quiet_turns"] >= self.COOLDOWN_TURNS:
                        del self._open[open_name]
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 039 — Progress nudge
# ---------------------------------------------------------------------------

# Tools whose success counts as "progress" for the nudge counter.
# Edit/Write successes reset it; Subagent/Task completion also counts —
# delegated work is progress too.
_PROGRESS_TOOLS = {"edit", "write", "subagent", "task", "agent"}

# Nudge schedule (brief 039): first nudge at 10 modification-free tool turns,
# then every 15 turns thereafter (10, 25, 40, 55, ...).
_NUDGE_FIRST_TURN = 10
_NUDGE_REPEAT_EVERY = 15


def is_progress_tool(tool_name: str) -> bool:
    try:
        return tool_name.lower() in _PROGRESS_TOOLS
    except Exception:
        return False


def build_progress_nudge(turns_without_progress: int) -> str:
    # Fail-open: never raises; a garbled count still yields a sane nudge.
    try:
        n = int(turns_without_progress)
    except Exception:
        n = _NUDGE_FIRST_TURN
    return (
        f"You've made {n} tool calls without modifying any files. "
        "Briefly state your plan — which files you'll change, what the "
        "change is, and what you're still looking for — before continuing."
    )


def should_nudge(turns_without_progress: int) -> bool:
    """True at 10 modification-free tool turns, then every 15 (10, 25, 40...).

    Fail-open: never raises — a missed nudge is just a missed nudge.
    """
    try:
        n = int(turns_without_progress)
    except Exception:
        return False
    return n >= _NUDGE_FIRST_TURN and (n - _NUDGE_FIRST_TURN) % _NUDGE_REPEAT_EVERY == 0
