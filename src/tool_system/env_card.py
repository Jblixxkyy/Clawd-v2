"""Brief 036 — environment capability card.

At agent-loop start the harness detects the machine's actual capabilities
(OS, which shells exist, shell-specific gotchas) and renders a compact
~8-line "environment card" for the turn-0 system message. This is MODULE
WORK ONLY: loop wiring (appending the card to the turn-0
``full_system_prompt``) happens in a later wave.

Never breaks the loop: all detection runs inside ``build_env_card`` under
a broad ``try/except`` — any detection failure returns ``""`` (no card)
instead of raising.
"""

from __future__ import annotations

import platform
import shutil
from typing import Callable, Optional

# Injectable detection hook. A detector returns a small dict:
#   {"os_name": str, "bash": str | None, "powershell": str | None}
# where the shell values are the resolved executable path or None.
# The default (None) runs live detection against the host.
Detector = Callable[[], dict]


def _default_detect() -> dict:
    os_name = platform.system() or "unknown"
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    return {
        "os_name": os_name,
        "bash": shutil.which("bash"),
        "powershell": powershell,
    }


def _workspace_root(tool_context) -> str:
    root = getattr(tool_context, "workspace_root", None)
    if root is None:
        return "unknown"
    try:
        return str(root)
    except Exception:
        return "unknown"


def build_env_card(tool_context, detector: Optional[Detector] = None) -> str:
    """Render the environment capability card for ``tool_context``.

    Returns a short (~8-line) multi-line string, or ``""`` when detection
    fails for any reason. Never raises.
    """
    try:
        detect = detector or _default_detect
        info = detect()
        os_name = str(info.get("os_name") or "unknown")
        has_bash = bool(info.get("bash"))
        has_powershell = bool(info.get("powershell"))

        lines = ["Environment capability card:"]
        lines.append(f"- OS: {os_name}")

        shells = []
        if has_bash:
            shells.append("bash")
        if has_powershell:
            shells.append("PowerShell")
        lines.append(
            "- Shells available: "
            + (", ".join(shells) if shells else "none detected")
        )

        if has_bash:
            lines.append("- DO: use Bash for shell commands.")
            if os_name == "Windows":
                # Git-Bash/Cygwin-style shell: it exists, so prefer it, but
                # flag the PowerShell gotchas anyway in case it is used.
                lines.append(
                    "- DON'T: use PowerShell `grep` — it has none; "
                    "use `Select-String` or the Grep tool instead."
                )
        else:
            if has_powershell:
                lines.append("- DO: use PowerShell for shell commands.")
                lines.append("- DON'T: use Bash — it is not installed, so Bash will fail.")
                lines.append(
                    "- PowerShell has no `grep`: use `Select-String` or the Grep tool."
                )
            else:
                lines.append("- DON'T: rely on shell tools — no supported shell detected.")

        lines.append(f"- Workspace root: {_workspace_root(tool_context)}")
        lines.append(
            "- NOTE: `cd` persists across Bash tool calls (the harness "
            "tracks it); the PowerShell tool has no `cd` tracking and no "
            "cwd parameter \u2014 chain commands with `;` or use absolute paths there."
        )
        return "\n".join(lines)
    except Exception:
        # Detection must never break the loop (brief 036).
        return ""
