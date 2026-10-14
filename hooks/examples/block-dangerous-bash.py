#!/usr/bin/env python3
"""PreToolUse hook example: block obviously dangerous Bash commands.

Reads the hook payload JSON from stdin:
    {"event": "PreToolUse", "tool": "Bash", "input": {"command": "..."}}

Writes a decision JSON object to stdout:
    {"allow": false, "message": "..."}  -> blocks the tool call
    {"allow": true}                     -> allows it

Wire it up in ~/.clawd/hooks.json:
    {"PreToolUse": [{"command": "python3 /path/to/block-dangerous-bash.py"}]}
"""

import json
import re
import sys

# Substrings / patterns that are never allowed through Bash.
BLOCKED_PATTERNS = [
    re.compile(r"\brm\s+.*-[a-z]*r[a-z]*f\b"),  # rm -rf anywhere in the command
    re.compile(r"\brm\s+-rf\s+/\b"),            # rm -rf / specifically
    re.compile(r"\bmkfs\b"),                    # filesystem formatting
    re.compile(r":\(\)\s*{\s*:\|\:&\s*}\s*;"),  # classic fork bomb
    re.compile(r"\bdd\s+.*\bof=/dev/"),         # dd writing to a device
]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        # Unreadable input -> fail open (allow).
        print(json.dumps({"allow": True}))
        return 0

    decision = {"allow": True}
    if payload.get("tool") == "Bash":
        command = str((payload.get("input") or {}).get("command", ""))
        for pattern in BLOCKED_PATTERNS:
            if pattern.search(command):
                decision = {
                    "allow": False,
                    "message": f"blocked by block-dangerous-bash hook: dangerous command '{command}'",
                }
                break

    print(json.dumps(decision))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
