#!/usr/bin/env python3
"""PostToolUse hook example: append every tool call to a JSONL log.

Reads the hook payload JSON from stdin:
    {"event": "PostToolUse", "tool": "...", "input": {...}, "output": {...}}

Appends one line per call to ~/.clawd/hook-log.jsonl:
    {"time": "2026-10-11T18:00:00+00:00", "tool": "Bash"}

PostToolUse hooks can never block; this script prints nothing meaningful
(the runner ignores PostToolUse hook output anyway).

Wire it up in ~/.clawd/hooks.json:
    {"PostToolUse": [{"command": "python3 /path/to/log-tool-use.py"}]}
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # unreadable input -> nothing to log

    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "tool": payload.get("tool"),
    }
    log_path = Path.home() / ".clawd" / "hook-log.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass  # logging must never break the tool call
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
