#!/usr/bin/env python3
"""Reference implementation (brief 031): plan-mode policy on BeforeToolSelection.

Migrates 016's plan-mode policy from a prompt suggestion into an
enforcement point: when the payload's ``plan_mode`` flag is true, this hook
rewrites the candidate tool list so the model only ever sees read-only
tools plus ``ExitPlanMode`` (the way out). Filtering happens *before* the
model call, so the model cannot be talked into using tools it never saw.

The read-only tool names arrive in the payload as ``read_only_tools``
(supplied by the loop from the live registry), so this script never
hardcodes the tool catalog and cannot drift out of sync with it.

Register in ~/.clawd/hooks.json::

    {"BeforeToolSelection": [{"command": "python3 plan_mode_tool_filter.py"}]}

Stdin payload: {"event": "BeforeToolSelection", "plan_mode": bool,
                "read_only_tools": [...], "tools": [{"name": ...}, ...]}
Stdout: {"decision": "rewrite", "tools": [...]} when filtering,
        {"decision": "allow"} otherwise.
"""

from __future__ import annotations

import json
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        # Malformed payload: fail open, do not touch the tool list.
        print(json.dumps({"decision": "allow"}))
        return 0

    if not isinstance(payload, dict):
        print(json.dumps({"decision": "allow"}))
        return 0

    tools = payload.get("tools")
    if not payload.get("plan_mode") or not isinstance(tools, list):
        print(json.dumps({"decision": "allow"}))
        return 0

    read_only = payload.get("read_only_tools") or []
    allowed = {str(name) for name in read_only if isinstance(name, str)}
    allowed.add("ExitPlanMode")

    filtered = [
        tool for tool in tools
        if isinstance(tool, dict) and str(tool.get("name")) in allowed
    ]
    print(json.dumps({"decision": "rewrite", "tools": filtered}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
