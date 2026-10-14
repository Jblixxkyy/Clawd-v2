#!/usr/bin/env python3
"""PostToolUse hook recipe: verify Python files after Edit/Write.

Reads the hook payload JSON from stdin:
    {"event": "PostToolUse", "tool": "Edit", "input": {"file_path": "..."}, "output": {...}}

For Edit/Write calls that touched a ``.py`` file, runs:
    - ``ruff check <file>`` on any Python file
    - ``pytest -x -q <file>`` when the file looks like a test module
      (``test_*.py`` / ``*_test.py``)

Each check is capped at ~60s. Results are appended to
``~/.clawd/verify-log.jsonl`` and printed to stdout (handy when run
manually; the hook runner ignores PostToolUse output anyway).

Fail-open by construction: this script ALWAYS exits 0. A missing ruff /
pytest, a timeout, or any internal error is recorded as "skipped" and the
tool call proceeds. PostToolUse hooks can never block a tool call.

Wire it up in ~/.clawd/hooks.json (see verify-hooks.json next to this file):
    {"PostToolUse": [{"command": "python3 /path/to/verify-after-edit.py"}]}

NOTE on the 10s runner budget: the hook runner kills hook commands after
10s by default, so keep per-file checks fast (ruff on one file is ms; a
single test file is usually <10s). For full-suite runs, background them
(e.g. ``nohup pytest -q &``) instead of blocking the hook.

FOLLOW-UP (not built): wire post-edit LSP diagnostics here for in-turn
compiler errors — see src/tool_system/tools/lsp.py. An LSP-backed check
would surface type errors the moment an edit lands, without shelling out.
"""

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VERIFY_TOOLS = ("Edit", "Write")
CHECK_TIMEOUT_S = 60
LOG_PATH = Path.home() / ".clawd" / "verify-log.jsonl"


def _is_test_file(path: str) -> bool:
    name = Path(path).name
    return name.startswith("test_") or name.endswith("_test.py")


def _run_check(exe: str, args: list[str]) -> dict:
    """Run one check; never raises — failures become a result record."""
    try:
        proc = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            timeout=CHECK_TIMEOUT_S,
        )
    except FileNotFoundError:
        return {"check": args[0], "status": "skipped", "reason": f"{exe} not installed"}
    except subprocess.TimeoutExpired:
        return {"check": args[0], "status": "skipped", "reason": f"timed out after {CHECK_TIMEOUT_S}s"}
    except Exception as exc:  # fail open: anything else is a skip, not a crash
        return {"check": args[0], "status": "skipped", "reason": str(exc)}
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return {
        "check": " ".join([exe, *args]),
        "status": "pass" if proc.returncode == 0 else "fail",
        "exit_code": proc.returncode,
        "output": "\n".join(tail[-15:]),  # last 15 lines only
    }


def _log(record: dict) -> None:
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:
        pass  # logging must never break the tool call


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # unreadable input -> fail open

    if payload.get("event") != "PostToolUse":
        return 0
    if payload.get("tool") not in VERIFY_TOOLS:
        return 0
    target = (payload.get("input") or {}).get("file_path")
    if not isinstance(target, str) or not target.endswith(".py"):
        return 0  # only Python files get verified
    if not Path(target).is_file():
        return 0  # deleted/renamed after the edit -> nothing to check

    record: dict = {
        "time": datetime.now(timezone.utc).isoformat(),
        "tool": payload.get("tool"),
        "file": target,
        "checks": [],
    }

    ruff = shutil.which("ruff")
    if ruff:
        record["checks"].append(_run_check(ruff, ["check", target]))
    else:
        record["checks"].append({"check": "ruff check", "status": "skipped", "reason": "ruff not installed"})

    pytest = shutil.which("pytest")
    if _is_test_file(target):
        if pytest:
            record["checks"].append(_run_check(pytest, ["-x", "-q", target]))
        else:
            record["checks"].append({"check": "pytest", "status": "skipped", "reason": "pytest not installed"})
    else:
        record["checks"].append({"check": "pytest", "status": "skipped", "reason": "not a test file"})

    _log(record)

    summary = ", ".join(f"{c['check'].split()[0]}={c['status']}" for c in record["checks"])
    print(f"[verify-after-edit] {target}: {summary}")
    return 0  # always: fail-open


if __name__ == "__main__":
    raise SystemExit(main())
