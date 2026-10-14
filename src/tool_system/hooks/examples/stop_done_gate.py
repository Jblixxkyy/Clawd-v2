#!/usr/bin/env python3
"""Reference implementation (brief 015): configurable checklist done-gate.

The default gate behind the ``Stop`` hook event. When the agent loop is
about to terminate via a terminal tool (``complete_task`` /
``ReportFailure`` / ``Exit``), the harness fires ``Stop`` and this hook
may veto the completion with ``{"decision": "block", "reason": ...}`` —
the loop then feeds the reason back as a continuation prompt and the
model keeps working instead of stopping.

Register in ``~/.clawd/hooks.json``::

    {"Stop": [{"command": "python3 stop_done_gate.py"}]}

Checklist config (``~/.clawd/done-gate.json`` — all optional)::

    {"enabled": false,               # master switch; default: OFF (fail-open allow)
     "test_command": "python3 -m pytest tests/ -q",  # run when enabled
     "test_timeout": 120,            # seconds for the test command
     "check_artifacts": true,        # verify complete_task "artifacts" exist
     "cwd": null}                    # working dir for checks (default: current dir)

Semantics:
  - ``enabled: false`` (or no config file at all) -> ``allow``. The gate
    never blocks a run just because no checklist was configured (fail-open).
  - Terminal *failure* exits (``ReportFailure`` / failed outcome) -> allow;
    gating a declared failure would just loop on it.
  - ``test_command`` runs only when enabled. Non-zero exit -> block with
    the failing command's tail output as the reason.
  - ``check_artifacts``: every path in the payload's ``artifacts`` list is
    resolved relative to ``cwd`` and must exist; a missing one blocks.
  - Any hook-side exception -> ``allow`` (fail-open; the harness also
    treats hook crashes/timeouts as allow).

Stdin payload: the loop's Stop payload — ``{"event": "Stop", "tool",
"outcome", "summary", "reason", "artifacts", "blocks_so_far",
"workspace_root", ...}``.
Stdout: ``{"decision": "block", "reason": ...}`` or ``{"decision": "allow"}``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_TEST_COMMAND = "python3 -m pytest tests/ -q"
DEFAULT_TEST_TIMEOUT = 120


def _config_path() -> Path:
    return Path.home() / ".clawd" / "done-gate.json"


def _load_config() -> dict:
    try:
        raw = _config_path().read_text(encoding="utf-8")
        config = json.loads(raw)
    except (FileNotFoundError, ValueError):
        return {}
    return config if isinstance(config, dict) else {}


def _allow() -> None:
    print(json.dumps({"decision": "allow"}))


def _block(reason: str) -> None:
    print(json.dumps({"decision": "block", "reason": reason}))


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        _allow()  # fail-open
        return 0
    if not isinstance(payload, dict):
        _allow()
        return 0

    config = _load_config()
    if not config.get("enabled", False):
        _allow()  # off-by-default: never block when no checklist configured
        return 0

    # A declared failure exit is not gated: blocking it would just loop on
    # a run the model already gave up on.
    outcome = str(payload.get("outcome") or "").lower()
    tool = str(payload.get("tool") or "")
    if outcome == "failed" or tool in ("ReportFailure",):
        _allow()
        return 0

    cwd = config.get("cwd") or os.getcwd()
    try:
        base = Path(cwd)
        if not base.is_dir():
            base = Path(os.getcwd())
    except Exception:
        base = Path(os.getcwd())

    # 1. Run the repo's configured test command.
    test_command = config.get("test_command") or DEFAULT_TEST_COMMAND
    test_timeout = config.get("test_timeout", DEFAULT_TEST_TIMEOUT)
    try:
        timeout_s = float(test_timeout)
    except (TypeError, ValueError):
        timeout_s = float(DEFAULT_TEST_TIMEOUT)
    try:
        proc = subprocess.run(
            test_command,
            shell=True,
            cwd=str(base),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        _block(f"done-gate: test command timed out after {timeout_s:g}s: {test_command}")
        return 0
    except Exception as exc:
        _allow()  # fail-open: the gate's own failure must not block a run
        return 0
    if proc.returncode != 0:
        tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
        tail_text = "\n".join(tail[-12:]) or "(no output)"
        _block(
            f"done-gate: test command failed (exit {proc.returncode}): "
            f"{test_command}\n{tail_text}"
        )
        return 0

    # 2. Verify claimed output files exist.
    if config.get("check_artifacts", True):
        artifacts = payload.get("artifacts") or []
        if isinstance(artifacts, list):
            for artifact in artifacts:
                # complete_task artifacts are {"name", "path", ...} dicts;
                # a bare string path is also accepted.
                path = artifact
                if isinstance(artifact, dict):
                    path = artifact.get("path") or artifact.get("name")
                if not isinstance(path, str) or not path:
                    continue
                if not (base / path).exists():
                    _block(f"done-gate: claimed artifact is missing: {path}")
                    return 0

    _allow()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        _allow()  # fail-open, never raise out of a hook
        sys.exit(0)
