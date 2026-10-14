"""Eval harness configuration: directories, defaults, env overrides."""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_TIMEOUT_S = 570
DEFAULT_MAX_TURNS = 20
DEFAULT_PROVIDER = "stub"
GRADER_TIMEOUT_S = 120


def default_tasks_dir() -> Path:
    """Directory holding the benchmark task fixtures (PROMPT.md, template/, check.sh)."""
    override = os.environ.get("CLAWD_EVAL_TASKS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "workspace" / "clawd-benchmark" / "tasks"


def default_runs_dir() -> Path:
    """Directory under which eval runs (traces, workdirs, summaries) are stored.

    Kept OUTSIDE the repo so the source tree stays clean.
    """
    override = os.environ.get("CLAWD_EVAL_RUNS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / "workspace" / "clawd-benchmark" / "eval-runs"


def sanitized_no_proxy() -> str:
    """Value matching the proven benchmark run pattern (vendored httpx2 no_proxy bug)."""
    return "localhost,127.0.0.1"
