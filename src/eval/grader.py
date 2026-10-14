"""Grader: runs a task's own checks against the run workdir -> pass/fail.

Grading is mechanical, never vibes:
- if check.sh exists: `bash check.sh <workdir>` (exit 0 = pass)
- elif test_hidden.py exists: `<sys.executable> -m pytest -q test_hidden.py`
  in the workdir (the harness's own interpreter, not PATH's python3)

Captured stdout/stderr is tail-truncated so the report stays readable.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import GRADER_TIMEOUT_S
from .tasks import TaskSpec

_MAX_LOG_CHARS = 6_000


def _tail(text: str, limit: int = _MAX_LOG_CHARS) -> str:
    if len(text) > limit:
        return "...[truncated]\n" + text[-limit:]
    return text


@dataclass(frozen=True)
class GradeResult:
    passed: bool
    method: str  # "check.sh" | "pytest" | "none"
    exit_code: int | None
    output: str


def grade_task(
    spec: TaskSpec,
    workdir: Path,
    timeout_s: float = GRADER_TIMEOUT_S,
) -> GradeResult:
    """Grade one completed run. The workdir is NOT modified by grading... in
    theory; check.sh files may write scratch files (e.g. stderr.txt) into it,
    same as the proven benchmark pattern."""
    workdir = Path(workdir)
    if spec.check_script is not None:
        try:
            proc = subprocess.run(
                ["bash", str(spec.check_script), str(workdir)],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = _tail((proc.stdout or "") + (proc.stderr or ""))
            return GradeResult(
                passed=proc.returncode == 0,
                method="check.sh",
                exit_code=proc.returncode,
                output=output,
            )
        except subprocess.TimeoutExpired as e:
            return GradeResult(
                passed=False,
                method="check.sh",
                exit_code=None,
                output=_tail(f"grader timed out after {timeout_s}s\n{(e.stdout or '')}{(e.stderr or '')}"),
            )
        except Exception as e:
            return GradeResult(
                passed=False, method="check.sh", exit_code=None,
                output=f"grader failed to run: {e}",
            )
    if spec.hidden_tests is not None:
        try:
            proc = subprocess.run(
                # sys.executable, not "python3": the grader must run the
                # hidden tests with the same interpreter as the harness
                # (PATH's python3 may not have pytest installed).
                [sys.executable, "-m", "pytest", "-q", str(spec.hidden_tests)],
                cwd=str(workdir),
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            output = _tail((proc.stdout or "") + (proc.stderr or ""))
            return GradeResult(
                passed=proc.returncode == 0,
                method="pytest",
                exit_code=proc.returncode,
                output=output,
            )
        except subprocess.TimeoutExpired as e:
            return GradeResult(
                passed=False,
                method="pytest",
                exit_code=None,
                output=_tail(f"grader timed out after {timeout_s}s\n{(e.stdout or '')}{(e.stderr or '')}"),
            )
        except Exception as e:
            return GradeResult(
                passed=False, method="pytest", exit_code=None,
                output=f"grader failed to run: {e}",
            )
    return GradeResult(
        passed=False,
        method="none",
        exit_code=None,
        output="no check.sh or test_hidden.py found for this task",
    )
