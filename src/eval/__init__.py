"""Eval harness with trace recording (roadmap B1).

Runs the agent headlessly against curated benchmark tasks, records a full
JSONL trace per run (prompt, model, tool calls, per-turn tokens, wall time,
final output), and grades pass/fail with each task's own checks — no vibes.

CLI: python3 -m src.eval run --tasks t1,t2 [--timeout 570] [--provider stub]
"""

from __future__ import annotations

from .runner import TaskRunner, run_task
from .tasks import TaskSpec, discover_tasks, select_tasks
from .trace import TraceRecorder
from .grader import GradeResult, grade_task
from .report import build_summary, print_report, write_summary

__all__ = [
    "TaskRunner",
    "run_task",
    "TaskSpec",
    "discover_tasks",
    "select_tasks",
    "TraceRecorder",
    "GradeResult",
    "grade_task",
    "build_summary",
    "print_report",
    "write_summary",
]
