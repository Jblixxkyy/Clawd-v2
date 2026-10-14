"""Task discovery and selection.

A task fixture directory looks like::

    tasks/t2/
        PROMPT.md        # prompt text; {WORKDIR} is substituted with the run dir
        template/        # optional; copied into the fresh workdir
        check.sh         # grader entry point: check.sh <run_dir> (exit 0 = pass)
        test_hidden.py   # fallback grader when no check.sh exists

The benchmark task fixtures themselves are never modified by the harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import default_tasks_dir


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    task_dir: Path
    prompt_path: Path
    template_dir: Path | None
    check_script: Path | None
    hidden_tests: Path | None


def discover_tasks(tasks_dir: Path | None = None) -> dict[str, TaskSpec]:
    """Scan a tasks dir and return {task_id: TaskSpec} for every valid fixture."""
    root = Path(tasks_dir) if tasks_dir else default_tasks_dir()
    specs: dict[str, TaskSpec] = {}
    if not root.is_dir():
        return specs
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith(("_", ".")):
            continue
        prompt = child / "PROMPT.md"
        if not prompt.is_file():
            continue
        template = child / "template"
        check = child / "check.sh"
        hidden = child / "test_hidden.py"
        specs[child.name] = TaskSpec(
            task_id=child.name,
            task_dir=child,
            prompt_path=prompt,
            template_dir=template if template.is_dir() else None,
            check_script=check if check.is_file() else None,
            hidden_tests=hidden if hidden.is_file() else None,
        )
    return specs


def select_tasks(selection: str, tasks_dir: Path | None = None) -> list[TaskSpec]:
    """Parse a task selection like 't1,t2' or 'all' into ordered TaskSpecs.

    Raises:
        ValueError: on unknown task ids (no partial runs).
    """
    available = discover_tasks(tasks_dir)
    sel = selection.strip().lower()
    if sel == "all":
        return [available[k] for k in sorted(available)]
    wanted = [t.strip() for t in selection.split(",") if t.strip()]
    if not wanted:
        raise ValueError("empty task selection")
    unknown = [t for t in wanted if t not in available]
    if unknown:
        raise ValueError(
            f"unknown task(s): {', '.join(unknown)} "
            f"(available: {', '.join(sorted(available)) or 'none'})"
        )
    return [available[t] for t in wanted]


def render_prompt(spec: TaskSpec, workdir: Path) -> str:
    """Substitute {WORKDIR} in the task prompt (mirrors run_clawd.sh)."""
    return spec.prompt_path.read_text(encoding="utf-8").replace("{WORKDIR}", str(workdir))
