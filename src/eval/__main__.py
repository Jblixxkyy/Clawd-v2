"""CLI entry: python3 -m src.eval

Subcommands:
    run     Run tasks:  python3 -m src.eval run --tasks t1,t2 [--timeout 570]
                        [--provider NAME] [--max-turns 20]
    list    List discovered tasks.

Examples:
    PYTHONPATH=~/workspace/Clawd-Code python3 -m src.eval run --tasks all --provider stub
    PYTHONPATH=~/workspace/Clawd-Code python3 -m src.eval run --tasks t2,t5 --provider nvidia --timeout 570
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DEFAULT_MAX_TURNS, DEFAULT_PROVIDER, DEFAULT_TIMEOUT_S
from .runner import TaskRunner
from .tasks import discover_tasks


def _cmd_list(args: argparse.Namespace) -> int:
    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else None
    specs = discover_tasks(tasks_dir)
    if not specs:
        print("no tasks found")
        return 1
    for task_id in sorted(specs):
        spec = specs[task_id]
        grader = "check.sh" if spec.check_script else ("pytest" if spec.hidden_tests else "none")
        tmpl = "template" if spec.template_dir else "no-template"
        print(f"{task_id}\tgrader={grader}\t{tmpl}\t{spec.prompt_path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    runner = TaskRunner(
        tasks_dir=Path(args.tasks_dir) if args.tasks_dir else None,
        runs_dir=Path(args.runs_dir) if args.runs_dir else None,
    )
    try:
        runner.run(
            args.tasks,
            provider_name=args.provider,
            timeout_s=args.timeout,
            max_turns=args.max_turns,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m src.eval",
        description="Eval harness: run the agent against benchmark tasks with trace recording.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List discovered benchmark tasks.")
    p_list.add_argument("--tasks-dir", default=None, help="Task fixtures dir (default: ~/workspace/clawd-benchmark/tasks).")

    p_run = sub.add_parser("run", help="Run selected tasks with trace recording.")
    p_run.add_argument("--tasks", default="all", help="Task selection: 't1,t2' or 'all' (default: all).")
    p_run.add_argument("--provider", default=DEFAULT_PROVIDER,
                       help="Provider name, or 'stub' for the no-credential smoke mode (default: stub).")
    p_run.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                       help=f"Per-task timeout in seconds (default: {DEFAULT_TIMEOUT_S}).")
    p_run.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS,
                       help=f"Max agent turns per task (default: {DEFAULT_MAX_TURNS}).")
    p_run.add_argument("--tasks-dir", default=None, help="Task fixtures dir override.")
    p_run.add_argument("--runs-dir", default=None, help="Eval runs dir override (default: ~/workspace/clawd-benchmark/eval-runs).")

    args = parser.parse_args(argv)
    if args.command == "list":
        return _cmd_list(args)
    if args.command == "run":
        return _cmd_run(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
