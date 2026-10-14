"""Report: console table + summary.json for an eval run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_summary(
    run_id: str,
    provider_name: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    passed = sum(1 for r in results if r.get("passed"))
    total_in = 0
    total_out = 0
    for r in results:
        usage = r.get("usage") or {}
        total_in += int(usage.get("input_tokens", 0) or 0)
        total_out += int(usage.get("output_tokens", 0) or 0)
    return {
        "run_id": run_id,
        "provider": provider_name,
        "num_tasks": len(results),
        "num_passed": passed,
        "num_failed": len(results) - passed,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "results": [
            {
                "task_id": r.get("task_id"),
                "passed": r.get("passed"),
                "grade_method": r.get("grade_method"),
                "num_turns": r.get("num_turns"),
                "usage": r.get("usage"),
                "wall_time_s": r.get("wall_time_s"),
                "timed_out": r.get("timed_out"),
                "cancelled": r.get("cancelled"),
                "error": r.get("error"),
                "trace_path": r.get("trace_path"),
                "workdir": r.get("workdir"),
            }
            for r in results
        ],
    }


def _fmt_tokens(usage: Any) -> str:
    usage = usage or {}
    return f"{usage.get('input_tokens', 0) or 0}/{usage.get('output_tokens', 0) or 0}"


def print_report(summary: dict[str, Any], eval_dir: Path) -> None:
    line = "=" * 78
    print(line)
    print(f"eval {summary['run_id']}  provider={summary['provider']}")
    print(f"{summary['num_passed']}/{summary['num_tasks']} passed")
    print("-" * 78)
    header = f"{'task':<8}{'result':<8}{'turns':<7}{'in/out tok':<16}{'time':<10}trace"
    print(header)
    print("-" * 78)
    for r in summary["results"]:
        verdict = "PASS" if r["passed"] else "FAIL"
        extra = ""
        if r.get("timed_out"):
            extra = " [timeout]"
        if r.get("error"):
            extra = f" [error]"
        print(
            f"{r['task_id']:<8}{verdict:<8}{r.get('num_turns', 0):<7}"
            f"{_fmt_tokens(r.get('usage')):<16}{r.get('wall_time_s', 0):<10}"
            f"{r.get('trace_path')}{extra}"
        )
    print("-" * 78)
    print(f"total tokens in/out: {summary['total_input_tokens']}/{summary['total_output_tokens']}")
    print(f"summary: {eval_dir / 'summary.json'}")
    print(line)


def write_summary(eval_dir: Path, summary: dict[str, Any]) -> Path:
    path = Path(eval_dir) / "summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return path
