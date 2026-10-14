"""Runner: fresh workdir per task, headless agent run with timeout, trace hook.

Mirrors the proven benchmark pattern (run_clawd.sh) but in-process so the
trace recorder can hook every tool event and provider call:

- fresh workdir per task under <runs>/<timestamp>/<task>/work/
- template copied in, {WORKDIR} substituted into the prompt
- no_proxy sanitized (same vendored-httpx2 workaround as run_clawd.sh)
- provider -> Session -> tool registry -> ToolContext(workspace_root=workdir)
- permission policy identical to print mode: tools that "ask" are auto-denied,
  so eval runs are non-interactive and comparable to the benchmark baseline
- timeout via a cooperative should_stop flag (checked between turns/tools)
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agent import Session
from ..config import get_default_provider, get_provider_config
from ..providers import get_provider_class
from ..repl.print_mode import _deny_permission
from ..tool_system.agent_loop import run_agent_loop
from ..tool_system.context import ToolContext
from ..tool_system.defaults import build_default_registry
from .config import (
    DEFAULT_MAX_TURNS,
    DEFAULT_TIMEOUT_S,
    default_runs_dir,
    sanitized_no_proxy,
)
from .grader import GradeResult, grade_task
from .report import build_summary, print_report, write_summary
from .stub_provider import StubProvider
from .tasks import TaskSpec, render_prompt, select_tasks
from .trace import TraceRecorder


def _build_provider(provider_name: str):
    """Build a provider instance (same config flow as print mode)."""
    if provider_name == "stub":
        return StubProvider()
    name = provider_name or get_default_provider()
    try:
        config = get_provider_config(name)
    except Exception:
        raise RuntimeError(f"unknown provider '{name}'")
    if not config.get("api_key"):
        raise RuntimeError(
            f"API key not configured for provider '{name}'. "
            "Run `clawd login` to configure it."
        )
    provider_class = get_provider_class(name)
    return provider_class(
        api_key=config["api_key"],
        base_url=config.get("base_url"),
        model=config.get("default_model"),
    )


def _install_turn_spy(provider, recorder: TraceRecorder) -> None:
    """Wrap provider.chat so every provider call is recorded as a turn.

    Wraps the bound method on the instance (not the class), so
    isinstance-based provider dispatch in the agent loop is unaffected.
    """
    orig_chat = provider.chat

    def chat_spy(messages, tools=None, **kwargs):
        recorder.begin_turn()
        try:
            response = orig_chat(messages, tools=tools, **kwargs)
        finally:
            pass
        recorder.end_turn(getattr(response, "usage", None))
        return response

    provider.chat = chat_spy  # type: ignore[method-assign]


def run_task(
    spec: TaskSpec,
    *,
    provider_name: str,
    eval_dir: Path,
    run_id: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_turns: int = DEFAULT_MAX_TURNS,
) -> dict[str, Any]:
    """Run one task end-to-end: setup -> agent -> grade. Returns a result dict."""
    task_dir = Path(eval_dir) / spec.task_id
    workdir = task_dir / "work"
    trace_path = task_dir / "trace.jsonl"
    result_path = task_dir / "result.json"

    # Fresh workdir (rm -rf like run_clawd.sh), template copied in.
    shutil.rmtree(task_dir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    if spec.template_dir is not None:
        for item in spec.template_dir.iterdir():
            dest = workdir / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

    prompt = render_prompt(spec, workdir)
    (workdir / "_prompt_given.txt").write_text(prompt, encoding="utf-8")

    # Sanitize proxy env exactly like the benchmark script; restore afterwards
    # so the harness never leaks process-global env changes into its caller.
    _saved_env = {k: os.environ.get(k) for k in ("no_proxy", "NO_PROXY")}
    os.environ["no_proxy"] = sanitized_no_proxy()
    os.environ["NO_PROXY"] = sanitized_no_proxy()
    try:
        return _run_task_inner(
            spec, workdir=workdir, prompt=prompt, task_dir=task_dir,
            trace_path=trace_path, result_path=result_path,
            provider_name=provider_name, run_id=run_id,
            timeout_s=timeout_s, max_turns=max_turns,
        )
    finally:
        for k, v in _saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _run_task_inner(
    spec: TaskSpec,
    *,
    workdir: Path,
    prompt: str,
    task_dir: Path,
    trace_path: Path,
    result_path: Path,
    provider_name: str,
    run_id: str,
    timeout_s: float,
    max_turns: int,
) -> dict[str, Any]:
    """Agent run + grading for one task. Workdir is already prepared."""
    recorder = TraceRecorder(trace_path)
    wall_t0 = time.monotonic()
    timed_out = False
    cancelled = False
    error: str | None = None
    response_text: str | None = None
    usage: Any = None
    num_turns = 0

    provider = _build_provider(provider_name)
    model = getattr(provider, "model", None)
    recorder.start_run(
        run_id=run_id,
        task_id=spec.task_id,
        prompt=prompt,
        provider_name=provider_name,
        model=model,
        timeout_s=timeout_s,
        max_turns=max_turns,
    )
    _install_turn_spy(provider, recorder)

    session = Session.create(provider_name, model)
    session.conversation.add_user_message(prompt)
    tool_registry = build_default_registry()
    tool_context = ToolContext(workspace_root=workdir)
    tool_context.permission_handler = _deny_permission

    stop_event = threading.Event()
    watchdog = threading.Timer(timeout_s, stop_event.set)
    old_cwd = os.getcwd()
    try:
        os.chdir(workdir)
        watchdog.start()
        result = run_agent_loop(
            session.conversation,
            provider,
            tool_registry,
            tool_context,
            max_turns=max_turns,
            stream=False,
            on_event=recorder.on_event,
            should_stop=stop_event.is_set,
        )
        timed_out = stop_event.is_set()
        cancelled = bool(result.cancelled)
        response_text = result.response_text
        usage = result.usage
        num_turns = result.num_turns
    except Exception as e:  # harness-level failure, still grade what exists
        error = f"{type(e).__name__}: {e}"
    finally:
        watchdog.cancel()
        os.chdir(old_cwd)

    wall_time_s = time.monotonic() - wall_t0
    recorder.finish(
        response_text=response_text,
        usage=usage,
        num_turns=num_turns,
        wall_time_s=wall_time_s,
        timed_out=timed_out,
        cancelled=cancelled,
        error=error,
    )

    grade: GradeResult = grade_task(spec, workdir)

    record = {
        "run_id": run_id,
        "task_id": spec.task_id,
        "provider": provider_name,
        "model": model,
        "passed": grade.passed,
        "grade_method": grade.method,
        "grade_exit_code": grade.exit_code,
        "grade_output": grade.output,
        "num_turns": num_turns,
        "usage": usage,
        "wall_time_s": round(wall_time_s, 3),
        "timed_out": timed_out,
        "cancelled": cancelled,
        "error": error,
        "workdir": str(workdir),
        "trace_path": str(trace_path),
    }
    result_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return record


class TaskRunner:
    """Runs a selected task set into one timestamped eval directory."""

    def __init__(
        self,
        *,
        tasks_dir: Path | None = None,
        runs_dir: Path | None = None,
    ) -> None:
        self.tasks_dir = Path(tasks_dir) if tasks_dir else None
        self.runs_dir = Path(runs_dir) if runs_dir else default_runs_dir()

    def run(
        self,
        selection: str = "all",
        *,
        provider_name: str = "stub",
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> dict[str, Any]:
        specs = select_tasks(selection, self.tasks_dir)
        if not specs:
            raise RuntimeError("no tasks discovered; check the tasks directory")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        eval_dir = self.runs_dir / stamp
        eval_dir.mkdir(parents=True, exist_ok=True)
        run_id = f"eval-{stamp}"

        results: list[dict[str, Any]] = []
        for spec in specs:
            print(f"[eval] {run_id}: running {spec.task_id} (provider={provider_name}) ...")
            record = run_task(
                spec,
                provider_name=provider_name,
                eval_dir=eval_dir,
                run_id=run_id,
                timeout_s=timeout_s,
                max_turns=max_turns,
            )
            status = "PASS" if record["passed"] else "FAIL"
            print(f"[eval] {spec.task_id}: {status} "
                  f"(turns={record['num_turns']}, {record['wall_time_s']}s)")
            results.append(record)

        summary = build_summary(run_id, provider_name, results)
        write_summary(eval_dir, summary)
        print_report(summary, eval_dir)
        return summary
