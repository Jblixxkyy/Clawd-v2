"""Unit tests for the eval harness (src/eval/).

All agent runs are mocked (fake run_agent_loop / stub provider); no network,
no credentials, no real model calls. Covers trace recording, grading
pass/fail, report generation, task selection, and the runner plumbing
(fresh workdir, {WORKDIR} substitution, timeout flagging).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.eval import grader as grader_mod
from src.eval import runner as runner_mod
from src.eval import tasks as tasks_mod
from src.eval.grader import GradeResult, grade_task
from src.eval.report import build_summary, print_report, write_summary
from src.eval.stub_provider import StubProvider
from src.eval.tasks import TaskSpec, discover_tasks, render_prompt, select_tasks
from src.eval.trace import TraceRecorder, read_trace
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import AgentLoopResult, ToolEvent


# --- fixtures ------------------------------------------------------------


def _make_task_dir(root: Path, task_id: str, *, with_template: bool = True,
                   check_exit: int = 0) -> Path:
    """Create a minimal fake benchmark task fixture. Returns the task dir."""
    tdir = root / task_id
    tdir.mkdir(parents=True)
    (tdir / "PROMPT.md").write_text(
        f"Work in {{WORKDIR}} for task {task_id}.", encoding="utf-8"
    )
    if with_template:
        tmpl = tdir / "template"
        tmpl.mkdir()
        (tmpl / "seed.txt").write_text("seed", encoding="utf-8")
    (tdir / "check.sh").write_text(
        f"#!/bin/bash\nexit {check_exit}\n", encoding="utf-8"
    )
    return tdir


def _spec_from(tdir: Path) -> TaskSpec:
    return TaskSpec(
        task_id=tdir.name,
        task_dir=tdir,
        prompt_path=tdir / "PROMPT.md",
        template_dir=tdir / "template" if (tdir / "template").is_dir() else None,
        check_script=tdir / "check.sh" if (tdir / "check.sh").is_file() else None,
        hidden_tests=tdir / "test_hidden.py" if (tdir / "test_hidden.py").is_file() else None,
    )


def _fake_loop_factory(events: list[ToolEvent] | None = None,
                       *, spin_on_stop: bool = False):
    """Build a fake run_agent_loop honoring the real signature subset we use."""
    events = events or []

    def fake_loop(conversation, provider, tool_registry, tool_context,
                  max_turns=20, stream=False, on_event=None, should_stop=None, **kw):
        if spin_on_stop:
            # Simulate an agent that never finishes: exits only via should_stop.
            while not (should_stop and should_stop()):
                time.sleep(0.005)
            return AgentLoopResult(response_text="", usage=None, num_turns=0, cancelled=True)
        if on_event:
            for ev in events:
                on_event(ev)
        return AgentLoopResult(
            response_text="fake done",
            usage={"input_tokens": 10, "output_tokens": 5},
            num_turns=2,
            cancelled=False,
        )

    return fake_loop


# --- task discovery / selection ------------------------------------------


class TestTaskDiscovery:
    def test_discover_and_select(self, tmp_path: Path):
        _make_task_dir(tmp_path, "t1")
        _make_task_dir(tmp_path, "t2")
        specs = discover_tasks(tmp_path)
        assert set(specs) == {"t1", "t2"}

        picked = select_tasks("t2,t1", tmp_path)
        assert [s.task_id for s in picked] == ["t2", "t1"]

        all_picked = select_tasks("all", tmp_path)
        assert [s.task_id for s in all_picked] == ["t1", "t2"]

    def test_select_unknown_raises(self, tmp_path: Path):
        _make_task_dir(tmp_path, "t1")
        with pytest.raises(ValueError, match="unknown task"):
            select_tasks("t9", tmp_path)

    def test_select_empty_raises(self, tmp_path: Path):
        _make_task_dir(tmp_path, "t1")
        with pytest.raises(ValueError):
            select_tasks("  , ", tmp_path)

    def test_render_prompt_substitutes_workdir(self, tmp_path: Path):
        tdir = _make_task_dir(tmp_path, "t1")
        spec = _spec_from(tdir)
        workdir = tmp_path / "work"
        rendered = render_prompt(spec, workdir)
        assert "{WORKDIR}" not in rendered
        assert str(workdir) in rendered

    def test_empty_dir_discovers_nothing(self, tmp_path: Path):
        assert discover_tasks(tmp_path) == {}


# --- trace recorder -------------------------------------------------------


class TestTraceRecorder:
    def _record(self, tmp_path: Path) -> Path:
        trace_path = tmp_path / "trace.jsonl"
        rec = TraceRecorder(trace_path)
        rec.start_run(run_id="r1", task_id="t1", prompt="do things",
                      provider_name="stub", model="stub/smoke",
                      timeout_s=60, max_turns=20)
        rec.begin_turn()
        rec.on_event(ToolEvent(kind="tool_use", tool_name="Bash",
                               tool_input={"command": "echo hi"},
                               tool_use_id="u1"))
        rec.on_event(ToolEvent(kind="tool_result", tool_name="Bash",
                               tool_output={"exit_code": 0, "output": "hi"},
                               tool_use_id="u1", is_error=False))
        rec.on_event(ToolEvent(kind="tool_error", tool_name="Write",
                               tool_use_id="u2", is_error=True,
                               error="boom"))
        rec.end_turn({"input_tokens": 10, "output_tokens": 5})
        rec.finish(response_text="done",
                   usage={"input_tokens": 10, "output_tokens": 5},
                   num_turns=1, wall_time_s=0.5, timed_out=False)
        return trace_path

    def test_line_types_and_order(self, tmp_path: Path):
        events = read_trace(self._record(tmp_path))
        types = [e["type"] for e in events]
        assert types == ["run", "tool_use", "tool_result", "tool_error",
                         "turn", "result"]

    def test_header_and_result_fields(self, tmp_path: Path):
        events = read_trace(self._record(tmp_path))
        header = events[0]
        assert header["run_id"] == "r1"
        assert header["task_id"] == "t1"
        assert header["provider"] == "stub"
        assert header["model"] == "stub/smoke"
        result = events[-1]
        assert result["num_turns"] == 1
        assert result["wall_time_s"] == 0.5
        assert result["timed_out"] is False

    def test_tool_events_carry_turn_and_summaries(self, tmp_path: Path):
        events = read_trace(self._record(tmp_path))
        use = events[1]
        assert use["tool_name"] == "Bash"
        assert use["turn"] == 1
        assert "echo hi" in use["input_summary"]
        res = events[2]
        assert res["is_error"] is False
        assert "exit=0" in res["result_summary"]
        err = events[3]
        assert err["is_error"] is True

    def test_large_payloads_truncated(self, tmp_path: Path):
        trace_path = tmp_path / "trace.jsonl"
        rec = TraceRecorder(trace_path)
        rec.start_run(run_id="r", task_id="t", prompt="p", provider_name="s",
                      model="m", timeout_s=1, max_turns=1)
        rec.on_event(ToolEvent(kind="tool_result", tool_name="Read",
                               tool_output={"text": "x" * 50_000},
                               tool_use_id="u", is_error=False))
        rec.finish(response_text="y" * 50_000, usage={}, num_turns=0,
                   wall_time_s=0.0, timed_out=False)
        events = read_trace(trace_path)
        result_ev = [e for e in events if e["type"] == "tool_result"][0]
        assert "[truncated" in result_ev["output"]
        final = [e for e in events if e["type"] == "result"][0]
        assert "[truncated" in final["response_text"]

    def test_recorder_never_breaks_run(self, tmp_path: Path):
        # Closing twice / writing after close must not raise.
        rec = TraceRecorder(tmp_path / "t.jsonl")
        rec.close()
        rec.close()  # idempotent-ish: must not raise


# --- grader ----------------------------------------------------------------


class TestGrader:
    def _workdir(self, tmp_path: Path) -> Path:
        wd = tmp_path / "work"
        wd.mkdir(exist_ok=True)
        return wd

    def test_check_sh_pass(self, tmp_path: Path):
        tdir = _make_task_dir(tmp_path, "t1", check_exit=0)
        res = grade_task(_spec_from(tdir), self._workdir(tmp_path))
        assert isinstance(res, GradeResult)
        assert res.passed is True
        assert res.method == "check.sh"
        assert res.exit_code == 0

    def test_check_sh_fail(self, tmp_path: Path):
        tdir = _make_task_dir(tmp_path, "t1", check_exit=1)
        (tdir / "check.sh").write_text(
            "#!/bin/bash\necho 'FAIL: bad thing'\nexit 1\n", encoding="utf-8"
        )
        res = grade_task(_spec_from(tdir), self._workdir(tmp_path))
        assert res.passed is False
        assert "FAIL: bad thing" in res.output

    def test_check_sh_receives_workdir_arg(self, tmp_path: Path):
        tdir = _make_task_dir(tmp_path, "t1", check_exit=0)
        (tdir / "check.sh").write_text(
            '#!/bin/bash\n[ -n "$1" ] && [ -d "$1" ]\n', encoding="utf-8"
        )
        res = grade_task(_spec_from(tdir), self._workdir(tmp_path))
        assert res.passed is True

    def test_pytest_fallback(self, tmp_path: Path):
        tdir = tmp_path / "t9"
        tdir.mkdir()
        (tdir / "PROMPT.md").write_text("x", encoding="utf-8")
        (tdir / "test_hidden.py").write_text(
            "def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8"
        )
        spec = _spec_from(tdir)
        assert spec.check_script is None
        res = grade_task(spec, tdir)
        assert res.method == "pytest"
        assert res.passed is True

    def test_no_grader(self, tmp_path: Path):
        tdir = tmp_path / "t0"
        tdir.mkdir()
        (tdir / "PROMPT.md").write_text("x", encoding="utf-8")
        spec = _spec_from(tdir)
        res = grade_task(spec, tdir)
        assert res.method == "none"
        assert res.passed is False

    def test_grader_timeout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        import subprocess as sp

        tdir = _make_task_dir(tmp_path, "t1", check_exit=0)
        monkeypatch.setattr(
            grader_mod.subprocess, "run",
            lambda *a, **k: (_ for _ in ()).throw(sp.TimeoutExpired("bash", 1)),
        )
        res = grade_task(_spec_from(tdir), self._workdir(tmp_path), timeout_s=1)
        assert res.passed is False
        assert "timed out" in res.output


# --- stub provider ----------------------------------------------------------


class TestStubProvider:
    def test_canned_response(self):
        p = StubProvider()
        resp = p.chat([{"role": "user", "content": "hi"}])
        assert isinstance(resp, ChatResponse)
        assert resp.tool_uses is None
        assert resp.usage["input_tokens"] == 4
        assert "stub" in resp.content

    def test_models(self):
        assert StubProvider().get_available_models() == ["stub/smoke"]


# --- runner (mocked agent) --------------------------------------------------


class TestRunner:
    def test_full_plumbing_with_mocked_loop(self, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _make_task_dir(tasks_dir, "t1", check_exit=0)
        runs_dir = tmp_path / "eval-runs"

        events = [
            ToolEvent(kind="tool_use", tool_name="Bash",
                      tool_input={"command": "true"}, tool_use_id="u1"),
            ToolEvent(kind="tool_result", tool_name="Bash",
                      tool_output={"exit_code": 0}, tool_use_id="u1",
                      is_error=False),
        ]
        monkeypatch.setattr(
            runner_mod, "run_agent_loop", _fake_loop_factory(events)
        )

        spec = select_tasks("t1", tasks_dir)[0]
        eval_dir = runs_dir / "run1"
        eval_dir.mkdir(parents=True)
        record = runner_mod.run_task(
            spec, provider_name="stub", eval_dir=eval_dir,
            run_id="run1", timeout_s=60, max_turns=5,
        )

        # Fresh workdir, template copied, prompt substituted.
        workdir = Path(record["workdir"])
        assert (workdir / "seed.txt").is_file()
        given = (workdir / "_prompt_given.txt").read_text(encoding="utf-8")
        assert "{WORKDIR}" not in given and str(workdir) in given

        # Trace recorded the mocked tool calls.
        trace_events = read_trace(record["trace_path"])
        kinds = [e["type"] for e in trace_events]
        assert "run" in kinds and "tool_use" in kinds and "tool_result" in kinds
        assert "result" in kinds
        assert trace_events[0]["provider"] == "stub"

        # Grading + result.json.
        assert record["passed"] is True
        assert record["grade_method"] == "check.sh"
        assert record["num_turns"] == 2
        assert record["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert (eval_dir / "t1" / "result.json").is_file()

    def test_timeout_flagged(self, tmp_path: Path,
                             monkeypatch: pytest.MonkeyPatch):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _make_task_dir(tasks_dir, "t1", check_exit=0)
        monkeypatch.setattr(
            runner_mod, "run_agent_loop", _fake_loop_factory(spin_on_stop=True)
        )
        spec = select_tasks("t1", tasks_dir)[0]
        eval_dir = tmp_path / "eval-runs" / "run2"
        eval_dir.mkdir(parents=True)
        record = runner_mod.run_task(
            spec, provider_name="stub", eval_dir=eval_dir,
            run_id="run2", timeout_s=0.05, max_turns=5,
        )
        assert record["timed_out"] is True
        assert record["cancelled"] is True
        trace_events = read_trace(record["trace_path"])
        assert trace_events[-1]["timed_out"] is True

    def test_task_runner_end_to_end(self, tmp_path: Path,
                                    monkeypatch: pytest.MonkeyPatch,
                                    capsys: pytest.CaptureFixture):
        tasks_dir = tmp_path / "tasks"
        tasks_dir.mkdir()
        _make_task_dir(tasks_dir, "t1", check_exit=0)
        _make_task_dir(tasks_dir, "t2", check_exit=1)
        runs_dir = tmp_path / "eval-runs"
        monkeypatch.setattr(runner_mod, "run_agent_loop", _fake_loop_factory())

        runner = runner_mod.TaskRunner(tasks_dir=tasks_dir, runs_dir=runs_dir)
        summary = runner.run("t1,t2", provider_name="stub", timeout_s=60)
        assert summary["num_tasks"] == 2
        assert summary["num_passed"] == 1
        assert summary["num_failed"] == 1

        out = capsys.readouterr().out
        assert "t1" in out and "t2" in out
        assert "PASS" in out and "FAIL" in out
        eval_subdirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        assert len(eval_subdirs) == 1
        summary_path = eval_subdirs[0] / "summary.json"
        assert summary_path.is_file()
        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
        assert loaded["num_passed"] == 1 and loaded["num_failed"] == 1


# --- report -----------------------------------------------------------------


def _fake_results() -> list[dict]:
    return [
        {"task_id": "t1", "passed": True, "grade_method": "check.sh",
         "num_turns": 3, "usage": {"input_tokens": 100, "output_tokens": 50},
         "wall_time_s": 1.5, "timed_out": False, "cancelled": False,
         "error": None, "trace_path": "/tmp/t1/trace.jsonl",
         "workdir": "/tmp/t1/work"},
        {"task_id": "t2", "passed": False, "grade_method": "check.sh",
         "num_turns": 5, "usage": {"input_tokens": 200, "output_tokens": 60},
         "wall_time_s": 9.9, "timed_out": True, "cancelled": True,
         "error": None, "trace_path": "/tmp/t2/trace.jsonl",
         "workdir": "/tmp/t2/work"},
    ]


class TestReport:
    def test_build_summary_counts(self):
        s = build_summary("eval-x", "stub", _fake_results())
        assert s["num_tasks"] == 2
        assert s["num_passed"] == 1
        assert s["num_failed"] == 1
        assert s["total_input_tokens"] == 300
        assert s["total_output_tokens"] == 110

    def test_print_report_table(self, capsys: pytest.CaptureFixture):
        s = build_summary("eval-x", "stub", _fake_results())
        print_report(s, Path("/tmp/eval-x"))
        out = capsys.readouterr().out
        assert "eval-x" in out
        assert "t1" in out and "PASS" in out
        assert "t2" in out and "FAIL" in out
        assert "summary.json" in out

    def test_write_summary_json(self, tmp_path: Path):
        s = build_summary("eval-x", "stub", _fake_results())
        path = write_summary(tmp_path, s)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["run_id"] == "eval-x"
        assert loaded["results"][0]["trace_path"] == "/tmp/t1/trace.jsonl"
