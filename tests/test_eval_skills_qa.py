"""Tests for the t6-skills-qa eval task (brief 004).

- Unit tests for src.eval.skills_qa.grade_skills_answer.
- The REAL check.sh fixture graded end-to-end against fabricated traces.
- Full run_task plumbing with a mocked agent loop: a canned correct answer
  passes, the documented 30B failure text fails.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
from pathlib import Path

import pytest

from src.eval import runner as runner_mod
from src.eval.skills_qa import BLOCKLISTED_GENERICS, grade_skills_answer
from src.eval.tasks import select_tasks
from src.tool_system.agent_loop import AgentLoopResult

REPO = Path("/home/hatch/workspace/Clawd-Code")
T6_FIXTURE = Path("/home/hatch/workspace/clawd-benchmark/tasks/t6-skills-qa")

PASSING_ANSWER = (
    "I have 6 skills:\n"
    "- /eval-alpha: Eval fixture skill alpha\n"
    "- /eval-beta: Eval fixture skill beta\n"
    "- /haiku: Write a haiku about any topic\n"
    "- /joke: Tell a short programming joke\n"
    "- /pdf: Handle PDF file operations\n"
    "- /skill-creator: Create or update a skill\n"
)

# The documented Oct 11 30B failure mode: inventory provably injected, model
# answered generic priors instead.
FAILING_ANSWER_30B = (
    "I have many engineering skills, including technical writing, "
    "thermodynamics, code review, and debugging."
)


class TestGradeSkillsAnswer(unittest.TestCase):
    def test_pass_names_inventory(self):
        ok, reason = grade_skills_answer(PASSING_ANSWER, ["eval-alpha", "eval-beta"])
        self.assertTrue(ok, reason)
        self.assertIn("PASS", "PASS")  # reason is human-readable

    def test_fail_missing_skill(self):
        ok, reason = grade_skills_answer("I have /eval-alpha.", ["eval-alpha", "eval-beta"])
        self.assertFalse(ok)
        self.assertIn("eval-beta", reason)

    def test_fail_blocklisted_generic(self):
        ok, reason = grade_skills_answer(FAILING_ANSWER_30B, ["eval-alpha", "eval-beta"])
        self.assertFalse(ok)
        # Missing names are reported first; a pure-invention answer with the
        # names present still fails on the blocklist.
        ok2, reason2 = grade_skills_answer(
            "I have /eval-alpha and /eval-beta, plus thermodynamics.",
            ["eval-alpha", "eval-beta"],
        )
        self.assertFalse(ok2)
        self.assertIn("thermodynamics", reason2)

    def test_case_insensitive_and_slash_optional(self):
        ok, _ = grade_skills_answer("I have EVAL-ALPHA and /Eval-Beta.", ["eval-alpha", "eval-beta"])
        self.assertTrue(ok)

    def test_none_and_empty_fail_closed(self):
        ok, _ = grade_skills_answer(None, ["eval-alpha"])
        self.assertFalse(ok)
        ok, _ = grade_skills_answer("", ["eval-alpha"])
        self.assertFalse(ok)

    def test_never_raises(self):
        ok, reason = grade_skills_answer(object(), ["eval-alpha"])  # type: ignore[arg-type]
        self.assertIsInstance(ok, bool)
        self.assertIsInstance(reason, str)

    def test_blocklist_documents_oct11_signature(self):
        self.assertIn("thermodynamics", BLOCKLISTED_GENERICS)
        self.assertIn("technical writing", BLOCKLISTED_GENERICS)


def _write_trace(task_dir: Path, response_text: str | None) -> Path:
    """Fabricate <task_dir>/trace.jsonl with a result event (or none)."""
    workdir = task_dir / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    trace = task_dir / "trace.jsonl"
    lines = [{"type": "run", "task_id": "t6-skills-qa"}]
    if response_text is not None:
        lines.append({"type": "result", "response_text": response_text})
    trace.write_text("\n".join(json.dumps(e) for e in lines) + "\n", encoding="utf-8")
    return workdir


class TestCheckShFixture(unittest.TestCase):
    """Grade with the REAL shipped check.sh against fabricated traces."""

    def setUp(self):
        self.tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "t6test"
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True)
        self.env = {
            "PYTHONPATH": str(REPO),
            "CLAWD_REPO": str(REPO),
            "PATH": os.environ["PATH"],
            "HOME": os.environ.get("HOME", "/home/hatch"),
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _grade(self, task_dir: Path) -> subprocess.CompletedProcess:
        workdir = task_dir / "work"
        return subprocess.run(
            ["bash", str(T6_FIXTURE / "check.sh"), str(workdir)],
            capture_output=True, text=True, timeout=60, env=self.env,
        )

    def test_passing_answer(self):
        task_dir = self.tmp / "pass"
        _write_trace(task_dir, PASSING_ANSWER)
        proc = self._grade(task_dir)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("PASS", proc.stdout)

    def test_30b_failure_answer(self):
        task_dir = self.tmp / "fail"
        _write_trace(task_dir, FAILING_ANSWER_30B)
        proc = self._grade(task_dir)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("FAIL", proc.stdout)

    def test_missing_trace_fails(self):
        task_dir = self.tmp / "notrace"
        (task_dir / "work").mkdir(parents=True)
        proc = self._grade(task_dir)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("FAIL", proc.stdout)


def _fake_loop_with_answer(answer: str):
    def fake_loop(conversation, provider, tool_registry, tool_context,
                  max_turns=20, stream=False, on_event=None, should_stop=None, **kw):
        return AgentLoopResult(
            response_text=answer,
            usage={"input_tokens": 10, "output_tokens": 5},
            num_turns=1,
        )
    return fake_loop


class TestSkillsQaEndToEnd(unittest.TestCase):
    """Full run_task plumbing against the real t6 fixture."""

    def setUp(self):
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.tasks_dir = self.root / "tasks"
        self.tasks_dir.mkdir()
        shutil.copytree(T6_FIXTURE, self.tasks_dir / "t6-skills-qa")
        self.runs_dir = self.root / "eval-runs"
        self._patchers = []

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        self._tmp.cleanup()

    def _run_with_answer(self, answer: str) -> dict:
        import unittest.mock as mock

        p1 = mock.patch.object(runner_mod, "run_agent_loop", _fake_loop_with_answer(answer))
        p1.start()
        self._patchers.append(p1)
        p2 = mock.patch.dict(os.environ, {
            "PYTHONPATH": str(REPO),
            "CLAWD_REPO": str(REPO),
        })
        p2.start()
        self._patchers.append(p2)

        spec = select_tasks("t6-skills-qa", self.tasks_dir)[0]
        # Template fixture skills copied into the fresh workdir.
        self.assertTrue((spec.template_dir / ".clawd" / "skills" / "eval-alpha" / "SKILL.md").is_file())
        eval_dir = self.runs_dir / "run1"
        eval_dir.mkdir(parents=True)
        return runner_mod.run_task(
            spec, provider_name="stub", eval_dir=eval_dir,
            run_id="run1", timeout_s=60, max_turns=5,
        )

    def test_correct_answer_passes(self):
        record = self._run_with_answer(PASSING_ANSWER)
        self.assertTrue(record["passed"], record["grade_output"])
        self.assertEqual(record["grade_method"], "check.sh")

    def test_30b_failure_answer_fails(self):
        record = self._run_with_answer(FAILING_ANSWER_30B)
        self.assertFalse(record["passed"])
        self.assertEqual(record["grade_method"], "check.sh")


if __name__ == "__main__":
    unittest.main()
