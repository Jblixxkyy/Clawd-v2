"""Tests for brief 018: reviewer loop (/review) + judge-driven goal loop.

Uses stub reviewer/judge runners — no network, no real LLM calls.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

from src.history import HistoryLog
from src.judge import (
    CriticConfig,
    GoalVerdict,
    JudgeController,
    build_followup_prompt,
    parse_goal_verdict,
    render_transcript,
    resolve_judge_config,
    should_judge_run,
)
from src.review import (
    ReviewVerdict,
    ReviewerSession,
    build_review_prompt,
    extract_json_blob,
    format_verdict_for_worker,
    make_verdict_event,
    parse_review_args,
    parse_review_verdict,
    resolve_review_model,
    snapshot_diff,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeConversation:
    """Minimal conversation: add_user_message + messages."""

    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    def add_user_message(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


def make_runner(responses: list[str]):
    """Stub reviewer/judge runner: returns scripted responses in order."""
    calls: list[tuple[str, Optional[str]]] = []

    def runner(prompt: str, model: Optional[str]) -> str:
        calls.append((prompt, model))
        if responses:
            return responses.pop(0)
        return '{"verdict": "pass", "confidence": 1.0, "findings": []}'

    runner.calls = calls  # type: ignore[attr-defined]
    return runner


def history_events_json(history: HistoryLog) -> list[dict[str, Any]]:
    return [json.loads(e.detail) for e in history.events]


# ---------------------------------------------------------------------------
# /review: clean diff -> pass, no iteration
# ---------------------------------------------------------------------------


class TestReviewerCleanDiff(unittest.TestCase):
    def test_clean_diff_pass_no_iteration(self):
        diff = "diff --git a/foo.py b/foo.py\n+print('ok')\n"
        runner = make_runner(['{"verdict": "pass", "confidence": 0.95, "findings": []}'])
        history = HistoryLog()
        convo = FakeConversation()
        session = ReviewerSession(
            reviewer_runner=runner, model="cheap-model", history=history
        )
        verdict = session.run(get_diff=lambda: diff, conversation=convo)

        self.assertEqual(verdict.status, "pass")
        self.assertAlmostEqual(verdict.confidence, 0.95)
        self.assertEqual(verdict.findings_count, 0)
        # reviewer spawned exactly once — no iteration on a pass
        self.assertEqual(len(runner.calls), 1)  # type: ignore[attr-defined]
        # the reviewer prompt carries the diff + rubric
        self.assertIn("foo.py", runner.calls[0][0])  # type: ignore[attr-defined]
        self.assertIn("P0", runner.calls[0][0])  # type: ignore[attr-defined]
        # verdict injected as a user-role message
        self.assertEqual(len(convo.messages), 1)
        self.assertEqual(convo.messages[0]["role"], "user")
        self.assertIn('verdict="pass"', convo.messages[0]["content"])


# ---------------------------------------------------------------------------
# /review: injected bug -> P0/P1 with correct file range, worker gets verdict
# ---------------------------------------------------------------------------

BUGGY_DIFF = """diff --git a/src/bug.py b/src/bug.py
index 111..222 100644
--- a/src/bug.py
+++ b/src/bug.py
@@ -8,6 +8,9 @@ def run(cmd):
     import subprocess
+    # TODO: sanitize
+    subprocess.call(cmd, shell=True)
+    return True
     return subprocess.check_output(cmd)
"""


class TestReviewerFindsBug(unittest.TestCase):
    def test_injected_bug_p0_correct_range_worker_iterates(self):
        fail_json = json.dumps({
            "verdict": "fail",
            "confidence": 0.9,
            "findings": [{
                "severity": "P0",
                "file": "src/bug.py",
                "start_line": 10,
                "end_line": 12,
                "description": "shell=True with unsanitized input: command injection",
            }],
            "patch": "--- a/src/bug.py\n+++ b/src/bug.py\n@@ -10,3 +10,3 @@\n-    subprocess.call(cmd, shell=True)\n+    subprocess.call(shlex.split(cmd))\n",
        })
        pass_json = '{"verdict": "pass", "confidence": 0.99, "findings": []}'
        runner = make_runner([fail_json, pass_json])
        history = HistoryLog()
        convo = FakeConversation()

        diffs = [BUGGY_DIFF, "+print('fixed')\n"]  # worker "fixes" between passes
        session = ReviewerSession(reviewer_runner=runner, model="cheap-model", history=history)
        verdict = session.run(get_diff=lambda: diffs.pop(0) if diffs else "", conversation=convo)

        self.assertEqual(verdict.status, "pass")  # second pass clean
        self.assertEqual(len(runner.calls), 2)  # type: ignore[attr-defined]

        # worker received the fail verdict as a user message on pass 1
        first_msg = convo.messages[0]["content"]
        self.assertIn('verdict="fail"', first_msg)
        self.assertIn("P0", first_msg)
        self.assertIn("src/bug.py:10-12", first_msg)
        self.assertIn("command injection", first_msg)

        # the first verdict was parsed with the correct file range
        events = history_events_json(history)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["kind"], "review_verdict")
        self.assertEqual(events[0]["verdict"], "fail")
        self.assertEqual(events[0]["findings_count"], 1)
        self.assertTrue(events[0]["patch_present"])

    def test_parse_infers_verdict_from_findings(self):
        # no verdict field: P1 finding -> fail
        verdict = parse_review_verdict(
            '{"findings": [{"severity": "P1", "file": "a.py", "description": "x"}]}'
        )
        self.assertEqual(verdict.status, "fail")
        self.assertEqual(verdict.findings[0].severity, "P1")


# ---------------------------------------------------------------------------
# Malformed reviewer JSON (prose-wrapped) -> parsed tolerantly
# ---------------------------------------------------------------------------


class TestTolerantParsing(unittest.TestCase):
    def test_prose_wrapped_json_parses(self):
        raw = (
            "Here is my review of the diff.\n\n"
            "```json\n"
            '{"verdict": "pass", "confidence": 0.8, "findings": []}\n'
            "```\n\n"
            "Hope this helps — the code looks fine to me."
        )
        verdict = parse_review_verdict(raw, model="cheap")
        self.assertEqual(verdict.status, "pass")
        self.assertAlmostEqual(verdict.confidence, 0.8)
        self.assertEqual(verdict.findings_count, 0)
        self.assertEqual(verdict.model, "cheap")

    def test_extract_json_blob_finds_balanced_object(self):
        blob = extract_json_blob('noise {"a": {"b": 1}} trailing {"x": 2}')
        self.assertEqual(json.loads(blob or ""), {"a": {"b": 1}})

    def test_unparseable_output_never_silently_passes(self):
        verdict = parse_review_verdict("I have no idea what this diff does.")
        self.assertEqual(verdict.status, "fail")

    def test_patch_fence_outside_json_captured(self):
        raw = (
            '{"verdict": "fail", "findings": [{"severity": "P2", "file": "a.py", '
            '"description": "nit"}], "confidence": 0.6}\n'
            "Suggested fix:\n```diff\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+y\n```"
        )
        verdict = parse_review_verdict(raw)
        self.assertEqual(verdict.status, "fail")
        self.assertIsNotNone(verdict.patch)
        self.assertIn("-x", verdict.patch or "")


# ---------------------------------------------------------------------------
# Verdict events in the run log
# ---------------------------------------------------------------------------


class TestVerdictEvents(unittest.TestCase):
    def test_review_event_shape(self):
        event = make_verdict_event(
            "review_verdict", "fail", "cheap-model", 0.9,
            findings_count=2, iteration=1,
        )
        self.assertEqual(event["kind"], "review_verdict")
        self.assertEqual(event["model"], "cheap-model")
        self.assertAlmostEqual(event["confidence"], 0.9)
        self.assertEqual(event["findings_count"], 2)
        self.assertEqual(event["iteration"], 1)

    def test_reviewer_session_logs_events_with_model_and_confidence(self):
        runner = make_runner(['{"verdict": "fail", "confidence": 0.7, '
                              '"findings": [{"severity": "P2", "file": "a.py", "description": "nit"}]}'])
        history = HistoryLog()
        session = ReviewerSession(
            reviewer_runner=runner, model="cheap-model", history=history, max_iterations=1
        )
        session.run(get_diff=lambda: "+x\n", conversation=FakeConversation())
        events = history_events_json(history)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["model"], "cheap-model")
        self.assertAlmostEqual(events[0]["confidence"], 0.7)
        self.assertEqual(events[0]["findings_count"], 1)


# ---------------------------------------------------------------------------
# Judge loop: early stop -> missing items -> resume
# ---------------------------------------------------------------------------


class TestJudgeLoop(unittest.TestCase):
    def test_missing_items_resume_then_complete(self):
        messages = [
            {"role": "user", "content": "write hello.py and test it"},
            {"role": "assistant", "content": "I wrote hello.py"},
        ]
        worker_prompts: list[str] = []

        def worker(followup: str) -> None:
            worker_prompts.append(followup)
            messages.append({"role": "assistant", "content": "I wrote the tests too"})

        judge_responses = [
            '{"complete": false, "confidence": 0.9, "missing": ["no tests written"]}',
            '{"complete": true, "confidence": 0.95, "missing": []}',
        ]
        runner = make_runner(judge_responses)
        history = HistoryLog()
        controller = JudgeController(
            judge_runner=runner,
            config={"judge": {"enabled": True}},
            history=history,
        )
        verdict = controller.run(get_messages=lambda: messages, worker=worker)

        self.assertTrue(verdict.complete)
        self.assertEqual(verdict.missing, [])
        self.assertEqual(len(runner.calls), 2)  # type: ignore[attr-defined]
        # transcript rendered as role: text, no system prompt
        first_transcript = runner.calls[0][0]  # type: ignore[attr-defined]
        self.assertIn("user: write hello.py", first_transcript)
        self.assertIn("assistant: I wrote hello.py", first_transcript)
        # worker got the "what's missing" follow-up
        self.assertEqual(len(worker_prompts), 1)
        self.assertIn("no tests written", worker_prompts[0])
        # judge events logged
        events = history_events_json(history)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["kind"], "judge_verdict")
        self.assertFalse(events[0]["verdict"])
        self.assertTrue(events[1]["verdict"])

    def test_third_iteration_incomplete_stops_and_reports(self):
        messages = [{"role": "user", "content": "do the thing"}]
        resumes = 0

        def worker(followup: str) -> None:
            nonlocal resumes
            resumes += 1
            messages.append({"role": "assistant", "content": "tried again"})

        runner = make_runner(
            ['{"complete": false, "missing": ["still broken"], "confidence": 0.6}'] * 10
        )
        controller = JudgeController(
            judge_runner=runner,
            config={"judge": {"enabled": True, "max_iterations": 3}},
            history=HistoryLog(),
        )
        verdict = controller.run(get_messages=lambda: messages, worker=worker)

        self.assertFalse(verdict.complete)
        self.assertEqual(verdict.missing, ["still broken"])
        # initial judge + 3 resumes = 4 judge calls, then stops
        self.assertEqual(len(runner.calls), 4)  # type: ignore[attr-defined]
        self.assertEqual(resumes, 3)

    def test_system_prompt_excluded_from_transcript(self):
        messages = [
            {"role": "system", "content": "SECRET SYSTEM PROMPT"},
            {"role": "user", "content": "hi"},
        ]
        text = render_transcript(messages)
        self.assertNotIn("SECRET SYSTEM PROMPT", text)
        self.assertIn("user: hi", text)

    def test_think_content_read_defensively(self):
        # dict-style think block (023 shape) is included without hard-requiring it
        messages = [
            {"role": "assistant", "content": [
                {"type": "think", "text": "plan: first read, then edit"},
                {"type": "text", "text": "done"},
            ]},
        ]
        text = render_transcript(messages)
        self.assertIn("plan: first read, then edit", text)
        # object-style think via getattr fallback
        class Thinkish:
            def __init__(self) -> None:
                self.thought = "considering edge cases"
        messages2 = [{"role": "assistant", "content": [Thinkish()]}]
        text2 = render_transcript(messages2)
        self.assertIn("considering edge cases", text2)
        # and plain messages without think still render
        self.assertIn("user: hello", render_transcript([{"role": "user", "content": "hello"}]))

    def test_goal_verdict_field_inference(self):
        v = parse_goal_verdict('{"missing": ["a", "b"]}')
        self.assertFalse(v.complete)
        self.assertEqual(v.missing, ["a", "b"])
        v2 = parse_goal_verdict('{"complete": true}')
        self.assertTrue(v2.complete)
        self.assertEqual(v2.missing, [])


# ---------------------------------------------------------------------------
# /review command registration + helpers
# ---------------------------------------------------------------------------


class TestReviewCommand(unittest.TestCase):
    def test_command_registered(self):
        from src.command_system.builtins import (
            REVIEW_COMMAND,
            get_builtin_commands,
            review_command_call,
        )

        names = [c.name for c in get_builtin_commands()]
        self.assertIn("review", names)
        self.assertEqual(REVIEW_COMMAND.name, "review")
        self.assertTrue(callable(review_command_call))

    def test_command_no_changes(self):
        from src.command_system.builtins import review_command_call
        from src.command_system.engine import create_command_context

        with tempfile.TemporaryDirectory() as tmp:
            ctx = create_command_context(
                workspace_root=tmp,
                conversation=FakeConversation(),
                history=HistoryLog(),
                config={},
            )
            result = review_command_call("", ctx)
            self.assertIn("No uncommitted changes", result.value)

    def test_command_fallback_emits_prompt_without_runner(self):
        from src.command_system.builtins import review_command_call
        from src.command_system.engine import create_command_context

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("x")
            ctx = create_command_context(
                workspace_root=tmp,
                conversation=FakeConversation(),
                history=HistoryLog(),
                config={},
            )
            # monkeypatch snapshot_diff at the import site (src.review)
            import src.review as review_mod

            orig = review_mod.snapshot_diff
            try:
                review_mod.snapshot_diff = lambda _root=None: "+x\n"
                result = review_command_call("--model cheap-x", ctx)
            finally:
                review_mod.snapshot_diff = orig
            self.assertIn("Subagent", result.value)
            self.assertIn("cheap-x", result.value)
            self.assertIn("P0", result.value)

    def test_command_with_runner(self):
        from src.command_system.builtins import review_command_call
        from src.command_system.engine import create_command_context

        convo = FakeConversation()
        runner = make_runner(['{"verdict": "pass", "confidence": 1.0, "findings": []}'])
        with tempfile.TemporaryDirectory() as tmp:
            ctx = create_command_context(
                workspace_root=tmp,
                conversation=convo,
                history=HistoryLog(),
                config={"review_subagent_runner": runner},
            )
            import src.review as review_mod

            orig = review_mod.snapshot_diff
            try:
                review_mod.snapshot_diff = lambda _root=None: "+x\n"
                result = review_command_call("", ctx)
            finally:
                review_mod.snapshot_diff = orig
            self.assertIn("PASS", result.value)
            self.assertEqual(len(convo.messages), 1)

    def test_parse_review_args(self):
        parsed = parse_review_args("--model cheap --max-iterations 5")
        self.assertEqual(parsed["model"], "cheap")
        self.assertEqual(parsed["max_iterations"], 5)
        parsed2 = parse_review_args("")
        self.assertIsNone(parsed2["model"])
        self.assertEqual(parsed2["max_iterations"], 2)

    def test_resolve_review_model_prefers_flag_then_config_then_tier(self):
        self.assertEqual(resolve_review_model("flag-model", {}), "flag-model")
        self.assertEqual(
            resolve_review_model(None, {"review": {"model": "cfg-model"}}), "cfg-model"
        )
        # utility_model with an empty config falls back to the passed model (None here)
        self.assertIsNone(resolve_review_model(None, {"models": {}}))

    def test_snapshot_diff_non_git_dir_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(snapshot_diff(tmp), "")

    def test_build_review_prompt_contains_rubric_and_diff(self):
        prompt = build_review_prompt("+print(1)\n")
        self.assertIn("P0", prompt)
        self.assertIn('"verdict"', prompt)
        self.assertIn("+print(1)", prompt)

    def test_format_verdict_for_worker(self):
        v = ReviewVerdict(
            status="fail",
            confidence=0.8,
            findings=[],
            model="m",
        )
        text = format_verdict_for_worker(v)
        self.assertIn('verdict="fail"', text)

    def test_should_judge_run_opt_in(self):
        self.assertFalse(should_judge_run({}))
        self.assertFalse(should_judge_run(None))
        self.assertTrue(should_judge_run({"judge": {"enabled": True}}))

    def test_resolve_judge_config(self):
        critic = resolve_judge_config({"judge": {"max_iterations": 5, "success_threshold": 0.9}})
        self.assertIsInstance(critic, CriticConfig)
        self.assertEqual(critic.max_iterations, 5)
        self.assertAlmostEqual(critic.success_threshold, 0.9)

    def test_followup_prompt_lists_missing(self):
        prompt = build_followup_prompt(GoalVerdict(complete=False, missing=["a", "b"]))
        self.assertIn("- a", prompt)
        self.assertIn("- b", prompt)


if __name__ == "__main__":
    unittest.main()
