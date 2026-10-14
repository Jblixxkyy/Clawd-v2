"""Tests for the scaffold tool tier (brief 017): Checklist, AskQuestion,
ReportFailure, Exit/Status.

The tool classes are tested directly; registry registration is a separate
integration step (PENDING, brief 014's builder owns defaults.py this wave).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError, ToolPermissionError
from src.tool_system.protocol import ToolResult
from src.tool_system.tools.scaffold import (
    CHECKLIST_STATUSES,
    SCAFFOLD_TOOLS,
    AskQuestionTool,
    ChecklistTool,
    ExitTool,
    QuestionDismissed,
    ReportFailureTool,
    ToolFailure,
    evaluate_run_end,
    is_terminal_result,
    render_question,
)


def _ctx(**kwargs) -> ToolContext:
    root = Path(tempfile.mkdtemp(prefix="scaffold-test-"))
    return ToolContext(workspace_root=root, **kwargs)


def _opts(n, recommended_idx=None):
    opts = [
        {"label": f"Option {i}", "description": f"Does thing {i}"}
        for i in range(1, n + 1)
    ]
    if recommended_idx is not None:
        opts[recommended_idx]["recommended"] = True
    return opts


class ChecklistTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ChecklistTool()
        self.ctx = _ctx()

    def test_valid_list_accepted_whole_list_overwrite(self):
        items = [
            {"label": "Plan", "status": "completed"},
            {"label": "Build", "status": "in_progress"},
            {"label": "Test", "status": "pending"},
            {"label": "Docs", "status": "blocked"},
        ]
        result = self.tool.run({"items": items}, self.ctx)
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["items"], items)
        self.assertEqual(result.output["widget"], "checklist")

    def test_two_in_progress_rejected(self):
        items = [
            {"label": "A", "status": "in_progress"},
            {"label": "B", "status": "in_progress"},
        ]
        with self.assertRaises(ToolInputError):
            self.tool.run({"items": items}, self.ctx)

    def test_invalid_status_rejected(self):
        with self.assertRaises(ToolInputError):
            self.tool.run({"items": [{"label": "A", "status": "wip"}]}, self.ctx)

    def test_empty_items_rejected(self):
        with self.assertRaises(ToolInputError):
            self.tool.run({"items": []}, self.ctx)

    def test_duplicate_labels_rejected(self):
        items = [
            {"label": "A", "status": "pending"},
            {"label": "A", "status": "completed"},
        ]
        with self.assertRaises(ToolInputError):
            self.tool.run({"items": items}, self.ctx)

    def test_statuses_cover_blocked(self):
        self.assertIn("blocked", CHECKLIST_STATUSES)

    def test_spec_is_read_only_no_side_effects(self):
        spec = self.tool.spec()
        self.assertEqual(spec.name, "Checklist")
        self.assertTrue(spec.is_read_only)
        self.assertFalse(spec.is_destructive)


class AskQuestionValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = AskQuestionTool()
        self.ctx = _ctx()

    def _base(self, n=3):
        return {"question": "Which approach?", "options": _opts(n)}

    def test_more_than_five_options_rejected(self):
        with self.assertRaises(ToolInputError):
            self.tool.run(self._base(6), self.ctx)

    def test_fewer_than_two_options_rejected(self):
        with self.assertRaises(ToolInputError):
            self.tool.run(self._base(1), self.ctx)

    def test_five_options_accepted(self):
        ctx = _ctx(ask_user=lambda qs: {qs[0]["question"]: qs[0]["options"][0]["label"]})
        result = self.tool.run(self._base(5), ctx)
        self.assertFalse(result.is_error)

    def test_mode_changing_option_rejected(self):
        bad = dict(self._base(2))
        bad["options"] = [
            {"label": "Keep going", "description": "Continue as normal"},
            {"label": "Toggle to Act mode", "description": "Switch the agent into Act mode"},
        ]
        with self.assertRaises(ToolInputError):
            self.tool.run(bad, self.ctx)

    def test_permission_changing_option_rejected(self):
        bad = dict(self._base(2))
        bad["options"] = [
            {"label": "Keep going", "description": "Continue as normal"},
            {"label": "Bypass permissions", "description": "Disable permission prompts and continue"},
        ]
        with self.assertRaises(ToolInputError):
            self.tool.run(bad, self.ctx)

    def test_elevation_option_rejected(self):
        bad = dict(self._base(2))
        bad["options"] = [
            {"label": "Keep going", "description": "Continue as normal"},
            {"label": "Elevate privileges", "description": "Gain root access to finish faster"},
        ]
        with self.assertRaises(ToolInputError):
            self.tool.run(bad, self.ctx)

    def test_benign_permission_mention_accepted(self):
        # "permission" alone in prose that does not change permissions is fine;
        # the contract targets mode/permission-CHANGING options.
        good = dict(self._base(2))
        good["options"] = [
            {"label": "Ask first", "description": "Ask the user for permission before deleting"},
            {"label": "Skip", "description": "Do not delete anything"},
        ]
        ctx = _ctx(ask_user=lambda qs: {qs[0]["question"]: "Skip"})
        result = self.tool.run(good, ctx)
        self.assertEqual(result.output["answer"], "Skip")

    def test_multiple_recommended_rejected(self):
        bad = self._base(3)
        bad["options"][0]["recommended"] = True
        bad["options"][1]["recommended"] = True
        with self.assertRaises(ToolInputError):
            self.tool.run(bad, self.ctx)

    def test_duplicate_labels_rejected(self):
        bad = self._base(2)
        bad["options"][1]["label"] = bad["options"][0]["label"]
        with self.assertRaises(ToolInputError):
            self.tool.run(bad, self.ctx)

    def test_recommended_rendered_first_with_suffix(self):
        rendered = render_question("Q?", _opts(3, recommended_idx=2))
        self.assertEqual(rendered["options"][0]["label"], "Option 3 (Recommended)")
        self.assertEqual(rendered["options"][1]["label"], "Option 1")
        self.assertEqual(rendered["options"][2]["label"], "Option 2")

    def test_no_handler_raises_permission_error(self):
        with self.assertRaises(ToolPermissionError):
            self.tool.run(self._base(2), self.ctx)

    def test_pre_supplied_answer_path(self):
        tool_input = dict(self._base(3))
        tool_input["answer"] = "Option 2"
        result = self.tool.run(tool_input, self.ctx)
        self.assertEqual(result.output["answer"], "Option 2")
        self.assertEqual(result.output["widget"], "ask_question")

    def test_pre_supplied_answer_must_match_label(self):
        tool_input = dict(self._base(2))
        tool_input["answer"] = "Not an option"
        with self.assertRaises(ToolInputError):
            self.tool.run(tool_input, self.ctx)


class AskQuestionDismissalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = AskQuestionTool()
        self.question = {"question": "Which approach?", "options": _opts(2)}

    def test_dismissal_raises_continuable_tool_failure(self):
        def dismiss(_qs):
            raise QuestionDismissed()

        result_ctx = _ctx(ask_user=dismiss)
        with self.assertRaises(ToolFailure) as cm:
            self.tool.run(self.question, result_ctx)
        failure = cm.exception
        # No dead end: the failure is explicitly continuable.
        self.assertTrue(failure.continuable)
        self.assertEqual(failure.kind, "dismissed")
        self.assertIn("dismissed", failure.reason)

    def test_dismissal_with_feedback_carries_feedback_as_reason(self):
        def dismiss(_qs):
            raise QuestionDismissed("none of these fit; check the logs first")

        result_ctx = _ctx(ask_user=dismiss)
        with self.assertRaises(ToolFailure) as cm:
            self.tool.run(self.question, result_ctx)
        failure = cm.exception
        self.assertTrue(failure.continuable)
        self.assertEqual(failure.kind, "dismissed_with_feedback")
        self.assertIn("none of these fit; check the logs first", failure.reason)

    def test_empty_handler_response_is_dismissal(self):
        result_ctx = _ctx(ask_user=lambda _qs: {})
        with self.assertRaises(ToolFailure) as cm:
            self.tool.run(self.question, result_ctx)
        self.assertTrue(cm.exception.continuable)

    def test_answer_returned_through_handler(self):
        def answer(_qs):
            return {"Which approach?": "Option 1"}

        result_ctx = _ctx(ask_user=answer)
        result = self.tool.run(self.question, result_ctx)
        self.assertEqual(result.output["answer"], "Option 1")

    def test_tool_failure_requires_reason(self):
        with self.assertRaises(ValueError):
            ToolFailure("")


class ReportFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ReportFailureTool()
        self.ctx = _ctx()

    def test_run_marked_failed_with_reason_in_log_record(self):
        result = self.tool.run(
            {
                "reason": "API key revoked; all retries exhausted",
                "attempted": ["retry with backoff", "fallback provider"],
            },
            self.ctx,
        )
        self.assertFalse(result.is_error)
        self.assertTrue(result.output["isComplete"])
        self.assertEqual(result.output["outcome"], "failed")
        self.assertEqual(result.output["reason"], "API key revoked; all retries exhausted")
        self.assertEqual(result.output["attempted"], ["retry with backoff", "fallback provider"])
        log_record = result.output["log_record"]
        self.assertEqual(log_record["tool"], "ReportFailure")
        self.assertEqual(log_record["outcome"], "failed")
        self.assertEqual(log_record["reason"], "API key revoked; all retries exhausted")
        self.assertEqual(result.output["widget"], "report_failure")

    def test_empty_reason_rejected(self):
        with self.assertRaises(ToolInputError):
            self.tool.run({"reason": "  ", "attempted": ["tried"]}, self.ctx)

    def test_empty_attempted_rejected(self):
        with self.assertRaises(ToolInputError):
            self.tool.run({"reason": "bad", "attempted": []}, self.ctx)

    def test_is_terminal_result(self):
        result = self.tool.run(
            {"reason": "nope", "attempted": ["tried"]}, self.ctx
        )
        self.assertTrue(is_terminal_result(result))
        plain = ToolResult(name="Bash", output={"x": 1})
        self.assertFalse(is_terminal_result(plain))


class ExitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = ExitTool()
        self.ctx = _ctx()

    def test_exit_done_sets_is_complete(self):
        result = self.tool.run({"outcome": "done", "summary": "all green"}, self.ctx)
        self.assertTrue(result.output["isComplete"])
        self.assertEqual(result.output["outcome"], "done")
        self.assertEqual(result.output["summary"], "all green")
        self.assertEqual(result.output["widget"], "exit")
        self.assertTrue(is_terminal_result(result))

    def test_exit_failed_sets_is_complete(self):
        result = self.tool.run({"outcome": "failed"}, self.ctx)
        self.assertTrue(result.output["isComplete"])
        self.assertEqual(result.output["outcome"], "failed")

    def test_invalid_outcome_rejected(self):
        with self.assertRaises(ToolInputError):
            self.tool.run({"outcome": "maybe"}, self.ctx)

    def test_spec_has_status_alias(self):
        spec = self.tool.spec()
        self.assertEqual(spec.name, "Exit")
        self.assertIn("Status", spec.aliases)


class LoopHygieneTests(unittest.TestCase):
    def test_exit_then_no_tool_calls_completes_cleanly(self):
        verdict = evaluate_run_end(
            exit_called=True,
            exit_outcome="done",
            final_message_had_tool_calls=False,
        )
        self.assertTrue(verdict["isComplete"])
        self.assertEqual(verdict["outcome"], "done")

    def test_exit_failed_completes_as_failed(self):
        verdict = evaluate_run_end(
            exit_called=True,
            exit_outcome="failed",
            final_message_had_tool_calls=False,
        )
        self.assertTrue(verdict["isComplete"])
        self.assertEqual(verdict["outcome"], "failed")

    def test_tool_call_loop_with_no_exit_is_stall_not_done(self):
        verdict = evaluate_run_end(
            exit_called=False,
            final_message_had_tool_calls=True,
        )
        self.assertFalse(verdict["isComplete"])
        self.assertEqual(verdict["outcome"], "stall")

    def test_no_exit_and_no_tool_calls_is_stall_not_done(self):
        # Silent stall: the model stopped calling tools but never signaled
        # completion. This must NOT be marked done.
        verdict = evaluate_run_end(
            exit_called=False,
            final_message_had_tool_calls=False,
        )
        self.assertFalse(verdict["isComplete"])
        self.assertEqual(verdict["outcome"], "stall")


class ModuleShapeTests(unittest.TestCase):
    def test_scaffold_tools_tuple(self):
        names = [t().spec().name for t in SCAFFOLD_TOOLS]
        self.assertEqual(names, ["Checklist", "AskQuestion", "ReportFailure", "Exit"])
        # All specs carry a description (prompt contract surface).
        for tool in SCAFFOLD_TOOLS:
            self.assertTrue(tool().spec().description.strip())


if __name__ == "__main__":
    unittest.main()
