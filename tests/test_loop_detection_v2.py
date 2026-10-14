"""Test loop-detection v2 (brief 029): nudge-before-terminate + per-tool
error counters + OpenHands detector family + doom_loop adjudication."""

import unittest
from unittest.mock import MagicMock
from pathlib import Path
import tempfile

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.defaults import build_default_registry
from src.tool_system.context import ToolContext
from src.tool_system.agent_loop import (
    run_agent_loop,
    AgentLoopResult,
    LoopDetector,
    StuckPattern,
    LOOP_NUDGE_TEMPLATE,
    TOOL_ERROR_STUCK_THRESHOLD,
)


def _tool_response(name="Bash", tool_input=None, uid="toolu_1"):
    return ChatResponse(
        content="",
        model="test-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="tool_use",
        tool_uses=[{"id": uid, "name": name, "input": tool_input or {}}],
    )


def _done_response(text="all done"):
    return ChatResponse(
        content=text,
        model="test-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="stop",
        tool_uses=None,
    )


class TestPerToolErrorCounter(unittest.TestCase):
    """029a: 3 consecutive errors from the same tool -> stuck (006 misses
    reworded-input retries)."""

    def test_three_consecutive_errors_reworded_inputs(self):
        d = LoopDetector()
        reps = 0
        for i in range(3):
            # Same failing tool, reworded input, same error text: the
            # SHA-256 signature never repeats, the error counter does.
            reps = d.record(
                "Bash",
                {"command": f"false # retry attempt number {i}"},
                "Error: command failed with exit 1",
                is_error=True,
            )
        pattern = d.detect_stuck("Bash", reps)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, "tool_error")
        self.assertIn("bash", pattern.tool_name)
        self.assertEqual(TOOL_ERROR_STUCK_THRESHOLD, 3)

    def test_two_errors_not_stuck(self):
        d = LoopDetector()
        reps = d.record("Bash", {"command": "x1"}, "err", is_error=True)
        reps = d.record("Bash", {"command": "x2"}, "err", is_error=True)
        self.assertIsNone(d.detect_stuck("Bash", reps))

    def test_error_run_resets_on_success(self):
        d = LoopDetector()
        d.record("Bash", {"c": "1"}, "e1", is_error=True)
        d.record("Bash", {"c": "2"}, "e2", is_error=True)
        d.record("Bash", {"c": "ok"}, "ok")  # success clears the run
        self.assertEqual(d.consecutive_errors_for("Bash"), 0)
        reps = d.record("Bash", {"c": "3"}, "e3", is_error=True)
        reps = d.record("Bash", {"c": "4"}, "e4", is_error=True)
        # Distinct error texts: the repeated-error family detector stays quiet.
        self.assertIsNone(d.detect_stuck("Bash", reps))
        reps = d.record("Bash", {"c": "5"}, "e5", is_error=True)
        pattern = d.detect_stuck("Bash", reps)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, "tool_error")

    def test_other_tool_error_resets_run(self):
        d = LoopDetector()
        d.record("Bash", {"c": "1"}, "e", is_error=True)
        d.record("Bash", {"c": "2"}, "e", is_error=True)
        d.record("Read", {"f": "x"}, "e", is_error=True)
        self.assertEqual(d.consecutive_errors_for("Bash"), 0)
        self.assertEqual(d.consecutive_errors_for("Read"), 1)

    def test_error_state_api_for_037(self):
        """The per-tool error state 037's circuit breaker will reuse."""
        d = LoopDetector()
        d.record("Bash", {"c": "1"}, "e", is_error=True)
        d.record("Bash", {"c": "2"}, "e", is_error=True)
        self.assertEqual(d.consecutive_errors_for("Bash"), 2)
        self.assertEqual(d.consecutive_errors_for("bash"), 2)  # case-insensitive
        self.assertEqual(d.consecutive_errors_for("Read"), 0)
        self.assertEqual(d.tool_error_counts, {"bash": 2})
        d.reset_tool_errors("Bash")
        self.assertEqual(d.consecutive_errors_for("Bash"), 0)
        d.record("Read", {"f": "x"}, "e", is_error=True)
        d.reset_tool_errors()
        self.assertEqual(d.tool_error_counts, {})

    def test_record_positional_shape_unchanged(self):
        """The 3-positional-arg call shape (006) still works; is_error
        defaults to False."""
        d = LoopDetector()
        reps = d.record("Bash", {"command": "echo hi"}, "ok")
        self.assertEqual(reps, 1)
        self.assertEqual(d.consecutive_errors_for("Bash"), 0)


class TestOpenHandsFamily(unittest.TestCase):
    """029b: OpenHands detector family over the 20-event journal."""

    def test_repeated_action_error_interleaved(self):
        """Same tool + same error 3x, interleaved with other-tool successes:
        the consecutive-error counter misses it, the family catches it."""
        d = LoopDetector()
        r = d.record("Bash", {"c": "a"}, "Error: boom", is_error=True)
        d.record("Read", {"f": "ok"}, "contents")  # success breaks the run
        r = d.record("Bash", {"c": "b"}, "Error: boom", is_error=True)
        d.record("Read", {"f": "ok2"}, "contents")
        r = d.record("Bash", {"c": "c"}, "Error: boom", is_error=True)
        self.assertEqual(d.consecutive_errors_for("Bash"), 1)  # not consecutive
        pattern = d.detect_stuck("Bash", r)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, "repeated_action_error")

    def test_ab_ab_alternation(self):
        d = LoopDetector()
        d.record("Bash", {"command": "echo A"}, "A\n")
        d.record("Read", {"file_path": "/x"}, "contents")
        d.record("Bash", {"command": "echo A"}, "A2\n")
        reps = d.record("Read", {"file_path": "/x"}, "contents2")
        pattern = d.detect_stuck("Read", reps)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, "ab_ab_alternation")

    def test_ab_ab_no_fire_when_inputs_differ(self):
        d = LoopDetector()
        d.record("Bash", {"command": "echo A1"}, "A1\n")
        d.record("Read", {"file_path": "/x"}, "c")
        d.record("Bash", {"command": "echo A2"}, "A2\n")
        reps = d.record("Read", {"file_path": "/y"}, "c2")
        self.assertIsNone(d.detect_stuck("Read", reps))

    def test_monologue(self):
        d = LoopDetector()
        for _ in range(3):
            d.record_assistant_text("I will now think step by step.")
        pattern = d.detect_stuck("Bash", 1)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, "monologue")

    def test_monologue_broken_by_tool_call(self):
        d = LoopDetector()
        d.record_assistant_text("same words")
        d.record_assistant_text("same words")
        d.record("Bash", {"command": "echo hi"}, "hi\n")
        d.record_assistant_text("same words")
        d.record_assistant_text("same words")
        self.assertIsNone(d.detect_stuck("Bash", 1))

    def test_monologue_ignores_empty_text(self):
        d = LoopDetector()
        d.record_assistant_text("")
        d.record_assistant_text("   ")
        d.record_assistant_text(None)
        self.assertIsNone(d.detect_stuck("Bash", 1))

    def test_exact_repeats_stay_with_signature(self):
        """Identical x3: the family's observation detector must not preempt
        006 -- identical x6 must still report signature."""
        d = LoopDetector()
        reps = 0
        for _ in range(3):
            reps = d.record("Bash", {"command": "echo stuck"}, "stuck\n")
        self.assertIsNone(d.detect_stuck("Bash", reps))
        for _ in range(3):
            reps = d.record("Bash", {"command": "echo stuck"}, "stuck\n")
        pattern = d.detect_stuck("Bash", reps)
        self.assertIsNotNone(pattern)
        self.assertEqual(pattern.pattern_type, "signature")

    def test_nudge_template_shape(self):
        text = LOOP_NUDGE_TEMPLATE.format(pattern="p", suggestion="s")
        self.assertTrue(text.startswith("You appear stuck: p."))
        self.assertIn("Try a different approach: s.", text)

    def test_stuck_pattern_fields(self):
        p = StuckPattern(
            pattern_type="signature", tool_name="Bash",
            detail="d", suggestion="s",
        )
        self.assertEqual(p.pattern_type, "signature")


class TestNudgeRecoveryState(unittest.TestCase):
    def test_mark_and_clear_nudge(self):
        d = LoopDetector()
        self.assertFalse(d.nudge_pending)
        p = StuckPattern("signature", "Bash", "d", "s")
        d.mark_nudged(p)
        self.assertTrue(d.nudge_pending)
        self.assertEqual(d.nudged_pattern, p)
        d.clear_nudge()
        self.assertFalse(d.nudge_pending)
        self.assertIsNone(d.nudged_pattern)

    def test_reset_clears_everything(self):
        d = LoopDetector()
        for _ in range(3):
            d.record("Bash", {"c": "x"}, "e", is_error=True)
        d.mark_nudged(StuckPattern("tool_error", "bash", "d", "s"))
        d.reset()
        self.assertFalse(d.nudge_pending)
        self.assertEqual(d.tool_error_counts, {})
        self.assertIsNone(d.detect_stuck("Bash", 1))


class TestLoopV2Integration(unittest.TestCase):
    """End-to-end through run_agent_loop."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, provider, **kw):
        return run_agent_loop(
            conversation=kw.pop("conversation", None) or Conversation(),
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
            **kw,
        )

    def _repeating_provider(self, tool_input, name="Bash", calls_before_done=None):
        """Provider that repeats one tool call, then optionally finishes."""
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        state = {"n": 0}

        def chat_side_effect(*args, **kwargs):
            state["n"] += 1
            if calls_before_done is not None and state["n"] > calls_before_done:
                return _done_response()
            return _tool_response(
                name=name, tool_input=tool_input, uid=f"toolu_{state['n']}"
            )

        mock_provider.chat.side_effect = chat_side_effect
        return mock_provider, state

    def test_error_counter_catches_reworded_retries(self):
        """Same tool, reworded inputs, 3 consecutive errors -> tool_error
        (the case 006's signature misses)."""
        attempts = iter([0, 1, 2, 3, 4, 5])
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()

        def chat_side_effect(*args, **kwargs):
            n = next(attempts)
            return _tool_response(
                tool_input={"command": f"false # retry attempt number {n}"},
                uid=f"toolu_e{n}",
            )

        mock_provider.chat.side_effect = chat_side_effect
        conversation = Conversation()
        conversation.add_user_message("retry the failing thing")

        result = self._run(mock_provider, conversation=conversation)

        self.assertTrue(
            result.response_text.startswith("[LoopDetected: tool_error"),
            result.response_text,
        )
        # Nudge at the 3rd consecutive error, one recovery turn, then stop.
        self.assertEqual(mock_provider.chat.call_count, 4)
        nudges = [e for e in result.run_log if e.get("event") == "loop_nudge"]
        self.assertEqual(len(nudges), 1)
        self.assertEqual(nudges[0]["pattern"], "tool_error")
        detected = [e for e in result.run_log if e.get("event") == "loop_detected"]
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0]["pattern"], "tool_error")
        self.assertIn("turn", detected[0])

    def test_ab_ab_alternation_integration(self):
        """Alternate two tools (Bash/Read) with fixed inputs -> the
        alternation detector fires."""
        probe = self.workspace / "probe.txt"
        probe.write_text("probe contents\n")
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        state = {"n": 0}

        def chat_side_effect(*args, **kwargs):
            state["n"] += 1
            if state["n"] % 2 == 1:
                use = {
                    "id": f"toolu_ab{state['n']}",
                    "name": "Bash",
                    "input": {"command": "echo ping"},
                }
            else:
                use = {
                    "id": f"toolu_ab{state['n']}",
                    "name": "Read",
                    "input": {"file_path": str(probe)},
                }
            return _tool_response(
                name=use["name"], tool_input=use["input"], uid=use["id"]
            )

        mock_provider.chat.side_effect = chat_side_effect

        result = self._run(mock_provider)
        self.assertTrue(
            result.response_text.startswith("[LoopDetected: ab_ab_alternation"),
            result.response_text,
        )
        # 4th tool call trips the alternation detector (nudge), 5th persists.
        self.assertEqual(mock_provider.chat.call_count, 5)
        nudges = [e for e in result.run_log if e.get("event") == "loop_nudge"]
        self.assertEqual(nudges[0]["pattern"], "ab_ab_alternation")

    def test_recovery_turn_breaks_pattern(self):
        """Nudge at x6, recovery turn does something different -> run
        continues normally, no abort, nudge state cleared."""
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        state = {"n": 0}

        def chat_side_effect(*args, **kwargs):
            state["n"] += 1
            if state["n"] <= 6:
                return _tool_response(
                    tool_input={"command": "echo stuck"},
                    uid=f"toolu_r{state['n']}",
                )
            if state["n"] == 7:
                # Recovery turn: a genuinely different call.
                return _tool_response(
                    tool_input={"command": "echo recovered"},
                    uid="toolu_r7",
                )
            return _done_response()

        mock_provider.chat.side_effect = chat_side_effect

        result = self._run(mock_provider)
        self.assertEqual(result.response_text, "all done")
        self.assertNotIn("LoopDetected", result.response_text)
        nudges = [e for e in result.run_log if e.get("event") == "loop_nudge"]
        self.assertEqual(len(nudges), 1)
        detected = [e for e in result.run_log if e.get("event") == "loop_detected"]
        self.assertEqual(detected, [])
        self.assertEqual(mock_provider.chat.call_count, 8)

    def test_interactive_doom_loop_adjudication_deny(self):
        """Interactive + permission_handler denying -> structured stop, and
        the handler was asked with the doom_loop action."""
        calls = []

        def handler(name, message, suggestion):
            calls.append((name, message, suggestion))
            return (False, False)

        self.context.permission_handler = handler
        mock_provider, _ = self._repeating_provider({"command": "echo stuck"})

        result = self._run(mock_provider)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], "doom_loop")
        self.assertIn("stuck", calls[0][1])
        self.assertTrue(
            result.response_text.startswith("[LoopDetected: signature"),
            result.response_text,
        )

    def test_interactive_doom_loop_adjudication_allow(self):
        """Interactive + permission_handler allowing -> run continues after
        a detector reset, no abort."""
        calls = []

        def handler(name, message, suggestion):
            calls.append(name)
            return (True, False)

        self.context.permission_handler = handler
        # 7 repeating calls (nudge at 6, adjudicate at 7), then done.
        mock_provider, _ = self._repeating_provider(
            {"command": "echo stuck"}, calls_before_done=7
        )

        result = self._run(mock_provider)
        self.assertEqual(calls, ["doom_loop"])
        adjudicated = [
            e for e in result.run_log if e.get("event") == "doom_loop_adjudicated"
        ]
        self.assertEqual(len(adjudicated), 1)
        self.assertEqual(adjudicated[0]["decision"], "continue")
        self.assertEqual(result.response_text, "all done")
        self.assertEqual(mock_provider.chat.call_count, 8)

    def test_non_interactive_structured_stop_report(self):
        """Non-interactive runs never invoke the permission handler: the
        persistent pattern auto-aborts with the structured report."""

        def handler(*args):
            raise AssertionError("permission_handler must not be called")

        self.context.permission_handler = handler
        self.context.non_interactive = True
        mock_provider, _ = self._repeating_provider({"command": "echo stuck"})

        result = self._run(mock_provider)
        self.assertTrue(
            result.response_text.startswith("[LoopDetected: signature"),
            result.response_text,
        )
        self.assertIn("Suggestion:", result.response_text)
        detected = [e for e in result.run_log if e.get("event") == "loop_detected"]
        self.assertEqual(len(detected), 1)
        self.assertEqual(detected[0]["pattern"], "signature")

    def test_recovery_judge_hook_rewrites_nudge(self):
        """The 028-style loop_recovery_judge hook can rewrite the nudge the
        model sees on its recovery turn."""
        self.context.loop_recovery_judge = (
            lambda pattern, base: f"CUSTOM-JUDGE-NUDGE for {pattern.pattern_type}"
        )
        mock_provider, _ = self._repeating_provider(
            {"command": "echo stuck"}, calls_before_done=6
        )
        conversation = Conversation()
        conversation.add_user_message("do it")

        result = self._run(mock_provider, conversation=conversation)
        nudge_texts = [
            m.content for m in conversation.messages
            if m.role == "user" and isinstance(m.content, str)
            and "CUSTOM-JUDGE-NUDGE" in m.content
        ]
        self.assertEqual(len(nudge_texts), 1)
        self.assertIn("signature", nudge_texts[0])
        self.assertEqual(result.response_text, "all done")

    def test_loop_turn_events_emitted(self):
        """loop_nudge / loop_detected are first-class turn events."""
        seen = []

        def on_event(event):
            seen.append(event.kind)

        mock_provider, _ = self._repeating_provider({"command": "echo stuck"})
        self._run(mock_provider, on_event=on_event)
        self.assertIn("loop_nudge", seen)
        self.assertIn("loop_detected", seen)
        self.assertLess(seen.index("loop_nudge"), seen.index("loop_detected"))


if __name__ == "__main__":
    unittest.main()
