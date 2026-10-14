"""Test tool-call loop detection (brief 006, crush parity)."""

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
    tool_call_signature,
    LOOP_WINDOW_SIZE,
    LOOP_THRESHOLD,
)


class TestToolCallSignature(unittest.TestCase):
    """The crush-style signature: sha256(name + \\x00 + canonical_json(input) + \\x00 + output)."""

    def test_identical_steps_hash_identically(self):
        first = tool_call_signature("Bash", {"command": "echo hi"}, {"exit_code": 0})
        second = tool_call_signature("Bash", {"command": "echo hi"}, {"exit_code": 0})
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)  # sha256 hex

    def test_input_key_order_does_not_matter(self):
        first = tool_call_signature("Bash", {"a": 1, "b": 2}, "out")
        second = tool_call_signature("Bash", {"b": 2, "a": 1}, "out")
        self.assertEqual(first, second)

    def test_different_name_input_or_output_hash_differently(self):
        base = tool_call_signature("Bash", {"command": "echo hi"}, "out")
        self.assertNotEqual(base, tool_call_signature("Read", {"command": "echo hi"}, "out"))
        self.assertNotEqual(base, tool_call_signature("Bash", {"command": "echo yo"}, "out"))
        self.assertNotEqual(base, tool_call_signature("Bash", {"command": "echo hi"}, "other"))


class TestLoopDetector(unittest.TestCase):
    """Unit tests for the sliding-window detector."""

    def test_six_identical_steps_trigger_detection(self):
        """Feed 6 identical fake steps -> detection fires with the tool named."""
        detector = LoopDetector()
        repetitions = 0
        for _ in range(6):
            repetitions = detector.record(
                "Bash",
                {"command": "echo stuck"},
                {"exit_code": 0, "stdout": "stuck\n", "stderr": ""},
            )
        self.assertEqual(repetitions, 6)
        self.assertGreater(repetitions, LOOP_THRESHOLD)
        self.assertTrue(detector.is_stuck(repetitions))

    def test_five_identical_steps_do_not_trigger(self):
        detector = LoopDetector()
        repetitions = 0
        for _ in range(5):
            repetitions = detector.record("Bash", {"command": "echo x"}, "ok")
        self.assertEqual(repetitions, 5)
        self.assertFalse(detector.is_stuck(repetitions))

    def test_varied_inputs_do_not_trigger(self):
        """Same tool with different inputs -> no detection."""
        detector = LoopDetector()
        repetitions = 0
        for i in range(LOOP_WINDOW_SIZE):
            repetitions = detector.record(
                "Bash",
                {"command": f"echo {i}"},
                {"exit_code": 0, "stdout": f"{i}\n", "stderr": ""},
            )
        self.assertEqual(repetitions, 1)
        self.assertFalse(detector.is_stuck(repetitions))

    def test_same_tool_different_outputs_do_not_trigger(self):
        """Output is part of the signature -> no detection when outputs differ."""
        detector = LoopDetector()
        repetitions = 0
        for i in range(LOOP_WINDOW_SIZE):
            repetitions = detector.record(
                "Read",
                {"file_path": "/tmp/x.py"},
                f"contents version {i}",
            )
        self.assertEqual(repetitions, 1)
        self.assertFalse(detector.is_stuck(repetitions))

    def test_window_slides_old_signatures_out(self):
        detector = LoopDetector(window_size=4, threshold=10)
        for _ in range(3):
            repetitions = detector.record("A", {}, "x")
            self.assertFalse(detector.is_stuck(repetitions))
        self.assertEqual(repetitions, 3)
        # Push 4 different steps through: the "A" signatures slide out.
        for _ in range(4):
            repetitions = detector.record("B", {}, "y")
            self.assertFalse(detector.is_stuck(repetitions))
        # Recording "A" again starts over at 1.
        self.assertEqual(detector.record("A", {}, "x"), 1)

    def test_custom_threshold_is_respected(self):
        detector = LoopDetector(window_size=10, threshold=2)
        detector.record("Bash", {"command": "echo x"}, "ok")
        detector.record("Bash", {"command": "echo x"}, "ok")
        self.assertFalse(detector.is_stuck(2))
        self.assertTrue(detector.is_stuck(detector.record("Bash", {"command": "echo x"}, "ok")))


class TestLoopDetectionIntegration(unittest.TestCase):
    """Integration: run_agent_loop breaks out of a repeated-tool-call loop."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()

    def test_run_agent_loop_breaks_on_repeated_tool_calls(self):
        """A provider stuck re-issuing the same Bash call stops at 6 with the break message."""
        conversation = Conversation()
        conversation.add_user_message("do the thing")

        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()

        def chat_side_effect(*args, **kwargs):
            return ChatResponse(
                content="",
                model="test-model",
                usage={"input_tokens": 5, "output_tokens": 5},
                finish_reason="tool_use",
                tool_uses=[{
                    "id": "toolu_loop",
                    "name": "Bash",
                    "input": {"command": "echo stuck"},
                }],
            )

        mock_provider.chat.side_effect = chat_side_effect

        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )

        self.assertIsInstance(result, AgentLoopResult)
        self.assertEqual(
            result.response_text,
            "[Loop detected: Bash repeated 6x with identical input \u2014 stopping. "
            "Try a different approach.]",
        )
        # The loop broke on the 6th identical step instead of burning max_turns.
        self.assertEqual(mock_provider.chat.call_count, 6)
        self.assertEqual(result.num_turns, 6)
        self.assertFalse(result.cancelled)

    def test_run_agent_loop_unaffected_by_varied_calls(self):
        """Distinct tool calls across turns never trip the detector."""
        conversation = Conversation()
        conversation.add_user_message("run three commands")

        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()

        commands = iter(["echo one", "echo two", "echo three"])

        def chat_side_effect(*args, **kwargs):
            try:
                command = next(commands)
            except StopIteration:
                return ChatResponse(
                    content="all done",
                    model="test-model",
                    usage={"input_tokens": 5, "output_tokens": 5},
                    finish_reason="stop",
                    tool_uses=None,
                )
            return ChatResponse(
                content="",
                model="test-model",
                usage={"input_tokens": 5, "output_tokens": 5},
                finish_reason="tool_use",
                tool_uses=[{
                    "id": "toolu_varied",
                    "name": "Bash",
                    "input": {"command": command},
                }],
            )

        mock_provider.chat.side_effect = chat_side_effect

        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )

        self.assertEqual(result.response_text, "all done")
        self.assertNotIn("Loop detected", result.response_text)


if __name__ == "__main__":
    unittest.main()
