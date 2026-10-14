"""Brief 038: semantic stuck detection — failing strategy trips even without exact repeats.

The detector watches a sliding window of the last 10 tool outcomes: if at
least 6 were errors and zero were file modifications (Edit/Write success),
the run is semantically stuck — the same failing *strategy* even though no
exact (tool, input, output) signature repeats. The loop then stops with a
distinct "[Stuck: ...]" verdict naming the pattern.
"""

import unittest
from unittest.mock import MagicMock
from pathlib import Path
import tempfile

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.defaults import build_default_registry
from src.tool_system.context import ToolContext
from src.tool_system.agent_loop import run_agent_loop, LoopDetector

SEMANTIC_MESSAGE = (
    "[Stuck: 6+ of the last 10 tool calls failed and no files "
    "were modified \u2014 stopping. Try a different approach.]"
)


class TestSemanticStuckDetector(unittest.TestCase):
    """Detector-level rules: 10-window, 6-error, 0-modification."""

    def _detector_with(self, outcomes):
        d = LoopDetector()
        for name, is_err, mod in outcomes:
            d.record_outcome(name, is_err, mod)
        return d

    def test_flailing_spiral_trips(self):
        # Josh's observed pattern: searches + shell errors, zero writes.
        outcomes = (
            [("Read", True, False)] * 6 + [("Grep", False, False)] * 4
        )
        self.assertTrue(self._detector_with(outcomes).is_semantically_stuck())

    def test_single_write_prevents_trip(self):
        outcomes = (
            [("Read", True, False)] * 6
            + [("Write", False, True)]
            + [("Grep", False, False)] * 3
        )
        self.assertFalse(self._detector_with(outcomes).is_semantically_stuck())

    def test_clean_reads_do_not_trip(self):
        outcomes = [("Read", False, False)] * 10
        self.assertFalse(self._detector_with(outcomes).is_semantically_stuck())

    def test_short_window_does_not_trip(self):
        # The gate needs a FULL 10-outcome window.
        outcomes = [("Bash", True, False)] * 9
        self.assertFalse(self._detector_with(outcomes).is_semantically_stuck())

    def test_failed_edit_does_not_count_as_modification(self):
        outcomes = [("Read", True, False)] * 5 + [("Edit", True, False)] * 5
        self.assertTrue(self._detector_with(outcomes).is_semantically_stuck())

    def test_exact_signature_path_still_works(self):
        # 006's signature detector is independent and untouched.
        d = LoopDetector()
        n = 0
        for _ in range(8):
            n = d.record("Read", {"file_path": "/a"}, {"type": "text"})
        self.assertTrue(d.is_stuck(n))
        self.assertFalse(d.is_semantically_stuck())

    def test_outcome_tracking_is_fail_open(self):
        d = LoopDetector()
        d.record_outcome(None, "not-a-bool", None)  # must not raise
        self.assertFalse(d.is_semantically_stuck())


def _tool_turn(tool_id, name, tool_input, text="Working."):
    return ChatResponse(
        content=text,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 20},
        finish_reason="tool_use",
        tool_uses=[{"id": tool_id, "name": name, "input": tool_input}],
    )


def _final_turn(text="Done."):
    return ChatResponse(
        content=text,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="stop",
        tool_uses=None,
    )


class TestSemanticStuckLoop(unittest.TestCase):
    """End-to-end: the loop verdict carries the distinct semantic message."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, turns):
        conversation = Conversation()
        conversation.add_user_message("Find the chat method.")
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        mock_provider.chat.side_effect = turns
        return run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )

    def _flailing_turns(self):
        # 6 Read errors (unique missing files) + 4 Grep no-matches
        # (unique patterns), interleaved so no 006/029 pattern fires:
        # inputs vary every call and no tool errors consecutively.
        turns = []
        seq = [
            ("Read", {"file_path": str(self.workspace / "nope_a1.py")}),
            ("Grep", {"pattern": "zzz_no_match_alpha", "path": str(self.workspace)}),
            ("Read", {"file_path": str(self.workspace / "nope_b2.py")}),
            ("Grep", {"pattern": "zzz_no_match_beta", "path": str(self.workspace)}),
            ("Read", {"file_path": str(self.workspace / "nope_c3.py")}),
            ("Read", {"file_path": str(self.workspace / "nope_d4.py")}),
            ("Grep", {"pattern": "zzz_no_match_gamma", "path": str(self.workspace)}),
            ("Read", {"file_path": str(self.workspace / "nope_e5.py")}),
            ("Grep", {"pattern": "zzz_no_match_delta", "path": str(self.workspace)}),
            ("Read", {"file_path": str(self.workspace / "nope_f6.py")}),
        ]
        for i, (name, tool_input) in enumerate(seq):
            turns.append(_tool_turn(f"toolu_{i}", name, tool_input, text=f"Searching ({i})."))
        return turns

    def test_loop_returns_semantic_verdict_on_flailing_spiral(self):
        result = self._run(self._flailing_turns())
        self.assertEqual(result.response_text, SEMANTIC_MESSAGE)

    def test_loop_continues_when_write_succeeds(self):
        target = self.workspace / "notes.txt"
        turns = self._flailing_turns()
        # A successful Write mid-window is progress: no semantic trip.
        turns[4] = _tool_turn(
            "toolu_write",
            "Write",
            {"file_path": str(target), "content": "found it"},
            text="Writing.",
        )
        turns.append(_final_turn("All done."))
        result = self._run(turns)
        self.assertEqual(result.response_text, "All done.")
        self.assertTrue(target.exists())


if __name__ == "__main__":
    unittest.main()
