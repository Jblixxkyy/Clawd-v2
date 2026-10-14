"""Tests for auto-compaction: context limits and the agent-loop trigger."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.conversation import Conversation
from src.context_limits import (
    AUTO_COMPACT_FRACTION,
    DEFAULT_CONTEXT_LIMIT,
    auto_compact_threshold,
    get_context_limit,
    should_auto_compact,
)
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import _maybe_auto_compact, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


def _summary_provider(summary="Summary of prior work."):
    """Mock provider whose summary call succeeds."""
    p = MagicMock()
    p.chat_async = AsyncMock(
        return_value=ChatResponse(
            content=summary, model="test", usage={}, finish_reason="stop"
        )
    )
    return p


class TestContextLimits(unittest.TestCase):
    """Per-model context limit table."""

    def test_known_models(self):
        self.assertEqual(get_context_limit("claude-sonnet-4-6"), 200_000)
        self.assertEqual(get_context_limit("gpt-4o"), 128_000)
        self.assertEqual(get_context_limit("gpt-5"), 400_000)
        self.assertEqual(get_context_limit("gemini-2.5-pro"), 1_000_000)
        self.assertEqual(get_context_limit("nvidia/nemotron-3.5-lightning-30b"), 128_000)
        self.assertEqual(get_context_limit("deepseek-chat"), 128_000)
        self.assertEqual(get_context_limit("glm-4.5"), 128_000)

    def test_case_insensitive(self):
        self.assertEqual(get_context_limit("Claude-Opus-4"), 200_000)
        self.assertEqual(get_context_limit("GPT-4O"), 128_000)

    def test_unknown_model_falls_back(self):
        self.assertEqual(
            get_context_limit("some-future-model-xyz"), DEFAULT_CONTEXT_LIMIT
        )
        self.assertEqual(get_context_limit(""), DEFAULT_CONTEXT_LIMIT)
        self.assertEqual(get_context_limit(None), DEFAULT_CONTEXT_LIMIT)

    def test_threshold_and_trigger(self):
        self.assertEqual(
            auto_compact_threshold("gpt-4o"), int(128_000 * AUTO_COMPACT_FRACTION)
        )
        self.assertTrue(should_auto_compact(int(128_000 * 0.95), "gpt-4o"))
        self.assertFalse(should_auto_compact(int(128_000 * 0.5), "gpt-4o"))
        self.assertFalse(should_auto_compact(0, "gpt-4o"))
        self.assertFalse(should_auto_compact(-5, "gpt-4o"))


class TestMaybeAutoCompact(unittest.TestCase):
    """Direct tests of the auto-compact trigger logic."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = ToolContext(workspace_root=Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def _conv(self):
        conv = Conversation()
        conv.add_user_message("hello")
        conv.add_assistant_message("hi there")
        conv.add_user_message("do the thing")
        return conv

    def test_no_compact_under_threshold(self):
        conv = self._conv()
        provider = _summary_provider()
        with patch("src.token_estimation.count_messages_tokens", return_value=1000):
            result = _maybe_auto_compact(conv, provider, [], "sys", self.ctx)
        self.assertFalse(result)
        self.assertEqual(len(conv.messages), 3)
        provider.chat_async.assert_not_called()

    def test_compact_over_threshold(self):
        conv = self._conv()
        provider = _summary_provider()
        with patch(
            "src.token_estimation.count_messages_tokens", return_value=200_000
        ):
            result = _maybe_auto_compact(conv, provider, [], "sys", self.ctx)
        self.assertTrue(result)
        provider.chat_async.assert_called_once()
        # Boundary marker present; original messages summarized away.
        # (tmp workspace is empty, so no survival message is appended.)
        self.assertTrue(
            any(getattr(m, "_is_internal", False) for m in conv.messages)
        )
        self.assertEqual(len(conv.messages), 2)

    def test_compact_failure_returns_false_never_raises(self):
        conv = self._conv()
        provider = _summary_provider()
        with patch(
            "src.token_estimation.count_messages_tokens", return_value=200_000
        ), patch(
            "src.compact_service.service.compact_conversation_sync",
            side_effect=RuntimeError("nope"),
        ):
            result = _maybe_auto_compact(conv, provider, [], "sys", self.ctx)
        self.assertFalse(result)


class TestAgentLoopAutoCompact(unittest.TestCase):
    """Auto-compact inside run_agent_loop: the session survives and continues."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.registry = build_default_registry()
        self.ctx = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.tmp.cleanup()

    def _provider(self, final="Done."):
        p = _summary_provider()
        p.chat = MagicMock(
            return_value=ChatResponse(
                content=final, model="test", usage={}, finish_reason="stop"
            )
        )
        return p

    def _conv(self):
        conv = Conversation()
        conv.add_user_message("please do the thing")
        conv.add_assistant_message("working on it")
        return conv

    def test_loop_auto_compacts_and_continues(self):
        conv = self._conv()
        provider = self._provider()
        with patch(
            "src.token_estimation.count_messages_tokens", return_value=200_000
        ):
            result = run_agent_loop(
                conv, provider, self.registry, self.ctx, max_turns=5
            )
        # Session survived: boundary marker present, loop produced an answer.
        self.assertTrue(
            any(getattr(m, "_is_internal", False) for m in conv.messages)
        )
        self.assertEqual(result.response_text, "Done.")
        self.assertGreaterEqual(result.num_turns, 1)
        # One summary call + at least one real turn call.
        provider.chat_async.assert_called_once()
        self.assertGreaterEqual(provider.chat.call_count, 1)

    def test_loop_no_auto_compact_when_disabled(self):
        conv = self._conv()
        provider = self._provider()
        with patch(
            "src.token_estimation.count_messages_tokens", return_value=200_000
        ):
            result = run_agent_loop(
                conv, provider, self.registry, self.ctx,
                max_turns=5, auto_compact=False,
            )
        self.assertFalse(
            any(getattr(m, "_is_internal", False) for m in conv.messages)
        )
        provider.chat_async.assert_not_called()
        self.assertEqual(result.response_text, "Done.")

    def test_loop_no_compact_under_threshold(self):
        conv = self._conv()
        provider = self._provider()
        with patch("src.token_estimation.count_messages_tokens", return_value=500):
            result = run_agent_loop(
                conv, provider, self.registry, self.ctx, max_turns=5
            )
        provider.chat_async.assert_not_called()
        self.assertEqual(result.response_text, "Done.")


if __name__ == "__main__":
    unittest.main()
