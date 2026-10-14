"""Tests for brief 035 — empty-response retry.

When a model call succeeds but returns no text and no tool calls
(empty/whitespace content), the loop retries the provider call up to 2 more
times (same messages) before accepting the turn. If all attempts come back
empty, the turn ends with an explicit notice instead of an empty bubble.

These tests live in a dedicated file per the brief; the helper-only checks
also live in tests/test_harness_guards.py::TestEmptyResponseRetry.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system import agent_loop
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


NOTICE = "[The model returned an empty response after 3 attempts. Try again.]"


def _resp(content="", tool_uses=None):
    return ChatResponse(
        content=content,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="stop",
        tool_uses=tool_uses,
    )


def _assistant_texts(conv: Conversation) -> list[str]:
    return [
        m.content if isinstance(m.content, str) else ""
        for m in conv.messages
        if m.role == "assistant"
    ]


class TestEmptyResponseRetry(unittest.TestCase):
    def _run(self, responses):
        conv = Conversation()
        conv.add_user_message("hello")
        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.side_effect = responses
        ctx = ToolContext(workspace_root=Path(tempfile.mkdtemp()))
        result = run_agent_loop(
            conversation=conv, provider=provider,
            tool_registry=build_default_registry(), tool_context=ctx,
            verbose=False,
        )
        return result, provider, conv

    def test_empty_twice_then_content_succeeds(self):
        """Mock provider returns empty twice, then content: turn succeeds,
        provider called 3 times, no empty message saved."""
        result, provider, conv = self._run(
            [_resp("  "), _resp(""), _resp("hi there")]
        )
        self.assertEqual(result.response_text, "hi there")
        self.assertEqual(provider.chat.call_count, 3)
        self.assertEqual(result.num_turns, 1)
        for text in _assistant_texts(conv):
            self.assertTrue(text.strip(), "no blank assistant message saved")

    def test_always_empty_returns_explicit_notice(self):
        """3 empty attempts in a row: result text is the notice, not ''."""
        result, provider, conv = self._run(
            [_resp(" "), _resp(""), _resp("\n")]
        )
        self.assertEqual(provider.chat.call_count, 3)
        self.assertEqual(result.response_text, NOTICE)
        self.assertTrue(result.response_text.strip())
        saved = _assistant_texts(conv)
        self.assertTrue(saved, "the notice is saved to the conversation")
        self.assertEqual(saved[-1], NOTICE)

    def test_non_empty_first_response_single_call(self):
        """Non-empty first response: exactly 1 provider call (no behavior
        change for healthy responses)."""
        result, provider, conv = self._run([_resp("hello!")])
        self.assertEqual(provider.chat.call_count, 1)
        self.assertEqual(result.response_text, "hello!")
        self.assertEqual(_assistant_texts(conv)[-1], "hello!")

    def test_empty_text_with_tool_calls_is_not_empty(self):
        """An empty-text response WITH tool calls is not treated as empty:
        the tool dispatches and the turn continues normally."""
        tool_use = {"id": "t1", "name": "Read",
                    "input": {"file_path": "/tmp/x"}}
        result, provider, _ = self._run([
            _resp(" ", tool_uses=[tool_use]),
            _resp("done"),
        ])
        self.assertEqual(provider.chat.call_count, 2)
        self.assertEqual(result.response_text, "done")

    def test_retries_do_not_burn_turns(self):
        """Empty retries do not increment the turn count."""
        result, _, _ = self._run(
            [_resp(""), _resp(""), _resp("final answer")]
        )
        self.assertEqual(result.response_text, "final answer")
        self.assertEqual(result.num_turns, 1)

    def test_is_empty_response_helper(self):
        self.assertTrue(agent_loop._is_empty_response(_resp("   ")))
        self.assertTrue(agent_loop._is_empty_response(_resp("")))
        self.assertFalse(agent_loop._is_empty_response(_resp("x")))
        self.assertFalse(agent_loop._is_empty_response(
            _resp("", tool_uses=[{"id": "t", "name": "Read", "input": {}}])))
        # A broken/missing-attribute response must not raise; it classifies
        # as empty (retry direction) rather than crashing the loop.
        self.assertTrue(agent_loop._is_empty_response(object()))


if __name__ == "__main__":
    unittest.main()
