"""Loop-wiring tests for brief 036 — environment capability card.

Verifies that ``_run_agent_loop_impl`` (via ``run_agent_loop``) appends the
card from ``src.tool_system.env_card.build_env_card`` to the turn-0 system
message exactly once, never re-appends it on turn-1+ deltas, and stays
fail-open when card construction blows up.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system import env_card
from src.tool_system.agent_loop import run_agent_loop, AgentLoopResult
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry

MARKER = "ENV-CARD-TEST-MARKER-036"


def _stop_response() -> ChatResponse:
    return ChatResponse(
        content="done",
        model="test-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="stop",
        tool_uses=None,
    )


def _write_then_stop(workspace: Path, mock_provider: MagicMock) -> None:
    mock_provider.chat_stream_response.side_effect = NotImplementedError()
    tool_use = {
        "id": "toolu_1",
        "name": "Write",
        "input": {
            "file_path": str(workspace / "wiring_probe.txt"),
            "content": "probe",
        },
    }
    mock_provider.chat.side_effect = [
        ChatResponse(
            content="writing",
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="tool_use",
            tool_uses=[tool_use],
        ),
        _stop_response(),
    ]


def _run_loop(workspace: Path, mock_provider: MagicMock) -> AgentLoopResult:
    conversation = Conversation()
    conversation.add_user_message("do the thing")
    registry = build_default_registry()
    context = ToolContext(workspace_root=workspace)
    return run_agent_loop(
        conversation=conversation,
        provider=mock_provider,
        tool_registry=registry,
        tool_context=context,
        verbose=False,
    )


def _messages_of(call) -> list:
    return list(call.args[0])


class TestEnvCardWiring(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self._orig = env_card.build_env_card

    def tearDown(self):
        env_card.build_env_card = self._orig
        self.temp_dir.cleanup()

    def test_card_appears_in_turn0_system_message(self):
        """Mocked card builder -> card text lands in the turn-0 system msg."""
        env_card.build_env_card = lambda ctx: MARKER  # noqa: E731
        mock_provider = MagicMock()
        mock_provider.chat.side_effect = [_stop_response()]

        result = _run_loop(self.workspace, mock_provider)

        self.assertIsInstance(result, AgentLoopResult)
        self.assertEqual(mock_provider.chat.call_count, 1)
        messages = _messages_of(mock_provider.chat.call_args_list[0])
        system_msgs = [m for m in messages if m.get("role") == "system"]
        self.assertTrue(system_msgs, "expected a system message on turn 0")
        self.assertIn(MARKER, system_msgs[0]["content"])

    def test_card_not_reappended_on_turn1_delta(self):
        """Turn 1+ deltas must not inject the card again (002 delta system)."""
        env_card.build_env_card = lambda ctx: MARKER  # noqa: E731
        mock_provider = MagicMock()
        _write_then_stop(self.workspace, mock_provider)

        _run_loop(self.workspace, mock_provider)

        self.assertEqual(mock_provider.chat.call_count, 2)
        turn1_messages = _messages_of(mock_provider.chat.call_args_list[1])
        marker_hits = [
            m for m in turn1_messages if MARKER in str(m.get("content", ""))
        ]
        # The turn-0 system message persists in history (intended); the delta
        # path must not add a second copy.
        self.assertEqual(
            len(marker_hits), 1,
            f"card re-appended on turn 1: {len(marker_hits)} hits",
        )

    def test_card_builder_exception_is_fail_open(self):
        """A raising card builder must not break loop start; no card ships."""

        def boom(ctx):
            raise RuntimeError("detection exploded")

        env_card.build_env_card = boom
        mock_provider = MagicMock()
        mock_provider.chat.side_effect = [_stop_response()]

        result = _run_loop(self.workspace, mock_provider)

        self.assertIsInstance(result, AgentLoopResult)
        self.assertEqual(mock_provider.chat.call_count, 1)
        messages = _messages_of(mock_provider.chat.call_args_list[0])
        system_msgs = [m for m in messages if m.get("role") == "system"]
        self.assertTrue(system_msgs)
        self.assertNotIn(MARKER, system_msgs[0]["content"])

    def test_empty_card_means_no_card_text(self):
        """build_env_card returning '' appends nothing (no stray newlines)."""
        env_card.build_env_card = lambda ctx: ""  # noqa: E731
        mock_provider = MagicMock()
        mock_provider.chat.side_effect = [_stop_response()]

        _run_loop(self.workspace, mock_provider)

        messages = _messages_of(mock_provider.chat.call_args_list[0])
        system_msgs = [m for m in messages if m.get("role") == "system"]
        self.assertTrue(system_msgs)
        self.assertFalse(
            system_msgs[0]["content"].endswith("\n\n"),
            "empty card must not append trailing newlines",
        )


if __name__ == "__main__":
    unittest.main()
