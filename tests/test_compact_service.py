"""
Tests for the compact service.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.conversation import (
    Conversation,
    Message,
    TextContentBlock,
    ToolResultContentBlock,
    ToolUseContentBlock,
)
from src.providers.base import ChatResponse


class MockConversation:
    """Mock conversation for testing (simple list-based)."""

    def __init__(self, messages=None):
        self.messages = messages or []

    def get_messages(self):
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.messages
        ]

    def clear(self):
        self.messages.clear()


class TestCompactBoundaryMessages(unittest.TestCase):
    """Tests for compact boundary message creation."""

    def test_creates_boundary_message(self):
        """Boundary message is created with correct properties."""
        from src.compact_service.messages import create_compact_boundary_message, is_compact_boundary_message
        msg = create_compact_boundary_message(
            trigger="manual",
            pre_compact_token_count=5000,
            messages_summarized=10,
        )
        self.assertTrue(is_compact_boundary_message(msg))
        self.assertEqual(msg.role, "system")
        self.assertTrue(getattr(msg, "_is_internal", False))

    def test_non_boundary_is_not_boundary(self):
        """Regular messages are not boundary messages."""
        from src.compact_service.messages import is_compact_boundary_message
        msg = Message(role="user", content="Hello")
        self.assertFalse(is_compact_boundary_message(msg))

    def test_get_messages_after_boundary_with_no_boundary(self):
        """Returns all messages when no boundary exists."""
        from src.compact_service.messages import get_messages_after_boundary
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
        ]
        result = get_messages_after_boundary(messages)
        self.assertEqual(len(result), 2)

    def test_get_messages_after_boundary_with_boundary(self):
        """Returns only messages after last boundary."""
        from src.compact_service.messages import create_compact_boundary_message, get_messages_after_boundary
        messages = [
            Message(role="user", content="Old"),
            Message(role="assistant", content="Old response"),
        ]
        boundary = create_compact_boundary_message(trigger="manual")
        messages.append(boundary)
        messages.append(Message(role="user", content="New"))
        messages.append(Message(role="assistant", content="New response"))

        result = get_messages_after_boundary(messages)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "New")
        self.assertEqual(result[1].content, "New response")

    def test_summary_message_format(self):
        """Summary message has correct role and content."""
        from src.compact_service.messages import create_compact_summary_message
        msg = create_compact_summary_message(
            "The user was working on a Python project. "
            "They asked about implementing a feature."
        )
        self.assertEqual(msg.role, "user")
        # msg.content is a list of ContentBlock, not a single TextContentBlock
        self.assertIn("continued from a previous conversation", msg.content[0].text)


class TestCompactConversation(unittest.TestCase):
    """Tests for compact_conversation()."""

    def setUp(self):
        """Set up test fixtures."""
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        """Clean up test fixtures."""
        self.tmpdir.cleanup()

    def _make_conversation(self, messages):
        """Create a Conversation with given messages."""
        conv = Conversation()
        conv.messages = messages
        return conv

    def test_not_enough_messages_raises(self):
        """Less than 2 messages raises ValueError."""
        from src.compact_service.service import compact_conversation
        conv = self._make_conversation([
            Message(role="user", content="Hello"),
        ])

        mock_provider = MagicMock()
        mock_provider.chat_async = AsyncMock(
            return_value=ChatResponse(content="Summary", model="test", usage={}, finish_reason="stop")
        )

        with self.assertRaises(ValueError) as ctx:
            asyncio.run(compact_conversation(conv, mock_provider, "claude-sonnet-4-6"))
        self.assertIn("Not enough messages", str(ctx.exception))

    def test_sync_fallback_on_llm_failure(self):
        """Falls back to sync extraction on LLM failure."""
        from src.compact_service.service import compact_conversation
        conv = self._make_conversation([
            Message(role="user", content="Hello world " * 50),
            Message(role="assistant", content="Hi there! " * 50),
            Message(role="user", content="What about the code? " * 50),
        ])

        mock_provider = MagicMock()
        mock_provider.chat_async = AsyncMock(side_effect=Exception("LLM failed"))
        mock_provider.chat = MagicMock(side_effect=Exception("Sync LLM failed"))

        result = asyncio.run(compact_conversation(conv, mock_provider, "claude-sonnet-4-6"))
        self.assertEqual(result.trigger, "manual")
        self.assertIn("Conversation had", result.summary_text)

    def test_boundary_and_summary_inserted(self):
        """Boundary and summary messages are inserted into conversation."""
        from src.compact_service.service import compact_conversation
        conv = self._make_conversation([
            Message(role="user", content="Hello world " * 50),
            Message(role="assistant", content="Hi there! " * 50),
            Message(role="user", content="What about the code? " * 50),
            Message(role="assistant", content="Here's the code... " * 50),
        ])
        original_count = len(conv.messages)

        mock_provider = MagicMock()
        mock_provider.chat_async = AsyncMock(
            return_value=ChatResponse(
                content="User worked on Python code. Assistant helped with implementation.",
                model="test",
                usage={},
                finish_reason="stop"
            )
        )

        result = asyncio.run(compact_conversation(conv, mock_provider, "claude-sonnet-4-6"))

        # Boundary and summary are added (2 new messages)
        self.assertEqual(len(conv.messages), result.post_compact_count)
        self.assertLess(len(conv.messages), original_count)

        # Check that internal boundary is present
        boundary = next(
            (m for m in conv.messages if getattr(m, "_is_internal", False)),
            None
        )
        self.assertIsNotNone(boundary)

    def test_custom_instructions_passed_to_llm(self):
        """Custom instructions are included in the prompt."""
        from src.compact_service.service import compact_conversation
        conv = self._make_conversation([
            Message(role="user", content="Hello " * 20),
            Message(role="assistant", content="Hi " * 20),
        ])

        captured_messages = []

        def capture_messages(*args, **kwargs):
            captured_messages.append(kwargs.get("messages", args[1] if len(args) > 1 else []))
            return ChatResponse(content="Summary", model="test", usage={}, finish_reason="stop")

        mock_provider = MagicMock()
        mock_provider.chat_async = AsyncMock(side_effect=capture_messages)

        asyncio.run(compact_conversation(
            conv, mock_provider, "claude-sonnet-4-6",
            custom_instructions="Focus on the Python code"
        ))

        # Custom instruction should appear in the last message
        self.assertGreater(len(captured_messages), 0)
        last_msg = captured_messages[0][-1] if captured_messages else {}
        self.assertIn("Focus on the Python code", last_msg.get("content", ""))


class TestCompactIntegration(unittest.TestCase):
    """Integration tests for compact with mock REPL flow."""

    def test_compact_preserves_boundary_marker_in_serialization(self):
        """Boundary markers are preserved when conversation is serialized."""
        from src.compact_service.messages import create_compact_boundary_message
        conv = Conversation()
        conv.messages.append(Message(role="user", content="Hello"))
        conv.messages.append(Message(role="assistant", content="Hi"))
        boundary = create_compact_boundary_message(trigger="manual", pre_compact_token_count=1000)
        conv.messages.append(boundary)

        # Serialize
        data = conv.to_dict()
        self.assertEqual(len(data["messages"]), 3)

        # Boundary should have _is_internal=True in serialized form
        boundary_data = data["messages"][2]
        self.assertTrue(boundary_data.get("_is_internal", False))

        # Deserialize
        restored = Conversation.from_dict(data)
        self.assertEqual(len(restored.messages), 3)

        # Boundary should still be internal
        self.assertTrue(getattr(restored.messages[2], "_is_internal", False))

        # get_messages() should skip internal messages
        api_messages = restored.get_messages()
        self.assertEqual(len(api_messages), 2)  # Only user + assistant

    def test_compact_result_dataclass(self):
        """CompactionResult dataclass has expected fields."""
        from src.command_system.types import CompactionResult
        result = CompactionResult(
            pre_compact_count=10,
            post_compact_count=2,
            tokens_saved=5000,
            trigger="manual",
            summary_preview="User worked on...",
        )
        self.assertEqual(result.pre_compact_count, 10)
        self.assertEqual(result.post_compact_count, 2)
        self.assertEqual(result.tokens_saved, 5000)
        self.assertEqual(result.trigger, "manual")
        self.assertIn("User worked on", result.summary_preview)


if __name__ == "__main__":
    unittest.main()


class TestSurvivalContext(unittest.TestCase):
    """Tests for post-compaction survival context re-injection."""

    def test_reinject_claude_md_and_recent_files(self):
        """CLAUDE.md and recently-modified files are re-injected."""
        from src.compact_service.service import reinject_survival_context
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAUDE.md").write_text("# Project rules\nAlways do X.")
            (root / "notes.txt").write_text("recent notes here")
            conv = Conversation()
            conv.messages.append(Message(role="user", content="hi"))
            msg = reinject_survival_context(conv, workspace_root=root)
            self.assertIsNotNone(msg)
            self.assertEqual(msg.role, "user")
            self.assertFalse(getattr(msg, "_is_internal", False))
            text = msg.content[0].text
            self.assertIn("Always do X.", text)
            self.assertIn("recent notes here", text)
            # Appended after existing messages
            self.assertIs(conv.messages[-1], msg)

    def test_reinject_empty_workspace_returns_none(self):
        """Nothing to re-inject -> None, conversation untouched."""
        from src.compact_service.service import reinject_survival_context
        with tempfile.TemporaryDirectory() as tmp:
            conv = Conversation()
            self.assertIsNone(reinject_survival_context(conv, workspace_root=tmp))
            self.assertEqual(len(conv.messages), 0)

    def test_reinject_never_raises(self):
        """Bogus roots and unreadable files are skipped, never raised."""
        from src.compact_service.service import reinject_survival_context
        conv = Conversation()
        # Must not raise on bogus roots; returns None when nothing found.
        self.assertIsNone(
            reinject_survival_context(conv, workspace_root="/nonexistent-xyz-123")
        )
        # workspace_root=None falls back to cwd — must not raise either.
        reinject_survival_context(conv, workspace_root=None)

    def test_skill_body_capped_at_5k_tokens(self):
        """Invoked skill bodies are capped at ~5000 tokens."""
        from src.compact_service.service import _load_skill_body
        from src.skills.model import PromptSkill
        from src.token_estimation import count_tokens
        big_body = "word " * 20000
        fake = PromptSkill(
            name="s", description="d", loaded_from="skills",
            user_invocable=True, disable_model_invocation=False,
            content_length=len(big_body), is_hidden=False, skill_root=None,
            when_to_use=None, version=None, model=None, allowed_tools=[],
            arg_names=[], context=None, agent=None, effort=None, paths=None,
            markdown_content=big_body,
        )
        with patch("src.skills.loader.find_skill", return_value=fake):
            body = _load_skill_body("s")
        self.assertIsNotNone(body)
        self.assertLessEqual(count_tokens(body), 5050)
        self.assertIn("[truncated]", body)

    def test_skill_body_missing_skill_returns_none(self):
        """Unknown skill names are skipped."""
        from src.compact_service.service import _load_skill_body
        with patch("src.skills.loader.find_skill", return_value=None):
            self.assertIsNone(_load_skill_body("nope-not-a-skill"))

    def test_invoked_skill_names_extracted(self):
        """Skill tool_use blocks are collected; other tools ignored."""
        from src.compact_service.service import _invoked_skill_names
        msgs = [
            Message(role="assistant", content=[
                ToolUseContentBlock(type="tool_use", id="1", name="Skill",
                                    input={"skill": "my-skill"}),
                ToolUseContentBlock(type="tool_use", id="2", name="Bash",
                                    input={"command": "ls"}),
                ToolUseContentBlock(type="tool_use", id="3", name="Skill",
                                    input={"skill": "my-skill"}),
            ]),
        ]
        self.assertEqual(_invoked_skill_names(msgs), ["my-skill"])

    def test_compact_conversation_reinjects_survival(self):
        """compact_conversation appends survival context when workspace_root given."""
        from src.compact_service.service import compact_conversation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAUDE.md").write_text("Project rule: be concise.")
            conv = Conversation()
            conv.messages = [
                Message(role="user", content="Hello world " * 50),
                Message(role="assistant", content="Hi there! " * 50),
            ]
            mock_provider = MagicMock()
            mock_provider.chat_async = AsyncMock(
                return_value=ChatResponse(
                    content="Summary.", model="test", usage={},
                    finish_reason="stop")
            )
            result = asyncio.run(compact_conversation(
                conv, mock_provider, "test-model", workspace_root=root))
            texts = " ".join(
                block.text
                for m in conv.messages
                for block in (m.content if isinstance(m.content, list) else [])
                if hasattr(block, "text")
            )
            self.assertIn("be concise", texts)
            self.assertEqual(len(conv.messages), result.post_compact_count)


class TestCompactHardFlag(unittest.TestCase):
    """Tests for /compact --hard and instruction passthrough in builtins."""

    def _context(self, conv, provider=None, model="t"):
        from src.command_system.engine import create_command_context
        config = {"model": model}
        if provider is not None:
            config["provider"] = provider
        return create_command_context(
            workspace_root="/tmp", conversation=conv, config=config)

    def _conv(self):
        conv = Conversation()
        conv.messages = [
            Message(role="user", content="a"),
            Message(role="assistant", content="b"),
        ]
        return conv

    def test_hard_clears_conversation(self):
        """--hard keeps the legacy clear-everything behavior."""
        from src.command_system.builtins import compact_command_call
        conv = self._conv()
        result = compact_command_call("--hard", self._context(conv))
        self.assertEqual(len(conv.messages), 0)
        self.assertIn("cleared", result.value.lower())

    def test_hard_with_instructions_still_clears(self):
        """--hard wins even when instructions are present."""
        from src.command_system.builtins import compact_command_call
        conv = self._conv()
        result = compact_command_call("focus on tests --hard", self._context(conv))
        self.assertEqual(len(conv.messages), 0)
        self.assertIn("cleared", result.value.lower())

    def test_instructions_steer_summary_via_command(self):
        """/compact <instructions> reaches the summarizer prompt."""
        from src.command_system.builtins import compact_command_call
        conv = Conversation()
        conv.messages = [
            Message(role="user", content="a " * 30),
            Message(role="assistant", content="b " * 30),
        ]
        captured = {}

        async def fake_chat_async(*args, **kwargs):
            captured["messages"] = kwargs.get("messages")
            return ChatResponse(content="S", model="t", usage={},
                                finish_reason="stop")

        provider = MagicMock()
        provider.chat_async = fake_chat_async
        result = compact_command_call(
            "focus on the database layer", self._context(conv, provider))
        self.assertEqual(result.type, "compact")
        last = captured["messages"][-1]
        self.assertIn("focus on the database layer", last["content"])

    def test_hard_does_not_call_llm(self):
        """--hard never touches the provider."""
        from src.command_system.builtins import compact_command_call
        conv = self._conv()
        provider = MagicMock()
        provider.chat_async = AsyncMock()
        compact_command_call("--hard", self._context(conv, provider))
        provider.chat_async.assert_not_called()
