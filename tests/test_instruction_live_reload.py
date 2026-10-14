"""Test instruction-file live reload (brief 012).

Project instruction files (AGENTS.md, CLAUDE.md) are snapshotted at session
start; a mid-session edit is surfaced to the model as a transient
<system-reminder> message naming the file plus a compact diff summary.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from src.agent.conversation import Conversation
from src.context_system.claude_md import (
    InstructionWatcher,
    format_instruction_change_reminder,
    snapshot_instruction_files,
)
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import AgentLoopResult, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


def _bash_response(command):
    """A harmless tool-use that keeps the loop going for another turn."""
    return ChatResponse(
        content="",
        model="test-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="tool_use",
        tool_uses=[{
            "id": "toolu_bash",
            "name": "Bash",
            "input": {"command": command},
        }],
    )


def _stop_response(text="all done"):
    return ChatResponse(
        content=text,
        model="test-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="stop",
        tool_uses=None,
    )


# Unique headers of the live-reload reminders (brief 012). The base system
# prompt documents the <system-reminder> convention itself, so matching on
# the tag alone would also match the turn-0 system message.
_REMINDER_MARKERS = (
    "The project instructions changed:",
    "New project instructions apply from:",
)


def _is_reload_reminder(content):
    return isinstance(content, str) and any(m in content for m in _REMINDER_MARKERS)


def _reminder_texts(mock_provider):
    """Every live-reload reminder text sent to the provider, per turn."""
    texts = []
    for call in mock_provider.chat.call_args_list:
        messages = call.args[0] if call.args else call.kwargs.get("api_messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if _is_reload_reminder(content):
                texts.append(content)
    return texts


class TestInstructionWatcherUnit(unittest.TestCase):
    """Unit tests for the snapshot / poll / debounce mechanics."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.instructions = Path(self.temp_dir.name) / "CLAUDE.md"
        self.instructions.write_text("# Rules\n\n- Be concise.\n", encoding="utf-8")
        self.watcher = InstructionWatcher([self.instructions])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_unchanged_file_no_notification(self):
        self.assertEqual(self.watcher.check_for_changes(), [])

    def test_mtime_only_change_no_notification(self):
        """A save that only touches the mtime (identical content) stays silent."""
        st = self.instructions.stat()
        os.utime(self.instructions, (st.st_atime + 5, st.st_mtime + 5))
        self.assertEqual(self.watcher.check_for_changes(), [])

    def test_modified_file_reports_filename_and_diff(self):
        self.instructions.write_text(
            "# Rules\n\n- Be concise.\n- Never use emojis.\n", encoding="utf-8"
        )
        reminders = self.watcher.check_for_changes()
        self.assertEqual(len(reminders), 1)
        self.assertIn("CLAUDE.md", reminders[0])
        self.assertIn("<system-reminder>", reminders[0])

    def test_rapid_double_save_single_notification(self):
        """Two saves inside the debounce window -> one notification."""
        self.instructions.write_text("# Rules v2\n", encoding="utf-8")
        first = self.watcher.check_for_changes()
        self.assertEqual(len(first), 1)
        # Second save lands well inside the ~2s debounce window.
        self.instructions.write_text("# Rules v3\n", encoding="utf-8")
        second = self.watcher.check_for_changes()
        self.assertEqual(second, [])

    def test_snapshot_records_mtime_and_hash(self):
        entry = self.watcher._entries[str(self.instructions)]
        self.assertIsNotNone(entry)
        mtime_ns, digest, content = entry
        self.assertEqual(mtime_ns, self.instructions.stat().st_mtime_ns)
        self.assertEqual(len(digest), 64)  # sha256 hex
        self.assertIn("Be concise", content)


class TestDiffSummaryCompact(unittest.TestCase):
    """The reminder carries a compact diff summary, not the full file."""

    def test_summary_counts_and_samples_not_full_file(self):
        old_lines = [f"unchanged line {i}" for i in range(30)]
        new_lines = old_lines + ["- New rule: be concise.", "- New rule: be kind."]
        old_text = "\n".join(old_lines) + "\n"
        new_text = "\n".join(new_lines) + "\n"
        reminder = format_instruction_change_reminder(
            "CLAUDE.md", old_text, new_text
        )
        self.assertIn("CLAUDE.md", reminder)
        self.assertIn("+2 lines", reminder)
        self.assertIn("- New rule: be concise.", reminder)
        # Compact: far shorter than the full new file, and an untouched
        # middle line is not echoed back.
        self.assertLess(len(reminder), len(new_text))
        self.assertNotIn("unchanged line 15", reminder)

    def test_new_file_reminder(self):
        reminder = format_instruction_change_reminder(
            "AGENTS.md", None, "# Agents\n\n- Follow the plan.\n"
        )
        self.assertIn("AGENTS.md", reminder)
        self.assertIn("New project instructions apply from", reminder)


class TestSnapshotInstructionFiles(unittest.TestCase):
    """snapshot_instruction_files covers loaded CLAUDE.md + AGENTS.md scheme."""

    def test_watches_loaded_claude_md_and_agents_scheme(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CLAUDE.md").write_text("# rules\n", encoding="utf-8")
            (root / "AGENTS.md").write_text("# agents\n", encoding="utf-8")
            watcher = snapshot_instruction_files(root)
            watched = set(watcher._entries)
            self.assertIn(str(root / "CLAUDE.md"), watched)
            self.assertIn(str(root / "AGENTS.md"), watched)
            # No false positive at rest.
            self.assertEqual(watcher.check_for_changes(), [])
            # A mid-session AGENTS.md edit is detected too.
            (root / "AGENTS.md").write_text("# agents\n\n- x\n", encoding="utf-8")
            reminders = watcher.check_for_changes()
            self.assertEqual(len(reminders), 1)
            self.assertIn("AGENTS.md", reminders[0])


class TestLiveReloadIntegration(unittest.TestCase):
    """Integration: run_agent_loop injects the notification mid-loop."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.claude_md = self.workspace / "CLAUDE.md"
        self.claude_md.write_text("# Rules\n\n- Be concise.\n", encoding="utf-8")
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, responses):
        conversation = Conversation()
        conversation.add_user_message("work on the task")
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        mock_provider.chat.side_effect = responses
        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )
        return result, mock_provider, conversation

    def _run_side_effect(self, chat_side_effect):
        conversation = Conversation()
        conversation.add_user_message("work on the task")
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        mock_provider.chat.side_effect = chat_side_effect
        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )
        return result, mock_provider, conversation

    def test_modified_file_injects_notification_with_filename(self):
        """Mid-loop edit -> reminder injected with the filename."""
        # The harness blocks Write to documentation files, so the edit is
        # made directly in the provider mock: it runs after turn 0's check
        # and before turn 1's check -- exactly a mid-loop edit.
        state = {"edited": False}

        def chat_side_effect(messages, **kwargs):
            if not state["edited"]:
                state["edited"] = True
                self.claude_md.write_text(
                    "# Rules\n\n- Be concise.\n- Never use emojis.\n",
                    encoding="utf-8",
                )
                return _bash_response("echo turn-zero")
            return _stop_response()

        result, provider, _conversation = self._run_side_effect(chat_side_effect)
        self.assertIsInstance(result, AgentLoopResult)
        reminders = _reminder_texts(provider)
        self.assertEqual(len(reminders), 1)
        self.assertIn("CLAUDE.md", reminders[0])
        self.assertIn("Never use emojis", reminders[0])

    def test_unchanged_file_no_notification_no_extra_tokens(self):
        """No edit -> no reminder injected on any turn (no extra prompt tokens)."""
        calls = {"n": 0}

        def chat_side_effect(messages, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _bash_response("echo turn-zero")
            return _stop_response()

        _result, provider, _conversation = self._run_side_effect(chat_side_effect)
        self.assertEqual(_reminder_texts(provider), [])
        for call in provider.chat.call_args_list:
            messages = call.args[0]
            for msg in messages:
                self.assertFalse(_is_reload_reminder(msg.get("content", "")))

    def test_rapid_double_save_single_notification(self):
        """Two quick mid-loop saves -> exactly one notification (debounce)."""
        # Turn 0's provider call saves v2 (seen by turn 1's check -> notify);
        # turn 1's provider call saves v3 within the debounce window, so
        # turn 2's check stays silent. One notification total.
        calls = {"n": 0}

        def chat_side_effect(messages, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                self.claude_md.write_text("# Rules v2\n", encoding="utf-8")
                return _bash_response("echo one")
            if calls["n"] == 2:
                self.claude_md.write_text("# Rules v3\n", encoding="utf-8")
                return _bash_response("echo two")
            return _stop_response()

        _result, provider, _conversation = self._run_side_effect(chat_side_effect)
        reminders = _reminder_texts(provider)
        self.assertEqual(len(reminders), 1)
        self.assertIn("CLAUDE.md", reminders[0])

    def test_notification_is_transient(self):
        """The reminder goes to the provider call but not the conversation history."""
        def chat_side_effect(messages, **kwargs):
            if not getattr(chat_side_effect, "done", False):
                chat_side_effect.done = True
                self.claude_md.write_text("# Rules v2\n", encoding="utf-8")
                return _bash_response("echo one")
            return _stop_response()

        _result, provider, conversation = self._run_side_effect(chat_side_effect)
        self.assertEqual(len(_reminder_texts(provider)), 1)
        self.assertFalse(
            any(_is_reload_reminder(m.content) for m in conversation.messages)
        )


class TestExitFiresSessionEnd(unittest.TestCase):
    """The REPL /exit path fires the SessionEnd lifecycle hook (brief 008 fix)."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / ".clawd"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        test_config = {
            "default_provider": "glm",
            "providers": {
                "glm": {
                    "api_key": "test_api_key_12345678",
                    "base_url": "https://open.bigmodel.cn/api/paas/v4",
                    "default_model": "glm-4.5",
                }
            },
        }
        with open(self.config_dir / "config.json", "w") as f:
            json.dump(test_config, f)

    def test_exit_fires_session_end_hook(self):
        """handle_command('/exit') fires SessionEnd before sys.exit."""
        from src.repl import ClawdREPL

        with patch('src.config.get_config_path',
                   return_value=self.config_dir / "config.json"):
            with patch('src.repl.core.Session.create'):
                with patch('src.repl.core.get_provider_class') as mock_provider_class:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_provider_class.return_value = mock_provider
                    repl = ClawdREPL(provider_name="glm")

                    with patch('src.repl.core._fire_lifecycle_hook') as fire:
                        with self.assertRaises(SystemExit):
                            repl.handle_command("/exit")

                    fire.assert_called_once()
                    event, extra = fire.call_args.args
                    self.assertEqual(event, "SessionEnd")
                    self.assertEqual(extra.get("reason"), "exit")


if __name__ == "__main__":
    unittest.main()
