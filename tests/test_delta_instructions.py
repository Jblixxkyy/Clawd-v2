"""Delta instruction system tests (brief 002, opencode parity).

Verifies that the agent loop sends the full system prompt on turn 0 and only
the static baseline + changed dynamic sections on later turns, with byte
counts as evidence.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.delta_instructions import (
    DeltaInstructionTracker,
    summarize_delta_report,
)


def _ctx_for(workspace: Path) -> ToolContext:
    ctx = ToolContext(workspace_root=workspace)
    ctx.cwd = workspace
    return ctx


def _resp(content: str, tool_uses=None) -> ChatResponse:
    return ChatResponse(
        content=content,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="tool_use" if tool_uses else "stop",
        tool_uses=tool_uses,
    )


class TestDeltaTracker(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.env = patch.dict(
            os.environ,
            {
                "CLAWD_SKILLS_DIR": str(self.root / "none"),
                "HOME": str(self.root / "none"),
            },
        )
        self.env.start()
        # Pre-create the project skills dir so installing a skill mid-session
        # does not change the workspace snapshot (context section stays stable).
        (self.root / ".clawd" / "skills").mkdir(parents=True)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_turn0_full_then_baseline_only(self):
        from src.outputStyles.base_prompt import BASE_SYSTEM_PROMPT

        tracker = DeltaInstructionTracker(BASE_SYSTEM_PROMPT, _ctx_for(self.root), {"read"})
        full = tracker.full_prompt()
        system2, report = tracker.next_system(turn=1)
        # Nothing changed: turn 2 carries only the baseline.
        self.assertFalse(report.changed)
        self.assertEqual(report.deltas, [])
        self.assertEqual(system2, tracker.baseline())
        self.assertIn("No skills are currently available.", full)
        self.assertNotIn("No skills are currently available.", system2)
        self.assertLess(report.system_bytes, report.full_bytes)

    def test_skills_change_produces_delta(self):
        from src.outputStyles.base_prompt import BASE_SYSTEM_PROMPT

        tracker = DeltaInstructionTracker(BASE_SYSTEM_PROMPT, _ctx_for(self.root), {"read"})
        tracker.full_prompt()
        # Install a skill mid-session (instant discovery).
        skill_dir = self.root / ".clawd" / "skills" / "newskill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: newskill\ndescription: Does new things\n---\nDo them."
        )
        system2, report = tracker.next_system(turn=1)
        self.assertTrue(report.changed)
        self.assertEqual(len(report.deltas), 1)
        self.assertEqual(report.deltas[0].name, "skills")
        # The "No skills are currently available." sentence is replaced by the
        # inventory, so the section counts as updated.
        self.assertEqual(report.deltas[0].kind, "updated")
        self.assertIn("## skills (updated)", system2)
        self.assertIn("- /newskill:", system2)
        # Unchanged context section is NOT re-sent.
        self.assertNotIn("## Runtime Context", system2)

    def test_report_evidence_shape(self):
        from src.outputStyles.base_prompt import BASE_SYSTEM_PROMPT

        tracker = DeltaInstructionTracker(BASE_SYSTEM_PROMPT, _ctx_for(self.root), {"read"})
        tracker.full_prompt()
        _, report = tracker.next_system(turn=1)
        evidence = summarize_delta_report(report)
        self.assertEqual(evidence["turn"], 1)
        self.assertFalse(evidence["changed"])
        self.assertGreater(evidence["full_bytes"], 0)
        self.assertEqual(
            evidence["saved_bytes"], evidence["full_bytes"] - evidence["system_bytes"]
        )
        self.assertGreater(evidence["saved_bytes"], 0)


class TestDeltaLiveSession(unittest.TestCase):
    """Two-turn live sessions proving turn 2 carries only the delta."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.registry = build_default_registry()
        self.context = _ctx_for(self.root)
        self.env = patch.dict(
            os.environ,
            {
                "CLAWD_SKILLS_DIR": str(self.root / "none"),
                "HOME": str(self.root / "none"),
            },
        )
        self.env.start()
        # Pre-create the project skills dir so installing a skill mid-session
        # does not change the workspace snapshot (context section stays stable).
        (self.root / ".clawd" / "skills").mkdir(parents=True)

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def _provider(self, responses):
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        mock_provider.chat.side_effect = responses
        return mock_provider

    def _install_skill_tool_use(self):
        # Write blocks documentation files (SKILL.md) unless allow_docs is on,
        # so the model installs the skill through Bash instead.
        return {
            "id": "toolu_1",
            "name": "Bash",
            "input": {
                "command": (
                    "mkdir -p .clawd/skills/liveskill && "
                    "printf -- '---\\nname: liveskill\\ndescription: Live delta proof\\n---\\nDo it.\\n'"
                    " > .clawd/skills/liveskill/SKILL.md"
                ),
            },
        }

    def test_openai_path_appends_delta_only_when_changed(self):
        conv = Conversation()
        conv.add_user_message("install the liveskill skill")
        provider = self._provider(
            [_resp("installing", [self._install_skill_tool_use()]), _resp("done")]
        )
        result = run_agent_loop(
            conversation=conv,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )
        self.assertEqual(result.response_text, "done")
        self.assertEqual(provider.chat.call_count, 2)
        calls = provider.chat.call_args_list
        # Turn 0: full prompt as the first system message.
        turn0_msgs = calls[0].args[0]
        self.assertEqual(turn0_msgs[0]["role"], "system")
        self.assertIn("No skills are currently available.", turn0_msgs[0]["content"])
        # Turn 1: exactly one appended delta system message, carrying only the change.
        turn1_msgs = calls[1].args[0]
        system_msgs = [m for m in turn1_msgs if m["role"] == "system"]
        self.assertEqual(len(system_msgs), 2)
        delta = system_msgs[1]["content"]
        self.assertIn("## skills (updated)", delta)
        self.assertIn("- /liveskill:", delta)
        self.assertNotIn("## Runtime Context", delta)
        # Evidence: the delta is far smaller than the full prompt.
        self.assertLess(len(delta.encode()), len(turn0_msgs[0]["content"].encode()) // 2)

    def test_openai_path_no_delta_when_unchanged(self):
        conv = Conversation()
        conv.add_user_message("say hi")
        provider = self._provider([_resp("hi")])
        result = run_agent_loop(
            conversation=conv,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )
        self.assertEqual(result.response_text, "hi")
        # Single turn: only the turn-0 system message exists.
        turn0_msgs = provider.chat.call_args_list[0].args[0]
        system_msgs = [m for m in turn0_msgs if m["role"] == "system"]
        self.assertEqual(len(system_msgs), 1)

    def test_anthropic_path_turn2_carries_baseline_only(self):
        conv = Conversation()
        conv.add_user_message("read a file")
        read_use = {
            "id": "toolu_1",
            "name": "Read",
            "input": {"file_path": str(self.root / "note.txt")},
        }
        (self.root / "note.txt").write_text("hello")
        provider = self._provider([_resp("reading", [read_use]), _resp("read it")])
        with patch(
            "src.tool_system.agent_loop._is_anthropic_provider", return_value=True
        ):
            result = run_agent_loop(
                conversation=conv,
                provider=provider,
                tool_registry=self.registry,
                tool_context=self.context,
                verbose=False,
            )
        self.assertEqual(result.response_text, "read it")
        self.assertEqual(provider.chat.call_count, 2)
        calls = provider.chat.call_args_list
        turn0_system = calls[0].kwargs["system"]
        turn1_system = calls[1].kwargs["system"]
        # Turn 0: full prompt (baseline + context + skills).
        self.assertIn("## Runtime Context", turn0_system)
        self.assertIn("No skills are currently available.", turn0_system)
        # Turn 1: baseline only -- unchanged dynamic sections are NOT re-sent.
        self.assertNotIn("## Runtime Context", turn1_system)
        self.assertNotIn("No skills are currently available.", turn1_system)
        self.assertIn("Today's date:", turn1_system)
        self.assertLess(len(turn1_system.encode()), len(turn0_system.encode()))
        # Byte evidence of the per-turn saving. (On a real workspace the
        # context section alone is KBs -- git status, key files, CLAUDE.md --
        # so the saving compounds every turn.)
        saved = len(turn0_system.encode()) - len(turn1_system.encode())
        self.assertGreater(saved, 0)


if __name__ == "__main__":
    unittest.main()
