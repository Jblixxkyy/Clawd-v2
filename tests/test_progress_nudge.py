"""Loop-level tests for brief 039: progress nudge.

The harness tracks tool turns since the last file modification. At 10
modification-free tool turns (then every 15: 10, 25, 40...) it injects a
planning nudge as a user-role message. The nudge steers, never blocks:
- 10 search-only turns -> nudge message present in conversation.
- Write on turn 5 -> counter resets; no nudge at turn 10.
- Nudge repeats at turn 25 with still no modifications.
- Nudge never blocks: the loop continues normally after injection.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.harness_guards import (
    build_progress_nudge,
    is_progress_tool,
    should_nudge,
)
from src.tool_system.protocol import ToolResult
from src.tool_system.registry import ToolRegistry, ToolSpec
from src.tool_system.tools.grep import GrepTool

_NUDGE_MARKER = "without modifying any files"


class _FakeSubagentTool:
    """Stands in for Subagent: succeeds instantly without spawning real work."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Subagent",
            description="fake delegated work",
            input_schema={"type": "object", "properties": {}},
        )

    def run(self, tool_input, context) -> ToolResult:
        return ToolResult(
            name="Subagent", output="delegated work complete", is_error=False
        )


def _resp(content="", tool_uses=None):
    return ChatResponse(
        content=content,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="stop",
        tool_uses=tool_uses,
    )


def _grep(i):
    # Distinct pattern per call so the 029 loop detector sees no exact
    # signature repeats; no matches -> success, no file modifications.
    return {
        "id": f"g{i}",
        "name": "Grep",
        "input": {"pattern": f"zzz_no_such_pattern_{i}_xyz"},
    }


def _write(i, workspace_root):
    return {
        "id": f"w{i}",
        "name": "Write",
        "input": {
            "file_path": str(workspace_root / f"plan_{i}.txt"),
            "content": "plan",
        },
    }


class TestProgressNudgeLoop(unittest.TestCase):
    def _run(
        self,
        tool_uses_sequence,
        final_text="Done searching.",
        registry=None,
        workspace_root=None,
    ):
        conv = Conversation()
        conv.add_user_message("find the chat method")
        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.side_effect = [
            _resp("", tool_uses=[tu]) for tu in tool_uses_sequence
        ] + [_resp(final_text)]
        workspace_root = workspace_root or Path(tempfile.mkdtemp())
        ctx = ToolContext(workspace_root=workspace_root)
        result = run_agent_loop(
            conversation=conv,
            provider=provider,
            tool_registry=registry or build_default_registry(),
            tool_context=ctx,
            verbose=False,
        )
        return result, conv, provider, workspace_root

    def _nudge_texts(self, conv):
        return [
            m.content
            for m in conv.messages
            if m.role == "user"
            and isinstance(m.content, str)
            and _NUDGE_MARKER in m.content
        ]

    def test_ten_search_only_turns_inject_nudge(self):
        result, conv, provider, _ = self._run([_grep(i) for i in range(10)])
        nudges = self._nudge_texts(conv)
        self.assertEqual(
            len(nudges), 1, f"expected exactly 1 nudge, got {len(nudges)}"
        )
        self.assertIn("10 tool calls", nudges[0])
        self.assertIn("plan", nudges[0])

    def test_write_on_turn_five_resets_counter(self):
        workspace_root = Path(tempfile.mkdtemp())
        seq = [_grep(i) for i in range(5)] + [_write(0, workspace_root)] + [
            _grep(100 + i) for i in range(5)
        ]
        result, conv, provider, _ = self._run(seq, workspace_root=workspace_root)
        self.assertEqual(
            self._nudge_texts(conv),
            [],
            "nudge fired even though Write reset the counter",
        )
        # The Write really ran — it wrote a file.
        self.assertTrue((workspace_root / "plan_0.txt").exists())

    def test_nudge_repeats_at_turn_25(self):
        result, conv, provider, _ = self._run([_grep(i) for i in range(25)])
        nudges = self._nudge_texts(conv)
        self.assertEqual(
            len(nudges), 2, f"expected nudges at 10 and 25, got {len(nudges)}"
        )
        self.assertIn("10 tool calls", nudges[0])
        self.assertIn("25 tool calls", nudges[1])

    def test_nudge_never_blocks(self):
        final = "Searches exhausted; stating my plan now."
        result, conv, provider, _ = self._run(
            [_grep(i) for i in range(10)], final_text=final
        )
        # The loop kept going after the injection: all 10 tool calls plus the
        # final text response were consumed, and the model's text is the result.
        self.assertEqual(provider.chat.call_count, 11)
        self.assertIn("stating my plan", result.response_text)

    def test_subagent_task_completion_resets(self):
        # Delegated work counts as progress: a successful Subagent resets.
        registry = ToolRegistry([GrepTool(), _FakeSubagentTool()])
        sub = {"id": "s0", "name": "Subagent", "input": {}}
        seq = [_grep(i) for i in range(9)] + [sub] + [
            _grep(100 + i) for i in range(9)
        ]
        result, conv, provider, _ = self._run(seq, registry=registry)
        self.assertEqual(
            self._nudge_texts(conv),
            [],
            "nudge fired even though Subagent completion reset the counter",
        )


class TestProgressNudgeHelpers(unittest.TestCase):
    def test_schedule_first_and_repeats(self):
        self.assertFalse(should_nudge(9))
        self.assertTrue(should_nudge(10))
        self.assertFalse(should_nudge(11))
        self.assertTrue(should_nudge(25))
        self.assertTrue(should_nudge(40))
        # Repeats every 15 thereafter — the old prototype stopped at 40.
        self.assertTrue(should_nudge(55))
        self.assertTrue(should_nudge(70))
        self.assertFalse(should_nudge(41))
        self.assertFalse(should_nudge(0))
        self.assertFalse(should_nudge(-1))

    def test_helpers_fail_open(self):
        self.assertFalse(should_nudge(None))
        self.assertFalse(should_nudge("garbage"))
        self.assertFalse(is_progress_tool(None))
        text = build_progress_nudge(None)
        self.assertIsInstance(text, str)
        self.assertIn("10 tool calls", text)

    def test_nudge_text_brief_shape(self):
        text = build_progress_nudge(10)
        self.assertIn("10 tool calls", text)
        self.assertIn("plan", text)
        # Brief: target files + intended change + what it's still looking for,
        # under 60 words, one short paragraph.
        self.assertIn("files", text)
        self.assertIn("looking for", text)
        self.assertLess(len(text.split()), 60)
        self.assertNotIn("\n", text)

    def test_progress_tool_set(self):
        for name in ("Edit", "Write", "Subagent", "Task", "Agent"):
            self.assertTrue(is_progress_tool(name), name)
        for name in ("Bash", "Grep", "Read", "Glob"):
            self.assertFalse(is_progress_tool(name), name)


if __name__ == "__main__":
    unittest.main()
