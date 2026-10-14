"""Tests for the richer hook lifecycle (brief 008).

Covers the new lifecycle events (SessionStart, SessionEnd, BeforeAgent,
AfterAgent, PreCompress, BeforeModel, AfterModel): per-event JSON payloads
delivered to stub hook scripts, block/ask decisions halting the turn with
the hook's message, malformed hook output failing open, and backward
compatibility of the existing PreToolUse/PostToolUse behavior.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.agent.session import Session
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import _maybe_auto_compact, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.hooks.runner import HOOK_EVENTS, run_hooks


CAPTURE_SCRIPT = """\
import json, sys
payload = json.load(sys.stdin)
with open(sys.argv[1], "w") as f:
    json.dump(payload, f)
"""

DECIDE_SCRIPT = """\
import json, sys
json.load(sys.stdin)  # consume the payload
message = sys.argv[2] if len(sys.argv) > 2 else "halted by test hook"
print(json.dumps({"decision": sys.argv[1], "message": message}))
"""

LEGACY_BLOCK_SCRIPT = """\
import json, sys
json.load(sys.stdin)  # consume the payload
print(json.dumps({"allow": False, "message": "legacy block"}))
"""

GARBAGE_SCRIPT = """\
import sys
sys.stdin.read()
print("this is not json {{{")
"""


def _write_script(tmpdir: Path, name: str, body: str) -> Path:
    path = tmpdir / name
    path.write_text(body)
    return path


class HookLifecycleTestCase(unittest.TestCase):
    """Tmp hooks.json + stub scripts, with the runner pointed at the tmp config."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.hooks_json = self.tmpdir / "hooks.json"
        self.capture_py = _write_script(self.tmpdir, "capture.py", CAPTURE_SCRIPT)
        self.decide_py = _write_script(self.tmpdir, "decide.py", DECIDE_SCRIPT)
        self._config_patcher = patch(
            "src.tool_system.hooks.runner._default_config_path",
            return_value=self.hooks_json,
        )
        self._config_patcher.start()

    def tearDown(self):
        self._config_patcher.stop()
        self.tmp.cleanup()

    def _write_config(self, config: dict) -> None:
        self.hooks_json.write_text(json.dumps(config))

    def _capture_hook(self, event: str) -> Path:
        """Configure *event* with a hook that records its stdin payload."""
        capture_file = self.tmpdir / f"capture-{event}.json"
        self._write_config(
            {event: [{"command": f"python3 {self.capture_py} {capture_file}"}]}
        )
        return capture_file

    def _read_capture(self, capture_file: Path) -> dict:
        return json.loads(capture_file.read_text())

    def _decide_hook(self, event: str, decision: str, message: str = "halted by test hook"):
        self._write_config(
            {event: [{"command": f"python3 {self.decide_py} {decision} {message!r}"}]}
        )


class TestLifecyclePayloads(HookLifecycleTestCase):
    """Unit test per new event: the stub hook script receives the right JSON."""

    def test_all_lifecycle_events_are_known(self):
        for event in (
            "SessionStart", "SessionEnd", "BeforeAgent", "AfterAgent",
            "PreCompress", "BeforeModel", "AfterModel",
        ):
            self.assertIn(event, HOOK_EVENTS)

    def test_session_start_payload(self):
        capture = self._capture_hook("SessionStart")
        extra = {"session_id": "s1", "provider": "glm", "model": "m",
                 "created_at": "t", "resumed": False}
        run_hooks("SessionStart", extra=extra)
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "SessionStart")
        for key, value in extra.items():
            self.assertEqual(payload[key], value)

    def test_session_end_payload(self):
        capture = self._capture_hook("SessionEnd")
        run_hooks("SessionEnd", extra={"session_id": "s1", "reason": "exit"})
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "SessionEnd")
        self.assertEqual(payload["session_id"], "s1")
        self.assertEqual(payload["reason"], "exit")

    def test_before_agent_payload(self):
        capture = self._capture_hook("BeforeAgent")
        run_hooks("BeforeAgent", extra={"session_id": "s1", "model": "m", "max_turns": 50})
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "BeforeAgent")
        self.assertEqual(payload["max_turns"], 50)

    def test_after_agent_payload(self):
        capture = self._capture_hook("AfterAgent")
        run_hooks(
            "AfterAgent",
            extra={"session_id": "s1", "response_text": "hi",
                   "num_turns": 2, "usage": None, "cancelled": False},
        )
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "AfterAgent")
        self.assertEqual(payload["num_turns"], 2)
        self.assertFalse(payload["cancelled"])

    def test_pre_compress_payload(self):
        capture = self._capture_hook("PreCompress")
        run_hooks(
            "PreCompress",
            extra={"session_id": "s1", "model": "m",
                   "estimated_tokens": 190_000, "trigger": "auto"},
        )
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "PreCompress")
        self.assertEqual(payload["estimated_tokens"], 190_000)
        self.assertEqual(payload["trigger"], "auto")

    def test_before_model_payload(self):
        capture = self._capture_hook("BeforeModel")
        messages = [{"role": "user", "content": "hi"}]
        run_hooks(
            "BeforeModel",
            extra={"session_id": "s1", "turn": 0, "model": "m",
                   "message_count": 1, "messages": messages},
        )
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "BeforeModel")
        self.assertEqual(payload["messages"], messages)
        self.assertEqual(payload["message_count"], 1)

    def test_after_model_payload(self):
        capture = self._capture_hook("AfterModel")
        run_hooks(
            "AfterModel",
            extra={"session_id": "s1", "turn": 0, "model": "m",
                   "content": "hello", "tool_uses": [],
                   "usage": {"input_tokens": 5}},
        )
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "AfterModel")
        self.assertEqual(payload["content"], "hello")
        self.assertEqual(payload["usage"], {"input_tokens": 5})


class TestHookDecisions(HookLifecycleTestCase):
    """allow / block / ask parsing and per-event enforcement."""

    def test_legacy_allow_false_still_blocks_pre_tool_use(self):
        script = _write_script(self.tmpdir, "legacy_block.py", LEGACY_BLOCK_SCRIPT)
        self._write_config({"PreToolUse": [{"command": f"python3 {script}"}]})
        result = run_hooks("PreToolUse", "Bash", {"command": "rm -rf /"})
        self.assertTrue(result["blocked"])
        self.assertEqual(result["decision"], "block")
        self.assertEqual(result["message"], "legacy block")

    def test_new_decision_block_pre_tool_use(self):
        self._decide_hook("PreToolUse", "block", "no way")
        result = run_hooks("PreToolUse", "Bash", {"command": "ls"})
        self.assertTrue(result["blocked"])
        self.assertEqual(result["decision"], "block")
        self.assertEqual(result["message"], "no way")

    def test_allow_decision_never_blocks(self):
        self._decide_hook("BeforeAgent", "allow")
        result = run_hooks("BeforeAgent", extra={})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")
        self.assertIsNone(result["message"])

    def test_block_ignored_on_informational_event(self):
        # AfterAgent cannot block: output ignored, always fail-open allow.
        self._decide_hook("AfterAgent", "block", "should be ignored")
        result = run_hooks("AfterAgent", extra={})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")
        self.assertIsNone(result["message"])

    def test_ask_enforced_like_block_on_blocking_event(self):
        self._decide_hook("BeforeAgent", "ask", "are you sure?")
        result = run_hooks("BeforeAgent", extra={})
        self.assertTrue(result["blocked"])
        self.assertEqual(result["decision"], "ask")
        self.assertEqual(result["message"], "are you sure?")

    def test_ask_ignored_on_informational_event(self):
        self._decide_hook("PostToolUse", "ask", "should be ignored")
        result = run_hooks("PostToolUse", "Bash", {"command": "ls"}, {"ok": True})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")

    def test_malformed_output_fails_open(self):
        script = _write_script(self.tmpdir, "garbage.py", GARBAGE_SCRIPT)
        self._write_config({"BeforeModel": [{"command": f"python3 {script}"}]})
        result = run_hooks("BeforeModel", extra={"turn": 0})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")

    def test_nonzero_exit_fails_open(self):
        self._write_config({"BeforeAgent": [{"command": "exit 3"}]})
        result = run_hooks("BeforeAgent", extra={})
        self.assertFalse(result["blocked"])

    def test_missing_config_fails_open(self):
        self.hooks_json.unlink(missing_ok=True)
        result = run_hooks("BeforeAgent", extra={})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")

    def test_empty_output_allows(self):
        script = _write_script(self.tmpdir, "empty.py", "import sys; sys.stdin.read()\n")
        self._write_config({"BeforeModel": [{"command": f"python3 {script}"}]})
        result = run_hooks("BeforeModel", extra={})
        self.assertFalse(result["blocked"])

    def test_pre_tool_use_payload_unchanged(self):
        # Backward compat: the exact stdin shape tool hooks have always seen.
        capture = self._capture_hook("PreToolUse")
        run_hooks("PreToolUse", "Bash", {"command": "ls"}, config_path=self.hooks_json)
        payload = self._read_capture(capture)
        self.assertEqual(
            payload,
            {"event": "PreToolUse", "tool": "Bash", "input": {"command": "ls"}},
        )

    def test_post_tool_use_payload_unchanged(self):
        capture = self._capture_hook("PostToolUse")
        run_hooks(
            "PostToolUse", "Bash", {"command": "ls"}, {"exit_code": 0},
            config_path=self.hooks_json,
        )
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "PostToolUse")
        self.assertEqual(payload["output"], {"exit_code": 0})


class AgentLoopHookTestCase(HookLifecycleTestCase):
    """run_agent_loop with a mock provider and tmp hook config."""

    def setUp(self):
        super().setUp()
        self.registry = build_default_registry()
        self.ctx = ToolContext(workspace_root=self.tmpdir)
        self.ctx.session_id = "test-session"

    def _provider(self, final="Done."):
        p = MagicMock()
        p.model = "test-model"
        p.chat = MagicMock(
            return_value=ChatResponse(
                content=final, model="test", usage={}, finish_reason="stop"
            )
        )
        return p

    def _run(self, provider):
        conv = Conversation()
        conv.add_user_message("hello")
        return run_agent_loop(conv, provider, self.registry, self.ctx, max_turns=5)


class TestBeforeAfterAgentHooks(AgentLoopHookTestCase):
    def test_before_agent_block_halts_run_with_message(self):
        self._decide_hook("BeforeAgent", "block", "not today")
        provider = self._provider()
        result = self._run(provider)
        self.assertEqual(result.response_text, "not today")
        self.assertEqual(result.num_turns, 0)
        provider.chat.assert_not_called()

    def test_before_agent_allow_runs_normally(self):
        self._decide_hook("BeforeAgent", "allow")
        provider = self._provider()
        result = self._run(provider)
        self.assertEqual(result.response_text, "Done.")
        provider.chat.assert_called()

    def test_after_agent_receives_run_summary(self):
        capture = self._capture_hook("AfterAgent")
        provider = self._provider()
        result = self._run(provider)
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "AfterAgent")
        self.assertEqual(payload["response_text"], result.response_text)
        self.assertEqual(payload["num_turns"], result.num_turns)
        self.assertFalse(payload["cancelled"])
        self.assertEqual(payload["session_id"], "test-session")

    def test_after_agent_fires_on_cancelled_run(self):
        capture = self._capture_hook("AfterAgent")
        provider = self._provider()
        conv = Conversation()
        conv.add_user_message("hello")
        result = run_agent_loop(
            conv, provider, self.registry, self.ctx, max_turns=5,
            should_stop=lambda: True,
        )
        self.assertTrue(result.cancelled)
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "AfterAgent")
        self.assertTrue(payload["cancelled"])


class TestBeforeAfterModelHooks(AgentLoopHookTestCase):
    def test_before_model_block_halts_turn_with_message(self):
        self._decide_hook("BeforeModel", "block", "no model calls right now")
        provider = self._provider()
        result = self._run(provider)
        self.assertEqual(result.response_text, "no model calls right now")
        provider.chat.assert_not_called()

    def test_before_model_receives_prompt_inspection_payload(self):
        capture = self._capture_hook("BeforeModel")
        provider = self._provider()
        self._run(provider)
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "BeforeModel")
        self.assertEqual(payload["turn"], 0)
        self.assertIn("messages", payload)
        self.assertEqual(payload["message_count"], len(payload["messages"]))
        self.assertGreater(payload["message_count"], 0)

    def test_after_model_cannot_block(self):
        self._decide_hook("AfterModel", "block", "should be ignored")
        provider = self._provider()
        result = self._run(provider)
        self.assertEqual(result.response_text, "Done.")
        provider.chat.assert_called()

    def test_malformed_before_model_output_fails_open(self):
        script = _write_script(self.tmpdir, "garbage.py", GARBAGE_SCRIPT)
        self._write_config({"BeforeModel": [{"command": f"python3 {script}"}]})
        provider = self._provider()
        result = self._run(provider)
        self.assertEqual(result.response_text, "Done.")
        provider.chat.assert_called()


class TestPreCompressHook(HookLifecycleTestCase):
    def setUp(self):
        super().setUp()
        self.ctx = ToolContext(workspace_root=self.tmpdir)

    def _conv(self):
        conv = Conversation()
        conv.add_user_message("hello")
        conv.add_assistant_message("hi there")
        return conv

    def _summary_provider(self):
        p = MagicMock()
        p.model = "gpt-4o"
        return p

    def test_pre_compress_block_vetoes_compaction(self):
        self._decide_hook("PreCompress", "block", "keep my context")
        conv = self._conv()
        provider = self._summary_provider()
        with patch(
            "src.token_estimation.count_messages_tokens", return_value=200_000
        ), patch(
            "src.compact_service.service.compact_conversation_sync"
        ) as compact_mock:
            result = _maybe_auto_compact(conv, provider, [], "sys", self.ctx)
        self.assertFalse(result)
        compact_mock.assert_not_called()

    def test_pre_compress_allow_compacts(self):
        self._decide_hook("PreCompress", "allow")
        conv = self._conv()
        provider = self._summary_provider()
        with patch(
            "src.token_estimation.count_messages_tokens", return_value=200_000
        ), patch(
            "src.compact_service.service.compact_conversation_sync"
        ) as compact_mock:
            result = _maybe_auto_compact(conv, provider, [], "sys", self.ctx)
        self.assertTrue(result)
        compact_mock.assert_called_once()

    def test_pre_compress_malformed_output_fails_open(self):
        script = _write_script(self.tmpdir, "garbage.py", GARBAGE_SCRIPT)
        self._write_config({"PreCompress": [{"command": f"python3 {script}"}]})
        conv = self._conv()
        provider = self._summary_provider()
        with patch(
            "src.token_estimation.count_messages_tokens", return_value=200_000
        ), patch(
            "src.compact_service.service.compact_conversation_sync"
        ) as compact_mock:
            result = _maybe_auto_compact(conv, provider, [], "sys", self.ctx)
        self.assertTrue(result)
        compact_mock.assert_called_once()


class TestSessionStartHook(HookLifecycleTestCase):
    def test_session_create_fires_session_start(self):
        capture = self._capture_hook("SessionStart")
        session = Session.create("glm", "glm-4.5")
        payload = self._read_capture(capture)
        self.assertEqual(payload["event"], "SessionStart")
        self.assertEqual(payload["session_id"], session.session_id)
        self.assertEqual(payload["provider"], "glm")
        self.assertEqual(payload["model"], "glm-4.5")
        self.assertFalse(payload["resumed"])

    def test_session_create_without_config_still_works(self):
        self.hooks_json.unlink(missing_ok=True)
        session = Session.create("glm", "glm-4.5")
        self.assertTrue(session.session_id)


if __name__ == "__main__":
    unittest.main()
