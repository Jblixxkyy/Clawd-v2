"""Tests for brief 026: idle auto-compaction.

Unit tests for :class:`IdleCompactor` (trigger discipline) and wiring
tests for the agent-loop idle check: at run start the loop measures the
gap since the previous run's last activity and, when the session sat idle
long enough with a big-enough context, launches a 022 background
compaction (never foreground). The 022 adopt checkpoints then adopt the
result, and the quiet "compacted while idle (X->Y tokens)" notice is
injected on the next interaction.

Uses the loop's test seams (``tool_context.idle_compactor`` /
``tool_context.idle_compaction_config`` /
``tool_context.idle_compaction_work_fn``) -- no LLM involved.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.compact_service.async_compaction import BackgroundCompactionResult
from src.compact_service.idle_compaction import (
    IdleCompactor,
    any_background_work_running,
    get_idle_compactor,
    read_idle_compaction_config,
    reset_idle_compactors,
)
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


def _idle_enabled_config(**overrides):
    cfg = {"enabled": True, "min_tokens": 0, "idle_minutes": 0.01}
    cfg.update(overrides)
    return cfg


def _idle_fast_work(summary="## Progress\ndone"):
    def work(snapshot):
        return BackgroundCompactionResult(
            summary_text=summary, generation=snapshot.generation
        )

    return work


def _final_response(text="done"):
    return ChatResponse(
        content=text,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="stop",
        tool_uses=[],
    )


class TestIdleCompactorDiscipline(unittest.TestCase):
    def setUp(self):
        self.now = [10_000.0]
        self.compactor = IdleCompactor(
            config={"enabled": True, "min_tokens": 100, "idle_minutes": 1},
            clock=lambda: self.now[0],
        )

    def test_trigger_when_idle_past_threshold_with_large_context(self):
        self.compactor.note_activity(9_000.0)  # idle 1000s >= 60s
        self.assertTrue(
            self.compactor.should_trigger(
                self.now[0], self.compactor.last_activity, 500, False
            )
        )

    def test_no_trigger_when_context_small(self):
        self.compactor.note_activity(9_000.0)
        self.assertFalse(
            self.compactor.should_trigger(
                self.now[0], self.compactor.last_activity, 50, False
            )
        )

    def test_no_trigger_when_not_idle_long_enough(self):
        self.compactor.note_activity(9_970.0)  # idle 30s < 60s
        self.assertFalse(
            self.compactor.should_trigger(
                self.now[0], self.compactor.last_activity, 500, False
            )
        )

    def test_no_trigger_when_background_busy(self):
        self.compactor.note_activity(9_000.0)
        self.assertFalse(
            self.compactor.should_trigger(
                self.now[0], self.compactor.last_activity, 500, True
            )
        )

    def test_no_trigger_when_disabled(self):
        compactor = IdleCompactor(
            config={"enabled": False}, clock=lambda: self.now[0]
        )
        self.assertFalse(compactor.should_trigger(self.now[0], 0.0, 10**9, False))

    def test_no_trigger_without_activity_history(self):
        self.assertFalse(
            self.compactor.should_trigger(self.now[0], None, 10**9, False)
        )

    def test_trigger_at_most_once_per_idle_window(self):
        self.compactor.note_activity(9_000.0)
        self.assertTrue(
            self.compactor.should_trigger(
                self.now[0], self.compactor.last_activity, 500, False
            )
        )
        self.compactor.mark_triggered(before_tokens=500)
        # Same idle window: suppressed.
        self.assertFalse(
            self.compactor.should_trigger(
                self.now[0], self.compactor.last_activity, 500, False
            )
        )
        # New activity ends the window: a later idle gap may trigger again.
        self.now[0] = 11_000.0
        self.compactor.note_activity()  # run completion
        self.now[0] = 11_000.0 + 3_600.0
        self.assertTrue(
            self.compactor.should_trigger(
                self.now[0], self.compactor.last_activity, 500, False
            )
        )

    def test_record_compaction_queues_quiet_notice_once(self):
        notice = self.compactor.record_compaction(500, 42)
        self.assertIn("compacted while idle (500\u219242 tokens)", notice)
        self.assertEqual(
            self.compactor.last_compaction["before_tokens"], 500
        )
        self.assertEqual(self.compactor.last_compaction["after_tokens"], 42)
        # The notice is delivered exactly once.
        self.assertEqual(self.compactor.take_pending_notice(), notice)
        self.assertIsNone(self.compactor.take_pending_notice())

    def test_should_trigger_never_raises(self):
        self.assertFalse(
            self.compactor.should_trigger("now", object(), "lots", "busy")
        )

    def test_configure_refreshes_config(self):
        self.compactor.configure({"enabled": False})
        self.assertFalse(self.compactor.is_enabled())


class TestIdleConfig(unittest.TestCase):
    def test_defaults_are_conservative(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home}):
                from src.config import idle_compaction_config

                cfg = idle_compaction_config()
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["min_tokens"], 200_000)
        self.assertEqual(cfg["idle_minutes"], 54)

    def test_default_config_includes_section(self):
        from src.config import get_default_config

        section = get_default_config()["idle_compaction"]
        self.assertFalse(section["enabled"])
        self.assertEqual(section["min_tokens"], 200_000)

    def test_read_probe_merges_override_over_defaults(self):
        cfg = read_idle_compaction_config({"enabled": True})
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["min_tokens"], 200_000)
        self.assertEqual(cfg["idle_minutes"], 54)

    def test_read_probe_never_creates_config_dir(self):
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home}):
                cfg = read_idle_compaction_config()
                self.assertFalse(cfg["enabled"])
            self.assertFalse(
                Path(home, ".clawd").exists(),
                "config probe created ~/.clawd",
            )

    def test_registry_caches_per_session(self):
        reset_idle_compactors()
        try:
            a = get_idle_compactor("s1")
            b = get_idle_compactor("s1")
            c = get_idle_compactor("s2")
            self.assertIs(a, b)
            self.assertIsNot(a, c)
        finally:
            reset_idle_compactors()


class TestBackgroundWorkProbe(unittest.TestCase):
    def tearDown(self):
        from src.tool_system.tools import bash as bash_tool
        from src.tool_system.tools import subagent as subagent_tool

        bash_tool._BG_JOBS.pop("test-idle-job", None)
        subagent_tool._BG_SUBAGENTS.pop("test-idle-task", None)

    def test_quiet_by_default(self):
        self.assertFalse(any_background_work_running())

    def test_running_bash_job_counts(self):
        from src.tool_system.tools import bash as bash_tool

        proc = MagicMock()
        proc.poll.return_value = None  # still running
        with bash_tool._BG_LOCK:
            bash_tool._BG_JOBS["test-idle-job"] = {
                "process": proc,
                "command": "sleep 999",
            }
        self.assertTrue(any_background_work_running())

    def test_finished_bash_job_does_not_count(self):
        from src.tool_system.tools import bash as bash_tool

        proc = MagicMock()
        proc.poll.return_value = 0  # exited
        with bash_tool._BG_LOCK:
            bash_tool._BG_JOBS["test-idle-job"] = {
                "process": proc,
                "command": "true",
            }
        self.assertFalse(any_background_work_running())

    def test_running_subagent_counts(self):
        from src.tool_system.tools import subagent as subagent_tool

        done = threading.Event()
        with subagent_tool._BG_LOCK:
            subagent_tool._BG_SUBAGENTS["test-idle-task"] = {
                "task_id": "test-idle-task",
                "status": "running",
                "done_event": done,
            }
        self.assertTrue(any_background_work_running())

    def test_finished_subagent_does_not_count(self):
        from src.tool_system.tools import subagent as subagent_tool

        done = threading.Event()
        done.set()
        with subagent_tool._BG_LOCK:
            subagent_tool._BG_SUBAGENTS["test-idle-task"] = {
                "task_id": "test-idle-task",
                "status": "done",
                "done_event": done,
            }
        self.assertFalse(any_background_work_running())


class IdleCompactionWiringBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)
        self.provider = MagicMock()
        self.provider.chat_stream_response.side_effect = NotImplementedError()

    def tearDown(self):
        self.tmp.cleanup()
        reset_idle_compactors()
        from src.tool_system.tools import bash as bash_tool

        bash_tool._BG_JOBS.pop("test-idle-job", None)
        # No async-compact worker threads may leak (they are non-daemon).
        deadline = time.time() + 10
        while time.time() < deadline:
            strays = [
                t for t in threading.enumerate() if t.name.startswith("async-compact")
            ]
            if not strays:
                break
            time.sleep(0.05)
        leaked = [
            t for t in threading.enumerate() if t.name.startswith("async-compact")
        ]
        self.assertFalse(leaked, "async-compact worker thread leaked")

    def _run(self, conversation, responses, **kwargs):
        self.provider.chat.side_effect = list(responses)
        return run_agent_loop(
            conversation=conversation,
            provider=self.provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
            **kwargs,
        )

    def _idle_events(self, result):
        return [
            e
            for e in (result.run_log or [])
            if str(e.get("event", "")).startswith("idle_compaction")
        ]

    def _chat_texts(self):
        """Text of the messages passed to the last provider.chat call."""
        msgs = self.provider.chat.call_args[0][0]
        texts = []
        for m in msgs:
            content = m.get("content", "") if isinstance(m, dict) else ""
            texts.append(content if isinstance(content, str) else str(content))
        return "\n".join(texts)


class TestIdleCompactionWiring(IdleCompactionWiringBase):
    def _enable_idle(self, **overrides):
        compactor = IdleCompactor(clock=time.time)
        self.context.idle_compactor = compactor
        self.context.idle_compaction_config = _idle_enabled_config(**overrides)
        self.context.idle_compaction_work_fn = _idle_fast_work()
        # Simulate a session that went quiet long ago.
        compactor.note_activity(time.time() - 120)
        return compactor

    def test_idle_triggers_background_compaction_with_token_log(self):
        self._enable_idle()
        conv = Conversation()
        conv.add_user_message("hello")
        result = self._run(conv, [_final_response("hi")])

        triggers = [
            e for e in self._idle_events(result)
            if e["event"] == "idle_compaction_trigger"
        ]
        self.assertEqual(len(triggers), 1, "expected exactly one idle trigger")
        self.assertIsInstance(triggers[0]["before_tokens"], int)
        self.assertGreaterEqual(triggers[0]["before_tokens"], 0)

        adopted = [
            e for e in self._idle_events(result)
            if e["event"] == "idle_compaction_adopted"
        ]
        self.assertEqual(len(adopted), 1, "expected the idle result to be adopted")
        self.assertIn("compacted while idle (", adopted[0]["notice"])
        self.assertRegex(
            adopted[0]["notice"], r"compacted while idle \(\d+\u2192\d+ tokens\)"
        )

    def test_next_interaction_starts_on_compacted_context_with_notice(self):
        compactor = self._enable_idle()
        conv = Conversation()
        conv.add_user_message("hello")
        self._run(conv, [_final_response("hi")])
        # The quiet notice is queued for the next interaction, not this one.
        self.assertIsNotNone(compactor._pending_notice)

        # The session conversation now starts from the compacted context
        # (same assertion style as the 022 adoption test).
        texts = []
        for m in conv.messages:
            content = (
                m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            )
            texts.append(str(content))
        self.assertIn("continued from a previous conversation", "\n".join(texts))

        conv.add_user_message("are you still there?")
        result2 = self._run(conv, [_final_response("yes")])
        texts = self._chat_texts()
        # ... and the next interaction sees the quiet idle notice exactly once.
        self.assertRegex(texts, r"compacted while idle \(\d+→\d+ tokens\)")
        self.assertIsNone(compactor.take_pending_notice())
        # No second idle trigger: the gap since run 1 is ~0s.
        triggers2 = [
            e for e in self._idle_events(result2)
            if e["event"] == "idle_compaction_trigger"
        ]
        self.assertEqual(triggers2, [])

    def test_small_context_no_compaction(self):
        self._enable_idle(min_tokens=10**9)
        conv = Conversation()
        conv.add_user_message("hello")
        result = self._run(conv, [_final_response("hi")])
        self.assertEqual(self._idle_events(result), [])

    def test_background_bash_job_defers_until_finished(self):
        from src.tool_system.tools import bash as bash_tool

        compactor = self._enable_idle()
        proc = MagicMock()
        proc.poll.return_value = None  # still running
        with bash_tool._BG_LOCK:
            bash_tool._BG_JOBS["test-idle-job"] = {
                "process": proc,
                "command": "sleep 999",
            }
        try:
            conv = Conversation()
            conv.add_user_message("hello")
            result = self._run(conv, [_final_response("hi")])
        finally:
            with bash_tool._BG_LOCK:
                bash_tool._BG_JOBS.pop("test-idle-job", None)
        self.assertEqual(
            [e for e in self._idle_events(result)
             if e["event"] == "idle_compaction_trigger"],
            [],
            "idle compaction must defer while a Bash job runs",
        )
        # The idle window is still open (no new trigger consumed): once the
        # job finishes, the next run compacts.
        compactor.note_activity(time.time() - 120)
        conv.add_user_message("hello again")
        result2 = self._run(conv, [_final_response("hi again")])
        self.assertEqual(
            len([e for e in self._idle_events(result2)
                 if e["event"] == "idle_compaction_trigger"]),
            1,
        )

    def test_disabled_never_triggers(self):
        compactor = IdleCompactor(clock=time.time)
        self.context.idle_compactor = compactor
        # Explicitly disabled: bypasses ~/.clawd/config.json for hermeticity.
        self.context.idle_compaction_config = {"enabled": False}
        self.context.idle_compaction_work_fn = _idle_fast_work()
        compactor.note_activity(time.time() - 3600)
        conv = Conversation()
        conv.add_user_message("hello")
        result = self._run(conv, [_final_response("hi")])
        self.assertEqual(self._idle_events(result), [])
        self.assertEqual(result.response_text, "hi")

    def test_double_idle_window_compacts_at_most_once(self):
        now = [5_000.0]
        compactor = IdleCompactor(clock=lambda: now[0])
        self.context.idle_compactor = compactor
        self.context.idle_compaction_config = _idle_enabled_config()
        self.context.idle_compaction_work_fn = _idle_fast_work()
        compactor.note_activity(0.0)  # quiet since the epoch (frozen clock)

        conv1 = Conversation()
        conv1.add_user_message("first")
        result1 = self._run(conv1, [_final_response("one")])
        conv2 = Conversation()
        conv2.add_user_message("second")
        result2 = self._run(conv2, [_final_response("two")])

        triggers = [
            e
            for result in (result1, result2)
            for e in (result.run_log or [])
            if e.get("event") == "idle_compaction_trigger"
        ]
        self.assertEqual(
            len(triggers), 1, "must compact at most once per idle window"
        )


class TestIdleCompactionNoSideEffects(IdleCompactionWiringBase):
    def test_loop_does_not_create_config_dir(self):
        # Regression: the idle wiring must never mkdir ~/.clawd as a side
        # effect of running the loop (no tool_context override here, so the
        # read-only probe path is exercised).
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home}):
                conv = Conversation()
                conv.add_user_message("hello")
                result = self._run(conv, [_final_response("hi")])
                self.assertEqual(result.response_text, "hi")
                self.assertEqual(
                    [e for e in (result.run_log or [])
                     if str(e.get("event", "")).startswith("idle_compaction")],
                    [],
                )
            self.assertFalse(
                Path(home, ".clawd").exists(),
                "running the loop created ~/.clawd",
            )


if __name__ == "__main__":
    unittest.main()
