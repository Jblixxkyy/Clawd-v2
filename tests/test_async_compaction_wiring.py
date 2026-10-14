"""Wiring tests for brief 022: async compaction in the agent-loop turn lifecycle.

Exercises the pre/mid/post-turn checkpoint wiring in
``src/tool_system/agent_loop.py`` using the loop's test seams
(``tool_context.async_compaction_config`` /
``tool_context.async_compaction_work_fn``) to inject a fake background
summarizer -- no LLM involved. The engine itself is covered by
tests/test_async_compaction.py.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.compact_service.async_compaction import (
    AdoptedState,
    BackgroundCompactionResult,
    CompactionSnapshot,
    is_evidence,
)
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


def _enabled_config(**overrides):
    cfg = {
        "enabled": True,
        "min_tokens": 0,
        "reserve_tokens": 16_000,
        "shrink_steps": [0.7, 0.5, 0.35],
        "summary_template": "TEMPLATE",
    }
    cfg.update(overrides)
    return cfg


def _fast_work(summary="## Progress\ndone"):
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


def _write_tool_use(workspace, name="hello.py", content="print('hello world')"):
    return {
        "id": "toolu_1",
        "name": "Write",
        "input": {"file_path": str(workspace / name), "content": content},
    }


def _tool_response(text="working", tool_uses=None):
    return ChatResponse(
        content=text,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="tool_use",
        tool_uses=tool_uses or [],
    )


class AsyncCompactionWiringBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)
        self.provider = MagicMock()
        self.provider.chat_stream_response.side_effect = NotImplementedError()
        self._releases: list[threading.Event] = []

    def tearDown(self):
        self.tmp.cleanup()
        for ev in self._releases:
            ev.set()
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

    def _async_events(self, result):
        return [
            e
            for e in (result.run_log or [])
            if str(e.get("event", "")).startswith("async_compact")
        ]

    def _async_statuses(self, result):
        return [
            e["status"]
            for e in self._async_events(result)
            if e["event"] == "async_compact_status"
        ]


class TestAsyncCompactionDefaultOff(AsyncCompactionWiringBase):
    def test_default_off_leaves_behavior_unchanged(self):
        # Explicitly disabled (bypasses ~/.clawd/config.json for hermeticity).
        self.context.async_compaction_config = {"enabled": False}
        conv = Conversation()
        conv.add_user_message("hello")
        result = self._run(conv, [_final_response("hi there")])
        self.assertEqual(result.response_text, "hi there")
        self.assertEqual(self._async_events(result), [])
        self.assertEqual(self.provider.chat.call_count, 1)


class TestAsyncCompactionNonBlocking(AsyncCompactionWiringBase):
    def test_background_launch_does_not_block_turn(self):
        self.context.async_compaction_config = _enabled_config()
        release = threading.Event()
        started = threading.Event()
        self._releases.append(release)

        def blocking_work(snapshot):
            started.set()
            release.wait(timeout=30)
            return BackgroundCompactionResult(
                summary_text="## Progress\ndone", generation=snapshot.generation
            )

        self.context.async_compaction_work_fn = blocking_work
        conv = Conversation()
        conv.add_user_message("hello")

        holder: dict = {}

        def target():
            holder["result"] = self._run(conv, [_final_response("hi")])

        t = threading.Thread(target=target, daemon=True)
        t.start()
        self.assertTrue(started.wait(timeout=10), "background compaction never launched")
        # The turn must finish while the background work is still blocked.
        t.join(timeout=10)
        self.assertFalse(t.is_alive(), "turn blocked on background compaction")
        release.set()

        result = holder["result"]
        self.assertEqual(result.response_text, "hi")
        self.assertIn("compacting", self._async_statuses(result))


class TestAsyncCompactionAdoption(AsyncCompactionWiringBase):
    def test_adopted_summary_preserves_post_snapshot_messages(self):
        self.context.async_compaction_config = _enabled_config()
        self.context.async_compaction_work_fn = _fast_work()
        big = "EVIDENCE-" + "x" * 6000
        (self.workspace / "big.txt").write_text(big)
        conv = Conversation()
        conv.add_user_message("read the file")
        read_use = {
            "id": "toolu_1",
            "name": "Read",
            "input": {"file_path": str(self.workspace / "big.txt")},
        }
        result = self._run(conv, [_tool_response("reading", [read_use]),
                                  _final_response("read it")])

        kinds = {e["event"] for e in self._async_events(result)}
        self.assertIn("async_compact_adopted", kinds)

        texts = []
        for m in conv.messages:
            content = (
                m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            )
            texts.append(str(content))
        blob = "\n".join(texts)
        # The tool result arrived after the snapshot and must be preserved
        # verbatim, not summarized away.
        self.assertIn("EVIDENCE-", blob)
        # Large tool outputs are marked as protected evidence.
        self.assertTrue(
            any(is_evidence(m) for m in conv.messages),
            "large tool output was not marked as evidence",
        )
        # A summary message from the background compaction is present.
        self.assertIn("continued from a previous conversation", blob)

    def test_failing_work_fn_never_breaks_turn(self):
        self.context.async_compaction_config = _enabled_config()

        def bad_work(snapshot):
            raise RuntimeError("boom")

        self.context.async_compaction_work_fn = bad_work
        conv = Conversation()
        conv.add_user_message("do the thing")
        result = self._run(
            conv,
            [
                _tool_response("writing", [_write_tool_use(self.workspace)]),
                _final_response("still fine"),
            ],
        )
        self.assertEqual(result.response_text, "still fine")
        self.assertIn("failed", self._async_statuses(result))


class TestAsyncCompactionCancellation(AsyncCompactionWiringBase):
    def test_cancelled_turn_launches_nothing(self):
        calls = []
        self.context.async_compaction_config = _enabled_config()

        def counting_work(snapshot):
            calls.append(snapshot)
            return BackgroundCompactionResult(
                summary_text="## Progress\ndone", generation=snapshot.generation
            )

        self.context.async_compaction_work_fn = counting_work
        conv = Conversation()
        conv.add_user_message("hello")
        result = self._run(conv, [_final_response("hi")], should_stop=lambda: True)
        self.assertTrue(result.cancelled)
        self.assertEqual(calls, [])
        self.assertNotIn("compacting", self._async_statuses(result))

    def test_cancelled_turn_keeps_inflight_without_new_launch(self):
        self.context.async_compaction_config = _enabled_config()
        release = threading.Event()
        started = threading.Event()
        self._releases.append(release)
        launches = []

        def blocking_work(snapshot):
            launches.append(snapshot)
            started.set()
            release.wait(timeout=30)
            return BackgroundCompactionResult(
                summary_text="## Progress\ndone", generation=snapshot.generation
            )

        self.context.async_compaction_work_fn = blocking_work
        conv = Conversation()
        conv.add_user_message("do the thing")

        stop_calls = {"n": 0}

        def stop():
            stop_calls["n"] += 1
            return stop_calls["n"] > 1

        result = self._run(
            conv,
            [_tool_response("writing", [_write_tool_use(self.workspace)])],
            should_stop=stop,
        )
        release.set()
        self.assertTrue(result.cancelled)
        # Launched once before the cancellation ...
        self.assertTrue(started.wait(timeout=10))
        self.assertEqual(len(launches), 1)
        # ... and never again after it (the cancelled turn is terminal, so
        # exactly one launch total proves no post-cancel launch).
        time.sleep(0.5)
        self.assertEqual(len(launches), 1)
        self.assertNotIn("launch_refused", self._async_statuses(result))


class TestAsyncCompactionForegroundFallback(AsyncCompactionWiringBase):
    def test_adopted_over_limit_runs_exactly_one_foreground_compaction(self):
        self.context.async_compaction_config = _enabled_config()
        summaries = {"n": 0}

        def work(snapshot):
            summaries["n"] += 1
            if summaries["n"] == 1:
                text = "S" * 200_000  # ~50k tokens > 40k limit - 16k reserve
            else:
                text = "## Progress\nsmall"
            return BackgroundCompactionResult(
                summary_text=text, generation=snapshot.generation
            )

        self.context.async_compaction_work_fn = work
        fg_calls = []

        def fake_foreground(conversation, provider, model, **kwargs):
            from src.agent.conversation import Message, TextContentBlock

            fg_calls.append(1)
            small = Message(
                role="user",
                content=[TextContentBlock(type="text", text="compacted")],
            )
            conversation.messages.clear()
            conversation.messages.append(small)
            return AdoptedState(
                messages=list(conversation.messages), state={}, routing={}
            )

        with (
            patch("src.context_limits.get_context_limit", return_value=40_000),
            patch(
                "src.compact_service.service.compact_conversation_sync",
                side_effect=fake_foreground,
            ),
        ):
            conv = Conversation()
            conv.add_user_message("do the thing")
            result = self._run(
                conv,
                [
                    _tool_response("writing", [_write_tool_use(self.workspace)]),
                    _final_response("all done"),
                ],
            )
        self.assertEqual(len(fg_calls), 1, "foreground fallback must run exactly once")
        self.assertEqual(result.response_text, "all done")
        self.assertIn("foreground_fallback", self._async_statuses(result))


class TestAsyncCompactionNoSideEffects(AsyncCompactionWiringBase):
    def test_loop_does_not_create_config_dir(self):
        # Regression: the wiring must never mkdir ~/.clawd as a side effect
        # of running the loop (load_config()/get_config_path() both do).
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home}):
                conv = Conversation()
                conv.add_user_message("hello")
                result = self._run(conv, [_final_response("hi")])
                self.assertEqual(result.response_text, "hi")
            self.assertFalse(
                Path(home, ".clawd").exists(),
                "running the loop created ~/.clawd",
            )


class TestAsyncCompactionWorkFn(unittest.TestCase):
    def test_background_work_fn_falls_back_without_llm(self):
        from src.tool_system.agent_loop import _make_async_compaction_work_fn

        provider = MagicMock()
        provider.chat.side_effect = RuntimeError("no llm")
        fn = _make_async_compaction_work_fn(provider, "test-model", _enabled_config())
        snap = CompactionSnapshot(
            messages=[], state={}, routing={}, generation=3, token_count=100
        )
        result = fn(snap)
        self.assertIsInstance(result, BackgroundCompactionResult)
        self.assertEqual(result.generation, 3)
        self.assertIn("## Progress", result.summary_text)


if __name__ == "__main__":
    unittest.main()
