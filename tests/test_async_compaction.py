"""Tests for the async/background compaction engine (brief 022, module side).

The engine is loop-agnostic: all tests inject fake work functions and
estimate functions -- no LLM, no agent loop.
"""

from __future__ import annotations

import concurrent.futures
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import tempfile

from src.agent.conversation import Conversation, Message, TextContentBlock
from src.compact_service.async_compaction import (
    EVIDENCE_MARK_KEY,
    STATUS_ADOPTED,
    STATUS_COMPACTING,
    STATUS_FOREGROUND_FALLBACK,
    STATUS_LAUNCH_REFUSED,
    STATUS_STALE_REJECTED,
    AdoptedState,
    AsyncCompactionManager,
    BackgroundCompactionResult,
    BackgroundCompactionTask,
    CompactionSnapshot,
    LiveState,
    AdoptOutcome,
    default_estimate_tokens,
    is_evidence,
    mark_evidence,
)
from src.config import (
    async_compaction_config,
    async_compaction_enabled,
    get_default_config,
    load_config,
    save_config,
)


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


def _text_msg(text):
    return Message(role="user", content=[TextContentBlock(type="text", text=text)])


def _ok_work(summary="## Progress\ndone\n\n## Decisions\nnone\n\n## Remaining\nrest\n\n## Critical references\nnone"):
    """Fake background work: returns a result for the snapshot's generation."""

    def work(snapshot):
        return BackgroundCompactionResult(
            summary_text=summary, generation=snapshot.generation
        )

    return work


def _wait_done(task, timeout=10.0):
    deadline = time.time() + timeout
    while not task.is_done():
        if time.time() > deadline:
            raise AssertionError("background task did not finish in time")
        time.sleep(0.01)
    return task


class TestAsyncCompactionConfig(unittest.TestCase):
    """The async_compaction config section: default OFF, codex/opencode values."""

    def _patched_home(self, tmp):
        return patch("src.config.Path.home", return_value=Path(tmp))

    def test_defaults(self):
        config = get_default_config()
        section = config["async_compaction"]
        self.assertFalse(section["enabled"])
        self.assertEqual(section["reserve_tokens"], 16_000)
        self.assertEqual(section["shrink_steps"], [0.7, 0.5, 0.35])
        template = section["summary_template"]
        for heading in ("Progress", "Decisions", "Remaining", "Critical references"):
            self.assertIn(heading, template)

    def test_enabled_accessor_defaults_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_home(tmp):
                self.assertFalse(async_compaction_enabled())

    def test_enabled_accessor_opt_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_home(tmp):
                config = load_config()
                config["async_compaction"]["enabled"] = True
                save_config(config)
                self.assertTrue(async_compaction_enabled())

    def test_load_config_merges_defaults_under_existing_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_home(tmp):
                config = load_config()
                # Simulate an older config file that predates the section.
                del config["async_compaction"]
                save_config(config)
                reloaded = load_config()
                section = reloaded["async_compaction"]
                self.assertFalse(section["enabled"])
                self.assertEqual(section["reserve_tokens"], 16_000)

    def test_accessor_merges_user_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self._patched_home(tmp):
                config = load_config()
                config["async_compaction"]["enabled"] = True
                config["async_compaction"]["reserve_tokens"] = 8_000
                save_config(config)
                merged = async_compaction_config()
                self.assertTrue(merged["enabled"])
                self.assertEqual(merged["reserve_tokens"], 8_000)
                # Untouched defaults survive.
                self.assertEqual(merged["shrink_steps"], [0.7, 0.5, 0.35])


class TestEvidenceMarking(unittest.TestCase):
    def test_mark_and_detect_dict(self):
        msg = {"role": "user", "content": "x"}
        self.assertFalse(is_evidence(msg))
        mark_evidence(msg)
        self.assertTrue(is_evidence(msg))
        self.assertTrue(msg["metadata"][EVIDENCE_MARK_KEY])

    def test_mark_and_detect_object(self):
        msg = _text_msg("tool output")
        self.assertFalse(is_evidence(msg))
        mark_evidence(msg)
        self.assertTrue(is_evidence(msg))

    def test_unmarked_message_not_evidence(self):
        self.assertFalse(is_evidence(_text_msg("plain")))
        self.assertFalse(is_evidence(None))
        self.assertFalse(is_evidence({"role": "user"}))


class TestBackgroundLaunch(unittest.TestCase):
    def setUp(self):
        self.managers = []

    def tearDown(self):
        for m in self.managers:
            m.shutdown()

    def _manager(self, **kwargs):
        kwargs.setdefault("config", _enabled_config())
        m = AsyncCompactionManager(**kwargs)
        self.managers.append(m)
        return m

    def _snapshot(self, manager, n=3, token_count=50_000):
        msgs = [_text_msg(f"message {i}") for i in range(n)]
        return manager.capture_snapshot(msgs, token_count=token_count)

    def test_launch_returns_without_blocking(self):
        gate = threading.Event()

        def slow_work(snapshot):
            gate.wait(timeout=30)
            return BackgroundCompactionResult(
                summary_text="s", generation=snapshot.generation
            )

        m = self._manager(work_fn=slow_work)
        task = m.start_background_compaction(self._snapshot(m))
        # Returned immediately even though the work is still blocked.
        self.assertIsNotNone(task)
        self.assertFalse(task.is_done())
        self.assertTrue(m.is_compaction_running())
        gate.set()
        _wait_done(task)

    def test_launch_disabled_returns_none(self):
        m = self._manager(config=_enabled_config(enabled=False))
        task = m.start_background_compaction(self._snapshot(m))
        self.assertIsNone(task)
        self.assertEqual(m.status, STATUS_LAUNCH_REFUSED)

    def test_launch_below_min_tokens_refused(self):
        m = self._manager(config=_enabled_config(min_tokens=100_000))
        task = m.start_background_compaction(self._snapshot(m, token_count=10))
        self.assertIsNone(task)
        self.assertEqual(m.status, STATUS_LAUNCH_REFUSED)

    def test_launch_requires_work_fn(self):
        m = AsyncCompactionManager(config=_enabled_config(), work_fn=None)
        self.managers.append(m)
        with self.assertRaises(ValueError):
            m.start_background_compaction(self._snapshot(m))

    def test_inflight_task_reused_not_relaunched(self):
        calls = []

        def counting_work(snapshot):
            calls.append(snapshot)
            return BackgroundCompactionResult(
                summary_text="s", generation=snapshot.generation
            )

        m = self._manager(work_fn=counting_work)
        snap = self._snapshot(m)
        task1 = m.start_background_compaction(snap)
        task2 = m.start_background_compaction(self._snapshot(m))
        self.assertIs(task1, task2)
        _wait_done(task1)
        # Give the worker a beat to prove no second submission happened.
        time.sleep(0.1)
        self.assertEqual(len(calls), 1)


class TestCancellationDiscipline(unittest.TestCase):
    def setUp(self):
        self.managers = []

    def tearDown(self):
        for m in self.managers:
            m.shutdown()

    def _manager(self, **kwargs):
        kwargs.setdefault("config", _enabled_config())
        m = AsyncCompactionManager(**kwargs)
        self.managers.append(m)
        return m

    def _snapshot(self, manager, n=2, token_count=50_000):
        return manager.capture_snapshot(
            [_text_msg(f"m{i}") for i in range(n)], token_count=token_count
        )

    def test_cancelled_turn_blocks_new_launch(self):
        m = self._manager(work_fn=_ok_work())
        m.mark_turn_cancelled()
        self.assertTrue(m.turn_cancelled)
        task = m.start_background_compaction(self._snapshot(m))
        self.assertIsNone(task)
        self.assertEqual(m.status, STATUS_LAUNCH_REFUSED)

    def test_cancelled_turn_reuses_inflight(self):
        gate = threading.Event()

        def slow_work(snapshot):
            gate.wait(timeout=30)
            return BackgroundCompactionResult(
                summary_text="s", generation=snapshot.generation
            )

        m = self._manager(work_fn=slow_work)
        task1 = m.start_background_compaction(self._snapshot(m))
        m.mark_turn_cancelled()
        # In-flight work is kept for later reuse, not relaunched.
        task2 = m.start_background_compaction(self._snapshot(m))
        self.assertIs(task1, task2)
        self.assertTrue(m.is_compaction_running())
        gate.set()
        _wait_done(task1)

    def test_clear_turn_cancelled_allows_launch_again(self):
        m = self._manager(work_fn=_ok_work())
        m.mark_turn_cancelled()
        self.assertIsNone(m.start_background_compaction(self._snapshot(m)))
        m.clear_turn_cancelled()
        task = m.start_background_compaction(self._snapshot(m))
        self.assertIsNotNone(task)
        _wait_done(task)

    def test_cancel_all_drops_inflight(self):
        gate = threading.Event()

        def slow_work(snapshot):
            gate.wait(timeout=30)
            return BackgroundCompactionResult(
                summary_text="s", generation=snapshot.generation
            )

        m = self._manager(work_fn=slow_work)
        task = m.start_background_compaction(self._snapshot(m))
        self.assertTrue(m.is_compaction_running())
        m.cancel_all()
        gate.set()
        self.assertFalse(m.is_compaction_running())
        outcome = m.adopt_result(
            task, LiveState(messages=list(task.snapshot.messages))
        )
        self.assertEqual(outcome.status, "cancelled")
        self.assertIsNone(outcome.adopted)


class TestAdoption(unittest.TestCase):
    def setUp(self):
        self.managers = []

    def tearDown(self):
        for m in self.managers:
            m.shutdown()

    def _manager(self, **kwargs):
        kwargs.setdefault("config", _enabled_config())
        m = AsyncCompactionManager(**kwargs)
        self.managers.append(m)
        return m

    def test_adopt_not_ready_does_not_block(self):
        gate = threading.Event()

        def slow_work(snapshot):
            gate.wait(timeout=30)
            return BackgroundCompactionResult(
                summary_text="s", generation=snapshot.generation
            )

        m = self._manager(work_fn=slow_work)
        msgs = [_text_msg("a"), _text_msg("b")]
        snap = m.capture_snapshot(msgs, token_count=50_000)
        task = m.start_background_compaction(snap)
        outcome = m.adopt_result(task, LiveState(messages=list(msgs)))
        self.assertEqual(outcome.status, "not_ready")
        self.assertIsNone(outcome.adopted)
        gate.set()
        _wait_done(task)

    def test_adopt_preserves_messages_arriving_after_snapshot(self):
        m = self._manager(work_fn=_ok_work())
        old = [_text_msg(f"old {i}") for i in range(3)]
        snap = m.capture_snapshot(old, token_count=50_000)
        task = m.start_background_compaction(snap)
        _wait_done(task)

        # Two messages arrive while the background work runs.
        new1, new2 = _text_msg("new 1"), _text_msg("new 2")
        live = LiveState(messages=old + [new1, new2])
        outcome = m.adopt_result(task, live)

        self.assertEqual(outcome.status, "adopted")
        adopted = outcome.adopted
        # boundary + summary, then the new messages verbatim (same objects).
        self.assertEqual(len(adopted.messages), 4)
        self.assertIs(adopted.messages[2], new1)
        self.assertIs(adopted.messages[3], new2)
        # Summarized-away messages are gone.
        for gone in old:
            self.assertNotIn(gone, adopted.messages)
        self.assertEqual(m.status, STATUS_ADOPTED)

    def test_marked_tool_outputs_survive_verbatim(self):
        conv = Conversation()
        conv.add_user_message("question")
        conv.add_tool_result_message("tool-1", "SECRET-OUTPUT-123")
        conv.add_assistant_message("answer")
        evidence_msg = conv.messages[1]
        mark_evidence(evidence_msg)
        self.assertTrue(is_evidence(evidence_msg))

        m = self._manager(work_fn=_ok_work())
        snap = m.capture_snapshot(list(conv.messages), token_count=50_000)
        task = m.start_background_compaction(snap)
        _wait_done(task)

        outcome = m.adopt_result(task, LiveState(messages=list(conv.messages)))
        self.assertEqual(outcome.status, "adopted")
        adopted = outcome.adopted
        # boundary + summary, then the evidence verbatim (same object).
        self.assertIs(adopted.messages[2], evidence_msg)
        # And it was not folded into the summary text.
        summary_text = adopted.messages[1].content[0].text
        self.assertNotIn("SECRET-OUTPUT-123", summary_text)

    def test_state_and_routing_updates_preserved(self):
        def work_with_updates(snapshot):
            return BackgroundCompactionResult(
                summary_text="s",
                generation=snapshot.generation,
                state_updates={"from_compaction": 1, "counter": 999},
            )

        m = self._manager(work_fn=work_with_updates)
        snap = m.capture_snapshot(
            [_text_msg("a")],
            state={"counter": 1, "keep": "yes"},
            routing={"model": "main"},
            token_count=50_000,
        )
        task = m.start_background_compaction(snap)
        _wait_done(task)

        # State updated after the snapshot (plus a brand-new key).
        live = LiveState(
            messages=list(snap.messages),
            state={"counter": 2, "keep": "yes", "new_key": "new"},
            routing={"model": "small"},
        )
        outcome = m.adopt_result(task, live)
        self.assertEqual(outcome.status, "adopted")
        # Post-snapshot updates win over the snapshot...
        self.assertEqual(outcome.adopted.state["counter"], 999)  # compaction patch applies
        self.assertEqual(outcome.adopted.state["new_key"], "new")
        self.assertEqual(outcome.adopted.state["keep"], "yes")
        self.assertEqual(outcome.adopted.state["from_compaction"], 1)
        # ...and live routing wins.
        self.assertEqual(outcome.adopted.routing, {"model": "small"})

    def test_stale_result_rejected_after_generation_moves(self):
        m = self._manager(work_fn=_ok_work())
        snap = m.capture_snapshot([_text_msg("a")], token_count=50_000)
        task = m.start_background_compaction(snap)
        _wait_done(task)

        first = m.adopt_result(task, LiveState(messages=list(snap.messages)))
        self.assertEqual(first.status, "adopted")
        # Adopting the same task again: it is no longer current -> stale.
        second = m.adopt_result(task, LiveState(messages=list(snap.messages)))
        self.assertEqual(second.status, "stale")
        self.assertIsNone(second.adopted)
        self.assertEqual(m.status, STATUS_STALE_REJECTED)

    def test_stale_generation_mismatch_rejected(self):
        m = self._manager(work_fn=_ok_work())
        snap = m.capture_snapshot([_text_msg("a")], token_count=50_000)
        task = m.start_background_compaction(snap)
        _wait_done(task)
        adopted_once = m.adopt_result(task, LiveState(messages=list(snap.messages)))
        self.assertEqual(adopted_once.status, "adopted")

        # A result computed against an older generation is stale.
        stale_future = concurrent.futures.Future()
        stale_future.set_result(
            BackgroundCompactionResult(summary_text="old", generation=0)
        )
        stale_task = BackgroundCompactionTask(
            snapshot=CompactionSnapshot(messages=[], generation=0),
            future=stale_future,
        )
        m._task = stale_task  # simulate a late-arriving old result
        outcome = m.adopt_result(stale_task, LiveState(messages=[]))
        self.assertEqual(outcome.status, "stale")
        self.assertIsNone(outcome.adopted)

    def test_background_work_exception_reports_failed(self):
        def bad_work(snapshot):
            raise RuntimeError("boom")

        m = self._manager(work_fn=bad_work)
        snap = m.capture_snapshot([_text_msg("a")], token_count=50_000)
        task = m.start_background_compaction(snap)
        _wait_done(task)
        outcome = m.adopt_result(task, LiveState(messages=list(snap.messages)))
        self.assertEqual(outcome.status, "failed")
        self.assertIsNone(outcome.adopted)

    def test_adopt_no_task(self):
        m = self._manager(work_fn=_ok_work())
        outcome = m.adopt_result(None, LiveState(messages=[]))
        self.assertEqual(outcome.status, "no_task")

    def test_status_hook_transitions(self):
        seen = []
        m = self._manager(work_fn=_ok_work())
        m.add_status_listener(seen.append)
        snap = m.capture_snapshot([_text_msg("a")], token_count=50_000)
        task = m.start_background_compaction(snap)
        _wait_done(task)
        m.adopt_result(task, LiveState(messages=list(snap.messages)))
        self.assertEqual(seen, [STATUS_COMPACTING, STATUS_ADOPTED])


class TestForegroundFallback(unittest.TestCase):
    def setUp(self):
        self.managers = []

    def tearDown(self):
        for m in self.managers:
            m.shutdown()

    def _manager(self, **kwargs):
        kwargs.setdefault("config", _enabled_config())
        m = AsyncCompactionManager(**kwargs)
        self.managers.append(m)
        return m

    def _adopted_task(self, manager, summary="s"):
        snap = manager.capture_snapshot([_text_msg("a")], token_count=50_000)
        task = manager.start_background_compaction(snap, work_fn=_ok_work(summary))
        _wait_done(task)
        return task, LiveState(messages=list(snap.messages))

    def test_over_limit_runs_exactly_one_foreground_compaction(self):
        m = self._manager()
        task, live = self._adopted_task(m)
        calls = []

        def foreground_compact(adopted: AdoptedState) -> AdoptedState:
            calls.append(adopted)
            return AdoptedState(
                messages=[_text_msg("foreground-compacted")],
                state=dict(adopted.state),
                routing=dict(adopted.routing),
                generation=adopted.generation,
            )

        outcome = m.adopt_with_foreground_fallback(
            task,
            live,
            limit=10_000,
            foreground_compact=foreground_compact,
            estimate_tokens=lambda messages: 1_000_000,  # way over limit
        )
        self.assertEqual(outcome.status, "adopted")
        self.assertTrue(outcome.foreground_ran)
        self.assertEqual(len(calls), 1)  # exactly one foreground pass
        self.assertEqual(
            outcome.adopted.messages[0].content[0].text, "foreground-compacted"
        )
        self.assertEqual(m.status, STATUS_FOREGROUND_FALLBACK)

    def test_under_limit_skips_foreground(self):
        m = self._manager()
        task, live = self._adopted_task(m)
        calls = []
        outcome = m.adopt_with_foreground_fallback(
            task,
            live,
            limit=200_000,
            foreground_compact=lambda adopted: calls.append(adopted) or adopted,
            estimate_tokens=lambda messages: 1_000,  # under limit - reserve
        )
        self.assertEqual(outcome.status, "adopted")
        self.assertFalse(outcome.foreground_ran)
        self.assertEqual(calls, [])

    def test_fallback_not_run_when_adoption_not_ready(self):
        gate = threading.Event()

        def slow_work(snapshot):
            gate.wait(timeout=30)
            return BackgroundCompactionResult(
                summary_text="s", generation=snapshot.generation
            )

        m = self._manager(work_fn=slow_work)
        snap = m.capture_snapshot([_text_msg("a")], token_count=50_000)
        task = m.start_background_compaction(snap)
        calls = []
        outcome = m.adopt_with_foreground_fallback(
            task,
            LiveState(messages=list(snap.messages)),
            limit=10_000,
            foreground_compact=lambda adopted: calls.append(adopted) or adopted,
            estimate_tokens=lambda messages: 1_000_000,
        )
        self.assertEqual(outcome.status, "not_ready")
        self.assertFalse(outcome.foreground_ran)
        self.assertEqual(calls, [])
        gate.set()
        _wait_done(task)

    def test_reserve_tokens_create_headroom(self):
        # adopted tokens between (limit - reserve) and limit -> fallback runs.
        m = self._manager(config=_enabled_config(reserve_tokens=16_000))
        task, live = self._adopted_task(m)
        calls = []
        outcome = m.adopt_with_foreground_fallback(
            task,
            live,
            limit=100_000,
            foreground_compact=lambda adopted: calls.append(adopted) or adopted,
            estimate_tokens=lambda messages: 90_000,  # > 84_000 effective limit
        )
        self.assertTrue(outcome.foreground_ran)
        self.assertEqual(len(calls), 1)


class TestEstimateTokens(unittest.TestCase):
    def test_default_estimate_scales_with_content(self):
        small = default_estimate_tokens([_text_msg("hi")])
        big = default_estimate_tokens([_text_msg("x" * 4000)])
        self.assertGreater(big, small)
        self.assertGreaterEqual(small, 1)

    def test_default_estimate_handles_dicts(self):
        msgs = [{"role": "user", "content": "hello world"}]
        self.assertGreaterEqual(default_estimate_tokens(msgs), 1)


if __name__ == "__main__":
    unittest.main()
