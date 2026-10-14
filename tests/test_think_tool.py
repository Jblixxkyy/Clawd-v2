"""Tests for the ``Think`` tool and ``ThinkEvent`` transcript support (brief 023)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError
from src.tool_system.tools.think import (
    ThinkEvent,
    ThinkTool,
    context_think_events,
)
from src.transcript import TranscriptStore


def _ctx(**kwargs) -> ToolContext:
    tmp = tempfile.TemporaryDirectory()
    ctx = ToolContext(workspace_root=Path(tmp.name).resolve())
    ctx._tmpdir = tmp  # keep alive for the life of the context
    for key, value in kwargs.items():
        setattr(ctx, key, value)
    return ctx


class TestThinkEvent(unittest.TestCase):
    def test_render_and_parse_roundtrip(self) -> None:
        event = ThinkEvent(thought="Double-check the plan before editing.", topic="pre-edit")
        line = event.render()
        self.assertTrue(line.startswith(ThinkEvent.MARKER))
        parsed = ThinkEvent.parse_line(line)
        self.assertEqual(parsed, event)

    def test_parse_line_defensive(self) -> None:
        self.assertIsNone(ThinkEvent.parse_line("hello world"))
        self.assertIsNone(ThinkEvent.parse_line("[think] not-json{{"))
        self.assertIsNone(ThinkEvent.parse_line("[think] [1, 2]"))
        self.assertIsNone(ThinkEvent.parse_line("[think] {}"))
        self.assertIsNone(ThinkEvent.parse_line("[think] {\"thought\": \"\"}"))
        self.assertIsNone(ThinkEvent.parse_line(None))
        self.assertIsNone(ThinkEvent.parse_line(123))

    def test_from_dict_validation(self) -> None:
        with self.assertRaises(ValueError):
            ThinkEvent.from_dict({"thought": ""})
        with self.assertRaises(ValueError):
            ThinkEvent.from_dict({"thought": "ok", "topic": 42})

    def test_checkpoint_roundtrip(self) -> None:
        """Think events survive JSON serialization (checkpoint-compatible)."""
        event = ThinkEvent(thought="Why is this failing?", topic="debug")
        restored = ThinkEvent.from_dict(json.loads(json.dumps(event.to_dict())))
        self.assertEqual(restored, event)


class TestThinkTool(unittest.TestCase):
    def test_spec_is_read_only_no_side_effects(self) -> None:
        spec = ThinkTool().spec()
        self.assertEqual(spec.name, "Think")
        self.assertTrue(spec.is_read_only)
        self.assertFalse(spec.is_destructive)

    def test_think_appears_in_transcript_as_distinct_replayable_event(self) -> None:
        store = TranscriptStore()
        ctx = _ctx(transcript_store=store)
        result = ThinkTool().run({"thought": "This is risky; reconsider.", "topic": "plan"}, ctx)

        self.assertEqual(result.name, "Think")
        self.assertFalse(result.is_error)
        self.assertTrue(result.output["recorded"])

        events = store.iter_think_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].thought, "This is risky; reconsider.")
        self.assertEqual(events[0].topic, "plan")

        # Distinct, replayable: the canonical marker line is in the replay.
        replay = store.replay()
        self.assertEqual(len(replay), 1)
        self.assertTrue(replay[0].startswith(ThinkEvent.MARKER))
        # Re-hydrate the transcript from the replayed lines and re-read.
        replayed = TranscriptStore(entries=list(replay))
        self.assertEqual(replayed.iter_think_events(), events)

    def test_think_without_topic(self) -> None:
        store = TranscriptStore()
        ctx = _ctx(transcript_store=store)
        result = ThinkTool().run({"thought": "Just thinking."}, ctx)
        self.assertFalse(result.is_error)
        self.assertEqual(store.iter_think_events()[0].topic, None)

    def test_think_events_survive_compaction_summarized_not_dropped(self) -> None:
        store = TranscriptStore()
        for i in range(8):
            store.append_event(ThinkEvent(thought=f"early thought number {i}"))
        for i in range(3):
            store.append(f"regular entry {i}")
        store.append_event(ThinkEvent(thought="late thought"))

        store.compact_preserve_think(keep_last=3)

        # Oldest entries were dropped, but the think contents survive as a summary.
        entries = store.replay()
        self.assertTrue(any(e.startswith("[think-summary]") for e in entries))
        summary = next(e for e in entries if e.startswith("[think-summary]"))
        self.assertIn("8 earlier thought(s)", summary)
        self.assertIn("early thought number 0", summary)
        self.assertIn("early thought number 7", summary)
        # In-window entries (incl. the late think event) kept verbatim.
        self.assertEqual(len(store.iter_think_events()), 1)
        self.assertEqual(store.iter_think_events()[0].thought, "late thought")

    def test_plain_compact_still_drops_think_events(self) -> None:
        """Baseline: the legacy compact() is unchanged (no think summary)."""
        store = TranscriptStore()
        store.append_event(ThinkEvent(thought="will vanish"))
        for i in range(12):
            store.append(f"entry {i}")
        store.compact(keep_last=10)
        self.assertEqual(store.iter_think_events(), ())

    def test_judge_accessor_reads_arbitrary_transcript(self) -> None:
        """The stable accessor scans a mixed, messy transcript defensively."""
        store = TranscriptStore(entries=[
            "user: hi",
            ThinkEvent(thought="agent should wait", topic="plan").render(),
            "[think] {broken json",
            "assistant: hello",
            "[think] []",
            ThinkEvent(thought="proceed").render(),
        ])
        events = store.iter_think_events()
        self.assertEqual([e.thought for e in events], ["agent should wait", "proceed"])

    def test_context_think_log_stamped(self) -> None:
        ctx = _ctx()
        ThinkTool().run({"thought": "one"}, ctx)
        ThinkTool().run({"thought": "two", "topic": "t2"}, ctx)
        log = context_think_events(ctx)
        self.assertEqual([e.thought for e in log], ["one", "two"])
        # Coordination stamp for subagent profiles (brief 023).
        self.assertTrue(getattr(ctx, "think_completed", False))

    def test_no_world_state_side_effects(self) -> None:
        root = Path(tempfile.mkdtemp())
        marker = root / "marker.txt"
        marker.write_text("untouched")
        before = sorted(p.name for p in root.iterdir())
        ctx = ToolContext(workspace_root=root)
        cwd_before = __import__("os").getcwd()

        result = ThinkTool().run({"thought": "safe"}, ctx)

        self.assertFalse(result.is_error)
        self.assertEqual(sorted(p.name for p in root.iterdir()), before)
        self.assertEqual(marker.read_text(), "untouched")
        self.assertEqual(__import__("os").getcwd(), cwd_before)
        # The only context changes are the think log and the coordination stamp.
        self.assertEqual([e.thought for e in context_think_events(ctx)], ["safe"])

    def test_validation(self) -> None:
        ctx = _ctx()
        with self.assertRaises(ToolInputError):
            ThinkTool().run({"thought": ""}, ctx)
        with self.assertRaises(ToolInputError):
            ThinkTool().run({"thought": "   "}, ctx)
        with self.assertRaises(ToolInputError):
            ThinkTool().run({"thought": 42}, ctx)
        with self.assertRaises(ToolInputError):
            ThinkTool().run({}, ctx)
        with self.assertRaises(ToolInputError):
            ThinkTool().run({"thought": "ok", "topic": 7}, ctx)

    def test_subagent_coordination_flag_defaults_off(self) -> None:
        # The flag exists for subagent profiles (005/024) to coordinate
        # through; the tool itself suggests rather than mandates.
        from src.tool_system.tools import think as think_module

        self.assertFalse(think_module.REQUIRE_THINK_BEFORE_FIRST_MUTATION)
        self.assertFalse(ThinkTool().require_before_first_mutation)


if __name__ == "__main__":
    unittest.main()
