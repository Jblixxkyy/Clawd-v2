"""Tests for Q9: tool-result truncation with context budgets.

Covers the helpers in src.tool_system.protocol, the ToolSpec default,
per-tool overrides, and budget enforcement in registry.dispatch.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.tool_system.context import ToolContext
from src.tool_system.protocol import (
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    ToolCall,
    ToolResult,
    format_truncation_marker,
    truncate_text,
)
from src.tool_system.registry import ToolRegistry, ToolSpec, _apply_result_budget


class TestTruncationHelpers(unittest.TestCase):
    def test_default_is_30k(self):
        self.assertEqual(DEFAULT_MAX_RESULT_SIZE_CHARS, 30_000)

    def test_marker_format(self):
        self.assertEqual(
            format_truncation_marker(10, 100),
            "\n\n[truncated — showing 10 of 100 chars]",
        )

    def test_short_text_unchanged(self):
        self.assertEqual(truncate_text("abc", 30_000), "abc")
        self.assertEqual(truncate_text("x" * 100, 100), "x" * 100)

    def test_truncated_fits_budget_and_counts_are_exact(self):
        text = "y" * 100
        out = truncate_text(text, 40)
        self.assertLessEqual(len(out), 40)
        self.assertIn("[truncated — showing ", out)
        self.assertTrue(out.endswith("of 100 chars]"))
        # N in the marker == chars actually shown before the marker
        kept = int(out.split("showing ")[1].split(" of ")[0])
        self.assertGreater(kept, 0)
        self.assertEqual(out, text[:kept] + format_truncation_marker(kept, 100))

    def test_zero_budget(self):
        self.assertEqual(truncate_text("hello", 0), "")

    def test_negative_budget_rejected(self):
        with self.assertRaises(ValueError):
            truncate_text("hello", -1)


class TestSpecDefault(unittest.TestCase):
    def test_toolspec_default_is_30k(self):
        spec = ToolSpec(name="T", description="d", input_schema={})
        self.assertEqual(spec.max_result_size_chars, DEFAULT_MAX_RESULT_SIZE_CHARS)

    def test_per_tool_override_still_works(self):
        spec = ToolSpec(name="T", description="d", input_schema={}, max_result_size_chars=1234)
        self.assertEqual(spec.max_result_size_chars, 1234)


class _BigTool:
    """Fake tool returning a huge string result."""

    def __init__(self, limit: int | None = None):
        self._limit = limit

    def spec(self) -> ToolSpec:
        kwargs = {}
        if self._limit is not None:
            kwargs["max_result_size_chars"] = self._limit
        return ToolSpec(name="BigTool", description="d", input_schema={"type": "object"}, **kwargs)

    def run(self, tool_input, context) -> ToolResult:
        return ToolResult(name="BigTool", output="z" * 100_000)


class TestApplyResultBudget(unittest.TestCase):
    def test_small_string_passes_through_untouched(self):
        spec = ToolSpec(name="T", description="d", input_schema={})
        result = ToolResult(name="T", output="small")
        self.assertIs(_apply_result_budget(spec, result), result)

    def test_big_string_truncated_with_marker(self):
        spec = ToolSpec(name="T", description="d", input_schema={})
        result = ToolResult(name="T", output="z" * 100_000)
        out = _apply_result_budget(spec, result)
        self.assertIsNot(out, result)
        self.assertLessEqual(len(out.output), 30_000)
        self.assertIn("[truncated — showing ", out.output)
        self.assertTrue(out.output.endswith("of 100000 chars]"))
        self.assertEqual(out.content_type, "text")

    def test_big_dict_rendered_and_truncated(self):
        spec = ToolSpec(name="T", description="d", input_schema={})
        result = ToolResult(name="T", output={"blob": "z" * 100_000})
        out = _apply_result_budget(spec, result)
        self.assertIsInstance(out.output, str)
        self.assertLessEqual(len(out.output), 30_000)
        self.assertIn("[truncated — showing ", out.output)
        self.assertEqual(out.content_type, "text")

    def test_small_dict_passes_through_untouched(self):
        spec = ToolSpec(name="T", description="d", input_schema={})
        payload = {"a": 1}
        result = ToolResult(name="T", output=payload)
        out = _apply_result_budget(spec, result)
        self.assertIs(out, result)
        self.assertEqual(out.output, payload)

    def test_per_tool_override_honored(self):
        spec = ToolSpec(name="T", description="d", input_schema={}, max_result_size_chars=500)
        result = ToolResult(name="T", output="z" * 1_000)
        out = _apply_result_budget(spec, result)
        self.assertLessEqual(len(out.output), 500)
        self.assertIn("[truncated — showing ", out.output)


class TestDispatchEnforcesBudget(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.ctx = ToolContext(workspace_root=Path(self.tmp.name).resolve())

    def tearDown(self):
        self.tmp.cleanup()

    def test_dispatch_truncates_huge_result(self):
        reg = ToolRegistry([_BigTool()])
        result = reg.dispatch(ToolCall(name="BigTool", input={}), self.ctx)
        self.assertIsInstance(result.output, str)
        self.assertLessEqual(len(result.output), 30_000)
        self.assertIn("[truncated — showing ", result.output)

    def test_dispatch_respects_small_override(self):
        reg = ToolRegistry([_BigTool(limit=200)])
        result = reg.dispatch(ToolCall(name="BigTool", input={}), self.ctx)
        self.assertLessEqual(len(result.output), 200)
        self.assertIn("[truncated — showing ", result.output)


if __name__ == "__main__":
    unittest.main()
