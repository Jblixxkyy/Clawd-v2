"""Tests for brief 013: the fuzzy edit fallback chain.

Covers the four failure modes the chain exists for (trailing whitespace,
indentation drift, collapsed whitespace, 1-2 char near-misses) plus the safety
rails: short needles never enter the chain, ambiguity raises instead of
guessing, exact matches always win, and multi-hunk stays atomic.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError
from src.tool_system.tools.edit import FileEditTool
from src.tool_system.tools.read import FileReadTool


class FuzzyEditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        self.tool = FileEditTool()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_and_read(self, name: str, content: str) -> Path:
        p = self.root / name
        p.write_text(content, encoding="utf-8")
        FileReadTool().run({"file_path": str(p), "limit": 200}, self.ctx)
        return p

    # -- strategy 1: line-trim ------------------------------------------

    def test_trailing_whitespace_old_string_matches_line_trim(self):
        p = self._write_and_read("a.py", "def main():\n    return 1  \n")
        out = self.tool.run(
            {"file_path": str(p),
             "old_string": "    return 1\n",
             "new_string": "    return 2\n"},
            self.ctx,
        ).output
        # old claimed the newline, so the line's trailing run goes with it
        self.assertEqual(p.read_text(encoding="utf-8"), "def main():\n    return 2\n")
        self.assertEqual(out["match"]["kind"], "fuzzy")
        self.assertEqual(out["match"]["strategy"], "line-trim")
        self.assertEqual(out["match"]["foundLine"], 2)
        self.assertIn("line-trim", out["notice"])

    def test_missing_final_newline_still_matches(self):
        p = self._write_and_read("b.py", "value = compute(42)")
        out = self.tool.run(
            {"file_path": str(p),
             "old_string": "value = compute(42)\n",
             "new_string": "value = compute(43)\n"},
            self.ctx,
        ).output
        self.assertEqual(p.read_text(encoding="utf-8"), "value = compute(43)\n")
        self.assertEqual(out["match"]["strategy"], "line-trim")

    # -- strategy 2: indent ---------------------------------------------

    def test_over_indented_old_string_keeps_file_indentation(self):
        # 8-space old vs 4-space file: not an exact substring, so fuzzy runs.
        p = self._write_and_read("c.py", "    result = compute(x)\n")
        out = self.tool.run(
            {"file_path": str(p),
             "old_string": "        result = compute(x)\n",
             "new_string": "        result = compute(y)\n"},
            self.ctx,
        ).output
        self.assertEqual(p.read_text(encoding="utf-8"), "    result = compute(y)\n")
        self.assertEqual(out["match"]["strategy"], "indent")
        self.assertEqual(out["match"]["kind"], "fuzzy")

    def test_dedented_old_string_still_edits_correctly(self):
        # Dedented old is a plain substring, so exact wins -- and the result is
        # still right: the file's indentation sits outside the replaced span.
        p = self._write_and_read("c2.py", "    result = compute(x)\n")
        out = self.tool.run(
            {"file_path": str(p),
             "old_string": "result = compute(x)\n",
             "new_string": "result = compute(y)\n"},
            self.ctx,
        ).output
        self.assertEqual(p.read_text(encoding="utf-8"), "    result = compute(y)\n")
        self.assertEqual(out["match"]["kind"], "exact")

    def test_indentation_drift_never_doubles_the_indent(self):
        # old_string has 4 spaces, the file has 8 -> the file's 8 must survive.
        p = self._write_and_read("d.py", "        result = compute(x)\n")
        self.tool.run(
            {"file_path": str(p),
             "old_string": "    result = compute(x)\n",
             "new_string": "    result = compute(y)\n"},
            self.ctx,
        )
        self.assertEqual(p.read_text(encoding="utf-8"), "        result = compute(y)\n")

    def test_file_indentation_is_preserved_over_model_indentation(self):
        p = self._write_and_read("d2.py", "        start()\n")
        self.tool.run(
            {"file_path": str(p),
             "old_string": "  start()\n",
             "new_string": "  begin()\n"},
            self.ctx,
        )
        self.assertEqual(p.read_text(encoding="utf-8"), "        begin()\n")

    # -- strategy 3: collapse-ws ----------------------------------------

    def test_collapsed_internal_whitespace_matches(self):
        # Neither old nor new claims the newline -> the file's own "  \\n" stays.
        p = self._write_and_read("e.py", "total = compute(a,    b)  \n")
        out = self.tool.run(
            {"file_path": str(p),
             "old_string": "total = compute(a, b)",
             "new_string": "total = compute(a, c)"},
            self.ctx,
        ).output
        self.assertEqual(p.read_text(encoding="utf-8"), "total = compute(a, c)  \n")
        self.assertEqual(out["match"]["strategy"], "collapse-ws")

    # -- strategy 4: near ------------------------------------------------

    def test_one_char_drift_on_long_line_matches_near(self):
        line = "    total = compute_value(alpha, beta) + 1"
        p = self._write_and_read("f.py", line + "\n")
        typo = "    total = compute_value(alpha, betq) + 1"
        out = self.tool.run(
            {"file_path": str(p), "old_string": typo, "new_string": line},
            self.ctx,
        ).output
        self.assertEqual(out["match"]["strategy"], "near")
        self.assertEqual(p.read_text(encoding="utf-8"), line + "\n")

    def test_one_char_drift_on_short_line_still_errors(self):
        p = self._write_and_read("g.py", "name = alphq\n")  # 12 chars, < 24
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run(
                {"file_path": str(p),
                 "old_string": "name = alpha",
                 "new_string": "name = alphb"},
                self.ctx,
            )
        self.assertIn("old_string not found", str(cm.exception))
        self.assertEqual(p.read_text(encoding="utf-8"), "name = alphq\n")

    # -- safety rails ----------------------------------------------------

    def test_short_needle_behaves_exactly_as_before(self):
        p = self._write_and_read("h.py", "foo\nbar\n")
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run(
                {"file_path": str(p), "old_string": "fooX", "new_string": "Y"},
                self.ctx,
            )
        msg = str(cm.exception)
        self.assertIn("old_string not found", msg)
        self.assertIn("Did you mean", msg)

    def test_ambiguous_fuzzy_match_refuses_to_guess(self):
        original = (
            "    result = compute(alpha)   \n"
            "    other = 1\n"
            "    result = compute(alpha)   \n"
        )
        p = self._write_and_read("i.py", original)
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run(
                {"file_path": str(p),
                 "old_string": "    result = compute(alpha)",
                 "new_string": "    result = compute(beta)"},
                self.ctx,
            )
        msg = str(cm.exception)
        self.assertIn("not unique", msg)
        self.assertIn("lines 1, 3", msg)
        self.assertEqual(p.read_text(encoding="utf-8"), original)

    def test_exact_match_wins_over_fuzzy(self):
        p = self._write_and_read("j.py", "foo bar\nfoo  bar\n")
        out = self.tool.run(
            {"file_path": str(p),
             "old_string": "foo bar",
             "new_string": "foo baz"},
            self.ctx,
        ).output
        self.assertEqual(out["match"]["kind"], "exact")
        self.assertEqual(out["match"]["strategy"], "exact")
        self.assertNotIn("notice", out)
        self.assertEqual(p.read_text(encoding="utf-8"), "foo baz\nfoo  bar\n")

    def test_replace_all_fuzzy_uses_one_strategy(self):
        original = "    a = compute(alpha)   \n    b = 1\n    a = compute(alpha)   \n"
        p = self._write_and_read("k.py", original)
        out = self.tool.run(
            {"file_path": str(p),
             "old_string": "    a = compute(alpha)\n",
             "new_string": "    a = compute(beta)\n",
             "replace_all": True},
            self.ctx,
        ).output
        self.assertEqual(out["replaced"], 2)
        self.assertEqual(out["match"]["strategy"], "line-trim")
        # old claimed the newline, so each line's trailing run goes with it
        self.assertEqual(
            p.read_text(encoding="utf-8"),
            "    a = compute(beta)\n    b = 1\n    a = compute(beta)\n",
        )

    def test_replace_all_fuzzy_reports_single_strategy_per_hunk(self):
        original = "    a = compute(alpha)   \n    b = 1\n"
        p = self._write_and_read("k2.py", original)
        out = self.tool.run(
            {"file_path": str(p), "edits": [
                {"old_string": "    a = compute(alpha)\n",
                 "new_string": "    a = compute(beta)\n",
                 "replace_all": True},
            ]},
            self.ctx,
        ).output
        self.assertEqual(out["hunks"][0]["match"]["strategy"], "line-trim")
        self.assertEqual(out["hunks"][0]["replaced"], 1)

    def test_expected_occurrences_applies_to_fuzzy_spans(self):
        original = "    a = compute(alpha)   \n    b = 1\n    a = compute(alpha)   \n"
        p = self._write_and_read("l.py", original)
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run(
                {"file_path": str(p),
                 "old_string": "    a = compute(alpha)\n",
                 "new_string": "    a = compute(beta)\n",
                 "expected_occurrences": 1},
                self.ctx,
            )
        self.assertIn("fuzzy-matched 2", str(cm.exception))
        self.assertEqual(p.read_text(encoding="utf-8"), original)

    def test_multi_hunk_failure_leaves_file_untouched(self):
        original = "aaa\nbbb\nccc\n"
        p = self._write_and_read("m.py", original)
        with self.assertRaises(ToolInputError):
            self.tool.run(
                {"file_path": str(p), "edits": [
                    {"old_string": "aaa", "new_string": "AAA"},
                    {"old_string": "zzz-definitely-absent", "new_string": "Z"},
                ]},
                self.ctx,
            )
        self.assertEqual(p.read_text(encoding="utf-8"), original)

    def test_multi_hunk_mixes_exact_and_fuzzy_atomically(self):
        original = "    start()  \n    middle = 1\n"
        p = self._write_and_read("n.py", original)
        out = self.tool.run(
            {"file_path": str(p), "edits": [
                {"old_string": "    middle = 1", "new_string": "    middle = 2"},
                {"old_string": "    start()\n", "new_string": "    begin()\n"},
            ]},
            self.ctx,
        ).output
        self.assertEqual(p.read_text(encoding="utf-8"), "    begin()\n    middle = 2\n")
        self.assertEqual(out["hunks"][0]["match"]["strategy"], "exact")
        self.assertEqual(out["hunks"][1]["match"]["strategy"], "line-trim")
        self.assertIn("line-trim", out["notice"])

    def test_never_enters_fuzzy_path_when_content_matches_exactly(self):
        p = self._write_and_read("o.py", "print('hi')\n")
        out = self.tool.run(
            {"file_path": str(p),
             "old_string": "print('hi')",
             "new_string": "print('hello')"},
            self.ctx,
        ).output
        self.assertEqual(out["match"]["kind"], "exact")
        self.assertEqual(out["occurrences"], 1)


if __name__ == "__main__":
    unittest.main()
