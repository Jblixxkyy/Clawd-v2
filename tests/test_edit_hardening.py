"""Tests for Q10: Edit tool hardening.

Covers expected_occurrences validation, nearest-match hints on
not-found/ambiguous errors, and the atomic multi-hunk (apply_patch-style)
'edits' mode.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError
from src.tool_system.tools.edit import FileEditTool
from src.tool_system.tools.read import FileReadTool


class EditHardeningTests(unittest.TestCase):
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
        FileReadTool().run({"file_path": str(p), "limit": 100}, self.ctx)
        return p

    # -- expected_occurrences -------------------------------------------

    def test_expected_occurrences_match_replaces_all(self):
        p = self._write_and_read("a.txt", "foo\nbar\nfoo\n")
        out = self.tool.run(
            {"file_path": str(p), "old_string": "foo", "new_string": "FOO",
             "expected_occurrences": 2},
            self.ctx,
        ).output
        self.assertEqual(out["replaced"], 2)
        self.assertEqual(p.read_text(encoding="utf-8"), "FOO\nbar\nFOO\n")

    def test_expected_occurrences_mismatch_errors(self):
        p = self._write_and_read("b.txt", "foo\nbar\nfoo\n")
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run(
                {"file_path": str(p), "old_string": "foo", "new_string": "FOO",
                 "expected_occurrences": 3},
                self.ctx,
            )
        self.assertIn("expected_occurrences=3 but found 2", str(cm.exception))
        # file untouched
        self.assertEqual(p.read_text(encoding="utf-8"), "foo\nbar\nfoo\n")

    def test_expected_occurrences_one_keeps_single_replace(self):
        p = self._write_and_read("c.txt", "aaa bbb aaa\n")
        out = self.tool.run(
            {"file_path": str(p), "old_string": "aaa", "new_string": "AAA",
             "expected_occurrences": 2},
            self.ctx,
        ).output
        self.assertEqual(out["replaced"], 2)

    def test_expected_occurrences_rejects_negative(self):
        p = self._write_and_read("d.txt", "x\n")
        with self.assertRaises(ToolInputError):
            self.tool.run(
                {"file_path": str(p), "old_string": "x", "new_string": "y",
                 "expected_occurrences": -1},
                self.ctx,
            )

    # -- not found / ambiguous hints ------------------------------------

    def test_not_found_error_has_nearest_match_hints(self):
        p = self._write_and_read("e.txt", "alpha = 1\nbeta = 2\ngamma = 3\n")
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run(
                {"file_path": str(p), "old_string": "alphaa = 9", "new_string": "x"},
                self.ctx,
            )
        msg = str(cm.exception)
        self.assertIn("old_string not found", msg)
        self.assertIn("Did you mean", msg)
        self.assertIn("alpha = 1", msg)

    def test_ambiguous_error_lists_lines_and_suggestions(self):
        p = self._write_and_read("f.txt", "dup\none\ndup\ntwo\ndup\n")
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run(
                {"file_path": str(p), "old_string": "dup", "new_string": "DUP"},
                self.ctx,
            )
        msg = str(cm.exception)
        self.assertIn("not unique", msg)
        self.assertIn("lines 1, 3, 5", msg)
        self.assertIn("replace_all=true", msg)
        self.assertIn("expected_occurrences=3", msg)

    # -- multi-hunk mode -------------------------------------------------

    def test_multi_hunk_applies_all_atomically(self):
        p = self._write_and_read("g.txt", "aaa\nbbb\nccc\n")
        out = self.tool.run(
            {"file_path": str(p), "edits": [
                {"old_string": "aaa", "new_string": "AAA"},
                {"old_string": "ccc", "new_string": "CCC"},
            ]},
            self.ctx,
        ).output
        self.assertEqual(p.read_text(encoding="utf-8"), "AAA\nbbb\nCCC\n")
        self.assertEqual(len(out["hunks"]), 2)
        self.assertEqual(out["hunks"][0]["replaced"], 1)
        self.assertTrue(out["structuredPatch"])  # one combined diff

    def test_multi_hunk_failure_leaves_file_untouched(self):
        original = "aaa\nbbb\nccc\n"
        p = self._write_and_read("h.txt", original)
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run(
                {"file_path": str(p), "edits": [
                    {"old_string": "aaa", "new_string": "AAA"},
                    {"old_string": "zzz-missing", "new_string": "Z"},
                ]},
                self.ctx,
            )
        self.assertIn("not found", str(cm.exception))
        self.assertEqual(p.read_text(encoding="utf-8"), original)

    def test_multi_hunk_rejects_empty_edits(self):
        p = self._write_and_read("i.txt", "aaa\n")
        with self.assertRaises(ToolInputError):
            self.tool.run({"file_path": str(p), "edits": []}, self.ctx)

    def test_multi_hunk_per_hunk_expected_occurrences(self):
        p = self._write_and_read("j.txt", "k=1\nk=2\nother\n")
        out = self.tool.run(
            {"file_path": str(p), "edits": [
                {"old_string": "k=", "new_string": "K=", "expected_occurrences": 2},
                {"old_string": "other", "new_string": "OTHER"},
            ]},
            self.ctx,
        ).output
        self.assertEqual(p.read_text(encoding="utf-8"), "K=1\nK=2\nOTHER\n")
        self.assertEqual(out["hunks"][0]["replaced"], 2)

    def test_multi_hunk_per_hunk_replace_all(self):
        p = self._write_and_read("k.txt", "x=1\nx=2\n")
        self.tool.run(
            {"file_path": str(p), "edits": [
                {"old_string": "x=", "new_string": "y=", "replace_all": True},
            ]},
            self.ctx,
        )
        self.assertEqual(p.read_text(encoding="utf-8"), "y=1\ny=2\n")

    # -- backward compatibility ------------------------------------------

    def test_single_replace_still_works(self):
        p = self._write_and_read("l.txt", "hello world\n")
        out = self.tool.run(
            {"file_path": str(p), "old_string": "world", "new_string": "there"},
            self.ctx,
        ).output
        self.assertEqual(p.read_text(encoding="utf-8"), "hello there\n")
        self.assertEqual(out["oldString"], "world")
        self.assertEqual(out["replaceAll"], False)

    def test_replace_all_still_works(self):
        p = self._write_and_read("m.txt", "a a a\n")
        self.tool.run(
            {"file_path": str(p), "old_string": "a", "new_string": "b", "replace_all": True},
            self.ctx,
        )
        self.assertEqual(p.read_text(encoding="utf-8"), "b b b\n")


if __name__ == "__main__":
    unittest.main()
