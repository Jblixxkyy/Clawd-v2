from __future__ import annotations

import difflib
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..permission_handler import PermissionResult
from ..protocol import ToolResult
from ..diff_utils import unified_diff_hunks
from ..registry import ToolSpec


# ---------------------------------------------------------------------------
# Edit hardening (Q10): expected_occurrences validation, nearest-match hints,
# and an atomic multi-hunk mode so one tool call can do N replacements.
# ---------------------------------------------------------------------------

def _line_number(content: str, index: int) -> int:
    """1-based line number of *index* in *content*."""
    return content.count("\n", 0, index) + 1


def _occurrence_lines(content: str, old: str) -> list[int]:
    """Line numbers (1-based) where *old* occurs in *content*."""
    lines: list[int] = []
    start = 0
    step = max(len(old), 1)
    while True:
        i = content.find(old, start)
        if i == -1:
            break
        lines.append(_line_number(content, i))
        start = i + step
    return lines


def _nearest_match_hints(content: str, old: str, n: int = 3) -> list[str]:
    """Closest file lines to *old_string*, for 'not found' hints.

    Matches each non-blank line of the search string against the file's
    lines with difflib; returns up to *n* ``"line X: <snippet>"`` hints.
    """
    file_lines = content.splitlines()
    if not file_lines:
        return []
    needles = [ln.strip() for ln in old.splitlines() if ln.strip()]
    if not needles:
        needles = [old.strip()]
    hints: list[str] = []
    seen: set[str] = set()
    for needle in needles[:3]:
        for match in difflib.get_close_matches(needle, file_lines, n=n, cutoff=0.55):
            if match in seen:
                continue
            seen.add(match)
            snippet = match.strip()
            if len(snippet) > 90:
                snippet = snippet[:87] + "..."
            hints.append(f"line {file_lines.index(match) + 1}: {snippet}")
            if len(hints) >= n:
                return hints
    return hints


def _format_lines(lines: list[int], limit: int = 10) -> str:
    shown = ", ".join(str(x) for x in lines[:limit])
    if len(lines) > limit:
        shown += f", ... (+{len(lines) - limit} more)"
    return shown


def _validate_replacement(
    content: str,
    old: str,
    replace_all: bool,
    expected_occurrences: Any,
    path_display: str,
) -> int:
    """Validate one replacement against *content*; return occurrence count.

    Raises ToolInputError with a specific, actionable message when the
    search string is missing, ambiguous, or violates *expected_occurrences*.
    """
    if not isinstance(old, str) or not old:
        raise ToolInputError("old_string must be a non-empty string")
    if expected_occurrences is not None and (
        not isinstance(expected_occurrences, int)
        or isinstance(expected_occurrences, bool)
        or expected_occurrences < 0
    ):
        raise ToolInputError("expected_occurrences must be a non-negative integer")

    count = content.count(old)
    lines = _occurrence_lines(content, old) if count else []

    if count == 0:
        msg = f"old_string not found in '{path_display}' (0 occurrences)."
        hints = _nearest_match_hints(content, old)
        if hints:
            msg += "\nDid you mean one of these?\n" + "\n".join(f"  {h}" for h in hints)
        else:
            msg += " Check the exact text (whitespace included) or read the file again."
        raise ToolInputError(msg)

    if expected_occurrences is not None and count != expected_occurrences:
        raise ToolInputError(
            f"expected_occurrences={expected_occurrences} but found {count} "
            f"occurrence{'s' if count != 1 else ''} of old_string in "
            f"'{path_display}' (lines {_format_lines(lines)})."
        )

    if count > 1 and not replace_all and expected_occurrences is None:
        raise ToolInputError(
            f"old_string is not unique in '{path_display}': found {count} "
            f"occurrences (lines {_format_lines(lines)}). Provide a larger "
            f"old_string, set replace_all=true, or pass "
            f"expected_occurrences={count} to replace a specific count."
        )
    return count


def _check_hunk(hunk: Any, index: int) -> dict[str, Any]:
    """Validate one entry of the ``edits`` array; return a normalized hunk."""
    if not isinstance(hunk, dict):
        raise ToolInputError(f"edits[{index}] must be an object")
    old = hunk.get("old_string")
    new = hunk.get("new_string")
    if not isinstance(old, str) or not isinstance(new, str):
        raise ToolInputError(f"edits[{index}]: old_string/new_string must be strings")
    replace_all = bool(hunk.get("replace_all", False))
    expected = hunk.get("expected_occurrences")
    return {
        "old_string": old,
        "new_string": new,
        "replace_all": replace_all,
        "expected_occurrences": expected,
    }


class FileEditTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Edit",
            description=(
                "Performs exact string replacements in files. Pass "
                "expected_occurrences to assert how many matches old_string "
                "should have, or pass an 'edits' array of {old_string, "
                "new_string, replace_all?, expected_occurrences?} hunks to "
                "apply N replacements atomically in one call."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                    "expected_occurrences": {"type": "integer", "minimum": 0},
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                                "replace_all": {"type": "boolean"},
                                "expected_occurrences": {"type": "integer", "minimum": 0},
                            },
                            "required": ["old_string", "new_string"],
                        },
                    },
                },
                "required": ["file_path"],
            },
            is_destructive=True,
            strict=True,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        """Check if edit permission is allowed for this file."""
        file_path = tool_input.get("file_path")
        if not isinstance(file_path, str):
            return PermissionResult.allow()  # Input validation happens in run()

        try:
            path = context.ensure_allowed_path(file_path)
        except ToolPermissionError:
            return PermissionResult.allow()  # Path validation happens in run()

        if path.suffix.lower() in {".md", ".markdown"} and not context.permission_context.allow_docs:
            return PermissionResult.ask(
                message="Editing documentation files is blocked unless allow_docs is enabled",
                suggestion="Enable allow_docs to edit .md files",
            )
        return PermissionResult.allow()

    def _load_for_edit(self, file_path: Any, context: ToolContext):
        if not isinstance(file_path, str):
            raise ToolInputError("file_path must be a string")
        path = context.ensure_allowed_path(file_path)
        if not path.exists():
            raise ToolInputError(f"file does not exist: {path}")
        if not context.was_file_read_and_unchanged(path):
            raise ToolInputError("refusing to edit: file must be read first and unchanged since last read")
        original = path.read_text(encoding="utf-8", errors="replace")
        return path, original

    def _write_and_diff(self, path, original: str, updated: str, context: ToolContext):
        path.write_text(updated, encoding="utf-8")
        context.mark_file_read(path)
        before_lines = original.splitlines(keepends=True)
        after_lines = updated.splitlines(keepends=True)
        diff_lines = list(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=str(path),
                tofile=str(path),
                n=3,
                lineterm="",
            )
        )
        return unified_diff_hunks(diff_lines)

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        file_path = tool_input["file_path"]
        edits = tool_input.get("edits")
        path, original = self._load_for_edit(file_path, context)
        display = str(path)

        if edits is not None:
            return self._run_multi_hunk(path, original, display, edits, context)

        old = tool_input.get("old_string")
        new = tool_input.get("new_string")
        if not isinstance(old, str) or not isinstance(new, str):
            raise ToolInputError("old_string/new_string must be strings (or use 'edits' for multi-hunk mode)")
        replace_all = bool(tool_input.get("replace_all", False))
        expected = tool_input.get("expected_occurrences")

        count = _validate_replacement(original, old, replace_all, expected, display)
        if replace_all or (expected is not None and expected > 1):
            updated = original.replace(old, new)
            replaced = count
        else:
            updated = original.replace(old, new, 1)
            replaced = 1

        hunks = self._write_and_diff(path, original, updated, context)
        return ToolResult(
            name="Edit",
            output={
                "filePath": str(path),
                "oldString": old,
                "newString": new,
                "occurrences": count,
                "replaced": replaced,
                "originalFile": original,
                "structuredPatch": hunks,
                "userModified": False,
                "replaceAll": bool(replace_all),
            },
        )

    def _run_multi_hunk(
        self,
        path,
        original: str,
        display: str,
        edits: Any,
        context: ToolContext,
    ) -> ToolResult:
        """Apply N replacements atomically: validate all against the original,
        then write once. Any validation failure leaves the file untouched."""
        if not isinstance(edits, list) or not edits:
            raise ToolInputError("'edits' must be a non-empty array of {old_string, new_string} hunks")
        hunks = [_check_hunk(h, i) for i, h in enumerate(edits)]

        # Phase 1: validate every hunk against the ORIGINAL content.
        counts = [
            _validate_replacement(
                original,
                h["old_string"],
                h["replace_all"],
                h["expected_occurrences"],
                display,
            )
            for h in hunks
        ]

        # Phase 2: apply in order to a working copy, then write once.
        updated = original
        replaced_counts: list[int] = []
        for h, count in zip(hunks, counts):
            if h["replace_all"] or (h["expected_occurrences"] is not None and h["expected_occurrences"] > 1):
                updated = updated.replace(h["old_string"], h["new_string"])
                replaced_counts.append(count)
            else:
                updated = updated.replace(h["old_string"], h["new_string"], 1)
                replaced_counts.append(1)

        patch_hunks = self._write_and_diff(path, original, updated, context)
        return ToolResult(
            name="Edit",
            output={
                "filePath": str(path),
                "hunks": [
                    {
                        "oldString": h["old_string"],
                        "newString": h["new_string"],
                        "occurrences": c,
                        "replaced": r,
                        "replaceAll": h["replace_all"],
                    }
                    for h, c, r in zip(hunks, counts, replaced_counts)
                ],
                "originalFile": original,
                "structuredPatch": patch_hunks,
                "userModified": False,
            },
        )
