"""Structured todo tracking with harness-enforced invariants (brief 020).

Ported semantics (gemini-cli ``write-todos.ts`` / codex ``plan_spec.rs`` /
crush ``todos.go``):

- Whole-list overwrite: every call submits the ENTIRE updated list; the
  harness replaces the persisted state wholesale.
- Server-side validation: at most one ``in_progress`` item, status enum
  (``pending``, ``in_progress``, ``completed``, ``blocked``) validated
  server-side — the schema is the last line of defense, the invariant
  check is the harness's.
- Persisted in session state: todos are written to
  ``~/.clawd/sessions/<session_id>.todos.json`` on every update and can
  be reloaded after a restart (:func:`load_session_todos`).
- Planning/execution disjoint: rejected while the context is in plan
  mode (016's plan artifact owns planning; this tool owns execution
  tracking).
- Response metadata carries list deltas (added / removed / status-changed)
  so the model can see exactly what moved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec

#: Valid todo statuses. ``blocked`` (gemini-cli parity) marks an item that
#: cannot proceed until something external unblocks it.
TODO_STATUSES = ("pending", "in_progress", "completed", "blocked")


def session_todos_path(session_id: str) -> Path:
    """Path of the per-session todo sidecar file.

    Lives next to the ``~/.clawd/sessions/<session_id>.json`` session file
    (``src/agent/session.py``) so todos survive restarts.
    """
    return Path.home() / ".clawd" / "sessions" / f"{session_id}.todos.json"


def save_session_todos(session_id: str, todos: list[dict[str, Any]]) -> None:
    """Persist todos for a session. Fail-open: never raises."""
    try:
        path = session_todos_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".todos.json.tmp")
        tmp.write_text(json.dumps(todos, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass


def load_session_todos(session_id: str) -> list[dict[str, Any]]:
    """Load persisted todos for a session. Returns [] on any failure."""
    try:
        path = session_todos_path(session_id)
        if not path.exists():
            return []
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            return []
        return [t for t in raw if isinstance(t, dict)]
    except Exception:
        return []


def _todo_deltas(
    old: list[dict[str, Any]], new: list[dict[str, Any]]
) -> dict[str, list[Any]]:
    """Diff two todo lists by ``content`` key.

    Returns ``{"added": [...], "removed": [...], "status_changed": [...]}``
    as response metadata (crush parity: "returns deltas as response
    metadata").
    """
    old_by_content = {t.get("content"): t for t in old if isinstance(t, dict)}
    new_by_content = {t.get("content"): t for t in new if isinstance(t, dict)}
    added = [c for c in new_by_content if c not in old_by_content]
    removed = [c for c in old_by_content if c not in new_by_content]
    status_changed = [
        {"content": c, "from": old_by_content[c].get("status"), "to": new_by_content[c].get("status")}
        for c in new_by_content
        if c in old_by_content
        and new_by_content[c].get("status") != old_by_content[c].get("status")
    ]
    return {"added": added, "removed": removed, "status_changed": status_changed}


class TodoWriteTool:
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="TodoWrite",
            description=(
                "Creates and updates a structured task list for the current session. Use it for tasks "
                "with three or more distinct steps or when the user gives several things to do; skip it "
                "for trivial single-step requests. Each todo needs content (imperative, e.g. \"Fix login "
                "bug\"), activeForm (present tense, e.g. \"Fixing login bug\") and a status of pending, "
                "in_progress, completed or blocked. Keep exactly one item in_progress at a time and mark "
                "items completed immediately when finished. Always send the full updated list — the "
                "harness replaces the whole list. Rejected while in plan mode (planning and execution "
                "tracking are deliberately disjoint); the todo list is persisted with the session and "
                "survives restarts."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "content": {"type": "string", "minLength": 1},
                                "status": {
                                    "type": "string",
                                    "enum": list(TODO_STATUSES),
                                },
                                "activeForm": {"type": "string", "minLength": 1},
                            },
                            "required": ["content", "status", "activeForm"],
                        },
                    }
                },
                "required": ["todos"],
            },
            is_read_only=True,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        # Planning / execution tracking are deliberately disjoint (020 +
        # codex plan_spec.rs parity): TodoWrite tracks EXECUTION; 016's plan
        # artifact owns planning. Enforced here (not just the registry's
        # plan-mode gate) so it holds on every dispatch path.
        if getattr(context, "plan_mode", False):
            raise ToolInputError(
                "TodoWrite is disabled while in plan mode: planning and execution "
                "tracking are deliberately disjoint. Finish planning with "
                "exit_plan_mode, then track implementation steps as todos."
            )
        todos = tool_input.get("todos")
        if not isinstance(todos, list):
            raise ToolInputError("todos must be an array")

        old = list(context.todos)
        in_progress = 0
        all_done = True
        normalized: list[dict[str, Any]] = []
        for i, t in enumerate(todos):
            if not isinstance(t, dict):
                raise ToolInputError(f"todos[{i}] must be an object")
            content = t.get("content")
            status = t.get("status")
            active_form = t.get("activeForm")
            if not isinstance(content, str) or not content.strip():
                raise ToolInputError(f"todos[{i}].content must be a non-empty string")
            if status not in TODO_STATUSES:
                raise ToolInputError(
                    f"todos[{i}].status must be one of {', '.join(TODO_STATUSES)}"
                )
            if not isinstance(active_form, str) or not active_form.strip():
                raise ToolInputError(f"todos[{i}].activeForm must be a non-empty string")
            if status == "in_progress":
                in_progress += 1
            all_done = all_done and status == "completed"
            normalized.append({"content": content, "status": status, "activeForm": active_form})

        # gemini-cli parity: reject more than one in_progress.
        if in_progress > 1:
            raise ToolInputError(
                f"at most one todo may be in_progress at a time (found {in_progress}); "
                "submit the whole list with the other items moved to pending, "
                "blocked, or completed"
            )

        context.todos = [] if all_done else normalized
        deltas = _todo_deltas(old, context.todos)
        # Persist in session state so the list survives restarts.
        session_id = getattr(context, "session_id", None)
        if isinstance(session_id, str) and session_id:
            save_session_todos(session_id, context.todos)
        return ToolResult(
            name="TodoWrite",
            output={
                "oldTodos": old,
                "newTodos": context.todos,
                "delta": deltas,
            },
        )
