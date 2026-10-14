"""Terminal success path: ``complete_task`` (brief 020).

gemini-cli ``complete-task.ts`` parity: this is the ONLY way to finish a
run cleanly — schema-validated (zod-style), so subagents can't ramble out.
OpenHands ``FinishTool`` parity: the typed schema forces a structured final
artifact (``outcome`` / ``summary`` / ``artifacts[]``), constraining answer
shape for weak models.

The loop treats a ``complete_task`` call as the completion signal
(``isComplete: True`` in the output → :func:`is_terminal_result`). The
absence of ``complete_task`` + no tool calls is a stall, not success —
see the completion clause wired in ``agent_loop.py`` and
``tools/subagent.py``.
"""

from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec

#: Structured outcomes the terminal success path can record.
COMPLETE_TASK_OUTCOMES = ("success", "partial")


class CompleteTaskTool:
    """The ONLY terminal success path. Schema-validated structured output."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="complete_task",
            description=(
                "TERMINAL. The ONLY way to finish a run successfully. Call it exactly once "
                "when the task is done, with a schema-validated structured outcome: "
                "'outcome' (success or partial), 'summary' (what was accomplished), and "
                "'artifacts' (files or outputs produced). Ending your turn without calling "
                "complete_task is logged as 'ended without explicit completion' — and a "
                "run that makes no tool calls and never calls complete_task is a stall, "
                "not success. If the task is unrecoverable, call ReportFailure instead."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": list(COMPLETE_TASK_OUTCOMES),
                        "description": "success: task fully accomplished. partial: best-effort result with known gaps.",
                    },
                    "summary": {
                        "type": "string",
                        "minLength": 1,
                        "description": "What was accomplished, in 1-3 sentences.",
                    },
                    "artifacts": {
                        "type": "array",
                        "description": "Files or outputs produced by the run.",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string", "minLength": 1},
                                "path": {"type": "string"},
                                "description": {"type": "string"},
                            },
                            "required": ["name"],
                        },
                    },
                },
                "required": ["outcome", "summary"],
            },
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        outcome = tool_input.get("outcome")
        if outcome not in COMPLETE_TASK_OUTCOMES:
            raise ToolInputError(
                f"outcome must be one of {', '.join(COMPLETE_TASK_OUTCOMES)}"
            )
        summary = tool_input.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ToolInputError("summary must be a non-empty string")
        artifacts = tool_input.get("artifacts", [])
        if not isinstance(artifacts, list):
            raise ToolInputError("artifacts must be an array")
        normalized_artifacts: list[dict[str, Any]] = []
        for i, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                raise ToolInputError(f"artifacts[{i}] must be an object")
            name = artifact.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ToolInputError(f"artifacts[{i}].name must be a non-empty string")
            entry: dict[str, Any] = {"name": name.strip()}
            for key in ("path", "description"):
                value = artifact.get(key)
                if value is not None:
                    if not isinstance(value, str):
                        raise ToolInputError(f"artifacts[{i}].{key} must be a string")
                    entry[key] = value
            normalized_artifacts.append(entry)
        summary = summary.strip()
        log_record = {
            "tool": "complete_task",
            "outcome": outcome,
            "summary": summary,
            "artifacts": normalized_artifacts,
        }
        return ToolResult(
            name="complete_task",
            output={
                "isComplete": True,
                "outcome": outcome,
                "summary": summary,
                "artifacts": normalized_artifacts,
                "widget": "complete_task",
                "log_record": log_record,
            },
        )
