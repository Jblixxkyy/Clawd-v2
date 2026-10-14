from __future__ import annotations

import uuid
from dataclasses import fields, replace
from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolCall, ToolResult
from ..registry import ToolRegistry, ToolSpec


class AgentTool:
    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Agent",
            description=(
                "Delegate a self-contained sequence of tool calls as a single "
                "atomic subagent task. Use for separable work (explore these files, "
                "run the tests, summarize the results) instead of doing many "
                "dependent steps yourself. Inside the task, permission prompts "
                "cannot reach the user: any tool call needing approval is "
                "auto-denied and reported back."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "calls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {"name": {"type": "string"}, "input": {"type": "object"}},
                            "required": ["name", "input"],
                        },
                    },
                    "stop_on_error": {"type": "boolean"},
                    "description": {"type": "string"},
                },
                "required": ["calls"],
            },
            aliases=("Task",),
            is_destructive=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        calls = tool_input["calls"]
        stop_on_error = bool(tool_input.get("stop_on_error", True))
        if not isinstance(calls, list):
            raise ToolInputError("calls must be an array")
        description = tool_input.get("description")
        if description is not None and not isinstance(description, str):
            raise ToolInputError("description must be a string when provided")

        task_id = uuid.uuid4().hex[:12]
        # Subagent execution context: shares tasks/todos/task_manager with
        # the parent, but can never prompt the user. Permission checks that
        # would ask are auto-denied (see registry.dispatch), and ask_user
        # is disabled so AskUserQuestion fails cleanly instead of blocking
        # on stdin. Denials are collected below and reported to the parent.
        sub_context = replace(context, non_interactive=True, ask_user=None)
        # dataclasses.replace only carries declared fields; preserve any
        # extra attributes set on the parent (e.g. session_id).
        _field_names = {f.name for f in fields(context)}
        for _k, _v in vars(context).items():
            if _k not in _field_names:
                setattr(sub_context, _k, _v)

        results: list[dict[str, Any]] = []
        permission_denials: list[dict[str, Any]] = []
        any_error = False
        for idx, call in enumerate(calls):
            if not isinstance(call, dict):
                raise ToolInputError(f"calls[{idx}] must be an object")
            name = call.get("name")
            inp = call.get("input")
            if not isinstance(name, str) or not isinstance(inp, dict):
                raise ToolInputError(f"calls[{idx}] must include name:string and input:object")
            result = self._registry.dispatch(ToolCall(name=name, input=inp), sub_context)
            results.append({"name": name, "is_error": result.is_error, "output": result.output})
            if result.is_error and isinstance(result.output, dict) and result.output.get(
                "subagent_permission_denied"
            ):
                permission_denials.append({"name": name, "error": result.output.get("error")})
            any_error = any_error or result.is_error
            if result.is_error and stop_on_error:
                break

        num_failed = sum(1 for r in results if r["is_error"])
        output: dict[str, Any] = {
            "task_id": task_id,
            "description": description or "",
            "num_calls": len(results),
            "num_succeeded": len(results) - num_failed,
            "num_failed": num_failed,
            "permission_denials": permission_denials,
            "results": results,
        }
        if permission_denials:
            output["note"] = (
                "Some tool calls were denied because they needed user approval, "
                "which is unavailable inside a subagent task. Delegate only work "
                "that needs no interactive approval."
            )
        return ToolResult(name="Agent", output=output, is_error=any_error)
