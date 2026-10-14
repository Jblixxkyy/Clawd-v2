from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol

from .context import ToolContext
from .input_repair import repair_input
from .permission_handler import PermissionResult, apply_permission_mode
from .permissions import PermissionMode
from .permission_store import is_always_allowed
from .protocol import (
    DEFAULT_MAX_RESULT_SIZE_CHARS,
    ToolCall,
    ToolResult,
    truncate_text,
)
from .schema_validation import validate_json_schema


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: Mapping[str, Any]
    aliases: tuple[str, ...] = ()
    is_read_only: bool = False
    is_destructive: bool = False
    strict: bool = False
    max_result_size_chars: int = DEFAULT_MAX_RESULT_SIZE_CHARS


class Tool(Protocol):
    def spec(self) -> ToolSpec: ...

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult: ...

    def check_permissions(
        self, tool_input: dict[str, Any], context: ToolContext
    ) -> PermissionResult:
        """Check if this tool has permission to run.

        Args:
            tool_input: The input arguments for the tool.
            context: The tool execution context.

        Returns:
            PermissionResult indicating allow, deny, or ask.
        """
        return PermissionResult.allow()


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] | None = None) -> None:
        self._tools: list[Tool] = []
        self._by_name: dict[str, Tool] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool: Tool) -> None:
        spec = tool.spec()
        key = spec.name.lower()
        if key in self._by_name:
            raise ValueError(f"duplicate tool name: {spec.name}")
        self._tools.append(tool)
        self._by_name[key] = tool
        for alias in spec.aliases:
            alias_key = alias.lower()
            if alias_key in self._by_name:
                raise ValueError(f"duplicate tool alias: {alias}")
            self._by_name[alias_key] = tool

    def list_specs(self) -> list[ToolSpec]:
        return [tool.spec() for tool in self._tools]

    def list_tools(self) -> list[Tool]:
        """Return the registered tool objects (for building filtered copies)."""
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name.lower())

    def dispatch(self, call: ToolCall, context: ToolContext) -> ToolResult:
        tool = self.get(call.name)
        if tool is None:
            return ToolResult(
                name=call.name,
                output={"error": f"unknown tool: {call.name}"},
                is_error=True,
                tool_use_id=call.tool_use_id,
            )
        spec = tool.spec()
        context.ensure_tool_allowed(spec.name)
        # Input repair (brief 007): coerce malformed inputs when the schema
        # unambiguously supports it, BEFORE validation. Fail open - a repair
        # bug must never break a tool call.
        try:
            repaired = repair_input(call.input, spec.input_schema)
            if repaired is not call.input:
                call = ToolCall(
                    name=call.name,
                    input=repaired,
                    tool_use_id=call.tool_use_id,
                )
        except Exception:
            pass
        validate_json_schema(call.input, spec.input_schema, root_name=spec.name)

        # Plan mode: while planning, only read-only tools (and ExitPlanMode,
        # the way out) may run. Writes/edits/shell are blocked, not prompted.
        if getattr(context, "plan_mode", False) and not spec.is_read_only and spec.name != "ExitPlanMode":
            return ToolResult(
                name=spec.name,
                output={
                    "error": (
                        "plan mode is on: this tool is disabled while planning. "
                        "Finish exploring, then call ExitPlanMode with the plan "
                        "for approval."
                    )
                },
                is_error=True,
                tool_use_id=call.tool_use_id,
            )

        # Hooks: PreToolUse runs after the plan-mode gate, before permissions.
        # Fail open — a broken hook never breaks a tool call.
        try:
            from .hooks.runner import run_hooks
            _hook_result = run_hooks("PreToolUse", spec.name, call.input)
            if _hook_result.get("blocked"):
                return ToolResult(
                    name=spec.name,
                    output={"error": _hook_result.get("message") or "blocked by PreToolUse hook"},
                    is_error=True,
                    tool_use_id=call.tool_use_id,
                )
        except Exception:
            pass

        # Check permissions before running
        permission_result = tool.check_permissions(call.input, context) if hasattr(tool, 'check_permissions') else PermissionResult.allow()
        # Session permission mode (stored on the context): deny rules always
        # win; acceptEdits auto-allows Edit/Write; dontAsk auto-denies
        # anything not pre-approved without prompting; bypassPermissions
        # skips prompts but keeps deny rules.
        _mode = getattr(context, "permission_mode", PermissionMode.DEFAULT)
        permission_result = apply_permission_mode(
            spec.name,
            permission_result,
            _mode,
            permission_context=context.permission_context,
            always_allowed=(
                _mode is PermissionMode.DONT_ASK and is_always_allowed(spec.name)
            ),
        )
        if permission_result.behavior.value == "allow" and permission_result.updated_input:
            # Mode-converted allow (e.g. a pre-approved tool in dontAsk mode):
            # carry the tool's updated input forward, mirroring the ASK
            # branch below.
            call = ToolCall(
                name=call.name,
                input=permission_result.updated_input,
                tool_use_id=call.tool_use_id,
            )
        if permission_result.behavior.value == "deny":
            return ToolResult(
                name=spec.name,
                output={"error": permission_result.message or "permission denied"},
                is_error=True,
                tool_use_id=call.tool_use_id,
            )
        if permission_result.behavior.value == "ask":
            # Subagent / delegated runs must never block on stdin waiting
            # for a user: deny outright and report, without invoking the
            # interactive permission handler.
            if getattr(context, "non_interactive", False):
                return ToolResult(
                    name=spec.name,
                    output={
                        "error": (
                            f"permission denied: '{spec.name}' requires user approval, "
                            "which is unavailable inside a subagent task. "
                            "Restructure the task to avoid steps that need "
                            "interactive approval."
                        ),
                        "subagent_permission_denied": True,
                    },
                    is_error=True,
                    tool_use_id=call.tool_use_id,
                )
            # Need user interaction
            if context.permission_handler is None:
                # No handler available, deny by default
                return ToolResult(
                    name=spec.name,
                    output={"error": permission_result.message or "permission required but no handler available"},
                    is_error=True,
                    tool_use_id=call.tool_use_id,
                )
            # Call the permission handler
            allowed, _ = context.permission_handler(
                spec.name,
                permission_result.message or f"Tool '{spec.name}' requires permission",
                permission_result.suggestion,
            )
            if not allowed:
                return ToolResult(
                    name=spec.name,
                    output={"error": "permission denied by user"},
                    is_error=True,
                    tool_use_id=call.tool_use_id,
                )
            # User allowed - proceed with potentially updated input
            if permission_result.updated_input:
                call = ToolCall(
                    name=call.name,
                    input=permission_result.updated_input,
                    tool_use_id=call.tool_use_id,
                )

        # Checkpoints: snapshot the file before any Write/Edit runs.
        if spec.name in ("Write", "Edit"):
            _fp = call.input.get("file_path")
            if isinstance(_fp, str):
                try:
                    from . import checkpoints as _checkpoints
                    _sid = getattr(context, "session_id", None) or "default"
                    _checkpoints.snapshot(_sid, _fp, context.workspace_root)
                except Exception:
                    pass

        result = tool.run(call.input, context)
        # Hooks: PostToolUse cannot block; output ignored; fail open.
        try:
            from .hooks.runner import run_hooks
            run_hooks("PostToolUse", spec.name, call.input, result.output)
        except Exception:
            pass
        # Context budget (Q9): cap what the model actually sees at the
        # tool’s max_result_size_chars. PostToolUse hooks above already
        # saw the full output. Oversized results are cut with a visible
        # ``[truncated — showing N of M chars]`` marker.
        result = _apply_result_budget(spec, result)
        if result.tool_use_id is None and call.tool_use_id is not None:
            return ToolResult(
                name=result.name,
                output=result.output,
                is_error=result.is_error,
                tool_use_id=call.tool_use_id,
                content_type=result.content_type,
            )
        return result


def _apply_result_budget(spec: ToolSpec, result: ToolResult) -> ToolResult:
    """Truncate an oversized tool result to the tool’s char budget.

    String outputs are cut in place. Anything else is rendered to JSON and,
    only when over budget, replaced by the truncated text plus the
    ``[truncated — showing N of M chars]`` marker (content_type flips
    to "text"). Results that already fit pass through untouched.
    """
    limit = spec.max_result_size_chars
    output = result.output
    if isinstance(output, str):
        if len(output) <= limit:
            return result
        return ToolResult(
            name=result.name,
            output=truncate_text(output, limit),
            is_error=result.is_error,
            tool_use_id=result.tool_use_id,
            content_type="text",
        )
    try:
        rendered = json.dumps(output, ensure_ascii=False)
    except (TypeError, ValueError):
        rendered = str(output)
    if len(rendered) <= limit:
        return result
    return ToolResult(
        name=result.name,
        output=truncate_text(rendered, limit),
        is_error=result.is_error,
        tool_use_id=result.tool_use_id,
        content_type="text",
    )
