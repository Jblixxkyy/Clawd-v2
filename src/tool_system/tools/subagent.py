"""True agentic subagents (brief 005).

Unlike the fixed call-list ``Agent`` tool, ``Subagent`` spawns a real
LLM-driven child: it runs its own ``run_agent_loop`` with a fresh
``Conversation``, a compact subagent system prompt, and a filtered copy of
the tool registry (session-scoped tools removed). Permission prompts inside
the child can never reach the user -- they auto-deny, same rule as the
existing ``Agent`` tool.

Background mode returns immediately with a task id; the parent fetches the
result with the ``SubagentOutput`` tool (poll-based, mirroring the
background-Bash pattern).
"""

from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import fields, replace
from typing import Any

from ..agent_loop import run_agent_loop
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolRegistry, ToolSpec
from ...agent.conversation import Conversation

# Tool names (lowercase spec.name) that are scoped to the interactive
# parent session and must not leak into a child subagent's registry.
_SESSION_SCOPED_TOOL_NAMES = frozenset(
    {
        # User interaction / session outbox
        "sendusermessage",
        "askuserquestion",
        "sendmessage",
        "structuredoutput",
        # User-level scheduling and team state
        "croncreate",
        "cronlist",
        "crondelete",
        "teamcreate",
        "teamdelete",
        # Interactive session modes / workspace mutation
        "enterplanmode",
        "exitplanmode",
        "enterworktree",
        "exitworktree",
        # User config mutation
        "config",
        # Parent's background tasks (shared task_manager would let the
        # child stop the parent's jobs)
        "taskstop",
        # No nesting in v1
        "subagent",
    }
)

# Tools that capture a registry reference in their constructor and must be
# re-bound to the child registry instead of being shared with the parent.
# (Handled in build_child_registry via lazy imports to avoid import cycles.)

_DEFAULT_CHILD_MAX_TURNS = 25
_MAX_CHILD_MAX_TURNS = 100
_OUTPUT_TRUNC_CHARS = 20000


def _sub_context(context: ToolContext) -> ToolContext:
    """Build the child's execution context from the parent's.

    Shares tasks/todos/task_manager with the parent, but can never prompt
    the user: permission checks that would ask are auto-denied (see
    registry.dispatch), and ask_user is disabled so AskUserQuestion-style
    paths fail cleanly instead of blocking on stdin. The child gets the
    compact subagent system prompt via output_style_name.
    """
    sub_context = replace(
        context,
        non_interactive=True,
        ask_user=None,
        output_style_name="subagent",
    )
    # dataclasses.replace only carries declared fields; preserve any extra
    # attributes set on the parent (e.g. session_id).
    _field_names = {f.name for f in fields(context)}
    for _k, _v in vars(context).items():
        if _k not in _field_names:
            setattr(sub_context, _k, _v)
    return sub_context


def build_child_registry(parent_registry: ToolRegistry) -> ToolRegistry:
    """Build a filtered copy of the parent registry for a subagent.

    Drops session-scoped tools and re-binds registry-capturing tools
    (Agent, ToolSearch) to the child registry so the child can never
    dispatch through the parent's full registry.
    """
    # Imported lazily to avoid a hard import cycle at module load
    # (subagent is imported from tools/__init__ via defaults).
    from .agent import AgentTool
    from .tool_search import ToolSearchTool

    child = ToolRegistry()
    for tool in parent_registry.list_tools():
        name = tool.spec().name.lower()
        if name in _SESSION_SCOPED_TOOL_NAMES:
            continue
        if isinstance(tool, (AgentTool, ToolSearchTool, SubagentTool)):
            continue  # re-bound below
        child.register(tool)
    child.register(AgentTool(child))
    child.register(ToolSearchTool(child))
    return child


# --- Background subagent records -------------------------------------------
# job_id -> {"status", "description", "model", "started_at", "done_event",
#            "response_text", "num_turns", "cancelled", "error"}
_BG_SUBAGENTS: dict[str, dict[str, Any]] = {}
_BG_LOCK = threading.Lock()


def _new_task_id() -> str:
    return "subagent_" + uuid.uuid4().hex[:12]


def _run_child_loop(
    *,
    task_id: str,
    description: str,
    prompt: str,
    provider: Any,
    parent_registry: ToolRegistry,
    parent_context: ToolContext,
    max_turns: int,
    model: str | None,
) -> dict[str, Any]:
    """Run one child agent loop; returns the record fields for the registry."""
    child_provider = copy.copy(provider)
    if model:
        child_provider.model = model
    child_registry = build_child_registry(parent_registry)
    sub_context = _sub_context(parent_context)
    conversation = Conversation()
    conversation.add_user_message(prompt)
    try:
        result = run_agent_loop(
            conversation=conversation,
            provider=child_provider,
            tool_registry=child_registry,
            tool_context=sub_context,
            max_turns=max_turns,
        )
        return {
            "status": "done",
            "response_text": result.response_text,
            "num_turns": result.num_turns,
            "cancelled": result.cancelled,
            "usage": result.usage,
            "error": None,
        }
    except Exception as exc:  # never let a thread die silently
        return {
            "status": "failed",
            "response_text": "",
            "num_turns": 0,
            "cancelled": False,
            "usage": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


class SubagentTool:
    """Delegate an autonomous task to an LLM-driven subagent."""

    def __init__(self, registry: ToolRegistry):
        self._registry = registry

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Subagent",
            description=(
                "Delegate an autonomous task to an LLM-driven subagent. The "
                "child runs its own agent loop with its own context and "
                "tools, then returns its final report. Use for explore-then-"
                "report work (investigate a bug, survey a codebase, draft a "
                "plan) instead of doing many dependent steps yourself. Give "
                "a precise, self-contained prompt: the child has no access "
                "to this conversation's history. Inside the child, permission "
                "prompts cannot reach the user: any tool call needing "
                "approval is auto-denied and reported. Pass background=true "
                "to run detached and fetch the result later with "
                "SubagentOutput."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Short (3-5 word) description of the task.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The full task for the subagent. Be specific about what to do and what to return.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Optional model override for the child (same provider).",
                    },
                    "background": {
                        "type": "boolean",
                        "description": "Run detached; returns a task id immediately.",
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "Max tool turns for the child (default 25).",
                    },
                },
                "required": ["description", "prompt"],
            },
            is_destructive=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        description = tool_input.get("description")
        prompt = tool_input.get("prompt")
        if not isinstance(description, str) or not description.strip():
            raise ToolInputError("description must be a non-empty string")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ToolInputError("prompt must be a non-empty string")
        model = tool_input.get("model")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ToolInputError("model must be a non-empty string when provided")
        background = bool(tool_input.get("background", False))
        max_turns = tool_input.get("max_turns", _DEFAULT_CHILD_MAX_TURNS)
        if not isinstance(max_turns, int) or isinstance(max_turns, bool):
            raise ToolInputError("max_turns must be an integer")
        max_turns = max(1, min(_MAX_CHILD_MAX_TURNS, max_turns))

        provider = getattr(context, "provider", None)
        if provider is None:
            return ToolResult(
                name="Subagent",
                output={"error": "no provider available for subagent loop"},
                is_error=True,
            )

        task_id = _new_task_id()

        if background:
            record: dict[str, Any] = {
                "task_id": task_id,
                "description": description,
                "model": model or getattr(provider, "model", None),
                "status": "running",
                "started_at": time.time(),
                "done_event": threading.Event(),
                "response_text": "",
                "num_turns": 0,
                "cancelled": False,
                "usage": None,
                "error": None,
            }
            with _BG_LOCK:
                _BG_SUBAGENTS[task_id] = record

            def _target() -> None:
                fields = _run_child_loop(
                    task_id=task_id,
                    description=description,
                    prompt=prompt,
                    provider=provider,
                    parent_registry=self._registry,
                    parent_context=context,
                    max_turns=max_turns,
                    model=model,
                )
                with _BG_LOCK:
                    record.update(fields)
                    record["done_event"].set()

            thread = threading.Thread(target=_target, name=f"subagent-{task_id}", daemon=True)
            thread.start()
            return ToolResult(
                name="Subagent",
                output={
                    "task_id": task_id,
                    "description": description,
                    "status": "running",
                    "note": (
                        "The subagent is running in the background. Use "
                        "SubagentOutput with this task_id to fetch the result "
                        "(it can block until done)."
                    ),
                },
                is_error=False,
            )

        record = _run_child_loop(
            task_id=task_id,
            description=description,
            prompt=prompt,
            provider=provider,
            parent_registry=self._registry,
            parent_context=context,
            max_turns=max_turns,
            model=model,
        )
        output: dict[str, Any] = {
            "task_id": task_id,
            "description": description,
            "model": model or getattr(provider, "model", None),
            "status": record["status"],
            "num_turns": record["num_turns"],
            "cancelled": record["cancelled"],
            "response_text": record["response_text"],
        }
        if record["usage"]:
            output["usage"] = record["usage"]
        if record["error"]:
            output["error"] = record["error"]
        return ToolResult(name="Subagent", output=output, is_error=record["status"] == "failed")


class SubagentOutputTool:
    """Fetch the result of a background subagent."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="SubagentOutput",
            description=(
                "Get the result of a background Subagent task. Pass "
                "block=true to wait until it finishes (up to timeout "
                "seconds)."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "task_id": {"type": "string"},
                    "block": {"type": "boolean"},
                    "timeout": {"type": "number"},
                },
                "required": ["task_id"],
            },
            is_read_only=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        task_id = tool_input.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            raise ToolInputError("task_id must be a non-empty string")
        block = bool(tool_input.get("block", False))
        timeout = tool_input.get("timeout", 30.0)
        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            raise ToolInputError("timeout must be a number")
        timeout = max(0.0, min(600.0, timeout))

        with _BG_LOCK:
            record = _BG_SUBAGENTS.get(task_id)
            done_event = record["done_event"] if record else None

        if record is None:
            return ToolResult(
                name="SubagentOutput",
                output={"retrieval_status": "success", "task": None},
                is_error=False,
            )

        if block and done_event is not None:
            done_event.wait(timeout)

        with _BG_LOCK:
            status = record["status"]
            response_text = record["response_text"]
            num_turns = record["num_turns"]
            cancelled = record["cancelled"]
            error = record["error"]
            description = record["description"]

        retrieval_status = "success" if status in ("done", "failed") else "not_ready"
        if len(response_text) > _OUTPUT_TRUNC_CHARS:
            response_text = (
                response_text[:_OUTPUT_TRUNC_CHARS]
                + f"\n... [truncated, {len(response_text) - _OUTPUT_TRUNC_CHARS} more chars]"
            )
        return ToolResult(
            name="SubagentOutput",
            output={
                "retrieval_status": retrieval_status,
                "task": {
                    "task_id": task_id,
                    "task_type": "subagent",
                    "status": status,
                    "description": description,
                    "num_turns": num_turns,
                    "cancelled": cancelled,
                    "error": error,
                    "response_text": response_text,
                },
            },
            is_error=False,
        )
