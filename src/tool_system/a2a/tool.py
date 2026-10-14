"""The ``DelegateAgent`` tool (brief 010): delegate a task to an external A2A agent.

Input: ``agent_card_url`` (fetched + validated), ``task`` (the work to
delegate), optional ``timeout`` (seconds, default 600). The whole exchange
is bounded by the timeout: on expiry the tool returns a clean error
instead of hanging.
"""

from __future__ import annotations

from typing import Any

from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec
from .client import (
    A2AError,
    AgentCard,
    DEFAULT_TIMEOUT,
    MAX_TIMEOUT,
    delegate_task,
    fetch_agent_card,
)

_FAILED_STATES = {"failed", "rejected", "canceled", "cancelled"}


class DelegateAgentTool:
    """Delegate a task to an external agent via the A2A protocol."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="DelegateAgent",
            description=(
                "Delegate a task to an external A2A agent. Fetches the "
                "agent's AgentCard from agent_card_url, sends the task via "
                "JSON-RPC (streaming when the card advertises it), and "
                "reassembles the incremental status/artifact updates into a "
                "final result. Use to hand off specialist work (code review, "
                "deep research) to another agent without caring what model "
                "it runs. The call is bounded by timeout seconds and returns "
                "a clean error instead of hanging."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "agent_card_url": {
                        "type": "string",
                        "description": (
                            "URL of the remote agent's AgentCard "
                            "(e.g. https://host/.well-known/agent-card.json)."
                        ),
                    },
                    "task": {
                        "type": "string",
                        "description": "The task to delegate, as a natural-language message.",
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Max seconds to wait for the remote agent "
                            f"(default {DEFAULT_TIMEOUT:g}, max {MAX_TIMEOUT:g})."
                        ),
                    },
                },
                "required": ["agent_card_url", "task"],
            },
            is_destructive=True,  # network egress + third-party agent action
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        card_url = tool_input.get("agent_card_url")
        if not isinstance(card_url, str) or not card_url.strip():
            raise ToolInputError("agent_card_url must be a non-empty string")
        task = tool_input.get("task")
        if not isinstance(task, str) or not task.strip():
            raise ToolInputError("task must be a non-empty string")
        timeout = tool_input.get("timeout", DEFAULT_TIMEOUT)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ToolInputError("timeout must be a number")
        timeout = float(timeout)
        if timeout <= 0:
            raise ToolInputError("timeout must be positive")
        timeout = min(timeout, MAX_TIMEOUT)

        try:
            card: AgentCard = fetch_agent_card(
                card_url, timeout=min(30.0, timeout)
            )
        except A2AError as exc:
            return ToolResult(
                name="DelegateAgent",
                output={"error": f"AgentCard error: {exc}"},
                is_error=True,
            )

        try:
            result = delegate_task(card, task, timeout=timeout)
        except A2AError as exc:
            return ToolResult(
                name="DelegateAgent",
                output={"error": f"delegation error: {exc}"},
                is_error=True,
            )
        except Exception as exc:  # never surface a raw traceback
            return ToolResult(
                name="DelegateAgent",
                output={"error": f"unexpected delegation error: {type(exc).__name__}: {exc}"},
                is_error=True,
            )

        return ToolResult(
            name="DelegateAgent",
            output={
                "agent_name": result.agent_name,
                "agent_url": result.agent_url,
                "state": result.state,
                "result_text": result.result_text,
                "artifacts": result.artifacts,
                "status_updates": result.status_updates,
            },
            is_error=result.state.lower() in _FAILED_STATES,
        )
