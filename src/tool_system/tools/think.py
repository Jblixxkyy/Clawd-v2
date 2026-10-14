from __future__ import annotations

from typing import Any, Optional

from ...transcript import ThinkEvent
from ..context import ToolContext
from ..errors import ToolInputError
from ..protocol import ToolResult
from ..registry import ToolSpec

__all__ = ["ThinkTool", "ThinkEvent", "context_think_events"]

#: Subagent coordination flag (brief 023, "optionally").
#:
#: Subagent profiles (005/024) may require a ``think`` call before the first
#: mutation tool call. That requirement lives in the profile, NOT in this
#: tool — this flag is the shared coordination point so profiles don't
#: hard-code the tool name. It defaults to False (suggest, don't mandate).
#: Individual profiles flip their own copy of the requirement; the tool
#: stamps ``context.think_completed = True`` on every successful call so a
#: profile can check for it without importing subagent internals.
REQUIRE_THINK_BEFORE_FIRST_MUTATION = False


def context_think_events(context: ToolContext) -> list[ThinkEvent]:
    """Defensive accessor for the think events recorded on a context.

    The ``Think`` tool appends a :class:`ThinkEvent` to this list on every
    successful call. Returns the (mutable) list, creating it on first use.
    The 018 judge loop can read this without touching the tool itself.
    """
    log = getattr(context, "think_events", None)
    if not isinstance(log, list):
        log = []
        setattr(context, "think_events", log)
    return log


class ThinkTool:
    """Explicit reasoning channel (brief 023).

    Calling it forces a pause: the reasoning lands in the transcript as a
    real, inspectable, replayable :class:`ThinkEvent` instead of invisible
    chain-of-thought. No side effects — no files, no processes, no network,
    no world-state changes of any kind.
    """

    #: Class-level copy of the coordination flag (see module constant).
    require_before_first_mutation: bool = REQUIRE_THINK_BEFORE_FIRST_MUTATION

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Think",
            description=(
                "Pause and record your reasoning as an explicit, inspectable "
                "transcript event. Use this when you need to think through "
                "something before acting. Consider calling it before "
                "high-stakes actions — multi-file edits, finishing a task, or "
                "answering an ambiguous question — but it is a suggestion, not "
                "a requirement. No side effects."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "The reasoning to record.",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Optional short label for the thought.",
                    },
                },
                "required": ["thought"],
            },
            is_read_only=True,
            max_result_size_chars=2000,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        thought = tool_input.get("thought")
        topic: Optional[str] = tool_input.get("topic")
        if not isinstance(thought, str) or not thought.strip():
            raise ToolInputError("thought must be a non-empty string")
        if topic is not None and not isinstance(topic, str):
            raise ToolInputError("topic must be a string when provided")

        event = ThinkEvent(thought=thought, topic=topic)

        # Record on the context: the stable, judge-readable think log.
        context_think_events(context).append(event)
        # Stamp for subagent-profile coordination ("think before first mutation").
        setattr(context, "think_completed", True)

        # Also land in the transcript when the caller attached one to the
        # context (query_engine pattern: context.transcript_store).
        transcript = getattr(context, "transcript_store", None)
        append_event = getattr(transcript, "append_event", None)
        if callable(append_event):
            append_event(event)

        return ToolResult(
            name="Think",
            output={"recorded": True, "topic": topic},
        )
