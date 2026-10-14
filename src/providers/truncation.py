"""Truncation recovery (brief 025): continue responses killed at output-token limits.

Mechanism (from opencode v2.0.28): detect the per-provider truncation signal,
re-prompt the model to continue, and concatenate the parts in order.

Provider truncation signals, normalized to one internal flag
(``ChatResponse.truncated``):
  - OpenAI / OpenAI-compatible (OpenAI, GLM/Zhipu, NVIDIA NIM): ``finish_reason == "length"``
  - Anthropic (Anthropic, MiniMax): ``stop_reason == "max_tokens"``
  - Gemini: ``finishReason == "MAX_TOKENS"`` (normalized to ``"max_tokens"``)

Recovery ordering: ``maybe_continue_truncated`` must run *before* any
tool-call JSON parse-failure handling, so a tool call cut off mid-JSON is
completed first. This composes with 007 input repair (repair handles malformed
input; this prevents the malformation).

The returned log lines always include a ``continued after truncation`` marker
when parts were stitched, and a ``STILL TRUNCATED`` marker when the limit was
hit, so the agent loop and the UI can surface both to the user.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from .base import BaseProvider, ChatMessage, ChatResponse, MessageInput

#: Config key under which recovery settings live in the harness config dict.
CONFIG_KEY = "truncation_recovery"

#: Default config values: enabled with up to 3 continuation turns.
DEFAULTS: dict[str, Any] = {"enabled": True, "max_continuations": 3}

#: Normalized finish reasons (lowercased) that mean "cut off by output limit".
#: Covers OpenAI "length", Anthropic "max_tokens", Gemini "MAX_TOKENS".
TRUNCATED_FINISH_REASONS = frozenset({"length", "max_tokens"})

#: What we ask the model when a response is truncated.
CONTINUATION_PROMPT = (
    "Continue exactly where you left off — do not repeat what you already wrote."
)

#: Log/marker prefix surfaced by the agent loop and the UI.
LOG_PREFIX = "[truncation_recovery]"

#: Human-readable marker shown when a response was stitched.
CONTINUED_MARKER = "continued after truncation"

#: Human-readable marker shown when continuations were exhausted.
STILL_TRUNCATED_MARKER = "STILL TRUNCATED after continuations"


def is_truncated_finish_reason(finish_reason: Any) -> bool:
    """Normalize a provider finish/stop reason and report truncation.

    Case-insensitive; ``None``/empty values are not truncation.
    """
    if not finish_reason:
        return False
    return str(finish_reason).strip().lower() in TRUNCATED_FINISH_REASONS


def mark_truncated(response: ChatResponse) -> ChatResponse:
    """Set ``response.truncated`` from its finish_reason. Returns the response."""
    response.truncated = is_truncated_finish_reason(response.finish_reason)
    return response


def _recovery_config(config: Optional[dict[str, Any]]) -> tuple[bool, int]:
    """Extract (enabled, max_continuations) from a harness config dict."""
    section = (config or {}).get(CONFIG_KEY, {}) or {}
    enabled = section.get("enabled", DEFAULTS["enabled"])
    try:
        max_continuations = int(section.get("max_continuations", DEFAULTS["max_continuations"]))
    except (TypeError, ValueError):
        max_continuations = DEFAULTS["max_continuations"]
    return bool(enabled), max(max_continuations, 0)


def _merge_usage(*usage_dicts: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Sum numeric usage fields across continuation turns."""
    merged: dict[str, Any] = {}
    for usage in usage_dicts:
        for key, value in (usage or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = merged.get(key, 0) + value
            elif key not in merged:
                merged[key] = value
    return merged


def _merge_tool_uses(responses: Sequence[ChatResponse]) -> Optional[list[dict[str, Any]]]:
    """Merge tool_uses across continuation turns, last wins per id.

    A truncated mid-tool-call turn may carry a partial/empty input; the
    continuation turn carries the complete version under the same id, which
    is why last-wins (not first-wins) is correct here.
    """
    merged: dict[Any, dict[str, Any]] = {}
    order: list[Any] = []
    for response in responses:
        for tool_use in response.tool_uses or []:
            tid = tool_use.get("id", tool_use.get("name"))
            if tid not in merged:
                order.append(tid)
            merged[tid] = tool_use
    return [merged[tid] for tid in order] or None


def _message_dicts(messages: Sequence[MessageInput]) -> list[dict[str, Any]]:
    """Normalize messages to dicts without mutating the caller's list."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, ChatMessage):
            out.append(msg.to_dict())
        elif isinstance(msg, dict):
            out.append(dict(msg))
        else:
            content = getattr(msg, "content", None)
            role = getattr(msg, "role", "user")
            out.append({"role": role, "content": content})
    return out


def maybe_continue_truncated(
    response: ChatResponse,
    provider: BaseProvider,
    messages: Sequence[MessageInput],
    *,
    tools: Optional[list[dict[str, Any]]] = None,
    config: Optional[dict[str, Any]] = None,
    call_kwargs: Optional[dict[str, Any]] = None,
    on_event: Optional[Callable[[str], None]] = None,
) -> tuple[ChatResponse, list[str]]:
    """Continue a truncated response and stitch the parts together.

    Args:
        response: The initial provider response (must have ``truncated`` set,
            e.g. by ``mark_truncated`` or the provider adapter).
        provider: Provider to issue continuation turns against.
        messages: The original turn messages; continuation turns are appended
            to a copy (assistant partial + "continue" user message).
        tools: Tool schemas to pass to continuation calls (same as original).
        config: Harness config dict; reads ``truncation_recovery`` section
            (``enabled``, ``max_continuations``; defaults: True, 3).
        call_kwargs: Extra kwargs forwarded to ``provider.chat`` (e.g. model).
        on_event: Optional callback receiving each log line as it happens,
            so the agent loop can stream them to the transcript/UI live.

    Returns:
        (stitched_response, log_lines). ``stitched_response.continuations``
        counts the continuation calls actually made; ``truncated`` reflects
        the final part. When nothing was truncated or recovery is disabled,
        the original response is returned unchanged with an empty log.
    """
    log: list[str] = []

    def emit(line: str) -> None:
        log.append(line)
        if on_event is not None:
            try:
                on_event(line)
            except Exception:
                pass

    if not response.truncated:
        return response, log

    enabled, max_continuations = _recovery_config(config)
    if not enabled:
        emit(f"{LOG_PREFIX} truncated response detected but recovery is disabled")
        return response, log

    emit(
        f"{LOG_PREFIX} truncated response detected "
        f"(finish_reason={response.finish_reason!r}); attempting up to "
        f"{max_continuations} continuation(s)"
    )

    parts: list[ChatResponse] = [response]
    history = _message_dicts(messages)
    continuation_kwargs = dict(call_kwargs or {})
    continuations_made = 0

    for attempt in range(1, max_continuations + 1):
        so_far = "".join(part.content or "" for part in parts)
        history = history + [
            {"role": "assistant", "content": so_far},
            {"role": "user", "content": CONTINUATION_PROMPT},
        ]
        try:
            continuation = provider.chat(history, tools=tools, **continuation_kwargs)
        except Exception as exc:
            emit(f"{LOG_PREFIX} continuation {attempt} failed: {exc}; keeping partial")
            break
        if not isinstance(continuation, ChatResponse):
            emit(f"{LOG_PREFIX} continuation {attempt} returned non-ChatResponse; stopping")
            break
        mark_truncated(continuation)
        parts.append(continuation)
        continuations_made += 1
        emit(
            f"{LOG_PREFIX} continuation {attempt}/{max_continuations} done "
            f"(finish_reason={continuation.finish_reason!r}, "
            f"+{len(continuation.content or '')} chars)"
        )
        if not continuation.truncated:
            break

    stitched = ChatResponse(
        content="".join(part.content or "" for part in parts),
        model=parts[0].model,
        usage=_merge_usage(*(part.usage for part in parts)),
        finish_reason=parts[-1].finish_reason,
        reasoning_content="".join(
            part.reasoning_content or "" for part in parts
        ) or None,
        tool_uses=_merge_tool_uses(parts),
        truncated=parts[-1].truncated,
        continuations=continuations_made,
    )

    if continuations_made and not stitched.truncated:
        emit(
            f"{LOG_PREFIX} {CONTINUED_MARKER}: stitched "
            f"{len(parts)} parts ({len(stitched.content)} chars total)"
        )
    elif stitched.truncated:
        emit(
            f"{LOG_PREFIX} {STILL_TRUNCATED_MARKER} "
            f"({continuations_made} attempt(s)); reporting partial result"
        )
    return stitched, log


__all__ = [
    "CONFIG_KEY",
    "DEFAULTS",
    "TRUNCATED_FINISH_REASONS",
    "CONTINUATION_PROMPT",
    "LOG_PREFIX",
    "CONTINUED_MARKER",
    "STILL_TRUNCATED_MARKER",
    "is_truncated_finish_reason",
    "mark_truncated",
    "maybe_continue_truncated",
]
