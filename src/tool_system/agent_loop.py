"""Agent loop for multi-turn tool calling."""

from __future__ import annotations

import hashlib
import json
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from .registry import ToolRegistry
from .delta_instructions import DeltaInstructionTracker
from .context import ToolContext
from ..agent.conversation import Conversation, TextContentBlock, ToolUseContentBlock
from ..outputStyles import resolve_output_style
from ..context_system.claude_md import snapshot_instruction_files
from ..providers.base import BaseProvider, ChatResponse
from ..providers.anthropic_provider import AnthropicProvider
from ..providers.minimax_provider import MinimaxProvider

# --- Tool-call loop detection (crush parity) ---------------------------------

# Sliding-window size: how many recent step signatures are remembered.
LOOP_WINDOW_SIZE = 10
# Repetitions of one signature that mark the model as stuck in a loop.
LOOP_THRESHOLD = 5
# Tool output is truncated to this many characters before signature hashing.
_LOOP_SIGNATURE_OUTPUT_LIMIT = 2000


def _canonical_json(value: Any) -> str:
    """Deterministic JSON for loop signatures (key order is irrelevant)."""
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
        )
    except Exception:
        return str(value)


def _truncated_output_text(output: Any) -> str:
    """String form of a tool output, truncated before signature hashing."""
    if isinstance(output, str):
        text = output
    else:
        text = _canonical_json(output)
    if len(text) > _LOOP_SIGNATURE_OUTPUT_LIMIT:
        text = text[:_LOOP_SIGNATURE_OUTPUT_LIMIT]
    return text


def tool_call_signature(tool_name: str, tool_input: Any, tool_output: Any) -> str:
    """SHA-256 signature of one tool step, crush-style.

    ``sha256(tool_name + "\\x00" + canonical_json(input) + "\\x00" + truncated_output)``.
    """
    payload = (
        tool_name
        + "\x00"
        + _canonical_json(tool_input)
        + "\x00"
        + _truncated_output_text(tool_output)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LoopDetector:
    """Sliding-window detector for repeated identical tool calls.

    Ports crush's loop detection (``internal/agent/loop_detection.go``): every
    completed tool step is hashed to a signature and the last
    ``LOOP_WINDOW_SIZE`` signatures are remembered. When one signature's count
    exceeds ``LOOP_THRESHOLD``, the model is stuck repeating itself.
    """

    def __init__(
        self,
        window_size: int = LOOP_WINDOW_SIZE,
        threshold: int = LOOP_THRESHOLD,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self._window: deque[tuple[str, str]] = deque(maxlen=window_size)

    def record(self, tool_name: str, tool_input: Any, tool_output: Any) -> int:
        """Record one completed tool call.

        Returns how many times this exact signature (tool name + input +
        output) now appears in the sliding window.
        """
        signature = tool_call_signature(tool_name, tool_input, tool_output)
        self._window.append((signature, tool_name))
        return sum(1 for sig, _ in self._window if sig == signature)

    def is_stuck(self, repetitions: int) -> bool:
        """True when a signature's repetition count exceeds the threshold."""
        return repetitions > self.threshold



def _is_anthropic_provider(provider: BaseProvider) -> bool:
    return isinstance(provider, (AnthropicProvider, MinimaxProvider))


def _build_openai_tool_result_content(result_output: Any) -> str:
    """Format tool result as string for OpenAI/GLM."""
    if isinstance(result_output, str):
        return result_output
    return json.dumps(result_output, ensure_ascii=False)

def summarize_tool_result(name: str, output: Any) -> str:
    """Create a concise, single-line summary for tool result output."""
    if not isinstance(output, dict):
        return str(output)
    if name.lower() == "write":
        path = output.get("filePath") or output.get("file_path")
        op = output.get("type")
        return f"{name} · {path} · {op}"
    if name.lower() == "edit":
        path = output.get("filePath") or output.get("file_path")
        replace_all = output.get("replaceAll")
        return f"{name} · {path} · replaceAll={replace_all}"
    if name.lower() == "read":
        if output.get("type") == "text" and isinstance(output.get("file"), dict):
            f = output["file"]
            path = f.get("filePath")
            num = f.get("numLines")
            total = f.get("totalLines")
            start = f.get("startLine")
            return f"{name} · {path} · lines={start}-{(start or 1) + (num or 0) - 1}/{total}"
        if output.get("type") == "file_unchanged" and isinstance(output.get("file"), dict):
            return f"{name} · {output['file'].get('filePath')} · unchanged"
        if output.get("type") in {"image", "pdf", "notebook"} and isinstance(output.get("file"), dict):
            return f"{name} · {output['file'].get('filePath')} · {output.get('type')}"
        return f"{name}"
    if name.lower() == "glob":
        n = output.get("numFiles")
        return f"{name} · matches={n}"
    if name.lower() == "grep":
        n = output.get("numFiles")
        mode = output.get("mode")
        return f"{name} · mode={mode} · files={n}"
    if name.lower() == "bash":
        code = output.get("exit_code")
        return f"{name} · exit={code}"
    if name.lower() == "webfetch":
        url = output.get("url")
        ct = output.get("content_type")
        return f"{name} · {url} · {ct}"
    if name.lower() == "websearch":
        q = output.get("query")
        results = output.get("results")
        n = len(results) if isinstance(results, list) else None
        return f"{name} · \"{q}\" · results={n}"
    if name.lower() == "config":
        op = output.get("operation")
        setting = output.get("setting")
        return f"{name} · {op} · {setting}"
    if name.lower() == "taskstop":
        tid = output.get("task_id")
        stopped = output.get("stopped")
        return f"{name} · {tid} · stopped={stopped}"
    if name.lower() == "sendusermessage":
        n = 0
        atts = output.get("attachments")
        if isinstance(atts, list):
            n = len(atts)
        return f"{name} · attachments={n}"
    # default: truncate dict keys for brevity
    keys = ", ".join(list(output.keys())[:3])
    return f"{name} · {keys}"


@dataclass(frozen=True)
class ToolEvent:
    kind: str
    tool_name: str
    tool_input: dict[str, Any] | None = None
    tool_output: Any | None = None
    tool_use_id: str | None = None
    is_error: bool = False
    error: str | None = None


class AgentLoopCancelled(Exception):
    """Raised when a caller requests the running agent loop to stop.

    Carries any text that was already streamed for the aborted turn so callers
    can persist or display the partial response.
    """

    def __init__(self, partial_text: str = ""):
        super().__init__("Agent loop cancelled")
        self.partial_text = partial_text


@dataclass(frozen=True)
class AgentLoopResult:
    """Result of running the agent loop."""
    response_text: str
    usage: dict[str, Any] | None = None  # {"input_tokens": int, "output_tokens": int}
    num_turns: int = 0
    cancelled: bool = False


ToolEventHandler = Callable[[ToolEvent], None]
TextChunkHandler = Callable[[str], None]
ActivityHandler = Callable[[str], None]


def _safe_call_handler(handler: ToolEventHandler | None, event: ToolEvent) -> None:
    if handler is None:
        return
    try:
        handler(event)
    except Exception:
        return


def _safe_call_activity_handler(handler: ActivityHandler | None, activity: str) -> None:
    if handler is None:
        return
    try:
        handler(activity)
    except Exception:
        return


def _stop_requested(should_stop: Callable[[], bool] | None) -> bool:
    """Check a caller-supplied stop hook without letting it break the loop."""
    if should_stop is None:
        return False
    try:
        return bool(should_stop())
    except Exception:
        return False


def _emit_text_chunks(handler: TextChunkHandler | None, text: str, *, chunk_size: int = 12) -> None:
    """Emit text in small chunks for user-visible streaming without changing loop semantics."""
    if handler is None or not text:
        return
    if chunk_size <= 0:
        chunk_size = len(text)
    for idx in range(0, len(text), chunk_size):
        try:
            handler(text[idx: idx + chunk_size])
        except Exception:
            return


def _call_provider_for_turn(
    *,
    provider: BaseProvider,
    api_messages: list[dict[str, Any]],
    call_kwargs: dict[str, Any],
    stream: bool,
    on_text_chunk: TextChunkHandler | None,
) -> tuple[Any, bool]:
    """Call the provider, preferring structured streaming when available.

    Returns (response, streamed_live_text).
    """
    if stream:
        try:
            response = provider.chat_stream_response(
                api_messages,
                on_text_chunk=on_text_chunk,
                **call_kwargs,
            )
            if not isinstance(response, ChatResponse):
                raise TypeError("Structured streaming must return ChatResponse")
            return response, True
        except NotImplementedError:
            pass
        except AgentLoopCancelled:
            # Cancellation must abort the turn, not fall back to a blocking call.
            raise
        except Exception:
            # Preserve existing stable behavior if streaming is unsupported or fails.
            pass

    response = provider.chat(api_messages, **call_kwargs)
    return response, False


# --- Provider-call retry ----------------------------------------------------

_MAX_PROVIDER_ATTEMPTS = 3
# Exponential backoff between attempts (~1s / 2s / 4s).
_PROVIDER_RETRY_DELAYS = (1.0, 2.0, 4.0)

# SDK-agnostic classification: openai/anthropic errors expose ``status_code``
# and httpx exposes it on ``response``; timeout/connection errors surface as
# builtins or SDK-specific subclasses. Matched by duck-typing so this works
# without importing optional SDK packages.
_TRANSIENT_EXCEPTION_NAMES = frozenset({
    "APITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout",
    "TimeoutException", "ConnectTimeoutError",
    "APIConnectionError", "ConnectionError", "ConnectError",
    "RateLimitError", "ServiceUnavailableError", "InternalServerError",
})


def _provider_error_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status extraction from provider SDK exceptions."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _is_transient_provider_error(exc: BaseException) -> bool:
    """True for failures worth retrying: 429, 5xx, timeouts, connection errors.

    4xx other than 429 are caller errors and are NEVER retried.
    """
    if isinstance(exc, AgentLoopCancelled):
        return False
    status = _provider_error_status(exc)
    if status is not None:
        return status == 429 or 500 <= status < 600
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in _TRANSIENT_EXCEPTION_NAMES


def _call_provider_with_retry(
    *,
    provider: BaseProvider,
    api_messages: list[dict[str, Any]],
    call_kwargs: dict[str, Any],
    stream: bool,
    on_text_chunk: TextChunkHandler | None,
) -> tuple[Any, bool]:
    """Call the provider with bounded retry on transient failures.

    Retries 429s, 5xx, timeouts, and connection errors up to
    ``_MAX_PROVIDER_ATTEMPTS`` total attempts with exponential backoff.
    Non-transient errors (other 4xx) fail immediately, as does cancellation.
    """
    last_exc: BaseException | None = None
    for attempt in range(1, _MAX_PROVIDER_ATTEMPTS + 1):
        try:
            return _call_provider_for_turn(
                provider=provider,
                api_messages=api_messages,
                call_kwargs=call_kwargs,
                stream=stream,
                on_text_chunk=on_text_chunk,
            )
        except AgentLoopCancelled:
            raise
        except Exception as exc:
            if not _is_transient_provider_error(exc) or attempt >= _MAX_PROVIDER_ATTEMPTS:
                raise
            last_exc = exc
            delay = _PROVIDER_RETRY_DELAYS[min(attempt - 1, len(_PROVIDER_RETRY_DELAYS) - 1)]
            time.sleep(delay)
    assert last_exc is not None  # unreachable: loop always raises or returns
    raise last_exc


def _clean_provider_error_message(exc: BaseException) -> str:
    """One-line, user-safe summary of a provider failure (no traceback)."""
    msg = str(exc).strip().splitlines()
    text = msg[0] if msg else type(exc).__name__
    return text if len(text) <= 200 else text[:197] + "..."


def _build_effective_system_prompt(
    style_prompt: str,
    tool_context: ToolContext,
    *,
    registered_tools: set[str] | None = None,
    script_mode: bool = False,
) -> str:
    """Assemble the full effective system prompt (turn-0 baseline).

    Delegates to DeltaInstructionTracker so there is exactly one assembler.
    """
    tracker = DeltaInstructionTracker(
        style_prompt, tool_context, registered_tools, script_mode
    )
    return tracker.full_prompt()



def summarize_tool_use(name: str, tool_input: dict[str, Any]) -> str:
    lowered = name.lower()
    if lowered == "bash":
        cmd = tool_input.get("command")
        if isinstance(cmd, str):
            s = cmd.strip().replace("\n", " ")
            return s if len(s) <= 80 else s[:77] + "..."
        return ""
    if lowered in {"read", "write", "edit"}:
        p = tool_input.get("file_path") or tool_input.get("filePath") or tool_input.get("path")
        if isinstance(p, str):
            extra = ""
            if lowered == "read":
                off = tool_input.get("offset")
                lim = tool_input.get("limit")
                if isinstance(off, int) or isinstance(lim, int):
                    start = off if isinstance(off, int) else 1
                    if isinstance(lim, int):
                        extra = f" · lines {start}-{start + lim - 1}"
            return f"{p}{extra}"
        return ""
    if lowered == "glob":
        pat = tool_input.get("pattern")
        base = tool_input.get("path")
        if isinstance(pat, str) and isinstance(base, str):
            return f"{pat} · {base}"
        if isinstance(pat, str):
            return pat
        return ""
    if lowered == "grep":
        pat = tool_input.get("pattern")
        base = tool_input.get("path")
        if isinstance(pat, str) and isinstance(base, str):
            return f"{pat} · {base}"
        if isinstance(pat, str):
            return pat
        return ""
    if lowered == "webfetch":
        url = tool_input.get("url")
        return url if isinstance(url, str) else ""
    if lowered == "websearch":
        q = tool_input.get("query")
        return q if isinstance(q, str) else ""
    if lowered == "toolsearch":
        q = tool_input.get("query")
        return q if isinstance(q, str) else ""
    if lowered == "askuserquestion":
        qs = tool_input.get("questions")
        if isinstance(qs, list):
            return f"{len(qs)} question(s)"
        return ""
    if lowered == "sendusermessage":
        status = tool_input.get("status")
        return status if isinstance(status, str) else ""
    return ""



def _add_cancelled_tool_results(
    *,
    conversation: Conversation,
    provider: BaseProvider,
    openai_messages: list[dict[str, Any]],
    pending_tool_uses: list[dict[str, Any]],
    note: str,
) -> None:
    """Close out tool calls that will never run so the next turn stays valid.

    Both message formats require every tool call to be answered; without this,
    stopping between the model reply and tool execution would make the follow-up
    API request fail.
    """
    for tool_use in pending_tool_uses:
        tool_id = tool_use.get("id") if isinstance(tool_use, dict) else None
        if not isinstance(tool_id, str):
            continue
        if _is_anthropic_provider(provider):
            conversation.add_tool_result_message(tool_id, note, is_error=True)
        else:
            openai_messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": note,
            })


def _format_max_turns_summary(
    turns_used: int,
    max_turns: int,
    tool_call_counts: dict,
    files_touched: list,
    last_tool_results: list,
) -> str:
    """Build a partial-state summary for max-turns exhaustion.

    Reports turns used, per-tool call counts, files touched, the last few
    tool results (truncated), and what succeeded -- so the user (or the next
    run) can see where the work got to instead of a bare marker.
    """
    lines = [f"[Max tool turns reached: {turns_used}/{max_turns}]", ""]
    lines.append(f"Turns used: {turns_used} of {max_turns}")
    if tool_call_counts:
        calls = ", ".join(
            f"{name} x{n}" for name, n in sorted(tool_call_counts.items())
        )
        total_calls = sum(tool_call_counts.values())
        lines.append(f"Tool calls ({total_calls}): {calls}")
    else:
        lines.append("Tool calls: none")
    if files_touched:
        shown = files_touched[:10]
        more = (
            f" (+{len(files_touched) - len(shown)} more)"
            if len(files_touched) > len(shown)
            else ""
        )
        lines.append(f"Files touched ({len(files_touched)}): " + ", ".join(shown) + more)
    else:
        lines.append("Files touched: none")
    if last_tool_results:
        succeeded = sum(1 for _, is_error, _ in last_tool_results if not is_error)
        lines.append(
            f"Last tool results ({succeeded}/{len(last_tool_results)} succeeded):"
        )
        for name, is_error, summary in last_tool_results[-5:]:
            status = "error" if is_error else "ok"
            short = summary if len(summary) <= 200 else summary[:197] + "..."
            lines.append(f"  - {name} ({status}): {short}")
    return "\n".join(lines)


def _provider_pricing_key(provider: Any) -> str:
    """Best-effort provider key for cost pricing (e.g. "anthropic")."""
    try:
        name = type(provider).__name__.removesuffix("Provider").lower()
        return {
            "anthropic": "anthropic",
            "openaicompatible": "openai",
            "openai": "openai",
            "glm": "glm",
            "minimax": "minimax",
        }.get(name, name or "default")
    except Exception:
        return "default"


def _record_turn_usage(cost_tracker: Any, provider: Any, usage: Any) -> None:
    """Record one turn's tokens on the cost tracker. Never raises.

    The loop must never break because of cost tracking, so every access
    here is defensive and all failures are swallowed.
    """
    if cost_tracker is None or not usage:
        return
    try:
        record_turn = getattr(cost_tracker, "record_turn", None)
        if not callable(record_turn):
            return
        record_turn(
            provider=_provider_pricing_key(provider),
            model=getattr(provider, "model", None),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cached_tokens=usage.get("cache_read_input_tokens", 0)
            or usage.get("cached_tokens", 0),
            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
        )
    except Exception:
        pass


def _flatten_to_openai_messages(conversation: Conversation) -> list[dict[str, Any]]:
    """Flatten a conversation to simple OpenAI-style {"role", "content"} dicts.

    Text blocks are joined; tool_use/tool_result blocks are dropped (auto-compact
    only runs between turns, when no tool calls are pending).
    """
    flattened: list[dict[str, Any]] = []
    try:
        for msg in conversation.get_messages():
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                if content:
                    flattened.append({"role": role, "content": content})
            elif isinstance(content, list):
                texts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                text = "\n".join(t for t in texts if t)
                if text:
                    flattened.append({"role": role, "content": text})
    except Exception:
        pass
    return flattened


def _estimate_context_tokens(
    conversation: Conversation,
    system_prompt: str | None,
    tool_schemas: list[dict[str, Any]] | None,
) -> int:
    """Rough token estimate for the next model call (messages + system + tools)."""
    try:
        from ..token_estimation import count_messages_tokens, count_tokens
    except Exception:
        return 0
    try:
        total = count_messages_tokens(conversation.get_messages())
        if system_prompt:
            total += count_tokens(system_prompt)
        if tool_schemas:
            total += count_tokens(json.dumps(tool_schemas, ensure_ascii=False))
        return total
    except Exception:
        return 0


def _fire_lifecycle_hook(event: str, extra: dict[str, Any] | None = None) -> dict:
    """Fire a lifecycle hook event. Never raises; always fails open.

    Returns the runner's result dict
    ``{"blocked": bool, "decision": str, "message": str | None}``.
    """
    try:
        from .hooks.runner import run_hooks

        return run_hooks(event, extra=extra)
    except Exception:
        return {"blocked": False, "decision": "allow", "message": None}


def _maybe_auto_compact(
    conversation: Conversation,
    provider: BaseProvider,
    tool_schemas: list[dict[str, Any]] | None,
    system_prompt: str | None,
    tool_context: ToolContext,
    on_activity: ActivityHandler | None = None,
) -> bool:
    """Compact the conversation when it exceeds ~90% of the model limit.

    Returns True when a compaction was performed (the caller must refresh any
    provider-specific message lists). Never raises: any failure returns False
    and the loop proceeds normally.
    """
    try:
        from ..compact_service.service import compact_conversation_sync
        from ..context_limits import should_auto_compact

        model = getattr(provider, "model", "") or ""
        estimated = _estimate_context_tokens(conversation, system_prompt, tool_schemas)
        if not should_auto_compact(estimated, model):
            return False

        # PreCompress hook (brief 008): a blocking hook vetoes this
        # compaction; the loop continues with the uncompacted conversation.
        pre_compress = _fire_lifecycle_hook(
            "PreCompress",
            {
                "session_id": getattr(tool_context, "session_id", None),
                "model": model,
                "estimated_tokens": estimated,
                "trigger": "auto",
            },
        )
        if pre_compress.get("blocked"):
            _safe_call_activity_handler(on_activity, "compaction skipped by hook")
            return False

        if on_activity is not None:
            try:
                on_activity("auto-compacting context")
            except Exception:
                pass

        workspace_root = getattr(tool_context, "workspace_root", None)
        compact_conversation_sync(
            conversation,
            provider,
            model,
            trigger="auto",
            workspace_root=workspace_root,
        )
        return True
    except Exception:
        return False


def _run_agent_loop_impl(
    conversation: Conversation,
    provider: BaseProvider,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    max_turns: int = 50,
    stream: bool = False,
    verbose: bool = False,
    on_event: ToolEventHandler | None = None,
    on_text_chunk: TextChunkHandler | None = None,
    on_activity: ActivityHandler | None = None,
    should_stop: Callable[[], bool] | None = None,
    cost_tracker: Any | None = None,
    script_mode: bool = False,
    auto_compact: bool = True,
) -> AgentLoopResult:
    """Run agent loop: LLM -> tools -> LLM until no more tools or max turns.

    Args:
        conversation: Conversation with initial user message
        provider: LLM provider
        tool_registry: Tool registry to use
        tool_context: Tool context
        max_turns: Maximum tool turns before stopping
        stream: Whether to stream responses
        verbose: Whether to print tool calls/results
        on_event: Optional callback for tool events
        on_text_chunk: Optional callback for incremental user-visible text chunks
        on_activity: Optional callback when requesting the next model response
        should_stop: Optional callback returning True to stop the loop at the
            next safe point (between turns, between tools, or mid-stream)
            cost_tracker: Optional cost tracker receiving per-turn
            input/output/cached tokens with provider+model for pricing.
            Recording never raises and never affects the loop.
        auto_compact: When True, automatically compact the conversation
            (summarize + re-inject survival context) before a model call if
            the estimated context exceeds ~90% of the model's context limit.
            The loop then continues with the compacted conversation.
            Disable with --no-auto-compact or the "auto_compact" config option.
        script_mode: When True, render the script-mode guidance line (permission
            prompts auto-deny; prefer read-only reconnaissance) -- set by the
            non-interactive `clawd -p` path.

    Returns:
        AgentLoopResult with final text response, usage info, and turn count.
        ``cancelled`` is True when the loop stopped early via ``should_stop``.
    """

    # Expose the provider on the context so tools that spawn child agent
    # loops (Subagent) can reach it; it rides along on sub-contexts via
    # dataclasses.replace.
    tool_context.provider = provider
    # Convert tools to schemas (Anthropic format)
    tool_schemas = []
    for spec in tool_registry.list_specs():
        tool_schemas.append({
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        })

    # For OpenAI/GLM, keep separate message list in OpenAI format
    openai_messages: list[dict[str, Any]] = []
    last_user_visible_message: str | None = None
    style_name = getattr(tool_context, "output_style_name", None)
    style_dir = getattr(tool_context, "output_style_dir", None)
    style_prompt = resolve_output_style(style_name, style_dir).prompt
    # Registered tool names drive the conditional per-tool guidance lines
    # (opencode's ${OPENCODE_TOOL_GUIDANCE} pattern).
    registered_tool_names = {spec.name for spec in tool_registry.list_specs()}
    # Delta instruction system (opencode parity): turn 0 sends the full
    # prompt; later turns send the static baseline plus only changed
    # dynamic sections (context, skills).
    instruction_tracker = DeltaInstructionTracker(
        style_prompt,
        tool_context,
        registered_tool_names,
        script_mode,
    )
    full_system_prompt = instruction_tracker.full_prompt()

    # Seed OpenAI messages from initial conversation messages
    for msg in conversation.messages:
        if isinstance(msg.content, str):
            openai_messages.append({"role": msg.role, "content": msg.content})
        else:
            # If there are already block messages, we are probably Anthropic; leave as is
            pass

    # Track usage across all turns
    total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}
    turn_count = 0
    streamed_text_parts: list[str] = []
    # Partial-state tracking for the max-turns summary
    tool_call_counts: dict[str, int] = {}
    files_touched: list[str] = []
    last_tool_results: list[tuple[str, bool, str]] = []  # (name, is_error, summary)
    # Tool-call loop detection: sliding window of recent step signatures.
    loop_detector = LoopDetector()
    # Instruction-file live reload (brief 012): record mtimes + hashes of
    # the loaded instruction files (AGENTS.md/CLAUDE.md) at session start.
    # Each turn below does a cheap stat check; on change a transient
    # system-reminder is injected so the model stops following stale rules.
    # Fail-open: the watcher must never break the loop.
    try:
        instruction_watcher = snapshot_instruction_files(
            getattr(tool_context, "workspace_root", "."),
            cwd=getattr(tool_context, "cwd", None),
        )
    except Exception:
        instruction_watcher = None


    def _loop_break_result(tool_name: str, repetitions: int) -> AgentLoopResult:
        """Build the result returned when a tool-call loop is detected."""
        return AgentLoopResult(
            response_text=(
                f"[Loop detected: {tool_name} repeated {repetitions}x "
                "with identical input \u2014 stopping. Try a different approach.]"
            ),
            usage=total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None,
            num_turns=turn_count,
        )

    def _cancelled_result() -> AgentLoopResult:
        return AgentLoopResult(
            response_text="".join(streamed_text_parts),
            usage=total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None,
            num_turns=turn_count,
            cancelled=True,
        )

    def guarded_text_chunk(text: str) -> None:
        """Forward streamed text and abort mid-stream once a stop is requested."""
        if text:
            streamed_text_parts.append(text)
        if on_text_chunk is not None:
            on_text_chunk(text)
        if _stop_requested(should_stop):
            raise AgentLoopCancelled("".join(streamed_text_parts))

    for turn in range(max_turns):
        if _stop_requested(should_stop):
            return _cancelled_result()
        # Auto-compact before the model call when the context is nearly
        # exhausted. The session survives: compaction summarizes the history
        # and re-injects survival context, then the loop continues.
        if auto_compact and _maybe_auto_compact(
            conversation, provider, tool_schemas, full_system_prompt,
            tool_context, on_activity,
        ):
            if not _is_anthropic_provider(provider):
                # openai_messages is a separate list; reseed it from the
                # compacted conversation so providers stay in sync.
                openai_messages = _flatten_to_openai_messages(conversation)
        if _is_anthropic_provider(provider):
            api_messages = conversation.get_messages()
        else:
            # Use OpenAI formatted messages for non-Anthropic
            api_messages = openai_messages

        call_kwargs: dict[str, Any] = {"tools": tool_schemas}
        if _is_anthropic_provider(provider):
            if turn == 0:
                call_kwargs["system"] = full_system_prompt
            else:
                # Delta: static baseline + only the changed sections.
                turn_system, _delta_report = instruction_tracker.next_system(turn)
                call_kwargs["system"] = turn_system
        else:
            if turn == 0:
                # Keep the system message in the persistent history so later
                # turns (and the delta appends below) build on it.
                openai_messages = [{"role": "system", "content": full_system_prompt}, *openai_messages]
                api_messages = openai_messages
            else:
                # Delta: the turn-0 system message is already in history, so
                # append only the changed-sections block when something changed.
                turn_system, _delta_report = instruction_tracker.next_system(turn)
                delta_block = turn_system[len(instruction_tracker.baseline()):].strip()
                if delta_block:
                    api_messages.append({"role": "system", "content": delta_block})
        # Instruction-file live reload (brief 012): cheap per-turn stat
        # check. Reminders are appended to a per-turn copy of the message
        # list so the notification is transient and adds zero tokens when
        # nothing changed. Fail-open: never break the turn on watcher error.
        instruction_reminders: list[str] = []
        if instruction_watcher is not None:
            try:
                instruction_reminders = instruction_watcher.check_for_changes()
            except Exception:
                instruction_reminders = []
        turn_messages = api_messages
        if instruction_reminders:
            turn_messages = [
                *api_messages,
                *({"role": "user", "content": text} for text in instruction_reminders),
            ]
        _safe_call_activity_handler(on_activity, "thinking")
        # BeforeModel hook (brief 008): a blocking hook halts the turn;
        # the hook's message becomes the turn's response text.
        before_model = _fire_lifecycle_hook(
            "BeforeModel",
            {
                "session_id": getattr(tool_context, "session_id", None),
                "turn": turn,
                "model": getattr(provider, "model", None),
                "message_count": len(api_messages),
                "messages": turn_messages,
            },
        )
        if before_model.get("blocked"):
            return AgentLoopResult(
                response_text=(
                    before_model.get("message") or "[Stopped by BeforeModel hook]"
                ),
                usage=total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None,
                num_turns=turn_count,
            )
        try:
            response, streamed_live_text = _call_provider_with_retry(
                provider=provider,
                api_messages=turn_messages,
                call_kwargs=call_kwargs,
                stream=stream,
                on_text_chunk=guarded_text_chunk,
            )
        except AgentLoopCancelled:
            # Keep the partial reply so the conversation history stays coherent.
            partial_text = "".join(streamed_text_parts)
            if partial_text.strip():
                conversation.add_assistant_message(partial_text)
            return _cancelled_result()
        except Exception as exc:
            # Final provider failure: surface a clean, one-line error to the
            # user — no raw traceback in chat output.
            return AgentLoopResult(
                response_text=(
                    f"[Provider error after {_MAX_PROVIDER_ATTEMPTS} attempts: "
                    f"{_clean_provider_error_message(exc)}]"
                ),
                usage=total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None,
                num_turns=turn_count,
            )
        turn_count += 1

        # Collect usage info
        if response.usage:
            total_usage["input_tokens"] += response.usage.get("input_tokens", 0)
            total_usage["output_tokens"] += response.usage.get("output_tokens", 0)
            total_usage["cached_tokens"] += (
                response.usage.get("cache_read_input_tokens", 0)
                or response.usage.get("cached_tokens", 0)
            )
            # Cost tracking must never break the loop.
            _record_turn_usage(cost_tracker, provider, response.usage)

        # AfterModel hook (brief 008): informational only; cannot block.
        _fire_lifecycle_hook(
            "AfterModel",
            {
                "session_id": getattr(tool_context, "session_id", None),
                "turn": turn,
                "model": getattr(provider, "model", None),
                "content": response.content,
                "tool_uses": response.tool_uses,
                "usage": response.usage,
            },
        )

        # Build assistant content for Anthropic or just text for OpenAI
        final_assistant_content = response.content or ""

        if _is_anthropic_provider(provider):
            assistant_blocks: list = []
            if response.content:
                assistant_blocks.append(TextContentBlock(type="text", text=response.content))

            tool_uses = response.tool_uses or []
            for tool_use in tool_uses:
                assistant_blocks.append(ToolUseContentBlock(
                    type="tool_use",
                    id=tool_use["id"],
                    name=tool_use["name"],
                    input=tool_use["input"],
                ))

            conversation.add_assistant_message(assistant_blocks if assistant_blocks else "")
        else:
            # Persist assistant text for session history features like /render-last
            # and for subsequent non-Anthropic turns seeded from conversation.
            conversation.add_assistant_message(final_assistant_content)
            # Add assistant message to OpenAI messages (text only)
            openai_assistant_msg: dict[str, Any] = {"role": "assistant", "content": final_assistant_content}
            # If there are tool_uses, add them in OpenAI format
            if response.tool_uses:
                # Build OpenAI tool_calls
                tool_calls = []
                for tu in response.tool_uses:
                    tool_calls.append({
                        "id": tu["id"],
                        "type": "function",
                        "function": {
                            "name": tu["name"],
                            "arguments": json.dumps(tu["input"], ensure_ascii=False)
                        }
                    })
                openai_assistant_msg["tool_calls"] = tool_calls
            openai_messages.append(openai_assistant_msg)

        tool_uses = response.tool_uses or []

        if not tool_uses:
            # No more tools, done
            if stream and final_assistant_content and not streamed_live_text:
                _emit_text_chunks(guarded_text_chunk, final_assistant_content)
            if (final_assistant_content or "").strip() == "" and last_user_visible_message is not None:
                return AgentLoopResult(
                    response_text=last_user_visible_message,
                    usage=total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None,
                    num_turns=turn_count,
                )
            return AgentLoopResult(
                response_text=final_assistant_content,
                usage=total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None,
                num_turns=turn_count,
            )

        send_user_message: str | None = None

        # Call each tool
        for tool_index, tool_use in enumerate(tool_uses):
            if _stop_requested(should_stop):
                _add_cancelled_tool_results(
                    conversation=conversation,
                    provider=provider,
                    openai_messages=openai_messages,
                    pending_tool_uses=tool_uses[tool_index:],
                    note="Stopped by the user before this tool ran.",
                )
                return _cancelled_result()
            tool_id = tool_use["id"]
            tool_name = tool_use["name"]
            tool_input = tool_use["input"]

            try:
                _safe_call_handler(
                    on_event,
                    ToolEvent(
                        kind="tool_use",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_use_id=tool_id,
                    ),
                )
                # Use dispatch to get proper validation and result wrapping
                from ..tool_system.protocol import ToolCall
                call = ToolCall(name=tool_name, input=tool_input, tool_use_id=tool_id)
                result = tool_registry.dispatch(call, tool_context)
                result_output = result.output
                # Partial-state tracking for the max-turns summary
                tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
                _touched = tool_input.get("file_path") if isinstance(tool_input, dict) else None
                if isinstance(_touched, str) and _touched not in files_touched:
                    files_touched.append(_touched)
                last_tool_results.append(
                    (tool_name, result.is_error, summarize_tool_result(tool_name, result_output))
                )
                if len(last_tool_results) > 5:
                    del last_tool_results[0]
                if tool_name.lower() == "sendusermessage" and isinstance(result_output, dict):
                    msg = result_output.get("message")
                    if isinstance(msg, str):
                        last_user_visible_message = msg
                        send_user_message = msg
                if tool_name.lower() == "structuredoutput" and isinstance(result_output, dict):
                    payload = result_output.get("structured_output")
                    try:
                        last_user_visible_message = json.dumps(payload, ensure_ascii=False, indent=2)
                    except Exception:
                        last_user_visible_message = str(payload)

                if verbose:
                    use_summary = summarize_tool_use(tool_name, tool_input)
                    if use_summary:
                        print(f"{tool_name} · {use_summary}")
                    summary = summarize_tool_result(tool_name, result_output)
                    print(f"{summary}")

                _safe_call_handler(
                    on_event,
                    ToolEvent(
                        kind="tool_result",
                        tool_name=tool_name,
                        tool_output=result_output,
                        tool_use_id=tool_id,
                        is_error=result.is_error,
                    ),
                )
                if _is_anthropic_provider(provider):
                    conversation.add_tool_result_message(tool_id, result_output)
                else:
                    # Add tool result in OpenAI format
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": _build_openai_tool_result_content(result_output)
                    })
                # Tool-call loop detection (crush parity): an identical
                # tool name + input + output repeated too often means the
                # model is stuck -- break out instead of burning turns.
                repetitions = loop_detector.record(tool_name, tool_input, result_output)
                if loop_detector.is_stuck(repetitions):
                    return _loop_break_result(tool_name, repetitions)
            except Exception as e:
                error_str = f"Error: {e}"
                if verbose:
                    print(f"[Tool Error] {error_str}")
                _safe_call_handler(
                    on_event,
                    ToolEvent(
                        kind="tool_error",
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_use_id=tool_id,
                        is_error=True,
                        error=error_str,
                    ),
                )
                if _is_anthropic_provider(provider):
                    conversation.add_tool_result_message(tool_id, error_str, is_error=True)
                else:
                    openai_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": error_str
                    })
                repetitions = loop_detector.record(tool_name, tool_input, error_str)
                if loop_detector.is_stuck(repetitions):
                    return _loop_break_result(tool_name, repetitions)

        if send_user_message is not None:
            conversation.add_assistant_message(send_user_message)
            _emit_text_chunks(guarded_text_chunk, send_user_message)
            return AgentLoopResult(
                response_text=send_user_message,
                usage=total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None,
                num_turns=turn_count,
            )

    # Reached max turns: report partial state instead of a bare marker.
    return AgentLoopResult(
        response_text=_format_max_turns_summary(
            turn_count,
            max_turns,
            tool_call_counts,
            files_touched,
            last_tool_results,
        ),
        usage=total_usage if total_usage["input_tokens"] > 0 or total_usage["output_tokens"] > 0 else None,
        num_turns=turn_count,
    )


def run_agent_loop(
    conversation: Conversation,
    provider: BaseProvider,
    tool_registry: ToolRegistry,
    tool_context: ToolContext,
    max_turns: int = 50,
    stream: bool = False,
    verbose: bool = False,
    on_event: ToolEventHandler | None = None,
    on_text_chunk: TextChunkHandler | None = None,
    on_activity: ActivityHandler | None = None,
    should_stop: Callable[[], bool] | None = None,
    cost_tracker: Any | None = None,
    script_mode: bool = False,
    auto_compact: bool = True,
) -> AgentLoopResult:
    """Run agent loop: LLM -> tools -> LLM until no more tools or max turns.

    Lifecycle hooks (brief 008): fires ``BeforeAgent`` once before the
    first turn and ``AfterAgent`` once when the run finishes. A blocking
    ``BeforeAgent`` hook halts the run before any model call; its message
    becomes the response text. ``BeforeModel``/``AfterModel`` fire around
    every model call and ``PreCompress`` fires before auto-compaction --
    see :func:`_run_agent_loop_impl` for the full argument docs.
    """
    session_id = getattr(tool_context, "session_id", None)
    before = _fire_lifecycle_hook(
        "BeforeAgent",
        {
            "session_id": session_id,
            "model": getattr(provider, "model", None),
            "max_turns": max_turns,
        },
    )
    if before.get("blocked"):
        return AgentLoopResult(
            response_text=before.get("message") or "[Stopped by BeforeAgent hook]",
            usage=None,
            num_turns=0,
        )
    try:
        result = _run_agent_loop_impl(
            conversation,
            provider,
            tool_registry,
            tool_context,
            max_turns=max_turns,
            stream=stream,
            verbose=verbose,
            on_event=on_event,
            on_text_chunk=on_text_chunk,
            on_activity=on_activity,
            should_stop=should_stop,
            cost_tracker=cost_tracker,
            script_mode=script_mode,
            auto_compact=auto_compact,
        )
    except Exception as exc:
        _fire_lifecycle_hook(
            "AfterAgent",
            {"session_id": session_id, "error": _clean_provider_error_message(exc)},
        )
        raise
    _fire_lifecycle_hook(
        "AfterAgent",
        {
            "session_id": session_id,
            "response_text": result.response_text,
            "num_turns": result.num_turns,
            "usage": result.usage,
            "cancelled": result.cancelled,
        },
    )
    return result
