from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass(frozen=True)
class ToolCall:
    name: str
    input: dict[str, Any]
    tool_use_id: Optional[str] = None


@dataclass(frozen=True)
class ToolResult:
    name: str
    output: Any
    is_error: bool = False
    tool_use_id: Optional[str] = None
    content_type: Literal["text", "json"] = "json"

# ---------------------------------------------------------------------------
# Tool-result context budget (Q9).
#
# Tool results are the most common way a single turn eats a whole context
# window (one big Read/Grep/Bash dump). Every tool result is capped at
# ``max_result_size_chars`` (ToolSpec; per-tool overrides still win) and
# anything over the cap is cut with a visible marker, so the agent sees
# that output was lost instead of silently missing it.
# ---------------------------------------------------------------------------

DEFAULT_MAX_RESULT_SIZE_CHARS: int = 30_000


def format_truncation_marker(kept: int, total: int) -> str:
    """Marker appended when a tool result is cut to fit the context budget."""
    return f"\n\n[truncated \u2014 showing {kept} of {total} chars]"


def truncate_text(text: str, max_chars: int = DEFAULT_MAX_RESULT_SIZE_CHARS) -> str:
    """Cut *text* to *max_chars* total, reserving room for the marker.

    Returns *text* unchanged when it already fits. Otherwise keeps the
    first ``max_chars - len(marker)`` chars and appends
    ``[truncated \u2014 showing N of M chars]``, so the result never exceeds
    *max_chars* and N is the exact number of chars shown.
    """
    if max_chars < 0:
        raise ValueError("max_chars must be >= 0")
    if len(text) <= max_chars:
        return text
    total = len(text)
    marker = format_truncation_marker(max_chars, total)  # upper-bound length
    kept = max_chars - len(marker)
    if kept <= 0:
        return format_truncation_marker(0, total)[-max_chars:] if max_chars else ""
    marker = format_truncation_marker(kept, total)
    return text[:kept] + marker
