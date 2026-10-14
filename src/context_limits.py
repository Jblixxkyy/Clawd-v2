"""Per-model context window limits for auto-compaction decisions.

Maps model name substrings to their context window sizes (in tokens).
The agent loop uses these to trigger automatic compaction *before* the
context window is exhausted, so long sessions degrade gracefully instead
of dying at the limit.

Values are conservative public figures for each model family; unknown
models fall back to DEFAULT_CONTEXT_LIMIT.
"""

from __future__ import annotations

#: Fraction of the context limit at which auto-compaction triggers.
AUTO_COMPACT_FRACTION = 0.90

#: Fallback limit when the model is unknown or unnamed.
DEFAULT_CONTEXT_LIMIT = 128_000

# Ordered (substring, limit) pairs — first case-insensitive match wins.
# Order matters: check the most specific names first.
_MODEL_CONTEXT_LIMITS: tuple[tuple[str, int], ...] = (
    ("gpt-5", 400_000),
    ("gpt-4.1", 1_047_576),
    ("gpt-4o", 128_000),
    ("gpt-4", 128_000),
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("claude", 200_000),
    ("gemini", 1_000_000),
    ("deepseek", 128_000),
    ("nemotron", 128_000),
    ("llama", 128_000),
    ("qwen", 128_000),
    ("mistral", 128_000),
    ("glm", 128_000),
    ("grok", 128_000),
)


def get_context_limit(model: str | None) -> int:
    """Return the context window size (tokens) for a model name.

    Matches case-insensitively on substrings; unknown models return
    DEFAULT_CONTEXT_LIMIT.
    """
    if not model:
        return DEFAULT_CONTEXT_LIMIT
    name = str(model).lower()
    for substring, limit in _MODEL_CONTEXT_LIMITS:
        if substring in name:
            return limit
    return DEFAULT_CONTEXT_LIMIT


def auto_compact_threshold(
    model: str | None, fraction: float = AUTO_COMPACT_FRACTION
) -> int:
    """Token estimate at which auto-compaction should trigger for a model."""
    return int(get_context_limit(model) * fraction)


def should_auto_compact(
    estimated_tokens: int,
    model: str | None,
    fraction: float = AUTO_COMPACT_FRACTION,
) -> bool:
    """True when the estimated context size has crossed the compact threshold."""
    if estimated_tokens <= 0:
        return False
    return estimated_tokens >= auto_compact_threshold(model, fraction)
