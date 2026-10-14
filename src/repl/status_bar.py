"""Persistent status bar for the Clawd Codex REPL.

Renders a compact one-line status: model, provider, session token usage, and
a plan-mode marker. Every attribute access is defensive -- this function must
never raise, even for a partially-initialized or mock REPL.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape


def _get(obj: Any, name: str) -> Any:
    """getattr that never raises and normalizes None/exceptions to None."""
    if obj is None:
        return None
    try:
        value = getattr(obj, name, None)
    except Exception:
        return None
    return value


def _format_units(units: Any) -> str:
    """Compact token count: 950 -> '950', 1234 -> '1.2k', 2_500_000 -> '2.5M'."""
    try:
        n = int(units)
    except Exception:
        return "?"
    if n < 0:
        return "?"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _format_cost(cost: Any) -> str | None:
    """Format a dollar cost if a numeric one is available; None otherwise.

    CostTracker populates ``cost_usd`` from per-turn token usage priced per
    provider/model, so this lights up once the tracker has seen a turn.
    """
    if cost is None or isinstance(cost, bool):
        return None
    if isinstance(cost, (int, float)):
        try:
            return f"${cost:.4f}"
        except Exception:
            return None
    return None


def render_status(repl: Any) -> str:
    """Return a one-line rich-markup status string for the REPL.

    Format: ``❯ <provider> · <model> · <N> tokens`` plus `` [plan]`` when
    plan mode is on. If the cost tracker exposes a numeric ``cost_usd``
    (or ``total_cost``), `` · $<cost>`` is appended after the token count.

    The caller applies the dim style at print time (e.g.
    ``console.print(f"[dim]{render_status(self)}[/dim]")``), so this string
    carries no styling of its own -- only escaped dynamic values.

    Never raises -- missing/None attributes degrade to placeholders.
    """
    try:
        provider_name = _get(repl, "provider_name")
        model = _get(_get(repl, "provider"), "model")

        head = "❯"
        if isinstance(provider_name, str) and provider_name:
            head += f" {escape(provider_name)}"
        if isinstance(model, str) and model:
            head += f" · {escape(model)}"

        tracker = _get(repl, "cost_tracker")
        tokens = _format_units(_get(tracker, "total_units")) if tracker is not None else "?"
        body = f"{head} · {tokens} tokens"

        cost = _format_cost(_get(tracker, "cost_usd"))
        if cost is None:
            cost = _format_cost(_get(tracker, "total_cost"))
        if cost is not None:
            body += f" · {cost}"

        tool_context = _get(repl, "tool_context")
        plan_mode = bool(_get(tool_context, "plan_mode")) if tool_context is not None else False
        if plan_mode:
            # Escaped: "[plan]" would otherwise be parsed as a style tag.
            body += " \\[plan]"

        # Session permission mode, shown only when it isn't the default.
        # Never raises: every access goes through _get.
        perm_mode = _get(tool_context, "permission_mode") if tool_context is not None else None
        mode_value = _get(perm_mode, "value")
        if isinstance(mode_value, str) and mode_value and mode_value != "default":
            body += f" \\[{escape(mode_value)}]"

        return body
    except Exception:
        return "❯ ?"
