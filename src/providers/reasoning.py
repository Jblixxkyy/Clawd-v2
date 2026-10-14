"""Reasoning-budget provider adapter (brief 027).

Maps the portable reasoning budget (see :mod:`src.reasoning_budgets`) onto
native provider thinking options where they exist, and onto *nothing* where
they don't. This module is the single place that decides whether a model
call may carry a reasoning parameter.

Design rules (from cline's portable-reasoning):
  - ``may_reason(model, provider)`` skips non-reasoning models: where the
    model does not advertise reasoning control, the adapter sends nothing --
    never a 400 from a guessing parameter.
  - Native user-specified options always win over the portable value: if the
    caller already set the provider's native reasoning key, the adapter
    leaves the call untouched.
  - Providers pass extra kwargs straight through to the API (see
    ``AnthropicProvider.chat`` / ``OpenAICompatibleProvider.chat``), so the
    adapter only needs to *add* native keys -- no provider-file edits.

This module deliberately duck-types the budget (``enabled`` / ``effort`` /
``budget_tokens`` attributes) so it never imports the policy layer and
cannot create an import cycle.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

# ---------------------------------------------------------------------------
# Capability table
# ---------------------------------------------------------------------------

#: (provider_name, model_name_regex) pairs whose models advertise a native
#: reasoning control. Anything not listed is treated as non-reasoning and
#: the adapter sends nothing -- this is the never-400 rule. Patterns are
#: matched case-insensitively against the model name.
_REASONING_CAPABLE: list[tuple[str, str]] = [
    # Anthropic extended-thinking models (Claude 4 family).
    ("anthropic", r"claude-(opus|sonnet|haiku)-4"),
    # OpenAI reasoning models (o-series style effort ladder on GPT-5+).
    ("openai", r"^gpt-5"),
    # GLM thinking mode (glm-4.5 family).
    ("glm", r"^glm-4(\.5|-plus)"),
    # Gemini thinking models (no in-tree provider yet; mapping kept ready).
    ("gemini", r"^gemini-(2\.5|3)"),
]


def register_reasoning_capable(provider_name: str, model_pattern: str) -> None:
    """Add a (provider, model-regex) pair to the reasoning-capable table.

    Extension point for new models/providers; used by tests to simulate
    capability without editing the table.
    """
    _REASONING_CAPABLE.append((str(provider_name).strip().lower(), model_pattern))


def may_reason(model: Optional[str], provider_name: Optional[str]) -> bool:
    """Whether this model advertises a native reasoning control.

    Unknown provider/model pairs return False -- the adapter sends nothing
    rather than risk a 400. Never raises.
    """
    try:
        provider = str(provider_name or "").strip().lower()
        name = str(model or "").strip().lower()
        if not provider or not name:
            return False
        for capable_provider, pattern in _REASONING_CAPABLE:
            if capable_provider == provider and re.search(pattern, name):
                return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Native option names per provider
# ---------------------------------------------------------------------------

#: Native kwargs that carry a reasoning control, by provider. Checked so a
#: user-specified native option always wins over the portable budget.
_NATIVE_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("thinking",),
    "openai": ("reasoning", "reasoning_effort"),
    "nvidia": ("reasoning", "reasoning_effort"),
    "glm": ("thinking",),
    "minimax": (),
    "gemini": ("thinking_config", "thinking_level"),
}


def native_reasoning_keys(provider_name: Optional[str]) -> tuple[str, ...]:
    """Native kwarg names that carry reasoning control for a provider."""
    return _NATIVE_KEYS.get(str(provider_name or "").strip().lower(), ())


# ---------------------------------------------------------------------------
# Effort -> native value mapping
# ---------------------------------------------------------------------------

#: Anthropic-style budget tokens per portable effort level.
EFFORT_BUDGET_TOKENS: dict[str, int] = {
    "low": 1024,
    "medium": 5000,
    "high": 16000,
    "max": 32000,
}

#: OpenAI reasoning-effort ladder. "max" has no documented OpenAI value, so
#: it clamps to "high" rather than invent a parameter the API would reject.
_OPENAI_EFFORT = {"low": "low", "medium": "medium", "high": "high", "max": "high"}

#: Gemini thinking levels (kept ready; no in-tree provider yet).
_GEMINI_LEVEL = {"low": "low", "medium": "medium", "high": "high", "max": "high"}


def _effort(budget: Any) -> str:
    return str(getattr(budget, "effort", "medium") or "medium").strip().lower()


def _budget_tokens(budget: Any) -> Optional[int]:
    value = getattr(budget, "budget_tokens", None)
    if value is None:
        return None
    try:
        return max(int(value), 0)
    except (TypeError, ValueError):
        return None


def native_options(
    provider_name: Optional[str],
    model: Optional[str],
    budget: Any,
    *,
    user_kwargs: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Build the native reasoning kwargs for one model call.

    Returns ``{}`` (send nothing) when:
      - the budget is disabled,
      - the model does not advertise reasoning control (``may_reason`` is
        False) -- the never-400 rule,
      - the caller already set a native reasoning option -- the native
        user-specified value always wins over the portable budget.

    Otherwise returns provider kwargs to merge into the chat call. Never
    raises.
    """
    try:
        if not getattr(budget, "enabled", False):
            return {}
        provider = str(provider_name or "").strip().lower()
        if not may_reason(model, provider):
            return {}
        keys = native_reasoning_keys(provider)
        if user_kwargs:
            for key in keys:
                if key in user_kwargs:
                    # Native user-specified option wins over the portable value.
                    return {}
        effort = _effort(budget)
        if effort not in EFFORT_BUDGET_TOKENS:
            effort = "medium"

        if provider == "anthropic":
            tokens = _budget_tokens(budget)
            if tokens is None:
                tokens = EFFORT_BUDGET_TOKENS[effort]
            # Anthropic requires budget_tokens >= 1024.
            tokens = max(tokens, 1024)
            return {"thinking": {"type": "enabled", "budget_tokens": tokens}}
        if provider in ("openai", "nvidia"):
            return {"reasoning": {"effort": _OPENAI_EFFORT[effort]}}
        if provider == "glm":
            return {"thinking": {"type": "enabled"}}
        if provider == "gemini":
            opts: dict[str, Any] = {
                "thinking_config": {"thinking_level": _GEMINI_LEVEL[effort]}
            }
            tokens = _budget_tokens(budget)
            if tokens is not None:
                opts["thinking_config"]["thinking_budget"] = tokens
            return opts
        # Provider advertises reasoning (capability table) but has no known
        # native key mapping: send nothing rather than guess.
        return {}
    except Exception:
        return {}


def apply_reasoning_kwargs(
    kwargs: Optional[dict[str, Any]],
    provider_name: Optional[str],
    model: Optional[str],
    budget: Any,
) -> dict[str, Any]:
    """Merge native reasoning options into call kwargs (copy, never mutate).

    Call sites use this right before ``provider.chat(...)``::

        kwargs = apply_reasoning_kwargs({"model": model}, "anthropic", model, budget)
        response = provider.chat(messages, tools=tools, **kwargs)

    Returns the input kwargs unchanged (as a copy) when the adapter decides
    to send nothing. Never raises.
    """
    base = dict(kwargs or {})
    extra = native_options(provider_name, model, budget, user_kwargs=base)
    base.update(extra)
    return base


__all__ = [
    "EFFORT_BUDGET_TOKENS",
    "apply_reasoning_kwargs",
    "may_reason",
    "native_options",
    "native_reasoning_keys",
    "register_reasoning_capable",
]
