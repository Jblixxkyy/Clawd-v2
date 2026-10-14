"""Per-session token + cost tracking for Clawd Codex.

CostTracker records per-turn token usage (input/output/cached) tagged with
provider+model for pricing, and exposes a numeric ``cost_usd`` (session
spend), ``last_turn_cost_usd``, plus the legacy token-unit API
(``record(label, units)`` / ``total_units``).

Prices are $ per 1M tokens (input, output, cached-input-read). Cache
creation (write) tokens are billed at 1.25x the input rate unless
overridden -- that mirrors Anthropic's published cache-write pricing.
Defaults can be overridden in ``~/.clawd/config.json`` under the
``"pricing"`` key, e.g.::

    "pricing": {
        "anthropic": {"input": 3.0, "output": 15.0, "cached": 0.3},
        "anthropic:claude-opus-4-6": {"input": 15.0, "output": 75.0, "cached": 1.5}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Prices: $ per 1M tokens -> (input, output, cached-input-read).
# Sensible defaults; the public API (Anthropic, OpenAI) is priced at the
# published list rates, GLM/Minimax/NVIDIA at their typical cheaper tiers.
_DEFAULT_PRICES: dict[str, tuple[float, float, float]] = {
    "anthropic": (3.00, 15.00, 0.30),
    "openai": (2.50, 10.00, 1.25),
    "glm": (0.60, 0.90, 0.06),
    "minimax": (0.30, 0.30, 0.03),
    "nvidia": (0.60, 1.20, 0.06),
    "default": (1.00, 2.00, 0.10),
}

# Model-name prefix -> prices ($/MTok in, out, cached-read). Longest prefix
# wins; checked before the provider default so a model that strongly names
# its family (e.g. a nemotron model reached via an OpenAI-compatible
# endpoint) gets the right price.
_MODEL_PRICE_OVERRIDES: dict[str, tuple[float, float, float]] = {
    # Anthropic
    "claude-opus-4": (15.00, 75.00, 1.50),
    "claude-sonnet-4": (3.00, 15.00, 0.30),
    "claude-haiku-4": (1.00, 5.00, 0.10),
    "claude-3-5-sonnet": (3.00, 15.00, 0.30),
    "claude-3-5-haiku": (0.80, 4.00, 0.08),
    # OpenAI
    "gpt-5": (5.00, 20.00, 2.50),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    "gpt-4o": (2.50, 10.00, 1.25),
    # GLM
    "glm-5": (1.00, 1.50, 0.10),
    "glm-4": (0.60, 0.90, 0.06),
    # NVIDIA NIM (usually reached via an OpenAI-compatible endpoint)
    "nemotron": (0.60, 1.20, 0.06),
    "deepseek": (0.27, 1.10, 0.07),
    "llama": (0.06, 0.12, 0.01),
}

_CACHE_WRITE_MULTIPLIER = 1.25


def _as_int(value: Any) -> int:
    """Coerce a token count to int. Never raises; negatives clamp to 0."""
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, n)


def _as_float(value: Any) -> float:
    """Coerce a price to float. Never raises."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _load_config_price_overrides() -> dict[str, tuple[float, float, float]]:
    """Read ``"pricing"`` overrides from ~/.clawd/config.json. Fail-open."""
    try:
        config_path = Path.home() / ".clawd" / "config.json"
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        pricing = raw.get("pricing")
        if not isinstance(pricing, dict):
            return {}
        overrides: dict[str, tuple[float, float, float]] = {}
        for key, value in pricing.items():
            if not isinstance(value, dict):
                continue
            overrides[str(key).lower()] = (
                _as_float(value.get("input")),
                _as_float(value.get("output")),
                _as_float(value.get("cached")),
            )
        return overrides
    except Exception:
        return {}


@dataclass
class CostTracker:
    total_units: int = 0
    events: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Session spend in USD (numeric -- the REPL status bar reads this).
        self.cost_usd: float = 0.0
        self.last_turn_cost_usd: float = 0.0
        # Per-turn records: provider, model, input/output/cached tokens, cost.
        self.turns: list[dict[str, Any]] = []
        # Per-tier spend attribution (brief 028): tier -> USD.
        self.cost_by_tier: dict[str, float] = {}
        self._price_overrides: Optional[dict[str, tuple[float, float, float]]] = None

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def set_prices(self, prices: dict[str, Any]) -> None:
        """Set/extend price overrides programmatically.

        Accepts the same shape as the config ``"pricing"`` dict, e.g.
        ``{"anthropic": {"input": 3.0, "output": 15.0, "cached": 0.3}}``.
        """
        overrides: dict[str, tuple[float, float, float]] = {}
        for key, value in (prices or {}).items():
            if isinstance(value, dict):
                overrides[str(key).lower()] = (
                    _as_float(value.get("input")),
                    _as_float(value.get("output")),
                    _as_float(value.get("cached")),
                )
            elif isinstance(value, (list, tuple)) and len(value) == 3:
                overrides[str(key).lower()] = (
                    _as_float(value[0]),
                    _as_float(value[1]),
                    _as_float(value[2]),
                )
        if self._price_overrides is None:
            self._price_overrides = {}
        self._price_overrides.update(overrides)

    @property
    def _overrides(self) -> dict[str, tuple[float, float, float]]:
        if self._price_overrides is None:
            self._price_overrides = _load_config_price_overrides()
        return self._price_overrides

    def price_for(self, provider: Any, model: Any) -> tuple[float, float, float]:
        """Return ($/MTok in, out, cached-read) for a provider+model.

        Never raises; falls back to the "default" tier for unknown input.
        """
        try:
            provider_key = str(provider or "").strip().lower()
            model_key = str(model or "").strip().lower()
            overrides = self._overrides

            # 1. Explicit "provider:model-prefix" override, longest prefix wins.
            best: Optional[tuple[float, float, float]] = None
            best_len = -1
            for key, prices in overrides.items():
                if ":" not in key:
                    continue
                p, m = key.split(":", 1)
                if p == provider_key and model_key.startswith(m) and len(m) > best_len:
                    best, best_len = prices, len(m)
            if best is not None:
                return best

            # 2. Provider-level override from config (user config beats
            #    built-in model-family defaults).
            if provider_key in overrides:
                return overrides[provider_key]

            # 3. Model-name family prefix (built-in).
            for prefix in sorted(_MODEL_PRICE_OVERRIDES, key=len, reverse=True):
                if model_key.startswith(prefix):
                    return _MODEL_PRICE_OVERRIDES[prefix]

            # 4. Provider default, else the global default.
            return _DEFAULT_PRICES.get(provider_key, _DEFAULT_PRICES["default"])
        except Exception:
            return _DEFAULT_PRICES["default"]

    def cost_for(
        self,
        provider: Any,
        model: Any,
        input_tokens: Any,
        output_tokens: Any,
        cached_tokens: Any = 0,
        cache_creation_tokens: Any = 0,
    ) -> float:
        """Price a turn in USD. Never raises.

        Cache-creation (write) tokens bill at 1.25x the input rate and
        cache-read tokens at the cached rate; the remainder of
        input_tokens bills at the base input rate.
        """
        try:
            in_price, out_price, cached_price = self.price_for(provider, model)
            read = _as_int(cached_tokens)
            created = _as_int(cache_creation_tokens)
            plain_input = _as_int(input_tokens) - read - created
            if plain_input < 0:
                plain_input = 0
            return (
                plain_input * in_price
                + created * in_price * _CACHE_WRITE_MULTIPLIER
                + read * cached_price
                + _as_int(output_tokens) * out_price
            ) / 1_000_000
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, label: str, units: int) -> None:
        """Legacy API: record raw token units. Behavior unchanged."""
        self.total_units += units
        self.events.append(f"{label}:{units}")

    def record_turn(
        self,
        provider: Any,
        model: Any,
        input_tokens: Any,
        output_tokens: Any,
        cached_tokens: Any = 0,
        cache_creation_tokens: Any = 0,
        label: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> float:
        """Record one model turn's tokens with pricing. Returns the turn cost.

        Also feeds ``total_units`` (so the status bar token count keeps
        working) and appends to ``events`` in the legacy ``label:units`` shape.
        ``tier`` attributes the spend to a model tier ("main"/"small"/"plan"/
        "act"); when omitted it is attributed best-effort from the model
        name via the cached tier snapshot (brief 028), falling back to
        "unattributed". Never raises -- on any error this is a no-op
        returning 0.0.
        """
        try:
            in_t = _as_int(input_tokens)
            out_t = _as_int(output_tokens)
            read_t = _as_int(cached_tokens)
            created_t = _as_int(cache_creation_tokens)
            cost = self.cost_for(provider, model, in_t, out_t, read_t, created_t)
            turn_index = len(self.turns) + 1
            turn_label = label or f"turn_{turn_index}_tokens"
            tier_label = self._attribute_tier(tier, model)
            self.turns.append({
                "label": turn_label,
                "provider": str(provider or ""),
                "model": str(model or ""),
                "tier": tier_label,
                "input_tokens": in_t,
                "output_tokens": out_t,
                "cached_tokens": read_t,
                "cache_creation_tokens": created_t,
                "cost_usd": cost,
            })
            self.record(turn_label, in_t + out_t)
            self.last_turn_cost_usd = cost
            self.cost_usd += cost
            self.cost_by_tier[tier_label] = self.cost_by_tier.get(tier_label, 0.0) + cost
            return cost
        except Exception:
            return 0.0

    @staticmethod
    def _attribute_tier(tier: Optional[str], model: Any) -> str:
        """Resolve the tier label for a turn. Never raises."""
        if tier and str(tier).strip():
            return str(tier).strip()
        try:
            from src.model_tiers import attribute_tier
            resolved = attribute_tier(model)
            if resolved:
                return resolved
        except Exception:
            pass
        return "unattributed"

    def spend_by_tier(self) -> dict[str, float]:
        """USD spend split by model tier (brief 028)."""
        return dict(self.cost_by_tier)

    # ------------------------------------------------------------------
    # Readouts
    # ------------------------------------------------------------------

    @property
    def session_cost_usd(self) -> float:
        """Total session spend in USD (alias for cost_usd)."""
        return self.cost_usd
