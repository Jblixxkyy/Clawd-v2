"""Model tiers for Clawd-Code (brief 028).

First-class model tiers: a cheap "small" model binding for utility calls
(session titles, compaction summaries, commit messages, reviewer/judge
roles, edit-correctors, file-mention classification) so the main model's
budget is preserved for reasoning -- plus per-mode binding (plan/act) and
the weak-model subagent profile.

Config shape (``~/.clawd/config.json``)::

    "models": {
        "main": "",    # primary model; "" falls back to provider default_model
        "small": "",   # utility model; "" falls back to main (zero change)
        "plan": "",    # plan-mode model; "" falls back to main (opt-in)
        "act": ""      # act-mode model; "" falls back to main (opt-in)
    }

Zero behavior change for single-model users: every resolution falls back
to the main model when the tier key is unset.

This module is the single place call sites ask "which model should I use
for X". Utility call sites that exist today route through it
(compaction); reviewer/judge (018), edit-corrector (019), commit messages
(032), session titles and file-mention classification wire in here when
they land.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional

from src.config import load_config

# ---------------------------------------------------------------------------
# Tier names
# ---------------------------------------------------------------------------

TIER_MAIN = "main"
TIER_SMALL = "small"
TIER_PLAN = "plan"
TIER_ACT = "act"

TIER_NAMES = (TIER_MAIN, TIER_SMALL, TIER_PLAN, TIER_ACT)

# Utility task roles that always resolve to the small tier.
UTILITY_TASKS = frozenset({
    "session_title",
    "compaction_summary",
    "commit_message",
    "reviewer",
    "judge",
    "edit_corrector",
    "file_mention_classification",
})


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _default_main_model(config: dict[str, Any]) -> str:
    """Best-effort main model: default provider's default_model. Never raises."""
    try:
        provider = config.get("default_provider") or "anthropic"
        providers = config.get("providers") or {}
        pconf = providers.get(provider) or {}
        model = (pconf.get("default_model") or "").strip()
        return model
    except Exception:
        return ""


def resolve_models(config: Optional[dict[str, Any]] = None) -> dict[str, str]:
    """Resolve the four model tiers from a config dict (or live config).

    ``small``/``plan``/``act`` fall back to ``main`` when unset; ``main``
    falls back to the default provider's ``default_model``. Never raises.
    """
    try:
        if config is None:
            config = load_config()
        models_cfg = (config or {}).get("models") or {}
        main = str(models_cfg.get("main") or "").strip()
        if not main:
            main = _default_main_model(config or {})
        small = str(models_cfg.get("small") or "").strip() or main
        plan = str(models_cfg.get("plan") or "").strip() or main
        act = str(models_cfg.get("act") or "").strip() or main
        return {
            TIER_MAIN: main,
            TIER_SMALL: small,
            TIER_PLAN: plan,
            TIER_ACT: act,
        }
    except Exception:
        return {t: "" for t in TIER_NAMES}


def model_for_tier(tier: str, config: Optional[dict[str, Any]] = None) -> str:
    """Model bound to a tier name; unknown tiers resolve to main."""
    models = resolve_models(config)
    return models.get(tier, models[TIER_MAIN])


def utility_model(
    model: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Model for a utility call (titles, summaries, commit messages, ...).

    Returns the configured small model when one is set; otherwise the
    passed ``model`` untouched (zero behavior change for single-model
    users); otherwise the resolved main model.
    """
    small = resolve_models(config)[TIER_SMALL]
    if small:
        return small
    return model or resolve_models(config)[TIER_MAIN]


def model_for_mode(mode: Optional[str], config: Optional[dict[str, Any]] = None) -> str:
    """Per-mode binding: plan mode may use a stronger model, act a cheaper
    one. Both are opt-in -- unset keys resolve to main."""
    models = resolve_models(config)
    mode = (mode or "").strip().lower()
    if mode == TIER_PLAN:
        return models[TIER_PLAN]
    if mode == TIER_ACT:
        return models[TIER_ACT]
    return models[TIER_MAIN]


# ---------------------------------------------------------------------------
# Tier attribution (cost accounting)
# ---------------------------------------------------------------------------

def tier_for_model(
    model: Optional[str],
    config: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Classify a model name back into a tier for cost attribution.

    A tier only claims a model when it is *distinct* from main, so a
    single-model setup attributes everything to "main". Returns None when
    the model matches no configured tier. Never raises.
    """
    try:
        m = str(model or "").strip()
        if not m:
            return None
        models = resolve_models(config)
        main = models[TIER_MAIN]
        for tier in (TIER_SMALL, TIER_PLAN, TIER_ACT):
            bound = models[tier]
            if bound and bound != main and m == bound:
                return tier
        if main and m == main:
            return TIER_MAIN
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Weak-model subagent profile
# ---------------------------------------------------------------------------

# Restricted read-only toolset for the weak profile, mirroring crush's
# sub-agent toolset (web_fetch, web_search, glob, grep, view). Tight
# scaffolding compensates for the weaker model.
WEAK_PROFILE_READ_TOOLS = frozenset({
    "read",
    "glob",
    "grep",
    "webfetch",
    "websearch",
})


def weak_profile_allowed_tools() -> frozenset:
    """Tool names (lowercased) a weak-profile subagent may use."""
    return WEAK_PROFILE_READ_TOOLS


def is_weak_profile_tool_allowed(tool_name: Optional[str]) -> bool:
    """Whether a tool name is inside the weak profile's read-only set."""
    return str(tool_name or "").strip().lower() in WEAK_PROFILE_READ_TOOLS


def weak_profile_model(config: Optional[dict[str, Any]] = None) -> str:
    """Model for the weak subagent profile: the small tier (== main when
    unset, so the profile still works for single-model users)."""
    return resolve_models(config)[TIER_SMALL]


def weak_read_permission_handler(
    tool_name: str,
    message: str,
    suggestion: Optional[str] = None,
) -> tuple[bool, bool]:
    """Permission handler for weak-profile subagents: auto-approve
    read-only tools, deny everything else -- never blocks on stdin.

    Returns (allowed, continue_without_caching) matching the
    ToolContext.permission_handler signature.
    """
    if is_weak_profile_tool_allowed(tool_name):
        return (True, False)
    return (False, False)


# ---------------------------------------------------------------------------
# Cached tier snapshot for hot paths (cost attribution)
# ---------------------------------------------------------------------------
# record_turn() runs per model turn; it must not do config file I/O on
# every call. The snapshot loads the config once and refreshes only when
# the config file's mtime changes. Never raises.

_snapshot_lock = threading.Lock()
_snapshot: Optional[dict[str, str]] = None
_snapshot_mtime: Optional[float] = None


def _config_mtime() -> Optional[float]:
    try:
        from src.config import get_config_path
        return get_config_path().stat().st_mtime
    except Exception:
        return None


def get_tier_snapshot() -> dict[str, str]:
    """Cached resolve_models(); refreshes when the config file changes."""
    global _snapshot, _snapshot_mtime
    mtime = _config_mtime()
    with _snapshot_lock:
        if _snapshot is not None and mtime == _snapshot_mtime:
            return dict(_snapshot)
    try:
        models = resolve_models()
    except Exception:
        models = {t: "" for t in TIER_NAMES}
    with _snapshot_lock:
        _snapshot = dict(models)
        _snapshot_mtime = mtime
    return dict(models)


def reset_tier_snapshot() -> None:
    """Forget the cached snapshot (tests)."""
    global _snapshot, _snapshot_mtime
    with _snapshot_lock:
        _snapshot = None
        _snapshot_mtime = None


def attribute_tier(model: Optional[str]) -> Optional[str]:
    """Best-effort tier attribution for a model name using the cached
    snapshot. Used by cost tracking on the hot path. Never raises."""
    try:
        snapshot = get_tier_snapshot()
        return tier_for_model(model, {"models": snapshot})
    except Exception:
        return None


# Re-exported for convenience; keeps the permission-handler type visible.
PermissionHandler = Callable[[str, str, Optional[str]], tuple[bool, bool]]
