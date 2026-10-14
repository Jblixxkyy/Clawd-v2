"""Configuration management for Clawd Codex."""

from __future__ import annotations

import json
import base64
import copy
import os
from pathlib import Path
from typing import Any, Optional


# Model tiers (brief 028). Keys of the "models" config dict:
#   main  - primary model ("" = fall back to provider default_model)
#   small - utility model ("" = fall back to main; zero behavior change)
#   plan  - plan-mode model ("" = fall back to main; opt-in)
#   act   - act-mode model ("" = fall back to main; opt-in)
MODEL_TIERS = ("main", "small", "plan", "act")
_DEFAULT_MODELS_CONFIG: dict[str, str] = {tier: "" for tier in MODEL_TIERS}


# 030: relevance-shaped skill loading defaults.
_SKILL_LOADING_DEFAULTS: dict[str, Any] = {
    # Master switch for marker/trigger auto-activation.
    "auto_activate": True,
    # Token budget for the whole rendered skill block (index + bodies).
    "token_budget": 8000,
    # Seconds allowed for a skill's context_cmd.
    "context_cmd_timeout": 5,
    # Per-model-family tool pruning: {family: [tool, ...]}.
    # Empty (default) = NO pruning, catalog passes through untouched.
    "prune_tools": {},
}


# 027: reasoning-budget defaults. Portable reasoning{enabled, effort,
# budget_tokens} per call site; phase/role/agent policies resolved by
# src.reasoning_budgets. Effort ladder: low|medium|high|max ("off" disables).
_REASONING_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "default_effort": "medium",
    "budget_tokens": None,
    # Prefer 023's think tool for the no-knob fallback when importable.
    "use_think_tool": True,
    # Planning gets a bigger think budget than execution.
    "phases": {"planning": "high", "execution": "medium", "compaction": "medium"},
    # Investigator/reviewer high; classifiers/summarizers/titles low or off.
    "roles": {
        "investigator": "high",
        "reviewer": "high",
        "classifier": "low",
        "summarizer": "low",
        "session_title": "off",
        "compaction": "medium",
    },
    # Per-agent overrides: {"my-agent": "low"} / {"my-agent": "off"}.
    "agents": {},
}


# 022: async/background compaction defaults. Mirrors codex's AsyncCompaction
# flag: default OFF until the turn-lifecycle wiring lands (a later wave).
_ASYNC_COMPACTION_DEFAULTS: dict[str, Any] = {
    # Master switch. Default off, like codex's AsyncCompaction flag.
    "enabled": False,
    # Background launch only when the snapshot holds at least this many
    # estimated tokens -- avoids paying for background work on tiny turns.
    "min_tokens": 20_000,
    # Token headroom kept free after adoption (opencode's RESERVE_MIN).
    "reserve_tokens": 16_000,
    # Successive target fractions of the context limit across foreground
    # passes (opencode's SHRINK_STEPS). The single foreground fallback
    # allowed per adoption cycle uses steps[0].
    "shrink_steps": [0.7, 0.5, 0.35],
    # Forced summary template: handoff framing (progress / decisions /
    # remaining / critical references), per codex's compact prompt +
    # opencode's forced SUMMARY_TEMPLATE. The summarizer MUST use it.
    "summary_template": (
        "You MUST use this exact format for the summary. Write it as a "
        "handoff so a fresh agent can continue seamlessly:\n"
        "\n"
        "## Progress\n"
        "What was accomplished so far.\n"
        "\n"
        "## Decisions\n"
        "Key decisions made and why.\n"
        "\n"
        "## Remaining\n"
        "What is still to do, including the immediate next step.\n"
        "\n"
        "## Critical references\n"
        "Files, commands, identifiers, and other references the next "
        "agent must know."
    ),
}


# 026: idle compaction defaults. Mirrors Claude Code 2.1.299's
# "idleCompaction" shape: a 200K+ token conversation idle ~54 minutes
# auto-compacts in the background. Default OFF (opt-in), like the
# async_compaction section.
_IDLE_COMPACTION_DEFAULTS: dict[str, Any] = {
    # Master switch. Default off -- idle compaction never runs unless the
    # user opts in, and it can always be turned back off (hard off switch).
    "enabled": False,
    # Idle-compact only when the estimated context holds at least this
    # many tokens -- avoids background work on small conversations.
    "min_tokens": 200_000,
    # Minutes with no user/model activity before the background compaction
    # is triggered (checked at the start of the next run).
    "idle_minutes": 54,
}


def async_compaction_config() -> dict[str, Any]:
    """022: config for async/background compaction.

    Merges the ``async_compaction`` section of ~/.clawd/config.json over
    the defaults. Conservative: never raises; ``enabled`` defaults to
    False so nothing changes for existing users.
    """
    merged = copy.deepcopy(_ASYNC_COMPACTION_DEFAULTS)
    try:
        config = load_config()
    except Exception:
        return merged
    section = config.get("async_compaction") or {}
    if isinstance(section, dict):
        merged.update(section)
    return merged


def async_compaction_enabled() -> bool:
    """Whether async/background compaction (brief 022) is enabled.

    Config key "async_compaction"."enabled" in ~/.clawd/config.json;
    default OFF (opt-in). Never raises -- returns False when the config
    can't be read.
    """
    try:
        config = load_config()
        section = config.get("async_compaction") or {}
        return bool(section.get("enabled", False))
    except Exception:
        return False


def idle_compaction_config() -> dict[str, Any]:
    """026: config for idle auto-compaction.

    Merges the ``idle_compaction`` section of ~/.clawd/config.json over
    the defaults. Conservative: never raises; ``enabled`` defaults to
    False so nothing changes for existing users.
    """
    merged = copy.deepcopy(_IDLE_COMPACTION_DEFAULTS)
    try:
        config = load_config()
    except Exception:
        return merged
    section = config.get("idle_compaction") or {}
    if isinstance(section, dict):
        merged.update(section)
    return merged


def idle_compaction_enabled() -> bool:
    """Whether idle auto-compaction (brief 026) is enabled.

    Config key "idle_compaction"."enabled" in ~/.clawd/config.json;
    default OFF (opt-in). Never raises -- returns False when the config
    can't be read.
    """
    try:
        config = load_config()
        section = config.get("idle_compaction") or {}
        return bool(section.get("enabled", False))
    except Exception:
        return False


def skill_loading_config() -> dict[str, Any]:
    """030: config for relevance-shaped skill loading.

    Merges the ``skills`` section of ~/.clawd/config.json over the defaults.
    Conservative: never raises, never prunes unless explicitly configured.
    """
    merged = dict(_SKILL_LOADING_DEFAULTS)
    try:
        config = load_config()
    except Exception:
        return merged
    section = config.get("skills") or {}
    if isinstance(section, dict):
        merged.update(section)
    return merged


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    config_dir = Path.home() / ".clawd"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "config.json"


def _get_default_config_from_providers() -> dict[str, Any]:
    """Build default config using provider info registry."""
    from src.providers import PROVIDER_INFO

    return {
        "default_provider": "anthropic",
        "providers": {
            name: {
                "api_key": "",
                "base_url": info["default_base_url"],
                "default_model": info["default_model"],
            }
            for name, info in PROVIDER_INFO.items()
        },
        "models": dict(_DEFAULT_MODELS_CONFIG),
        "session": {
            "auto_save": True,
            "max_history": 100
        },
        "agent": {
            "max_turns": 50
        },
        "skills": dict(_SKILL_LOADING_DEFAULTS),
        "reasoning": dict(_REASONING_DEFAULTS),
        "async_compaction": copy.deepcopy(_ASYNC_COMPACTION_DEFAULTS),
        "idle_compaction": copy.deepcopy(_IDLE_COMPACTION_DEFAULTS),
    }


def get_default_config() -> dict[str, Any]:
    """Generate default configuration."""
    return _get_default_config_from_providers()


def _encode_api_key(api_key: str) -> str:
    """Encode API key for basic obfuscation."""
    return base64.b64encode(api_key.encode()).decode()


def _decode_api_key(encoded_key: str) -> str:
    """Decode API key."""
    try:
        return base64.b64decode(encoded_key.encode()).decode()
    except Exception:
        # If decoding fails, return as-is (might be plain text)
        return encoded_key


def load_config() -> dict[str, Any]:
    """Load configuration from file.

    Returns:
        Configuration dictionary
    """
    config_path = get_config_path()

    if not config_path.exists():
        # Create default config
        config = get_default_config()
        save_config(config)
        return config

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        providers = config.setdefault("providers", {})
        from src.providers import PROVIDER_INFO

        needs_save = False
        for provider_name, info in PROVIDER_INFO.items():
            provider_config = providers.setdefault(provider_name, {})
            if "api_key" not in provider_config:
                provider_config["api_key"] = ""
                needs_save = True
            if "base_url" not in provider_config:
                provider_config["base_url"] = info["default_base_url"]
                needs_save = True
            if "default_model" not in provider_config:
                provider_config["default_model"] = info["default_model"]
                needs_save = True

        # Decode API keys
        for provider_name, provider_config in providers.items():
            if provider_config.get("api_key"):
                provider_config["api_key"] = _decode_api_key(provider_config["api_key"])

        # Model tiers (brief 028): main/small/plan/act; small/plan/act
        # default to main when unset (zero behavior change for
        # single-model users).
        models_config = config.setdefault("models", {})
        for tier_key, tier_default in _DEFAULT_MODELS_CONFIG.items():
            if tier_key not in models_config:
                models_config[tier_key] = tier_default
                needs_save = True

        # Reasoning budgets (brief 027): portable reasoning{enabled, effort,
        # budget_tokens} + phase/role/agent policies. Merge defaults under
        # any existing section so user overrides survive upgrades.
        reasoning_config = config.setdefault("reasoning", {})
        for key, default in _REASONING_DEFAULTS.items():
            if key not in reasoning_config:
                reasoning_config[key] = (
                    dict(default) if isinstance(default, dict) else default
                )
                needs_save = True
        for bucket in ("phases", "roles", "agents"):
            section_bucket = reasoning_config.setdefault(bucket, {})
            for name, spec in _REASONING_DEFAULTS[bucket].items():
                if name not in section_bucket:
                    section_bucket[name] = spec
                    needs_save = True

        # Async/background compaction (brief 022): merge defaults under
        # any existing section so user overrides survive upgrades.
        # Default OFF (opt-in); nothing changes for existing users.
        async_compaction_section = config.setdefault("async_compaction", {})
        for key, default in _ASYNC_COMPACTION_DEFAULTS.items():
            if key not in async_compaction_section:
                async_compaction_section[key] = copy.deepcopy(default)
                needs_save = True

        # Idle auto-compaction (brief 026): merge defaults under any
        # existing section so user overrides survive upgrades.
        # Default OFF (opt-in); nothing changes for existing users.
        idle_compaction_section = config.setdefault("idle_compaction", {})
        for key, default in _IDLE_COMPACTION_DEFAULTS.items():
            if key not in idle_compaction_section:
                idle_compaction_section[key] = copy.deepcopy(default)
                needs_save = True

        if config.get("default_provider") is None:
            config["default_provider"] = "anthropic"
            needs_save = True

        if needs_save:
            save_config(config)

        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return get_default_config()


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to file.

    Args:
        config: Configuration dictionary to save
    """
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a copy to avoid modifying the original
    config_copy = json.loads(json.dumps(config))

    # Encode API keys
    for provider_name, provider_config in config_copy.get("providers", {}).items():
        if provider_config.get("api_key"):
            provider_config["api_key"] = _encode_api_key(provider_config["api_key"])

    if os.name == "nt":
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_copy, f, indent=2, ensure_ascii=False)
    else:
        fd = os.open(
            config_path,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(config_copy, f, indent=2, ensure_ascii=False)
        os.chmod(config_path, 0o600)


def get_provider_config(provider: str) -> dict[str, Any]:
    """Get configuration for a specific provider.

    Args:
        provider: Provider name (anthropic, openai, nvidia, glm, minimax)

    Returns:
        Provider configuration dictionary
    """
    config = load_config()
    providers = config.get("providers", {})

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}")

    return providers[provider]


def set_api_key(provider: str, api_key: str, base_url: Optional[str] = None,
                default_model: Optional[str] = None) -> None:
    """Set API key for a provider.

    Args:
        provider: Provider name (anthropic, openai, nvidia, glm, minimax)
        api_key: API key to set
        base_url: Optional base URL override
        default_model: Optional default model override
    """
    config = load_config()

    if provider not in config.get("providers", {}):
        # Add new provider if it doesn't exist
        if "providers" not in config:
            config["providers"] = {}
        config["providers"][provider] = {}

    config["providers"][provider]["api_key"] = api_key

    if base_url is not None:
        config["providers"][provider]["base_url"] = base_url

    if default_model is not None:
        config["providers"][provider]["default_model"] = default_model

    save_config(config)


def set_default_provider(provider: str) -> None:
    """Set the default provider.

    Args:
        provider: Provider name
    """
    config = load_config()
    config["default_provider"] = provider
    save_config(config)


def get_default_provider() -> str:
    """Get the default provider.

    Returns:
        Default provider name
    """
    config = load_config()
    return config.get("default_provider", "anthropic")


def get_model_tier(tier: str) -> str:
    """Get the configured model for a tier (main/small/plan/act).

    Returns the raw configured value; "" means unset (resolution falls
    back to main -- see src.model_tiers.resolve_models).

    Raises:
        ValueError: for an unknown tier name.
    """
    if tier not in MODEL_TIERS:
        raise ValueError(f"Unknown model tier: {tier!r} (expected one of {MODEL_TIERS})")
    config = load_config()
    models = config.get("models") or {}
    return str(models.get(tier) or "")


def set_model_tier(tier: str, model: str) -> None:
    """Bind a model name to a tier (main/small/plan/act).

    Pass "" to clear a tier back to its fallback.

    Raises:
        ValueError: for an unknown tier name.
    """
    if tier not in MODEL_TIERS:
        raise ValueError(f"Unknown model tier: {tier!r} (expected one of {MODEL_TIERS})")
    config = load_config()
    models = config.setdefault("models", {})
    models[tier] = str(model or "")
    save_config(config)


def powershell_opt_in() -> bool:
    """Whether the PowerShell tool is enabled.

    Config key "enable_powershell" in ~/.clawd/config.json; default ON
    (PowerShell is the default shell on Windows, where most users are).
    Set "enable_powershell": false to opt out. Never raises -- returns
    True when the config can't be read.
    """
    try:
        config = load_config()
    except Exception:
        return True
    return bool(config.get("enable_powershell", True))


def git_checkpoints_enabled() -> bool:
    """Whether git-backed checkpoints (snapshot-before-write, brief 032) are on.

    Config key "git_checkpoints" in ~/.clawd/config.json; default OFF
    (opt-in: the snapshots run git plumbing in the target repo).
    Set "git_checkpoints": true to opt in. Never raises -- returns
    False when the config can't be read.
    """
    try:
        config = load_config()
    except Exception:
        return False
    return bool(config.get("git_checkpoints", False))
