"""Hook system: shell commands fired around agent lifecycle events.

Events: SessionStart, SessionEnd, BeforeAgent, AfterAgent, PreCompress,
BeforeModel, AfterModel, PreToolUse, PostToolUse, BeforeToolSelection,
ToolExecuteBefore, SessionRetry, Stop. Config lives at ~/.clawd/hooks.json.
Run hooks through ``run_hooks`` in :mod:`runner`.

Ordering (brief 031): BeforeToolSelection -> ToolExecuteBefore ->
PreToolUse -> execute -> PostToolUse. Example policy hooks live in the
``examples/`` package (reference implementations only; they are inert
until registered in hooks.json).

Isolation (brief 033 — crash isolation, not a security boundary):
hook/plugin code never runs on the agent loop's main thread; each command
executes in a dedicated worker-thread context (:mod:`isolation`) with a
per-plugin timeout. Crashes and timeouts are fail-open, audit-logged, and
the plugin is marked failed. Load failures are loud: missing
bootstrap/config surfaces a named ``PluginLoadError`` in /doctor output
and session-start warnings, never a quiet skip.
"""

from .isolation import (
    IsolatedResult,
    PluginBootstrapMissingError,
    PluginConfigError,
    PluginLoadError,
    plugin_load_warnings,
    run_command_isolated,
    run_in_worker,
    validate_hook_config,
    validate_hook_entry,
)
from .runner import HOOK_EVENTS, run_hooks

__all__ = [
    "HOOK_EVENTS",
    "IsolatedResult",
    "PluginBootstrapMissingError",
    "PluginConfigError",
    "PluginLoadError",
    "plugin_load_warnings",
    "run_command_isolated",
    "run_hooks",
    "run_in_worker",
    "validate_hook_config",
    "validate_hook_entry",
]
