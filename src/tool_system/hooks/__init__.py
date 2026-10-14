"""Hook system: shell commands fired around agent lifecycle events.

Events: SessionStart, SessionEnd, BeforeAgent, AfterAgent, PreCompress,
BeforeModel, AfterModel, PreToolUse, PostToolUse. Config lives at
~/.clawd/hooks.json. Run hooks through ``run_hooks`` in :mod:`runner`.
"""

from .runner import HOOK_EVENTS, run_hooks

__all__ = ["HOOK_EVENTS", "run_hooks"]
