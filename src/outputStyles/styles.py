from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base_prompt import BASE_SYSTEM_PROMPT, EXPLANATORY_ADDENDUM


@dataclass(frozen=True)
class OutputStyle:
    name: str
    prompt: str
    source_path: Path | None = None


BUILTIN_OUTPUT_STYLES: dict[str, OutputStyle] = {
    "default": OutputStyle(
        name="default",
        prompt=BASE_SYSTEM_PROMPT,
    ),
    "explanatory": OutputStyle(
        name="explanatory",
        prompt=BASE_SYSTEM_PROMPT + EXPLANATORY_ADDENDUM,
    ),
    # Compact system prompt for LLM-driven subagents (brief 005). The child
    # has no access to the parent conversation, cannot interact with the
    # user, and its final message is returned verbatim to the parent.
    "subagent": OutputStyle(
        name="subagent",
        prompt=BASE_SYSTEM_PROMPT
        + """

## You are a subagent
- A parent agent delegated this task to you. Complete it autonomously with
  your tools and return your findings as your final message -- it goes back
  to the parent verbatim, so report concisely: outcome first, then key
  details (file paths as path/to/file.py:123, commands run, findings).
- You cannot interact with the user. Permission prompts are auto-denied:
  adapt (narrower scope, read-only path, different tool) instead of
  stalling or retrying the identical call.
- Interactive tools (messaging the user, asking questions, scheduling,
  plan/worktree/config changes) are unavailable to you.
- If you are blocked, say exactly what is missing. Do not invent results.
- No filler, no flattery, no questions back -- just the result.""",
    ),
}

