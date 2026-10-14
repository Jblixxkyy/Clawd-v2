"""Skill ``$``-mentions in user input (harness brief 009).

Lets the user explicitly invoke a skill from chat with a ``$`` sigil --
``$pdf`` or ``$skill://pdf`` -- instead of relying solely on the model to
pick the right skill. The input is pre-processed in the REPL *before* the
agent loop: mentions are resolved against the skill index through the exact
same path the ``Skill`` tool uses (:func:`src.command_surface.run_skill_tool`),
the skill's SKILL.md content is injected as context for the turn, and the
model is told the user explicitly requested it.

Unknown names produce the same helpful error the Skill tool returns
(including the available-skills listing). Input without any mention passes
through byte-identical.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ``$skill://name`` wins over the bare ``$name`` form when both could match
# at the same position (alternation is ordered). A bare ``$`` or ``$ `` can
# never match: at least one name character is required.
MENTION_RE = re.compile(r"\$skill://([a-zA-Z0-9_-]+)|\$([a-zA-Z0-9_-]+)")


def find_skill_mentions(text: str) -> list[str]:
    """Return skill names mentioned as ``$name`` / ``$skill://name``.

    Names come back in order of first appearance, de-duplicated. Input
    without any mention yields ``[]``. A malformed ``$skill://`` (no name)
    is ignored rather than parsed as a bare ``$skill`` mention.
    """
    names: list[str] = []
    for match in MENTION_RE.finditer(text):
        name = match.group(1) or match.group(2)
        if not name or name in names:
            continue
        if match.group(2) is not None and text[match.end() : match.end() + 3] == "://":
            # "$skill://" — malformed skill-uri mention, not a bare $skill.
            continue
        names.append(name)
    return names


@dataclass
class MentionResolution:
    """Outcome of expanding ``$``-mentions in one user message."""

    input_text: str
    """Final text to send to the agent loop (unchanged when no mentions)."""
    skills: list[str] = field(default_factory=list)
    """Skill names that were resolved, in mention order."""
    error: str | None = None
    """Set when a mention could not be resolved; the turn should be skipped."""


def resolve_skill_mentions(
    text: str,
    *,
    tool_registry: Any,
    tool_context: Any,
) -> MentionResolution:
    """Resolve ``$``-mentions in ``text`` through the Skill tool path.

    On a hit, the skill's expanded prompt (the same content the Skill tool
    returns) is appended after the user's original message, preceded by a
    note telling the model the user explicitly requested the skill. The
    user's original text is preserved verbatim.

    Unknown (or non-invocable) names abort the turn: ``error`` carries the
    Skill tool's message, which lists available skills.
    """
    names = find_skill_mentions(text)
    if not names:
        return MentionResolution(input_text=text)

    from src.command_surface import run_skill_tool

    resolved: list[tuple[str, str]] = []
    for name in names:
        try:
            result = run_skill_tool(name, "", tool_registry, tool_context)
        except Exception as exc:  # pragma: no cover - defensive
            return MentionResolution(
                input_text=text, error=f"Skill error for ${name}: {exc}"
            )
        payload = result.output if isinstance(result.output, dict) else {}
        if result.is_error or not payload.get("success"):
            error = payload.get("error")
            message = (
                error
                if isinstance(error, str) and error
                else f"Unknown skill: {name}"
            )
            return MentionResolution(input_text=text, error=message)
        prompt = payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return MentionResolution(
                input_text=text, error=f"Skill '${name}' produced empty prompt"
            )
        resolved.append((name, prompt.strip()))

    parts = [text]
    for name, prompt in resolved:
        parts.append(
            f"[The user explicitly invoked skill '{name}' via ${name}. "
            f"Follow the skill instructions below for this request.]\n{prompt}"
        )
    return MentionResolution(
        input_text="\n\n".join(parts),
        skills=[name for name, _ in resolved],
    )
