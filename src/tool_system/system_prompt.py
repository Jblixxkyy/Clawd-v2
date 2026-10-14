"""Agent-facing guidance injected into the effective system prompt.

Built by ``_build_effective_system_prompt`` in ``agent_loop.py``.

Prompt-diet architecture (opencode parity, 2026-10-13): the static core lives
in ``src/outputStyles/base_prompt.py`` (BASE_SYSTEM_PROMPT) as ONE merged
layer. This module renders only the dynamic/conditional parts:

- ``build_tool_guidance``: per-tool guidance lines (opencode's
  ``${OPENCODE_TOOL_GUIDANCE}`` pattern) rendered only when the tool is
  registered -- weight costs tokens only when the tool exists.
- ``build_env_block``: opencode's ``<env>`` block with working dir, git
  status, platform, and tmp preference.
- ``build_skills_section``: the live skill inventory, so the model answers
  from real state instead of guessing from the tool roster.
- ``build_agent_guidelines``: legacy entry point, now only the Windows-only
  shell section (kept for backwards compatibility).
"""

from __future__ import annotations

import platform
import tempfile
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import ToolContext


def _build_windows_shell_section() -> str:
    """Windows-only shell guidance; returns "" on other platforms.

    The Bash tool is bash-only: on stock Windows there is no bash executable
    and the tool fails with an actionable error (install Git Bash). The
    PowerShell tool is a separate, explicitly opt-in tool -- when it is used,
    the syntax must be PowerShell, never bash. Bash syntax is NEVER silently
    translated to PowerShell.
    """
    try:
        if platform.system().lower() != "windows":
            return ""
    except Exception:
        return ""
    return """## Shell on Windows

- The Bash tool requires a bash executable and runs bash syntax only. On stock Windows it is not installed -- the Bash tool returns a clear error telling you to install Git Bash (or see docs/guide/SETUP_GUIDE.md).
- The PowerShell tool is enabled by default on Windows: use it with PowerShell syntax. Never write bash syntax (&& chains, $VAR, $(...), export, backticks, forward-slash paths assumed) for the PowerShell tool -- the semantics differ silently and failures can be quiet. Match the syntax to the tool, always."""


def build_agent_guidelines() -> str:
    """Legacy entry point: now only the Windows-only shell section.

    The old multi-section judgment/capability guidance was merged into the
    single core (BASE_SYSTEM_PROMPT) and conditional per-tool guidance
    (``build_tool_guidance``) as part of the prompt diet. Kept so existing
    imports keep working.
    """
    return _build_windows_shell_section()


# --- Conditional per-tool guidance (opencode's ${OPENCODE_TOOL_GUIDANCE}) ---
#
# Each line renders only when the tool is registered, so prompt weight costs
# tokens only when the tool exists. Keys are lowercase tool names.

_TOOL_GUIDANCE: dict[str, str] = {
    # Prompt-level medicine for the "hella noisy" complaint.
    "bash": (
        "- Do not chain shell commands with separators like `echo \"====\"`; "
        "the output becomes noisy in a way that makes the user's side of the "
        "conversation worse."
    ),
    "write": (
        "- Write creates files or completely replaces their content; prefer Edit "
        "for targeted changes. A checkpoint snapshot is taken before every write "
        "-- /rewind can restore the pre-edit state. Don't re-read files you just "
        "wrote -- trust the write result."
    ),
    "edit": (
        "- Edit replaces the exact text in old_string with new_string, and the "
        "values must differ. old_string must occur exactly once; include more "
        "surrounding context to make it unique. A checkpoint snapshot is taken "
        "before every edit."
    ),
    "enterplanmode": (
        "- Plan mode: for complex or multi-step tasks, call EnterPlanMode first to "
        "explore and design an approach before editing files. When the plan is "
        "ready, call ExitPlanMode with a clear `plan` -- the user reviews and "
        "approves it before anything is executed, so write plans the user can "
        "read and judge."
    ),
    "agent": (
        "- Use the Agent tool to delegate a self-contained sequence of tool calls "
        "as one atomic subagent step (e.g. \"read these three files and summarize\"). "
        "Batch independent Agent calls in the same turn so subagents explore in "
        "parallel. Inside a delegated task, permission prompts are auto-denied -- "
        "give delegated work everything it needs up front (paths, constraints) and "
        "avoid steps that would need interactive approval."
    ),
    "taskcreate": (
        "- Use TaskCreate / TaskList / TaskUpdate to track multi-step plans you are "
        "executing yourself; the task list is your memory, the Agent tool is your "
        "worker."
    ),
    "askuserquestion": (
        "- Ask with AskUserQuestion before destructive actions, ambiguous requests, "
        "or anything externally visible (sends, publishing, purchases). If a request "
        "is genuinely ambiguous and a wrong guess would be costly, ask one short "
        "question; otherwise make a reasonable assumption, state it, and continue."
    ),
    "skill": (
        "- Before starting a task, check whether a skill listed under "
        "## Available Skills matches it. If the skill's instructions already "
        "appear as a `<skill_content>` block in this conversation, don't invoke "
        "it again."
    ),
    "websearch": (
        "- Use WebSearch/WebFetch for current or external information you can't get "
        "from the codebase (docs, versions, error messages, changelogs), and say "
        "where it came from."
    ),
    "webfetch": (
        "- Use WebSearch/WebFetch for current or external information you can't get "
        "from the codebase (docs, versions, error messages, changelogs), and say "
        "where it came from."
    ),
}

# Deterministic render order (dict order is stable, but explicit beats implicit).
_TOOL_GUIDANCE_ORDER = (
    "bash", "write", "edit", "enterplanmode", "agent", "taskcreate",
    "askuserquestion", "skill", "websearch", "webfetch",
)

_SCRIPT_MODE_LINE = (
    "- Script mode: running non-interactively -- permission prompts auto-deny; "
    "prefer read-only reconnaissance and avoid edits unless asked."
)


def build_tool_guidance(
    registered_tools: set[str] | None,
    *,
    script_mode: bool = False,
) -> str:
    """Render conditional per-tool guidance for the registered tools.

    ``registered_tools`` is a set of tool names (any case). Identical lines
    are deduplicated so e.g. WebSearch/WebFetch share one line. Returns ""
    when nothing applies.
    """
    if not registered_tools:
        names = set()
    else:
        names = {str(n).lower() for n in registered_tools}
    lines: list[str] = []
    for key in _TOOL_GUIDANCE_ORDER:
        if key in names:
            line = _TOOL_GUIDANCE[key]
            if line not in lines:
                lines.append(line)
    if script_mode:
        lines.append(_SCRIPT_MODE_LINE)
    if not lines:
        return ""
    return "## Tool guidance\n" + "\n".join(lines)


def todays_date_line() -> str:
    """First-class date line, rendered at the core head (opencode parity)."""
    return f"Today's date: {date.today().isoformat()}"


def build_env_block(tool_context: "ToolContext") -> str:
    """Opencode-style <env> block: working dir, git status, platform, tmp pref."""
    try:
        root = str(Path(tool_context.workspace_root).expanduser().resolve())
    except Exception:
        root = str(getattr(tool_context, "workspace_root", ""))
    try:
        cwd = str(Path(tool_context.cwd).expanduser().resolve()) if tool_context.cwd else root
    except Exception:
        cwd = root
    try:
        is_git = "yes" if (Path(root) / ".git").exists() else "no"
    except Exception:
        is_git = "no"
    try:
        plat = platform.system() or "unknown"
    except Exception:
        plat = "unknown"
    try:
        tmpdir = tempfile.gettempdir()
    except Exception:
        tmpdir = "/tmp"
    return (
        "Here is some useful information about the environment you are running in:\n"
        "<env>\n"
        f"  Working directory: {cwd}\n"
        f"  Workspace root: {root}\n"
        f"  Is directory a git repo: {is_git}\n"
        f"  Platform: {plat}\n"
        f"  Prefer {tmpdir} over generic system temporary directories such as /tmp; "
        "it is pre-created and approved for external access.\n"
        "</env>"
    )


_SKILL_DESC_MAX = 250


def _skill_desc(s) -> str:
    """One-line description plus when_to_use trigger text, capped for prompt budget."""
    desc = " ".join((s.description or "").split())
    when = " ".join((getattr(s, "when_to_use", "") or "").split())
    if when:
        desc = f"{desc} {when}"
    if len(desc) > _SKILL_DESC_MAX:
        desc = desc[: _SKILL_DESC_MAX - 1].rstrip() + "\u2026"
    return desc


# Opencode's explicit empty state: small models handle the explicit sentence
# better than an absent section.
NO_SKILLS_AVAILABLE = "No skills are currently available."


def build_skills_section(tool_context: "ToolContext") -> str:
    """Render the live skill inventory as an '## Available Skills' section.

    Uses the thread-safe loader (private registry, no shared-state side
    effects), filtered to skills the model may actually invoke. One line per
    skill (``- /<name>: <description>``) plus a usage line pointing at the
    Skill tool. Returns the explicit "No skills are currently available."
    sentence when no invocable skills exist.
    """
    try:
        from ..skills.loader import find_all_skills
    except Exception:
        return NO_SKILLS_AVAILABLE
    try:
        cwd = tool_context.cwd or tool_context.workspace_root
        skills = list(find_all_skills(project_root=cwd))
    except Exception:
        return NO_SKILLS_AVAILABLE
    usable = [
        s
        for s in skills
        if getattr(s, "user_invocable", True)
        and not getattr(s, "disable_model_invocation", False)
    ]
    usable.sort(key=lambda s: s.name.lower())
    if not usable:
        return NO_SKILLS_AVAILABLE
    lines = [
        "## Available Skills",
        "These are the ONLY skills you have. When asked about your skills or",
        "capabilities, list ONLY these - never invent others and never answer",
        "with generic capabilities.",
        "",
        "- Before starting a task, check whether a skill's description matches it. If one does,",
        "  call the Skill tool with its name first, then follow the returned instructions.",
        "- Several skills can apply to one task; load each relevant one.",
        "- Do not call a skill that does not clearly fit, and do not reload one already loaded this task.",
    ]
    for s in usable:
        lines.append(f"- /{s.name}: {_skill_desc(s)}")
    lines.append('To use a skill, call the Skill tool with {"skill": "<name>"}.')
    lines.append("")
    lines.append("## Installing Skills")
    lines.append("A skill is just a folder containing a SKILL.md file. You CAN install")
    lines.append("new skills yourself with your normal tools (Bash, Write) - skills are")
    lines.append("not fixed, and you are not limited to the list above.")
    lines.append("- User skills live in ~/.clawd/skills/<name>/SKILL.md")
    lines.append("- Project skills live in <project>/.clawd/skills/<name>/SKILL.md")
    lines.append("- To download one from GitHub: git clone --depth 1 <repo-url>, then")
    lines.append("  copy the skill folder into one of the locations above.")
    lines.append("- From a skill index (a URL serving an index.json listing skills): run")
    lines.append("  python3 -c \"from src.skills.install import install_from_index;")
    lines.append("  install_from_index('<index-url>', '<skill-name>')\" to download and")
    lines.append("  install it atomically (safe extraction, version tracking).")
    lines.append("- New skills are discovered instantly - no restart needed.")
    lines.append("When the user asks you to download or install a skill, do it - do not")
    lines.append("say you have no mechanism for it.")
    # Few-shot shape example: small models answer "what skills do you have?" from
    # base-model priors unless shown the expected shape. Generic placeholder
    # skills, not a re-list of the full inventory (prompt diet).
    lines.append("")
    lines.append("Example:")
    lines.append("User: what skills do you have?")
    lines.append("Assistant: I have 2 skills:")
    lines.append("- /deploy: deploys the app to production")
    lines.append("- /debug: helps debug failing tests")
    return "\n".join(lines)
