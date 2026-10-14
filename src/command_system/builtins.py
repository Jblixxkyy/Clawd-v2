"""
Built-in commands for Clawd Code.

Implements core commands like /help, /clear, /exit, /skills, etc.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from ..context_system.builder import build_context_prompt
from ..context_system.context_analyzer import (
    analyze_context,
    format_context_as_markdown,
    get_context_window_for_model,
)
from ..context_system.microcompact import microcompact_messages, strip_images_from_messages
from ..cost_tracker import CostTracker
from ..history import HistoryLog
from ..providers.base import BaseProvider
from ..setup import run_setup
from .engine import CommandContext, CommandResult, LocalCommandResult
from .registry import CommandRegistry, get_command_registry, list_commands
from .types import Command, CommandType, CompactionResult, LocalCommand, PromptCommand


# Official Claude Code /init prompts (Simplified)
NEW_INIT_PROMPT = """Set up a CLAUDE.md file for this repo. CLAUDE.md is loaded into every Claude Code session, so it must be concise — only include what Claude would get wrong without it.

## Step 1: Ask what to set up

Use AskUserQuestion to ask the user:
- "Which CLAUDE.md files should /init set up?" with options: "Project CLAUDE.md" | "Personal CLAUDE.local.md" | "Both project + personal"

Use AskUserQuestion to ask:
- "Also set up skills and hooks?" with options: "Skills + hooks" | "Skills only" | "Hooks only" | "Neither, just CLAUDE.md"

## Step 2: Explore the codebase

Use tools to understand the project:
- Read key files: README, package.json, pyproject.toml, Cargo.toml, Makefile, existing CLAUDE.md
- Detect: build/test/lint commands, languages, frameworks, project structure
- Detect: code style rules, required env vars, gotchas
- Check for formatter config (ruff, black, prettier, etc.)

## Step 3: Ask follow-up questions (if needed)

Use AskUserQuestion to ask only things you CAN'T figure out from code:
- User's role (e.g., "backend engineer", "new hire")
- Non-obvious workflows or commands
- Communication preferences (terse vs detailed)

## Step 4: Write CLAUDE.md

Write a minimal CLAUDE.md at the project root.

Include:
- Build/test/lint commands that aren't standard (e.g., "uv run pytest" not just "pytest")
- Code style rules that DIFFER from defaults
- Required env vars or setup steps
- Non-obvious gotchas

Exclude:
- File structure (Claude can discover this)
- Standard conventions Claude already knows
- Generic advice

Prefix with:
```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```

If CLAUDE.md exists: read it, propose specific improvements.

## Step 5: Write CLAUDE.local.md (if user chose personal or both)

Write CLAUDE.local.md at project root. Add it to .gitignore.

Include:
- User's role and familiarity with codebase
- Personal sandbox URLs, test accounts
- Communication preferences

## Step 6: Create skills (if user chose skills)

Create skills at `.claude/skills/<name>/SKILL.md`:
```yaml
---
name: <skill-name>
description: <what it does>
---

<Instructions>
```

## Step 7: Summary

Tell the user what was set up and suggest any additional optimizations."""

# Fallback prompt for simpler initialization
OLD_INIT_PROMPT = """Please analyze this codebase and create a CLAUDE.md file, which will be given to future instances of Claude Code to operate in this repository.

What to add:
1. Commands that will be commonly used, such as how to build, lint, and run tests. Include the necessary commands to develop in this codebase, such as how to run a single test.
2. High-level code architecture and structure so that future instances can be productive more quickly. Focus on the "big picture" architecture that requires reading multiple files to understand.

Usage notes:
- If there's already a CLAUDE.md, suggest improvements to it.
- When you make the initial CLAUDE.md, do not repeat yourself and do not include obvious instructions like "Provide helpful error messages to users", "Write unit tests for all new utilities", "Never include sensitive information (API keys, tokens) in code or commits".
- Avoid listing every component or file structure that can be easily discovered.
- Don't include generic development practices.
- If there are Cursor rules (in .cursor/rules/ or .cursorrules) or Copilot rules (in .github/copilot-instructions.md), make sure to include the important parts.
- If there is a README.md, make sure to include the important parts.
- Do not make up information such as "Common Development Tasks", "Tips for Development", "Support and Documentation" unless this is expressly included in other files that you read.
- Be sure to prefix the file with the following text:

```
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
```"""


def clear_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /clear command - clear conversation history.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    if hasattr(context.conversation, "clear"):
        context.conversation.clear()

    if hasattr(context.history, "events"):
        context.history.events.clear()

    return LocalCommandResult(
        type="text",
        value="Conversation cleared.",
    )


def help_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /help command - show available commands.

    Args:
        args: Command arguments (optional search query)
        context: Command context

    Returns:
        LocalCommandResult
    """
    registry = get_command_registry()
    query = args.strip()

    if query:
        commands = registry.find_commands(query, limit=50)
        header = f"Commands matching '{query}':"
    else:
        commands = registry.list_commands(include_hidden=False)
        header = "Available commands:"

    lines = [header, ""]

    for cmd in commands:
        alias_str = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
        lines.append(f"  /{cmd.name}{alias_str}")
        lines.append(f"      {cmd.description}")
        if cmd.argument_hint:
            lines.append(f"      Usage: /{cmd.name} {cmd.argument_hint}")
        lines.append("")

    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


def skills_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /skills command - list available skills.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    try:
        from ..skills.loader import get_all_skills
        # Pass project_root to find skills in project directories
        skills = get_all_skills(project_root=context.cwd or context.workspace_root)
    except Exception:
        skills = []

    if not skills:
        return LocalCommandResult(
            type="text",
            value="No skills available. Add skills to ~/.clawd/skills/ or ./.clawd/skills/.",
        )

    lines = ["Available skills:", ""]
    for skill in skills:
        lines.append(f"  {skill.name}")
        lines.append(f"      {skill.description}")
        if skill.when_to_use:
            lines.append(f"      When to use: {skill.when_to_use}")
        lines.append("")

    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


def exit_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /exit command - exit the application.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    return LocalCommandResult(
        type="text",
        value="Goodbye!",
    )


def cost_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /cost command - show session cost.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    tracker = context.cost_tracker
    if tracker is None:
        return LocalCommandResult(
            type="text",
            value="Cost tracking not available.",
        )

    lines = ["Session Cost:", ""]
    lines.append(f"  Total units: {tracker.total_units}")

    if tracker.events:
        lines.append("")
        lines.append("  Recent events:")
        for event in tracker.events[-10:]:
            lines.append(f"    - {event}")

    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


def context_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /context command - show token usage breakdown.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult with Markdown table of context usage
    """
    try:
        # Get conversation messages in API format
        conversation_api: list[dict[str, Any]] = []
        if hasattr(context.conversation, "get_messages"):
            conversation_api = context.conversation.get_messages()
        elif hasattr(context.conversation, "messages"):
            # Fall back for simple mock conversations
            for msg in context.conversation.messages:
                role = getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', '')
                conversation_api.append({"role": role, "content": content})

        # Get system prompt from config
        system_prompt = context.config.get("system_prompt", "")

        # Get tool schemas from config
        tool_schemas = context.config.get("tool_schemas", [])

        # Get MCP tools info from config
        mcp_tools = context.config.get("mcp_tools", [])

        # Get custom agents info from config
        custom_agents = context.config.get("custom_agents", [])

        # Get CLAUDE.md content
        claude_md_content = ""
        try:
            from ..context_system.claude_md import load_claude_md_context
            claude_md = load_claude_md_context(context.workspace_root, cwd=context.cwd)
            if claude_md.files:
                parts = []
                for f in claude_md.files:
                    if hasattr(f, 'read_text'):
                        parts.append(f.read_text())
                    elif hasattr(f, 'content'):
                        parts.append(f.content)
                    elif isinstance(f, dict):
                        parts.append(f.get("content", ""))
                claude_md_content = "\n".join(parts)
        except Exception:
            pass

        # Get model from config
        model = context.config.get("model", "claude-sonnet-4-6")

        # Get skills info from config
        skills_frontmatter_tokens = context.config.get("skills_tokens", 0)
        skills_count = context.config.get("skills_count", 0)

        # Get API usage from cost tracker
        api_usage = None
        if hasattr(context.cost_tracker, "last_usage"):
            api_usage = context.cost_tracker.last_usage

        # Get auto-compact info from config
        auto_compact_threshold = context.config.get("auto_compact_threshold")
        is_auto_compact_enabled = context.config.get("is_auto_compact_enabled", False)

        data = analyze_context(
            conversation_api_messages=conversation_api,
            model=model,
            system_prompt=system_prompt,
            tool_schemas=tool_schemas,
            claude_md_content=claude_md_content,
            skills_frontmatter_tokens=skills_frontmatter_tokens,
            skills_count=skills_count,
            api_usage=api_usage,
            mcp_tools=mcp_tools,
            custom_agents=custom_agents,
            auto_compact_threshold=auto_compact_threshold,
            is_auto_compact_enabled=is_auto_compact_enabled,
        )

        markdown = format_context_as_markdown(data)
        return LocalCommandResult(type="text", value=markdown)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return LocalCommandResult(type="text", value=f"Context analysis failed: {e}")


async def _compact_async(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Async implementation of compact command.
    """
    if not hasattr(context.conversation, "messages"):
        return LocalCommandResult(
            type="text",
            value="No conversation to compact.",
        )

    messages = context.conversation.messages
    if len(messages) < 2:
        return LocalCommandResult(
            type="text",
            value=f"Nothing to compact: only {len(messages)} messages.",
        )

    # Get provider from config
    provider = context.config.get("provider")
    if provider is None:
        return LocalCommandResult(
            type="text",
            value="Compact requires an LLM provider (not available in this context).",
        )

    model = context.config.get("model", "claude-sonnet-4-6")
    arg_tokens = (args or "").split()
    if "--hard" in arg_tokens:
        # Legacy behavior: clear everything, no summarization.
        try:
            context.conversation.clear()
        except Exception:
            if hasattr(context.conversation, "messages"):
                context.conversation.messages.clear()
        return LocalCommandResult(
            type="text",
            value="Conversation cleared (--hard).",
        )
    custom_instructions = " ".join(t for t in arg_tokens if t != "--hard") or None
    workspace_root = getattr(context, "workspace_root", None)

    try:
        # Import here to avoid circular imports
        from ..compact_service.service import compact_conversation

        result = await compact_conversation(
            conversation=context.conversation,
            provider=provider,
            model=model,
            custom_instructions=custom_instructions,
            trigger="manual",
            workspace_root=workspace_root,
        )
        return LocalCommandResult(
            type="compact",
            value=result.user_display_message or "Conversation compacted.",
            compaction_result=CompactionResult(
                pre_compact_count=result.pre_compact_count,
                post_compact_count=result.post_compact_count,
                tokens_saved=result.tokens_saved,
                trigger=result.trigger,
                summary_preview=result.summary_text[:200] if len(result.summary_text) > 200 else result.summary_text,
            ),
        )
    except ValueError as e:
        return LocalCommandResult(type="text", value=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return LocalCommandResult(
            type="text",
            value=f"Compact failed: {e}",
        )


def compact_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /compact command - compact conversation context.

    Args:
        args: Command arguments
        context: Command context

    Returns:
        LocalCommandResult
    """
    # --hard: legacy clear-everything behavior, no summarization.
    if "--hard" in (args or "").split():
        try:
            context.conversation.clear()
        except Exception:
            if hasattr(context.conversation, "messages"):
                context.conversation.messages.clear()
        return LocalCommandResult(
            type="text",
            value="Conversation cleared (--hard).",
        )

    # Run the async version in a new event loop
    try:
        loop = asyncio.get_running_loop()
        # If we're already in an async context, we can't use asyncio.run
        # Fall back to sync path
        return _sync_compact_fallback(context)
    except RuntimeError:
        # No running event loop — safe to use asyncio.run
        try:
            return asyncio.run(_compact_async(args, context))
        except Exception as e:
            import traceback
            traceback.print_exc()
            return _sync_compact_fallback(context)


def _sync_compact_fallback(context: CommandContext) -> LocalCommandResult:
    """Synchronous fallback when async provider is not available."""
    if not hasattr(context.conversation, "messages"):
        return LocalCommandResult(type="text", value="No conversation to compact.")

    messages = context.conversation.messages
    if len(messages) < 2:
        return LocalCommandResult(
            type="text",
            value=f"Nothing to compact: only {len(messages)} messages.",
        )

    # Get messages after last boundary
    try:
        from ..compact_service.messages import (
            create_compact_boundary_message,
            create_compact_summary_message,
            get_messages_after_boundary,
            is_compact_boundary_message,
        )
        from ..token_estimation import count_messages_tokens

        after_boundary = get_messages_after_boundary(messages)
        if len(after_boundary) < 2:
            return LocalCommandResult(
                type="text",
                value=f"Nothing to compact: only {len(after_boundary)} messages after boundary.",
            )

        # Count tokens
        api_messages = context.conversation.get_messages()
        pre_tokens = count_messages_tokens(api_messages)

        # Strip images and microcompact
        stripped = strip_images_from_messages(api_messages)
        compacted, saved = microcompact_messages(stripped)

        # Find boundary position
        boundary_indices = [
            i for i, m in enumerate(messages)
            if is_compact_boundary_message(m)
        ]

        if boundary_indices:
            insert_pos = max(boundary_indices) + 1
        else:
            insert_pos = 0

        # Create simple text summary
        summary_parts = [f"Conversation had {len(after_boundary)} messages ({pre_tokens:,} tokens)."]
        summary_text = "\n".join(summary_parts)

        boundary = create_compact_boundary_message(
            trigger="manual",
            pre_compact_token_count=pre_tokens,
        )
        summary = create_compact_summary_message(summary_text)

        # Rebuild conversation
        if insert_pos == 0:
            context.conversation.messages.clear()
            context.conversation.messages.append(boundary)
            context.conversation.messages.append(summary)
        else:
            context.conversation.messages = list(messages[:insert_pos])
            context.conversation.messages.append(boundary)
            context.conversation.messages.append(summary)

        # Re-inject survival context (CLAUDE.md, memory, recent files).
        # Defensive: never raises; missing files are skipped.
        try:
            from ..compact_service.service import reinject_survival_context
            reinject_survival_context(
                context.conversation,
                workspace_root=getattr(context, "workspace_root", None),
            )
        except Exception:
            pass

        return LocalCommandResult(
            type="compact",
            value=f"Compacted: removed {len(after_boundary) - 2} messages ({pre_tokens:,} tokens → ~{saved} saved).",
            compaction_result=CompactionResult(
                pre_compact_count=len(messages),
                post_compact_count=len(context.conversation.messages),
                tokens_saved=saved,
                trigger="manual",
                summary_preview=summary_text[:200],
            ),
        )
    except Exception as e:
        # Last resort: just clear old messages
        original_count = len(messages)
        if original_count > 10:
            context.conversation.messages = list(messages[-10:])
            return LocalCommandResult(
                type="compact",
                value=f"Compacted: removed {original_count - 10} messages (fallback mode).",
                compaction_result=CompactionResult(
                    pre_compact_count=original_count,
                    post_compact_count=10,
                    tokens_saved=0,
                    trigger="manual",
                ),
            )
        return LocalCommandResult(
            type="text",
            value="Nothing to compact.",
        )


# Command definitions
HELP_COMMAND = LocalCommand(
    name="help",
    description="Show available commands",
    aliases=["?"],
    argument_hint="[search_query]",
    supports_non_interactive=True,
)

CLEAR_COMMAND = LocalCommand(
    name="clear",
    description="Clear conversation history",
    aliases=["reset", "new"],
    supports_non_interactive=False,
)

EXIT_COMMAND = LocalCommand(
    name="exit",
    description="Exit the application",
    aliases=["quit", "q"],
    supports_non_interactive=True,
)

SKILLS_COMMAND = LocalCommand(
    name="skills",
    description="List available skills",
    argument_hint="",
    supports_non_interactive=True,
)

COST_COMMAND = LocalCommand(
    name="cost",
    description="Show session cost and usage",
    argument_hint="",
    supports_non_interactive=True,
)

CONTEXT_COMMAND = LocalCommand(
    name="context",
    description="Show current workspace context",
    argument_hint="",
    supports_non_interactive=True,
)

COMPACT_COMMAND = LocalCommand(
    name="compact",
    description="Compact conversation to save context space",
    argument_hint="",
    supports_non_interactive=True,
)

# ---------------------------------------------------------------------------
# /doctor — local diagnostics (Q7). Pure helpers take an explicit *home*
# so tests can point them at a temp dir; the command itself uses Path.home().
# ---------------------------------------------------------------------------

_DOCTOR_PROVIDER_ENV_VARS = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "nvidia": ("NVIDIA_API_KEY", "NVIDIA_NIM_API_KEY"),
    "glm": ("GLM_API_KEY", "ZHIPU_API_KEY"),
    "minimax": ("MINIMAX_API_KEY",),
}


def _doctor_home(home=None) -> Path:
    return Path(home).expanduser() if home is not None else Path.home()


def _doctor_load_json(path: Path):
    """Load a JSON file; return None when missing or unparseable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _doctor_check(ok: bool, detail: str, fix: str | None = None) -> dict:
    return {"ok": ok, "detail": detail, "fix": fix}


def doctor_check_provider_keys(home=None) -> dict:
    """Default provider must have an API key (env var or config.json)."""
    from ..providers import PROVIDER_INFO

    home = _doctor_home(home)
    config = _doctor_load_json(home / ".clawd" / "config.json") or {}
    providers_cfg = config.get("providers") or {}
    default = config.get("default_provider") or "anthropic"

    statuses = []
    default_ok = False
    for name in PROVIDER_INFO:
        env_hit = next(
            (var for var in _DOCTOR_PROVIDER_ENV_VARS.get(name, ()) if os.environ.get(var)),
            None,
        )
        cfg_key = (providers_cfg.get(name) or {}).get("api_key")
        has_key = bool(env_hit) or bool(cfg_key)
        statuses.append(f"{name}: {'key set' + (' via ' + env_hit if env_hit else '') if has_key else 'no key'}")
        if name == default:
            default_ok = has_key

    detail = f"default provider '{default}' — " + "; ".join(statuses)
    fix = None
    if not default_ok:
        env_vars = "/".join(_DOCTOR_PROVIDER_ENV_VARS.get(default, ("<PROVIDER>_API_KEY",)))
        fix = (
            f"No API key for the default provider '{default}'. "
            f"Set it in ~/.clawd/config.json (providers.{default}.api_key) "
            f"or export {env_vars}."
        )
    return _doctor_check(default_ok, detail, fix)


def doctor_check_config_valid(home=None) -> dict:
    """~/.clawd/config.json must exist and parse."""
    home = _doctor_home(home)
    path = home / ".clawd" / "config.json"
    if not path.exists():
        return _doctor_check(
            False,
            f"{path} not found.",
            "Start the app once to generate a default config, then add your provider API key.",
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return _doctor_check(
            False,
            f"{path} is not valid JSON: {exc}.",
            f"Fix the JSON syntax in {path}, or delete it to regenerate defaults on next start.",
        )
    if not isinstance(data, dict) or "providers" not in data:
        return _doctor_check(
            False,
            f"{path} is missing the 'providers' section.",
            f"Delete {path} to regenerate defaults on next start.",
        )
    return _doctor_check(True, f"{path} parses and has a providers section.")


def doctor_check_clawd_dir_writable(home=None) -> dict:
    """~/.clawd must exist and accept writes."""
    home = _doctor_home(home)
    clawd = home / ".clawd"
    probe = clawd / ".doctor-write-test"
    try:
        clawd.mkdir(parents=True, exist_ok=True)
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return _doctor_check(
            False,
            f"{clawd} is not writable: {exc}.",
            f"Check ownership/permissions of {clawd} (and disk space).",
        )
    return _doctor_check(True, f"{clawd} is writable.")


def doctor_check_bash(home=None) -> dict:
    """bash must be on PATH — BashTool hardcodes `bash -lc`."""
    if shutil.which("bash"):
        return _doctor_check(True, f"bash found: {shutil.which('bash')}.")
    return _doctor_check(
        False,
        "bash not found on PATH.",
        "BashTool runs every command as `bash -lc`. On Windows install Git Bash "
        "(https://git-scm.com/downloads) and make sure bash is on PATH, or use "
        "the PowerShell tool instead. macOS/Linux ship bash by default.",
    )


def doctor_check_mcp_servers(home=None) -> dict:
    """Every MCP server in ~/.clawd/mcp.json must point at a runnable command."""
    home = _doctor_home(home)
    path = home / ".clawd" / "mcp.json"
    if not path.exists():
        return _doctor_check(True, "no MCP servers configured (mcp.json absent).")
    data = _doctor_load_json(path)
    if data is None:
        return _doctor_check(
            False,
            f"{path} is not valid JSON.",
            f"Fix the JSON syntax in {path} (see /mcp for the expected shape).",
        )
    servers = data.get("mcpServers") or {}
    if not servers:
        return _doctor_check(True, "mcp.json has no servers configured.")
    problems = []
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            problems.append(f"{name}: entry is not an object")
            continue
        command = str(spec.get("command") or "")
        if not command:
            problems.append(f"{name}: no 'command' configured")
            continue
        first = shlex.split(command)[0] if command.strip() else ""
        runnable = (
            os.path.isabs(first) and os.path.isfile(first) and os.access(first, os.X_OK)
        ) or (not os.path.isabs(first) and shutil.which(first))
        if not runnable:
            problems.append(f"{name}: command not runnable: {first!r}")
    if problems:
        return _doctor_check(
            False,
            f"{len(problems)} MCP server(s) unreachable: " + "; ".join(problems),
            "Install the missing server command(s) or fix the entries in "
            "~/.clawd/mcp.json (see /mcp).",
        )
    return _doctor_check(True, f"{len(servers)} MCP server(s) configured and runnable.")


def _doctor_hook_script_problems(command: str) -> list[str]:
    """Problems with one hooks.json command string (missing script/interpreter)."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    problems = []
    if not tokens:
        return ["empty command"]
    interp, rest = tokens[0], tokens[1:]
    script_tokens = [tok for tok in rest if not tok.startswith("-")]
    scripts = [tok for tok in script_tokens if tok.endswith((".py", ".sh", ".js", ".mjs", ".ts"))]
    if scripts:
        # interpreter + script form, e.g. "python3 /path/to/hook.py"
        if not (os.path.isabs(interp) or shutil.which(interp)):
            problems.append(f"interpreter not found: {interp!r}")
        for script in scripts:
            spath = Path(script).expanduser()
            if not spath.is_file():
                problems.append(f"hook script not found: {script}")
    else:
        # bare executable form
        if not (os.path.isabs(interp) and os.access(interp, os.X_OK) or shutil.which(interp)):
            problems.append(f"hook command not runnable: {interp!r}")
    return problems


def doctor_check_hook_scripts(home=None) -> dict:
    """Hook commands in ~/.clawd/hooks.json must reference runnable scripts."""
    home = _doctor_home(home)
    path = home / ".clawd" / "hooks.json"
    data = _doctor_load_json(path)
    if data is None:
        return _doctor_check(True, "no hooks configured (hooks.json absent or invalid).")
    problems = []
    for event in ("PreToolUse", "PostToolUse"):
        raw = data.get(event)
        if raw is None:
            continue
        entries = [raw] if isinstance(raw, str) else (raw if isinstance(raw, list) else [])
        for entry in entries:
            command = entry if isinstance(entry, str) else (entry.get("command") if isinstance(entry, dict) else None)
            if not command:
                problems.append(f"{event}: entry without a command: {entry!r}")
                continue
            for prob in _doctor_hook_script_problems(str(command)):
                problems.append(f"{event}: {prob}")
    if problems:
        return _doctor_check(
            False,
            f"{len(problems)} hook problem(s): " + "; ".join(problems[:5]),
            "Fix the paths in ~/.clawd/hooks.json — hook scripts must exist and be runnable. "
            "Hooks fail open, so a broken hook only warns, but it also never runs.",
        )
    return _doctor_check(True, "hook scripts in hooks.json are runnable.")


def doctor_check_sessions_dir(home=None) -> dict:
    """~/.clawd/sessions must exist and be writable."""
    home = _doctor_home(home)
    sessions = home / ".clawd" / "sessions"
    if not sessions.is_dir():
        return _doctor_check(
            False,
            f"{sessions} does not exist.",
            f"Create it with: mkdir -p {sessions}",
        )
    if not os.access(sessions, os.W_OK):
        return _doctor_check(
            False,
            f"{sessions} is not writable.",
            f"Check ownership/permissions of {sessions}.",
        )
    return _doctor_check(True, f"{sessions} exists and is writable.")


def doctor_run_checks(home=None) -> list[dict]:
    """Run every diagnostic; each item is {name, ok, detail, fix}."""
    checks = [
        ("provider keys", doctor_check_provider_keys),
        ("config valid", doctor_check_config_valid),
        ("~/.clawd writable", doctor_check_clawd_dir_writable),
        ("bash present", doctor_check_bash),
        ("MCP servers", doctor_check_mcp_servers),
        ("hook scripts", doctor_check_hook_scripts),
        ("sessions dir", doctor_check_sessions_dir),
    ]
    return [{"name": name, **fn(home)} for name, fn in checks]


def doctor_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """
    Handle /doctor command - diagnose the local setup.

    Args:
        args: Command arguments (ignored)
        context: Command context

    Returns:
        LocalCommandResult
    """
    results = doctor_run_checks()
    lines = ["Clawd-Code diagnostics (/doctor)", ""]
    failed = 0
    for res in results:
        mark = "\u2713" if res["ok"] else "\u2717"
        lines.append(f"{mark} {res['name']}: {res['detail']}")
        if not res["ok"]:
            failed += 1
            if res["fix"]:
                lines.append(f"  Fix: {res['fix']}")
        lines.append("")
    lines.append(f"{len(results) - failed}/{len(results)} checks passed.")
    if failed:
        lines.append("Fix the items above, then run /doctor again.")
    return LocalCommandResult(
        type="text",
        value="\n".join(lines),
    )


DOCTOR_COMMAND = LocalCommand(
    name="doctor",
    description="Diagnose your setup: provider keys, config, bash, MCP servers, hooks, sessions",
    aliases=["checkup"],
    argument_hint="",
    supports_non_interactive=True,
)

INIT_COMMAND = PromptCommand(
    name="init",
    description="Initialize new CLAUDE.md file(s) and optional skills/hooks with codebase documentation",
    markdown_content=NEW_INIT_PROMPT,
    progress_message="analyzing your codebase",
    content_length=0,
    source="builtin",
)


# Synchronous versions for REPL integration
def execute_command_sync(cmd_name: str, args: str, context: CommandContext) -> tuple[bool, str | None, str | None]:
    """
    Execute a command synchronously.

    Returns:
        Tuple of (success: bool, result_text: str | None, error: str | None)
    """
    cmd = None
    for builtin_cmd in get_builtin_commands():
        if builtin_cmd.name.lower() == cmd_name.lower() or cmd_name.lower() in [a.lower() for a in builtin_cmd.aliases]:
            cmd = builtin_cmd
            break

    if cmd is None:
        return False, None, f"Unknown command: {cmd_name}"

    try:
        # This is a synchronous wrapper - we directly call the underlying function
        # instead of going through the async call() method
        if cmd is HELP_COMMAND:
            result = help_command_call(args, context)
        elif cmd is CLEAR_COMMAND:
            result = clear_command_call(args, context)
        elif cmd is EXIT_COMMAND:
            result = exit_command_call(args, context)
        elif cmd is SKILLS_COMMAND:
            result = skills_command_call(args, context)
        elif cmd is COST_COMMAND:
            result = cost_command_call(args, context)
        elif cmd is CONTEXT_COMMAND:
            result = context_command_call(args, context)
        elif cmd is COMPACT_COMMAND:
            result = compact_command_call(args, context)
        elif cmd is DOCTOR_COMMAND:
            result = doctor_command_call(args, context)
        else:
            return False, None, f"Command not implemented for sync execution: {cmd_name}"

        return True, result.value, None
    except Exception as e:
        return False, None, str(e)


# Set the call implementations
HELP_COMMAND.set_call(help_command_call)
CLEAR_COMMAND.set_call(clear_command_call)
EXIT_COMMAND.set_call(exit_command_call)
SKILLS_COMMAND.set_call(skills_command_call)
COST_COMMAND.set_call(cost_command_call)
CONTEXT_COMMAND.set_call(context_command_call)
COMPACT_COMMAND.set_call(compact_command_call)
DOCTOR_COMMAND.set_call(doctor_command_call)


def get_builtin_commands() -> list[Command]:
    """Get all built-in commands."""
    return [
        HELP_COMMAND,
        CLEAR_COMMAND,
        EXIT_COMMAND,
        SKILLS_COMMAND,
        COST_COMMAND,
        CONTEXT_COMMAND,
        COMPACT_COMMAND,
        DOCTOR_COMMAND,
        INIT_COMMAND,
    ]


def register_builtin_commands(registry: CommandRegistry | None = None) -> None:
    """
    Register all built-in commands.

    Args:
        registry: Optional registry to use (uses global if None)
    """
    reg = registry or get_command_registry()
    for cmd in get_builtin_commands():
        reg.register(cmd)


async def execute_command_async(
    cmd_name: str,
    args: str,
    context: CommandContext,
) -> CommandResult:
    """
    Execute a command asynchronously.

    This function handles both LocalCommand and PromptCommand types.
    For PromptCommand, it returns the prompt content that should be sent to the LLM.

    Args:
        cmd_name: Name of the command to execute
        args: Arguments for the command
        context: Command context

    Returns:
        CommandResult with the execution result
    """
    from .engine import CommandEngine

    registry = get_command_registry()
    cmd = registry.get(cmd_name)

    if cmd is None:
        return CommandResult.error(cmd_name, f"Unknown command: {cmd_name}")

    if not cmd.is_enabled():
        return CommandResult.error(cmd_name, f"Command {cmd_name} is disabled")

    engine = CommandEngine(
        registry=registry,
        workspace_root=context.workspace_root,
        context=context,
    )

    # Create a fake command input string for the engine
    command_input = f"/{cmd_name}"
    if args:
        command_input += f" {args}"

    return await engine.execute(command_input)


# ---------------------------------------------------------------------------
# /init — CLAUDE.md draft generation (workspace inspection)
# ---------------------------------------------------------------------------

import json
import re


def _toml_string_field(text: str, field: str) -> Optional[str]:
    """Best-effort extraction of a simple `field = "value"` from TOML text."""
    import re

    m = re.search(rf'^\s*{re.escape(field)}\s*=\s*["\']([^"\']+)["\']', text, re.M)
    return m.group(1).strip() if m else None


def _toml_string_list(text: str, field: str) -> list[str]:
    """Best-effort extraction of a `field = ["a", "b"]` list from TOML text."""
    import re

    m = re.search(rf'{re.escape(field)}\s*=\s*\[(.*?)\]', text, re.S)
    if not m:
        return []
    return re.findall(r'["\']([^"\']+)["\']', m.group(1))


def generate_claude_md_draft(root: str | Path) -> str:
    """Inspect a workspace and generate a CLAUDE.md draft.

    Looks at project manifests (pyproject.toml / package.json / go.mod /
    Cargo.toml), README.md, and the top-level directory layout. Only includes
    facts actually detected in the workspace.

    Args:
        root: Workspace root to inspect.

    Returns:
        Markdown draft string, prefixed with the standard CLAUDE.md header.
    """
    root = Path(root)

    project_name = root.name
    description = ""
    stack_bits: list[str] = []
    commands: list[tuple[str, str]] = []  # (label, command)

    pyproject = root / "pyproject.toml"
    package_json = root / "package.json"
    go_mod = root / "go.mod"
    cargo = root / "Cargo.toml"

    if pyproject.exists():
        try:
            text = pyproject.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        name = _toml_string_field(text, "name")
        desc = _toml_string_field(text, "description")
        requires_python = _toml_string_field(text, "requires-python")
        if name:
            project_name = name
        if desc:
            description = desc
        deps = _toml_string_list(text, "dependencies")
        lang = "Python"
        if requires_python:
            lang += f" ({requires_python})"
        stack_bits.append(lang)
        if deps:
            stack_bits.append("Dependencies: " + ", ".join(deps[:12]))
        test_cmd = "pytest"
        if (root / "uv.lock").exists():
            test_cmd = "uv run pytest"
        commands.append(("Test", f"{test_cmd} tests/"))
        scripts = re.findall(r'^\s*([\w.-]+)\s*=\s*["\']([^"\']+)["\']', text, re.M)
        entry = [v for k, v in scripts if k.lower() in ("clawd", "main", "cli")]
        if entry:
            commands.append(("Run", f"python -m {entry[0]}"))

    if package_json.exists():
        try:
            pkg = json.loads(package_json.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError):
            pkg = {}
        if pkg.get("name"):
            project_name = str(pkg["name"])
        if pkg.get("description") and not description:
            description = str(pkg["description"])
        deps = pkg.get("dependencies") or {}
        if deps:
            stack_bits.append("Node.js — Dependencies: " + ", ".join(sorted(deps)[:12]))
        elif not stack_bits:
            stack_bits.append("Node.js")
        scripts = pkg.get("scripts") or {}
        for label, script_key in (("Test", "test"), ("Build", "build"), ("Lint", "lint")):
            if script_key in scripts:
                cmd = f"npm run {script_key}" if script_key != "test" else "npm test"
                commands.append((label, cmd))

    if go_mod.exists():
        try:
            text = go_mod.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        m = re.search(r"^\s*module\s+(\S+)", text, re.M)
        if m:
            project_name = m.group(1)
        m = re.search(r"^\s*go\s+(\S+)", text, re.M)
        stack_bits.append(f"Go {m.group(1)}" if m else "Go")
        commands.append(("Test", "go test ./..."))
        commands.append(("Build", "go build ./..."))

    if cargo.exists():
        try:
            text = cargo.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        name = _toml_string_field(text, "name")
        if name:
            project_name = name
        stack_bits.append("Rust")
        commands.append(("Test", "cargo test"))
        commands.append(("Build", "cargo build"))

    # README: first non-empty paragraph as the project summary
    if not description:
        readme = root / "README.md"
        if readme.exists():
            try:
                lines = readme.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            para: list[str] = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if not stripped:
                    if para:
                        break
                    continue
                para.append(stripped)
            description = " ".join(para)[:400]

    # Directory layout: top-level directories only
    dirs: list[str] = []
    try:
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir() and not child.name.startswith("."):
                dirs.append(child.name)
    except OSError:
        pass

    lines = [
        "# CLAUDE.md",
        "",
        "This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.",
        "",
        f"## {project_name}",
        "",
    ]
    if description:
        lines += [description, ""]
    if stack_bits:
        lines += ["### Stack", ""]
        lines += [f"- {bit}" for bit in stack_bits]
        lines += [""]
    if commands:
        lines += ["### Commands", ""]
        seen: set[str] = set()
        for label, cmd in commands:
            if cmd not in seen:
                seen.add(cmd)
                lines.append(f"- {label}: `{cmd}`")
        lines += [""]
    if dirs:
        lines += ["### Structure", ""]
        lines += [f"- `{d}/`" for d in dirs[:20]]
        lines += [""]

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# /model — provider/model inventory from src/config.py
# ---------------------------------------------------------------------------


def list_configured_providers() -> list[dict[str, Any]]:
    """List providers with models, config state, and the current default.

    Reads from src.config (load_config / get_default_provider) and the
    src.providers PROVIDER_INFO registry.

    Returns:
        List of dicts with keys: name, label, default_model, available_models,
        api_key_configured, is_default.
    """
    from ..config import load_config
    from ..providers import PROVIDER_INFO

    config = load_config()
    providers_cfg = config.get("providers") or {}
    default_provider = config.get("default_provider")

    out: list[dict[str, Any]] = []
    for name, info in PROVIDER_INFO.items():
        pcfg = providers_cfg.get(name) or {}
        out.append(
            {
                "name": name,
                "label": info.get("label", name),
                "default_model": pcfg.get("default_model") or info.get("default_model"),
                "available_models": list(info.get("available_models") or []),
                "api_key_configured": bool(pcfg.get("api_key")),
                "is_default": name == default_provider,
            }
        )
    return out


# ---------------------------------------------------------------------------
# /mcp — MCP server config (read-only list + validated add)
# ---------------------------------------------------------------------------

MCP_CONFIG_FILENAME = "mcp.json"


def get_mcp_config_path() -> Path:
    """Path to the MCP servers config file (~/.clawd/mcp.json)."""
    config_dir = Path.home() / ".clawd"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / MCP_CONFIG_FILENAME


def load_mcp_config() -> dict[str, Any]:
    """Load MCP config. Missing or corrupt files yield an empty config."""
    path = get_mcp_config_path()
    if not path.exists():
        return {"mcpServers": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"mcpServers": {}}
    if not isinstance(data, dict):
        return {"mcpServers": {}}
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        data["mcpServers"] = {}
    return data


def save_mcp_config(config: dict[str, Any]) -> None:
    """Persist MCP config to disk."""
    path = get_mcp_config_path()
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def list_mcp_servers() -> list[dict[str, Any]]:
    """List configured MCP servers.

    Returns:
        List of dicts with keys: name, command, args (list of str), url.
    """
    servers = load_mcp_config().get("mcpServers") or {}
    out: list[dict[str, Any]] = []
    for name, spec in servers.items():
        if not isinstance(spec, dict):
            continue
        args = spec.get("args") or []
        out.append(
            {
                "name": name,
                "command": str(spec.get("command") or ""),
                "args": [str(a) for a in args] if isinstance(args, list) else [],
                "url": str(spec.get("url") or ""),
            }
        )
    return sorted(out, key=lambda s: s["name"].lower())


def add_mcp_server(name: str, command: str, args: list[str] | None = None) -> dict[str, Any]:
    """Add a stdio MCP server to the config.

    If command looks like http(s)://, it is stored as a remote {url} server.
    """
    name = (name or "").strip()
    command = (command or "").strip()
    arg_list = [str(a) for a in (args or []) if str(a).strip()]

    if not name:
        raise ValueError("Server name must be non-empty.")
    if not command:
        raise ValueError("Server command must be non-empty.")

    config = load_mcp_config()
    servers = config.setdefault("mcpServers", {})
    if name in servers:
        raise ValueError(f"MCP server '{name}' is already configured.")

    if command.lower().startswith(("http://", "https://")) and not arg_list:
        entry: dict[str, Any] = {"url": command}
        servers[name] = entry
        save_mcp_config(config)
        return {"name": name, "command": "", "args": [], "url": command}

    entry = {"command": command}
    if arg_list:
        entry["args"] = arg_list
    servers[name] = entry
    save_mcp_config(config)
    return {"name": name, "command": command, "args": arg_list, "url": ""}


def add_mcp_server_url(name: str, url: str) -> dict[str, Any]:
    """Add a remote HTTP/SSE MCP server."""
    name = (name or "").strip()
    url = (url or "").strip()
    if not name:
        raise ValueError("Server name must be non-empty.")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://.")
    config = load_mcp_config()
    servers = config.setdefault("mcpServers", {})
    if name in servers:
        raise ValueError(f"MCP server '{name}' is already configured.")
    servers[name] = {"url": url}
    save_mcp_config(config)
    return {"name": name, "command": "", "args": [], "url": url}


def _derive_mcp_name(url: str, hint: str | None = None) -> str:
    import re
    from urllib.parse import urlsplit

    if hint and hint.strip():
        base = re.sub(r"[^a-zA-Z0-9-]+", "-", hint.strip()).strip("-").lower()
        if base:
            return base[:64]
    try:
        parts = [p for p in urlsplit(url).path.split("/") if p]
        if parts:
            last = parts[-1]
            last = re.sub(r"\.git$", "", last, flags=re.I)
            base = re.sub(r"[^a-zA-Z0-9-]+", "-", last).strip("-").lower()
            if base:
                return base[:64]
    except Exception:
        pass
    return "mcp-server"


def _fetch_url_text(url: str, timeout: int = 12, max_chars: int = 60000) -> str:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "clawd-mcp-installer/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read(max_chars + 1)
    text = raw[:max_chars].decode("utf-8", errors="replace")
    return text


def auto_install_mcp_from_link(url: str, name_hint: str | None = None) -> dict[str, Any]:
    """Install an MCP server from just a link (GitHub repo, README, install-mcp URL, or remote endpoint).

    Figures out stdio vs remote, writes ~/.clawd/mcp.json, returns the entry plus a note.
    """
    import base64
    import re
    from urllib.parse import parse_qs, urlsplit

    url = (url or "").strip().rstrip(".,;:!?)]}")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("Provide an http(s) link to the MCP server.")
    name = _derive_mcp_name(url, name_hint)

    # 1) install-mcp config link: ?config=<base64 {command,args,name}>
    try:
        qs = parse_qs(urlsplit(url).query)
        cfg_b64 = (qs.get("config") or [""])[0]
        if cfg_b64:
            padded = cfg_b64 + "=" * (-len(cfg_b64) % 4)
            cfg = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8", errors="replace"))
            cmd = str(cfg.get("command") or "").strip()
            args = [str(a) for a in (cfg.get("args") or []) if str(a).strip()]
            cfg_name = str(cfg.get("name") or name).strip() or name
            if cmd:
                entry = add_mcp_server(cfg_name, cmd, args)
                entry["note"] = f"installed from install-mcp link: {url}"
                return entry
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        pass  # fall through to GitHub heuristics

    # 2) GitHub owner/repo(.git) links: fetch package.json / server.json / README
    m = re.match(r"https?://github\.com/([^/]+)/([^/#?]+?)(?:\.git)?/?(?:[#?].*)?$", url, re.I)
    if m:
        owner, repo = m.group(1), re.sub(r"\.git$", "", m.group(2), flags=re.I)
        repo_name = _derive_mcp_name(repo, name_hint)
        for branch in ("main", "master"):
            base = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"
            # package.json -> npm package -> npx -y <pkg>
            try:
                pkg = json.loads(_fetch_url_text(f"{base}/package.json"))
                npm_name = str(pkg.get("name") or "").strip()
                if npm_name and "/" not in npm_name or npm_name.startswith("@"):
                    entry = add_mcp_server(repo_name, "npx", ["-y", npm_name])
                    entry["note"] = f"installed from GitHub {owner}/{repo} (npm package {npm_name})"
                    return entry
            except Exception:
                pass
            # server.json (MCP registry metadata) may carry the run command
            try:
                srv = json.loads(_fetch_url_text(f"{base}/server.json"))
                for key in ("command", "run", "start"):
                    if isinstance(srv.get(key), str) and srv[key].strip():
                        entry = add_mcp_server(repo_name, srv[key].strip(), [])
                        entry["note"] = f"installed from {owner}/{repo}/server.json"
                        return entry
            except Exception:
                pass
            # README: look for install-mcp config link or npx/uvx one-liner
            try:
                readme = _fetch_url_text(f"{base}/README.md")
                cfg_m = re.search(r"install-mcp/\?config=([A-Za-z0-9\-_=]+)", readme)
                if cfg_m:
                    return auto_install_mcp_from_link(f"https://adamjones.me/install-mcp/?config={cfg_m.group(1)}", repo_name)
                npx_m = re.search(r"npx\s+(?:-y\s+)?(@?[A-Za-z0-9@/_.-]+)", readme)
                if npx_m:
                    pkg_guess = npx_m.group(1).strip().rstrip(".,;:'\")]}")
                    if pkg_guess and len(pkg_guess) < 120 and pkg_guess not in {"-y"}:
                        try:
                            entry = add_mcp_server(repo_name, "npx", ["-y", pkg_guess])
                            entry["note"] = f"installed from {owner}/{repo} README (npx {pkg_guess})"
                            return entry
                        except ValueError:
                            pass
                uvx_m = re.search(r"uvx\s+([A-Za-z0-9@/_.-]+)", readme)
                if uvx_m:
                    tool = uvx_m.group(1).strip().rstrip(".,;:'\")]}")
                    if tool:
                        try:
                            entry = add_mcp_server(repo_name, "uvx", [tool])
                            entry["note"] = f"installed from {owner}/{repo} README (uvx {tool})"
                            return entry
                        except ValueError:
                            pass
            except Exception:
                pass
        # Fallback for GitHub: npx -y <repo> (works for JS MCPs like computer-use-mcp)
        try:
            entry = add_mcp_server(repo_name, "npx", ["-y", repo])
            entry["note"] = f"installed from GitHub {owner}/{repo} (default npx -y {repo}; fix in /mcp if the README says otherwise)"
            return entry
        except ValueError as exc:
            raise ValueError(str(exc))

    # 3) Anything else http(s): treat as a remote MCP endpoint.
    entry = add_mcp_server_url(name, url)
    entry["note"] = f"installed as remote MCP endpoint: {url}"
    return entry


# ---------------------------------------------------------------------------
# /resume — saved session inventory
# ---------------------------------------------------------------------------


def get_sessions_dir() -> Path:
    """Directory where the agent Session class persists sessions."""
    return Path.home() / ".clawd" / "sessions"


def list_saved_sessions() -> list[dict[str, Any]]:
    """List saved sessions with timestamps, newest first.

    Mirrors the persistence format written by src.agent.Session.save()
    (session_id, provider, model, conversation, created_at, updated_at,
    workspace). Corrupt files are skipped.

    Returns:
        List of dicts with keys: session_id, provider, model, created_at,
        updated_at, workspace, message_count.
    """
    directory = get_sessions_dir()
    if not directory.exists():
        return []

    sessions: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not data.get("session_id"):
            continue
        conv = data.get("conversation")
        messages = conv.get("messages") if isinstance(conv, dict) else []
        sessions.append(
            {
                "session_id": str(data.get("session_id")),
                "provider": str(data.get("provider") or "?"),
                "model": str(data.get("model") or "?"),
                "created_at": str(data.get("created_at") or ""),
                "updated_at": str(data.get("updated_at") or ""),
                "workspace": data.get("workspace"),
                "message_count": len(messages) if isinstance(messages, list) else 0,
            }
        )

    sessions.sort(key=lambda s: s["updated_at"] or s["created_at"], reverse=True)
    return sessions
