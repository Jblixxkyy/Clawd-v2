"""
Compact service for /compact command.

Handles the full compaction flow:
1. Pre-process messages (microcompact, strip images)
2. Call LLM to generate summary
3. Return boundary marker + summary messages
4. Update conversation in place
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ..agent.conversation import Conversation, Message, TextContentBlock
from ..context_system.microcompact import microcompact_messages, strip_images_from_messages
from ..providers.base import BaseProvider
from .messages import (
    create_compact_boundary_message,
    create_compact_summary_message,
    get_messages_after_boundary,
    is_compact_boundary_message,
)

logger = logging.getLogger(__name__)

# Maximum output tokens for the summary model
COMPACT_MAX_OUTPUT_TOKENS = 4096

# No-tools preamble prepended to the summary prompt
NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
Do NOT use Read, Bash, Grep, or any other tool. Your entire response must be plain text.
"""

SUMMARY_PROMPT_TEMPLATE = NO_TOOLS_PREAMBLE + """
Your task is to create a detailed summary of the conversation so far.

Include:
1. Primary Request and Intent: What the user asked for
2. Key Technical Concepts: Technologies, frameworks, patterns discussed
3. Files and Code Sections: Files examined, modified, or created (with snippets)
4. Errors and Fixes: Problems encountered and how they were resolved
5. Problem Solving: What was accomplished
6. All User Messages: All non-tool-result user messages
7. Pending Tasks: Tasks explicitly requested but not yet done
8. Current Work: What was being worked on immediately before this summary
9. Optional Next Step: Next step in line with recent requests

Respond ONLY with plain text. No tools. No XML tags.
"""


@dataclass
class CompactResult:
    """Result of a compaction operation."""
    boundary_message: Message
    summary_message: Message
    tokens_saved: int
    pre_compact_count: int
    post_compact_count: int
    summary_text: str
    trigger: str = "manual"
    user_display_message: Optional[str] = None


async def compact_conversation(
    conversation: Conversation,
    provider: BaseProvider,
    model: str,
    custom_instructions: Optional[str] = None,
    trigger: str = "manual",
    workspace_root: Optional[Path] = None,
) -> CompactResult:
    """
    Compact a conversation by summarizing older messages.

    This is the Python equivalent of the TypeScript compactConversation().
    It is NOT a 1:1 port — it omits features unavailable in a Python CLI:
    - Forked agent with cache sharing (no subprocess isolation in Python REPL)
    - Session memory (no disk-backed session memory extraction)
    - Reactive compact (no background compaction agent)
    - Hook system (no pre/post compact hooks in Clawd-Code)

    Args:
        conversation: The live Conversation object (mutated in place)
        provider: The LLM provider for generating the summary
        model: Model name for the summary call
        custom_instructions: Optional user instructions for the summarizer
        trigger: "manual" or "auto"
        workspace_root: Project root used for survival-context re-injection
            (CLAUDE.md, memory files, recent files). Defaults to the current
            working directory.

    Returns:
        CompactResult with boundary, summary, and metadata
    """
    # Step 1: Get messages after the last boundary (skip already-summarized)
    messages = get_messages_after_boundary(conversation.messages)

    if len(messages) < 2:
        raise ValueError("Not enough messages to compact.")

    # Step 2: Count pre-compact tokens
    from ..token_estimation import count_messages_tokens
    api_messages = conversation.get_messages()
    pre_compact_tokens = count_messages_tokens(api_messages)
    pre_compact_count = len(conversation.messages)

    # Step 3: Microcompact — strip images and clear old tool results
    api_messages_stripped = strip_images_from_messages(api_messages)
    compacted_api, tokens_saved = microcompact_messages(api_messages_stripped)

    # Step 4: Build summary prompt
    prompt = SUMMARY_PROMPT_TEMPLATE
    if custom_instructions:
        prompt += f"\n\nAdditional instructions: {custom_instructions}"

    # Build messages for the summary API call
    # System prompt + recent messages + summary request
    summary_request_messages: list[dict[str, Any]] = []

    # Add the last N messages as context (after microcompact)
    # Avoid sending too many old messages to stay within context limits
    context_messages = compacted_api[-20:] if len(compacted_api) > 20 else compacted_api
    for msg in context_messages:
        summary_request_messages.append(msg)

    # Add the summary request as the last user message
    summary_request_messages.append({
        "role": "user",
        "content": prompt,
    })

    # Step 5: Call the LLM to generate summary
    summary_text = ""
    try:
        response = await provider.chat_async(
            messages=summary_request_messages,
            tools=None,  # No tools during compaction
            model=model,
            max_tokens=COMPACT_MAX_OUTPUT_TOKENS,
        )
        summary_text = response.content.strip()
    except Exception as e:
        # Try sync fallback
        try:
            response = provider.chat(
                messages=summary_request_messages,
                tools=None,
                model=model,
                max_tokens=COMPACT_MAX_OUTPUT_TOKENS,
            )
            summary_text = response.content.strip()
        except Exception as e2:
            logger.warning(f"Compact LLM call failed: {e}, sync fallback: {e2}, using text extraction")
            summary_text = _fallback_summary(messages)

    if not summary_text:
        summary_text = _fallback_summary(messages)

    # Step 6: Create boundary marker
    last_msg_uuid = None
    if messages:
        last_msg = messages[-1]
        if hasattr(last_msg, 'uuid'):
            last_msg_uuid = getattr(last_msg, 'uuid', None)
        elif isinstance(last_msg, dict):
            last_msg_uuid = last_msg.get("uuid")

    boundary_msg = create_compact_boundary_message(
        trigger=trigger,
        pre_compact_token_count=pre_compact_tokens,
        last_message_uuid=last_msg_uuid,
    )

    # Step 7: Create summary message
    summary_msg = create_compact_summary_message(summary_text)

    # Step 8: Insert boundary + summary into conversation
    # Find position after the last boundary marker
    boundary_indices = [
        i for i, m in enumerate(conversation.messages)
        if is_compact_boundary_message(m)
    ]

    if boundary_indices:
        insert_pos = max(boundary_indices) + 1
    else:
        insert_pos = 0

    # Remove messages that were summarized (everything from insert_pos onward)
    # Then insert boundary + summary
    if insert_pos == 0:
        # No existing boundary — clear all and start fresh
        conversation.messages.clear()
        conversation.messages.append(boundary_msg)
        conversation.messages.append(summary_msg)
    else:
        # Keep messages before boundary, then boundary + summary
        conversation.messages = list(conversation.messages[:insert_pos])
        conversation.messages.append(boundary_msg)
        conversation.messages.append(summary_msg)

    # Step 8b: Re-inject survival context (CLAUDE.md, memory files, invoked
    # skill bodies, recently-touched files) so the session doesn't get amnesia.
    invoked_skill_names = _invoked_skill_names(messages)
    reinject_survival_context(
        conversation,
        workspace_root=workspace_root,
        invoked_skill_names=invoked_skill_names,
    )

    post_compact_count = len(conversation.messages)

    # Step 9: Build user display message
    saved_str = f"~{tokens_saved:,}" if tokens_saved > 0 else "some"
    user_display = (
        f"Compacted conversation ({saved_str} tokens saved). "
        f"Pre-compact: {pre_compact_tokens:,} tokens. "
        f"Press Ctrl+O to see full summary."
    )

    return CompactResult(
        boundary_message=boundary_msg,
        summary_message=summary_msg,
        tokens_saved=tokens_saved,
        pre_compact_count=pre_compact_count,
        post_compact_count=post_compact_count,
        summary_text=summary_text,
        trigger=trigger,
        user_display_message=user_display,
    )


def _fallback_summary(messages: list[Message]) -> str:
    """
    Generate a simple text fallback summary when the LLM call fails.
    """
    user_msgs: list[str] = []
    assistant_msgs: list[str] = []

    for msg in messages:
        role = getattr(msg, 'role', None)
        if role is None and isinstance(msg, dict):
            role = msg.get("role")

        content = getattr(msg, 'content', '')
        if content is None and isinstance(msg, dict):
            content = msg.get("content", "")

        if role == "user":
            if isinstance(content, str):
                user_msgs.append(content[:200])
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        user_msgs.append(block.get("text", "")[:200])
        elif role == "assistant":
            if isinstance(content, str):
                assistant_msgs.append(content[:200])
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        assistant_msgs.append(block.get("text", "")[:200])

    summary_parts = [f"Conversation had {len(messages)} messages."]

    # Extract tool use info
    tool_uses = []
    for msg in messages:
        content = getattr(msg, 'content', [])
        if isinstance(msg, dict):
            content = msg.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_uses.append(block.get("name", ""))

    if tool_uses:
        summary_parts.append(f"Tools used: {', '.join(tool_uses[:10])}")

    if user_msgs:
        summary_parts.append(f"Last user message: {user_msgs[-1][:150]}")
    if assistant_msgs:
        summary_parts.append(f"Last assistant message: {assistant_msgs[-1][:150]}")

    return "\n".join(summary_parts)


# ---------------------------------------------------------------------------
# Survival context re-injection
# ---------------------------------------------------------------------------

# Cap for re-injected skill bodies (~5k tokens each).
SURVIVAL_SKILL_MAX_TOKENS = 5000

# Caps for re-injected file content.
_SURVIVAL_MEMORY_MAX_CHARS = 4000
_SURVIVAL_RECENT_FILE_MAX_CHARS = 6000
_SURVIVAL_RECENT_FILE_COUNT = 5

_EXCLUDED_SCAN_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".clawd", "dist", "build", ".pytest_cache", ".mypy_cache",
})

_EXCLUDED_SCAN_NAMES = frozenset({
    "claude.md", "memory.md",  # covered by their own sections
})


def _invoked_skill_names(messages: list) -> list[str]:
    """Collect Skill tool invocations from pre-compact messages.

    Fully defensive: any unexpected shape is skipped.
    """
    names: list[str] = []
    try:
        for msg in messages or []:
            content = getattr(msg, "content", None)
            if content is None and isinstance(msg, dict):
                content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                try:
                    if isinstance(block, dict):
                        btype = block.get("type")
                        bname = block.get("name")
                        binput = block.get("input") or {}
                    else:
                        btype = getattr(block, "type", None)
                        bname = getattr(block, "name", None)
                        binput = getattr(block, "input", None) or {}
                    if btype != "tool_use":
                        continue
                    if not isinstance(bname, str) or bname.lower() != "skill":
                        continue
                    if isinstance(binput, dict):
                        skill_name = binput.get("skill") or binput.get("name")
                        if isinstance(skill_name, str) and skill_name.strip():
                            names.append(skill_name.strip())
                except Exception:
                    continue
    except Exception:
        pass
    # De-dupe, preserve order
    seen: set[str] = set()
    unique: list[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


def _load_skill_body(skill_name: str, max_tokens: int = SURVIVAL_SKILL_MAX_TOKENS) -> Optional[str]:
    """Load an invoked skill's SKILL.md body, capped at ~max_tokens tokens."""
    try:
        from ..skills.loader import find_skill
        from ..token_estimation import count_tokens
        skill = find_skill(skill_name)
        if skill is None:
            return None
        body = getattr(skill, "markdown_content", "") or ""
        if not body.strip():
            return None
        if count_tokens(body) > max_tokens:
            body = body[: max_tokens * 4].rstrip() + "\n...[truncated]"
        return body
    except Exception:
        return None


def _load_memory_files(root: Path) -> list[tuple[str, str]]:
    """Find memory files in/around the workspace. Missing files are skipped."""
    found: list[tuple[str, str]] = []

    def _read_capped(path: Path) -> Optional[str]:
        try:
            if not path.is_file():
                return None
            text = path.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                return None
            if len(text) > _SURVIVAL_MEMORY_MAX_CHARS:
                text = text[: _SURVIVAL_MEMORY_MAX_CHARS].rstrip() + "\n...[truncated]"
            return text
        except Exception:
            return None

    candidates: list[Path] = [
        root / "MEMORY.md",
        root / "memory.md",
        root / ".clawd" / "MEMORY.md",
    ]
    try:
        user_mem = Path.home() / ".clawd" / "memory"
        if user_mem.is_dir():
            for child in sorted(user_mem.iterdir()):
                if child.is_file() and child.suffix.lower() in (".md", ".txt"):
                    candidates.append(child)
                if len(candidates) >= 8:
                    break
    except Exception:
        pass

    for path in candidates:
        text = _read_capped(path)
        if text is not None:
            found.append((str(path), text))
    return found


def _recently_modified_files(root: Path, limit: int = _SURVIVAL_RECENT_FILE_COUNT) -> list[tuple[str, str]]:
    """Return the `limit` most-recently-modified text files under root.

    Proxy for "files the session touched recently". Skips hidden/tooling
    directories and binary files. Fully defensive.
    """
    import os

    scored: list[tuple[float, Path]] = []
    examined = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune excluded / hidden dirs in place
            dirnames[:] = [
                d for d in dirnames
                if d not in _EXCLUDED_SCAN_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                if fname.lower() in _EXCLUDED_SCAN_NAMES:
                    continue
                fpath = Path(dirpath) / fname
                try:
                    st = fpath.stat()
                except OSError:
                    continue
                if not st.st_size or st.st_size > 200_000:
                    continue
                scored.append((st.st_mtime, fpath))
                examined += 1
                if examined > 3000:
                    break
            if examined > 3000:
                break
    except Exception:
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    results: list[tuple[str, str]] = []
    for _, fpath in scored[:limit]:
        try:
            raw = fpath.read_bytes()[: _SURVIVAL_RECENT_FILE_MAX_CHARS + 1024]
            if b"\x00" in raw[:8192]:
                continue  # binary
            text = raw.decode("utf-8", errors="replace")
            if len(text) > _SURVIVAL_RECENT_FILE_MAX_CHARS:
                text = text[: _SURVIVAL_RECENT_FILE_MAX_CHARS].rstrip() + "\n...[truncated]"
            if text.strip():
                results.append((str(fpath), text))
        except Exception:
            continue
    return results


def reinject_survival_context(
    conversation: Conversation,
    workspace_root: str | Path | None = None,
    invoked_skill_names: Optional[list[str]] = None,
) -> Optional[Message]:
    """Re-inject survival context after a compaction.

    Appends a single user-role message carrying:
      1. project instructions from CLAUDE.md (via load_claude_md_context),
      2. memory files (workspace + ~/.clawd/memory),
      3. bodies of skills invoked before the compact (capped at ~5k tokens),
      4. the 5 most-recently-modified files in the workspace.

    Fully defensive: missing files or any error skips that section — this
    function never raises. Returns the appended message, or None when there
    was nothing to re-inject.

    Args:
        conversation: The live Conversation object (mutated in place)
        workspace_root: Project root; defaults to the current working directory
        invoked_skill_names: Skill names seen in the pre-compact messages
    """
    try:
        root = Path(workspace_root).expanduser() if workspace_root else Path.cwd()
    except Exception:
        return None

    sections: list[str] = []

    # 1. Project instructions (CLAUDE.md)
    try:
        from ..context_system.claude_md import load_claude_md_context
        claude_ctx = load_claude_md_context(root)
        for claude_file in claude_ctx.files:
            sections.append(
                f'<project-instructions path="{claude_file.path}">\n'
                f"{claude_file.content}\n</project-instructions>"
            )
    except Exception:
        pass

    # 2. Memory files
    try:
        for mem_path, mem_text in _load_memory_files(root):
            sections.append(f'<memory path="{mem_path}">\n{mem_text}\n</memory>')
    except Exception:
        pass

    # 3. Invoked skill bodies (cap ~5k tokens each)
    try:
        for skill_name in invoked_skill_names or []:
            body = _load_skill_body(skill_name)
            if body:
                sections.append(f'<skill name="{skill_name}">\n{body}\n</skill>')
    except Exception:
        pass

    # 4. Most-recently-modified files
    try:
        for fpath, ftext in _recently_modified_files(root, limit=_SURVIVAL_RECENT_FILE_COUNT):
            sections.append(f'<recent-file path="{fpath}">\n{ftext}\n</recent-file>')
    except Exception:
        pass

    if not sections:
        return None

    header = (
        "Context restored after compaction. The conversation above was summarized; "
        "the following project context was re-injected so it is not lost:"
    )
    content = TextContentBlock(
        type="text",
        text=header + "\n\n" + "\n\n".join(sections),
    )
    msg = Message(
        role="user",
        content=[content],
        timestamp=datetime.now().isoformat(),
    )
    try:
        conversation.messages.append(msg)
    except Exception:
        return None
    return msg


def compact_conversation_sync(
    conversation: Conversation,
    provider: BaseProvider,
    model: str,
    **kwargs: Any,
) -> CompactResult:
    """Synchronous wrapper around :func:`compact_conversation`.

    Uses ``asyncio.run`` when no event loop is running in this thread;
    otherwise isolates the coroutine in a worker thread with its own loop
    (e.g. when called from an async web-app context).
    """
    import asyncio
    import concurrent.futures

    async def _runner() -> CompactResult:
        return await compact_conversation(conversation, provider, model, **kwargs)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_runner())
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(_runner())).result()
