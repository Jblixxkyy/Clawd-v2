from __future__ import annotations

from datetime import date
from pathlib import Path

from .claude_md import load_claude_md_context
from .git_context import collect_git_context
from .workspace_snapshot import build_workspace_snapshot


def build_context_prompt(
    workspace_root: str | Path,
    *,
    cwd: str | Path | None = None,
) -> str:
    root = Path(workspace_root).expanduser().resolve()
    current = Path(cwd).expanduser().resolve() if cwd is not None else root

    workspace = build_workspace_snapshot(root, cwd=current)
    git = collect_git_context(root)
    claude_md = load_claude_md_context(root, cwd=current)

    sections: list[str] = []

    sections.append("\n".join(_render_workspace_section(workspace)))

    git_lines = _render_git_section(git, root)
    if git_lines:
        sections.append("\n".join(git_lines))

    md_lines = _render_claude_md_section(claude_md, root)
    if md_lines:
        sections.append("\n".join(md_lines))

    return "\n\n".join(section for section in sections if section.strip())


def _render_workspace_section(workspace) -> list[str]:
    lines = [
        "## Runtime Context",
        f"- Today's date: {date.today().isoformat()}",
        f"- Workspace root: {workspace.workspace_root}",
        f"- Current directory: {workspace.current_directory}",
        f"- Python files: {workspace.python_file_count}",
        f"- Test files: {workspace.test_file_count}",
    ]
    if workspace.key_files:
        lines.append(f"- Key files: {', '.join(workspace.key_files)}")
    if workspace.top_level_entries:
        lines.append(f"- Top-level entries: {', '.join(workspace.top_level_entries)}")
    return lines


def _render_git_section(git, workspace_root: Path) -> list[str]:
    if not git.available:
        return []
    lines = ["## Git Context"]
    if git.repo_root is not None:
        lines.append(f"- Repository root: {git.repo_root}")
    if git.branch:
        lines.append(f"- Current branch: {git.branch}")
    if git.recent_commit:
        lines.append(f"- Latest commit: {git.recent_commit}")
    if git.status:
        lines.extend([
            "- Git status snapshot:",
            "```text",
            git.status,
            "```",
        ])
    return lines


def _render_claude_md_section(claude_md, workspace_root: Path) -> list[str]:
    if not claude_md.files:
        return []
    lines = ["## Project Instructions"]
    for item in claude_md.files:
        try:
            rel = item.path.relative_to(workspace_root)
            label = f"./{rel}"
        except ValueError:
            label = str(item.path)
        lines.extend([
            f"### {label}",
            "```md",
            item.content,
            "```",
        ])
    if claude_md.truncated:
        lines.append("- Additional instruction files were truncated to stay within prompt budget.")
    return lines
