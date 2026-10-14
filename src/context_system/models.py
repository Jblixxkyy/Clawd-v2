from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClaudeMdFile:
    path: Path
    content: str


@dataclass(frozen=True)
class ClaudeMdContext:
    files: tuple[ClaudeMdFile, ...]
    truncated: bool = False


@dataclass(frozen=True)
class GitContext:
    available: bool
    repo_root: Path | None = None
    branch: str | None = None
    status: str | None = None
    recent_commit: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class WorkspaceSnapshot:
    workspace_root: Path
    current_directory: Path
    top_level_entries: tuple[str, ...]
    key_files: tuple[str, ...]
    python_file_count: int
    test_file_count: int
