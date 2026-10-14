from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .errors import ToolPermissionError
from .permissions import PermissionMode, ToolPermissionContext
from .task_manager import TaskManager


def _default_skill_dirs() -> list[Path]:
    """Global skill roots that must stay readable regardless of workspace."""
    import os

    dirs: list[Path] = []
    for _env in ("CLAWD_SKILLS_DIR", "CLAUDE_SKILLS_DIR"):
        _v = os.environ.get(_env)
        if _v:
            try:
                _p = Path(_v).expanduser().resolve()
                if _p not in dirs:
                    dirs.append(_p)
            except Exception:
                pass
    for _d in (Path.home() / ".clawd" / "skills", Path.home() / ".claude" / "skills"):
        try:
            _p = _d.expanduser().resolve()
            if _p not in dirs:
                dirs.append(_p)
        except Exception:
            pass
    _managed = os.environ.get("CLAWD_MANAGED_SKILLS_DIR")
    if _managed:
        try:
            _p = Path(_managed).expanduser().resolve()
            if _p not in dirs:
                dirs.append(_p)
        except Exception:
            pass
    else:
        try:
            # src/tool_system/context.py -> src/ -> repo/managed_skills
            _fallback = Path(__file__).resolve().parent.parent.parent / "managed_skills"
            if _fallback not in dirs:
                dirs.append(_fallback)
        except Exception:
            pass
    return dirs


def _default_config_dirs() -> list[Path]:
    """Clawd config roots the AI is allowed to manage (mcp.json etc)."""
    import os

    dirs: list[Path] = []
    try:
        _c = (Path.home() / ".clawd").expanduser().resolve()
        dirs.append(_c)
    except Exception:
        pass
    return dirs


def is_managed_skill_path(path: Path) -> bool:
    """True when path is a SKILL.md inside any known skill root."""
    try:
        rp = Path(path).expanduser().resolve()
    except Exception:
        return False
    if rp.name != "SKILL.md":
        return False
    parts = {p.lower() for p in rp.parts}
    if "managed_skills" in parts:
        return True
    # .../.clawd/skills/<name>/SKILL.md or .../.claude/skills/<name>/SKILL.md
    for i, p in enumerate([x.lower() for x in rp.parts]):
        if p in {".clawd", ".claude"} and i + 1 < len(rp.parts) and rp.parts[i + 1].lower() == "skills":
            return True
    # Any globally-known skill root
    try:
        for root in _default_skill_dirs():
            try:
                rp.relative_to(root)
                return True
            except ValueError:
                pass
    except Exception:
        pass
    return False


@dataclass
class ToolContext:
    workspace_root: Path
    permission_context: ToolPermissionContext = field(default_factory=ToolPermissionContext)
    cwd: Path | None = None
    read_file_fingerprints: dict[Path, tuple[int, int]] = field(default_factory=dict)
    task_manager: TaskManager = field(default_factory=TaskManager)
    mcp_clients: dict[str, Any] = field(default_factory=dict)
    lsp_client: Any | None = None
    todos: list[dict[str, Any]] = field(default_factory=list)
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    plan_mode: bool = False
    worktree_root: Path | None = None
    outbox: list[dict[str, Any]] = field(default_factory=list)
    ask_user: Callable[[list[dict[str, Any]]], dict[str, str]] | None = None
    crons: dict[str, dict[str, Any]] = field(default_factory=dict)
    team: dict[str, Any] | None = None
    output_style_name: str | None = None
    output_style_dir: Path | None = None

    # Non-interactive execution (e.g. inside a subagent Task/Agent run).
    # When True, permission checks that would ask the user are auto-denied
    # WITHOUT invoking permission_handler, so a delegated run can never
    # block on stdin waiting for input. The denial is returned as a normal
    # tool error so the caller can report it to the parent context.
    non_interactive: bool = False

    # Permission handler callback: called when a tool needs user consent.
    # Signature: (tool_name: str, message: str, suggestion: str | None)
    #           -> tuple[bool, bool] (allowed: bool, continue_without_caching: bool)
    # If not set, permission errors will be raised as exceptions.
    permission_handler: Callable[[str, str, Optional[str]], tuple[bool, bool]] | None = None

    # Session-level permission mode (PermissionMode.DEFAULT unless changed via
    # /permissions mode or --permission-mode). Deny rules always win,
    # regardless of mode.
    permission_mode: PermissionMode = field(default_factory=lambda: PermissionMode.DEFAULT)

    # Explicit opt-in for the PowerShell tool (Windows only). Default off;
    # populated from "enable_powershell" in ~/.clawd/config.json.
    # Default ON: PowerShell is the default shell on Windows.
    enable_powershell: bool = True

    # LLM provider for the current session. Set by run_agent_loop on entry;
    # tools like Subagent read it to spawn child loops. None when the
    # context is built without a provider (tests, standalone dispatch).
    provider: Any = None

    def __post_init__(self) -> None:
        self.workspace_root = Path(self.workspace_root).resolve()
        if self.cwd is None:
            self.cwd = self.workspace_root
        else:
            self.cwd = Path(self.cwd).resolve()
        if self.permission_context.workspace_root is None:
            self.permission_context = ToolPermissionContext.from_iterables(
                self.permission_context.deny_names,
                self.permission_context.deny_prefixes,
                workspace_root=self.workspace_root,
                additional_working_directories=self.permission_context.additional_working_directories,
                allow_docs=self.permission_context.allow_docs,
            )
        # Skills must be readable from anywhere: managed + user skill dirs
        # live outside the workspace (e.g. D:\clawd\...\managed_skills while
        # cwd is C:\Users\...). Pre-whitelist them so Read/Glob on
        # "Base directory for this skill" never hits
        # "path is outside allowed working directories".
        # Per-skill whitelisting also happens in SkillTool.run for any
        # custom location; this covers discovery without a Skill call.
        # Global skill roots are whitelisted even when missing so the AI can
        # CREATE new skills (Write needs an allowed parent). ~/.clawd is
        # whitelisted so the AI can manage mcp.json.
        try:
            for _d in _default_skill_dirs() + _default_config_dirs():
                if _d not in self.permission_context.additional_working_directories:
                    self.permission_context.additional_working_directories = (
                        *self.permission_context.additional_working_directories,
                        _d,
                    )
            # Repo-owned skills: .clawd/skills / .claude/skills in workspace
            # or any ancestor (repo root while cwd is a subdir). Whitelist
            # existing ones so Read/Glob work even when ancestor is outside
            # workspace_root.
            for _base in {self.workspace_root, self.cwd}:
                if _base is None:
                    continue
                _cur: Path | None = _base
                while _cur is not None:
                    for _cand in (_cur / ".clawd" / "skills", _cur / ".claude" / "skills"):
                        try:
                            if _cand.exists() and _cand.is_dir() and _cand.resolve() not in self.permission_context.additional_working_directories:
                                self.permission_context.additional_working_directories = (
                                    *self.permission_context.additional_working_directories,
                                    _cand.resolve(),
                                )
                        except Exception:
                            pass
                    _nxt = _cur.parent
                    if _nxt == _cur:
                        break
                    _cur = _nxt
        except Exception:
            pass

    def mark_file_read(self, path: Path) -> None:
        stat = path.stat()
        self.read_file_fingerprints[path.resolve()] = (int(stat.st_mtime), int(stat.st_size))

    def was_file_read_and_unchanged(self, path: Path) -> bool:
        resolved = path.resolve()
        fingerprint = self.read_file_fingerprints.get(resolved)
        if fingerprint is None:
            return False
        stat = resolved.stat()
        return fingerprint == (int(stat.st_mtime), int(stat.st_size))

    def ensure_allowed_path(self, path: str | Path) -> Path:
        p = Path(path).expanduser() if isinstance(path, str) else path.expanduser()
        if not p.is_absolute():
            base = self.cwd or self.workspace_root
            p = (base / p).resolve()
        return self.permission_context.ensure_path_allowed(p)

    def ensure_tool_allowed(self, tool_name: str) -> None:
        if self.permission_context.blocks_tool(tool_name):
            raise ToolPermissionError(f"tool is blocked by permission context: {tool_name}")
