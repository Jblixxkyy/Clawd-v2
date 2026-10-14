from __future__ import annotations

import hashlib
import time
from difflib import SequenceMatcher
from pathlib import Path

from .models import ClaudeMdContext, ClaudeMdFile

_PROJECT_CANDIDATES = ("CLAUDE.md", ".clawd/CLAUDE.md", ".claude/CLAUDE.md")
_USER_CANDIDATES = (".clawd/CLAUDE.md", ".claude/CLAUDE.md")
# Watch AGENTS.md with the same candidate scheme as CLAUDE.md (brief 012).
_PROJECT_AGENTS_CANDIDATES = ("AGENTS.md", ".clawd/AGENTS.md", ".claude/AGENTS.md")
_USER_AGENTS_CANDIDATES = (".clawd/AGENTS.md", ".claude/AGENTS.md")

# --- Instruction-file live reload (brief 012) -------------------------------------------------

# Debounce window: at most one notification per this many seconds, so a rapid
# burst of saves produces a single notification instead of one per save.
_INSTRUCTION_RELOAD_DEBOUNCE_SECONDS = 2.0
# Compact diff summary: keep reminders small -- at most this many sample
# changed lines, each truncated to this many characters (never the full file).
_DIFF_SAMPLE_LINES = 3
_DIFF_LINE_CHARS = 120


def load_claude_md_context(
    workspace_root: str | Path,
    *,
    cwd: str | Path | None = None,
    max_files: int = 6,
    max_chars_per_file: int = 4_000,
    max_total_chars: int = 12_000,
) -> ClaudeMdContext:
    root = Path(workspace_root).expanduser().resolve()
    current = Path(cwd).expanduser().resolve() if cwd is not None else root

    candidates: list[Path] = []

    home = Path.home()
    for rel in _USER_CANDIDATES:
        path = (home / rel).resolve()
        if path not in candidates:
            candidates.append(path)

    for base in _walk_up_to_root(current, root):
        for rel in _PROJECT_CANDIDATES:
            path = (base / rel).resolve()
            if path not in candidates:
                candidates.append(path)

    files: list[ClaudeMdFile] = []
    total_chars = 0
    truncated = False
    for path in candidates:
        if len(files) >= max_files or total_chars >= max_total_chars:
            truncated = True
            break
        if not path.exists() or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not content:
            continue
        if len(content) > max_chars_per_file:
            content = content[: max_chars_per_file - 32].rstrip() + "\n...[truncated]"
            truncated = True
        remaining = max_total_chars - total_chars
        if remaining <= 0:
            truncated = True
            break
        if len(content) > remaining:
            content = content[: max(0, remaining - 32)].rstrip() + "\n...[truncated]"
            truncated = True
        total_chars += len(content)
        files.append(ClaudeMdFile(path=path, content=content))

    return ClaudeMdContext(files=tuple(files), truncated=truncated)


def _walk_up_to_root(current: Path, root: Path) -> list[Path]:
    current = current.resolve()
    root = root.resolve()
    if current != root:
        try:
            current.relative_to(root)
        except ValueError:
            current = root

    bases: list[Path] = []
    node = current
    while True:
        if node not in bases:
            bases.append(node)
        if node == root:
            break
        parent = node.parent
        if parent == node:
            break
        node = parent
    return bases


def _summarize_diff(old_text: str, new_text: str) -> tuple[int, int, list[str]]:
    """Compact diff summary: (added, removed, sample_lines).

    Counts added/removed lines and returns up to ``_DIFF_SAMPLE_LINES``
    truncated sample lines prefixed with ``+``/``-`` -- a summary of the
    change, never the full file.
    """
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    matcher = SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    added = 0
    removed = 0
    added_samples: list[str] = []
    removed_samples: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            added += j2 - j1
            added_samples.extend(new_lines[j1:j2])
        elif tag == "delete":
            removed += i2 - i1
            removed_samples.extend(old_lines[i1:i2])
        elif tag == "replace":
            added += j2 - j1
            removed += i2 - i1
            added_samples.extend(new_lines[j1:j2])
            removed_samples.extend(old_lines[i1:i2])
    samples: list[str] = []
    for line in added_samples[:_DIFF_SAMPLE_LINES]:
        samples.append("+ " + line[:_DIFF_LINE_CHARS])
    for line in removed_samples[:_DIFF_SAMPLE_LINES]:
        samples.append("- " + line[:_DIFF_LINE_CHARS])
    return added, removed, samples


def format_instruction_change_reminder(
    path: str | Path,
    old_text: str | None,
    new_text: str,
) -> str:
    """Render a system-reminder-style notification for a changed file.

    Names the file and carries a compact diff summary (line counts plus a few
    sample lines), never the full file. ``old_text=None`` means the file
    appeared mid-session.
    """
    name = Path(path).name
    if old_text is None:
        added, _, samples = _summarize_diff("", new_text)
        header = (
            f"New project instructions apply from: {name} ({added} lines). "
            "Read and follow them from this turn on."
        )
    else:
        added, removed, samples = _summarize_diff(old_text, new_text)
        header = (
            f"The project instructions changed: {name} "
            f"(+{added} lines, -{removed} lines). "
            "The updated instructions apply from this turn; "
            "the previously loaded version no longer applies."
        )
    reminder = f"<system-reminder>{header}"
    if samples:
        reminder += "\nChange summary:\n" + "\n".join(samples)
    return reminder + "</system-reminder>"


class InstructionWatcher:
    """Watch project instruction files (AGENTS.md, CLAUDE.md) mid-session.

    Construction records ``(mtime_ns, sha256)`` for every watched file.
    :meth:`check_for_changes` is meant to be called once per agent-loop turn:
    it does a cheap ``stat()`` per file and only reads + hashes files whose
    mtime moved, so an unchanged session costs almost nothing. On a real
    content change (the hash check skips no-op saves that only touched the
    mtime) it returns a ``<system-reminder>`` text naming the file plus a
    compact diff summary. Rapid saves are debounced: at most one notification
    per change burst (see ``debounce_seconds``).

    Never raises: unreadable or deleted files are skipped.
    """

    def __init__(
        self,
        paths: list[str | Path] | tuple[str | Path, ...],
        *,
        debounce_seconds: float = _INSTRUCTION_RELOAD_DEBOUNCE_SECONDS,
    ) -> None:
        self._debounce_seconds = debounce_seconds
        self._last_notified = 0.0  # time.monotonic()
        self._entries: dict[str, tuple[int, str, str] | None] = {}
        for path in paths:
            resolved = str(Path(path))
            if resolved not in self._entries:
                self._entries[resolved] = self._snapshot(Path(resolved))

    @staticmethod
    def _snapshot(path: Path) -> tuple[int, str, str] | None:
        """(mtime_ns, sha256_hex, text), or None when unreadable/missing."""
        try:
            st = path.stat()
            content = path.read_text(encoding="utf-8")
        except Exception:
            return None
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return (st.st_mtime_ns, digest, content)

    def check_for_changes(self) -> list[str]:
        """Return ``<system-reminder>`` texts for changed files (usually [])."""
        changed: list[tuple[Path, str | None, str]] = []
        for path_str, old in self._entries.items():
            path = Path(path_str)
            try:
                mtime_ns = path.stat().st_mtime_ns
            except OSError:
                continue  # deleted or unreadable mid-session: skip, never raise
            if old is not None and mtime_ns == old[0]:
                continue  # cheap path: stat unchanged, no read, no hash
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            new = (mtime_ns, digest, content)
            if old is None:
                # File appeared mid-session: new instructions.
                changed.append((path, None, content))
            elif digest != old[1]:
                # mtime moved and content differs (the hash check skips
                # no-op saves that only touched the mtime).
                changed.append((path, old[2], content))
            # Always refresh the snapshot so suppressed bursts never re-fire.
            self._entries[path_str] = new
        if not changed:
            return []
        now = time.monotonic()
        if now - self._last_notified < self._debounce_seconds:
            return []  # debounce: one notification per change burst
        self._last_notified = now
        return [
            format_instruction_change_reminder(path, old_text, new_text)
            for path, old_text, new_text in changed
        ]


def snapshot_instruction_files(
    workspace_root: str | Path,
    *,
    cwd: str | Path | None = None,
    context: ClaudeMdContext | None = None,
) -> InstructionWatcher:
    """Build an :class:`InstructionWatcher` for the session's instruction files.

    Watches the loaded CLAUDE.md files (from ``load_claude_md_context``) plus
    AGENTS.md candidates resolved with the same scheme, so edits to project
    rules made mid-session can be surfaced by the agent loop. Call once at
    session start, then :meth:`InstructionWatcher.check_for_changes` per turn.
    """
    root = Path(workspace_root).expanduser().resolve()
    current = Path(cwd).expanduser().resolve() if cwd is not None else root
    if context is None:
        try:
            context = load_claude_md_context(root, cwd=current)
        except Exception:
            context = None
    paths: list[Path] = []
    if context is not None:
        paths.extend(f.path for f in context.files)
    home = Path.home()
    for rel in _USER_AGENTS_CANDIDATES:
        paths.append((home / rel).resolve())
    for base in _walk_up_to_root(current, root):
        for rel in _PROJECT_AGENTS_CANDIDATES:
            paths.append((base / rel).resolve())
    return InstructionWatcher(paths)
