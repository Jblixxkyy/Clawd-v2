"""Per-session file checkpoints for the `/rewind` command (feature #6).

Before a Write/Edit tool run, the tool dispatch snapshots the target file's
bytes so the user can later restore an earlier version with `/rewind <n>`.

Checkpoints live workspace-relative at::

    <workspace_root>/.clawd/checkpoints/<session_id>/<n>-<relpath>.bak

where ``<n>`` is a zero-padded incrementing counter and ``<relpath>`` is the
file's path relative to the workspace root with path separators replaced by
``__``.  A per-session ``manifest.json`` records metadata (timestamp, label,
and a new-file tombstone) for each snapshot.  Only the newest 20 snapshots
per session are kept; older ``.bak`` files are pruned.

Every function never raises on filesystem trouble except ``restore()``, which
raises a clear ``ValueError`` when the requested checkpoint does not exist.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

#: Maximum number of snapshot files kept per session.
MAX_SNAPSHOTS_PER_SESSION = 20

#: Marker written into the .bak of a file that did not exist yet when
#: snapshotted. Restoring such a checkpoint deletes the file again.
NEW_FILE_MARKER = b"__NEW_FILE__"


def _session_dir(session_id: str, workspace_root: Path) -> Path:
    return Path(workspace_root).resolve() / ".clawd" / "checkpoints" / str(session_id)


def _manifest_path(session_dir: Path) -> Path:
    return session_dir / "manifest.json"


def _load_manifest(session_dir: Path) -> dict[str, dict]:
    try:
        raw = _manifest_path(session_dir).read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def _save_manifest(session_dir: Path, manifest: dict[str, dict]) -> None:
    try:
        session_dir.mkdir(parents=True, exist_ok=True)
        tmp = _manifest_path(session_dir).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(_manifest_path(session_dir))
    except Exception:
        pass


def _relpath_flat(path: Path, root: Path) -> tuple[str, str]:
    """Return (relpath display string, flattened filename-safe string)."""
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError:
        # Outside the workspace: fall back to the full absolute path.
        rel = resolved
    flat = "__".join(rel.parts) or "__root__"
    return rel.as_posix(), flat


def _existing_numbers(session_dir: Path) -> list[int]:
    numbers: list[int] = []
    if not session_dir.is_dir():
        return numbers
    for entry in session_dir.iterdir():
        if entry.is_file() and entry.suffix == ".bak":
            stem = entry.name[:4]
            if stem.isdigit():
                numbers.append(int(stem))
    return numbers


def _prune_old(session_dir: Path, manifest: dict[str, dict]) -> None:
    """Keep only the newest MAX_SNAPSHOTS_PER_SESSION .bak files."""
    bak_files = sorted(
        (e for e in session_dir.iterdir() if e.is_file() and e.suffix == ".bak"),
        key=lambda e: e.name,
    )
    excess = len(bak_files) - MAX_SNAPSHOTS_PER_SESSION
    if excess <= 0:
        return
    for entry in bak_files[:excess]:
        try:
            entry.unlink()
        except Exception:
            continue
        key = entry.name[:4].lstrip("0") or "0"
        manifest.pop(key, None)
        manifest.pop(entry.name[:4], None)


def snapshot(session_id: str, file_path: str | Path, workspace_root: Path) -> str | None:
    """Snapshot the current bytes of ``file_path`` before a write/edit.

    Returns a label like ``"#0003 notes.txt"`` or ``None`` when nothing could
    be snapshotted. Never raises.
    """
    try:
        root = Path(workspace_root).resolve()
        path = Path(file_path)
        if not path.is_absolute():
            path = root / path
        rel, flat = _relpath_flat(path, root)

        sdir = _session_dir(session_id, root)
        sdir.mkdir(parents=True, exist_ok=True)

        existing = _existing_numbers(sdir)
        n = (max(existing) + 1) if existing else 1
        fname = f"{n:04d}-{flat}.bak"
        bak_path = sdir / fname

        is_new = not path.exists()
        data = NEW_FILE_MARKER if is_new else path.read_bytes()
        bak_path.write_bytes(data)

        manifest = _load_manifest(sdir)
        label = f"#{n:04d} {rel}{' (new file)' if is_new else ''}"
        manifest[str(n)] = {
            "n": n,
            "relpath": rel,
            "flat": flat,
            "created": datetime.now().isoformat(timespec="seconds"),
            "label": label,
            "new_file": is_new,
        }
        _save_manifest(sdir, manifest)
        _prune_old(sdir, manifest)
        _save_manifest(sdir, manifest)
        return label
    except Exception:
        return None


def list_checkpoints(session_id: str, workspace_root: Path) -> list[dict]:
    """List checkpoints for a session, oldest first.

    Each dict has keys ``n``, ``relpath``, ``created``, ``label``, ``new_file``.
    """
    try:
        root = Path(workspace_root).resolve()
        sdir = _session_dir(session_id, root)
        if not sdir.is_dir():
            return []
        manifest = _load_manifest(sdir)
        entries: list[dict] = []
        for entry in sorted(sdir.iterdir(), key=lambda e: e.name):
            if not (entry.is_file() and entry.suffix == ".bak"):
                continue
            key = entry.name[:4]
            if not key.isdigit():
                continue
            n = int(key)
            meta = manifest.get(str(n), {})
            relpath = str(meta.get("relpath") or entry.name[5:-4].replace("__", "/"))
            entries.append(
                {
                    "n": n,
                    "relpath": relpath,
                    "created": str(meta.get("created", "")),
                    "label": str(meta.get("label", f"#{n:04d} {relpath}")),
                    "new_file": bool(meta.get("new_file", False)),
                }
            )
        return entries
    except Exception:
        return []


def restore(session_id: str, n: int, workspace_root: Path) -> str:
    """Restore the file snapshot ``n`` for ``session_id``.

    Writes the snapshot bytes back to the file, or deletes the file when the
    checkpoint records a file that did not exist yet. Returns the relpath that
    was restored. Raises ValueError when the checkpoint does not exist.
    """
    root = Path(workspace_root).resolve()
    sdir = _session_dir(session_id, root)
    matches = [e for e in sdir.glob(f"{int(n):04d}-*.bak")] if sdir.is_dir() else []
    if not matches:
        raise ValueError(
            f"no checkpoint #{int(n):04d} for session {session_id!r} "
            f"(workspace {root})"
        )
    bak_path = matches[0]
    data = bak_path.read_bytes()

    manifest = _load_manifest(sdir)
    meta = manifest.get(str(int(n)), {})
    relpath = str(meta.get("relpath") or bak_path.name[5:-4].replace("__", "/"))
    new_file = bool(meta.get("new_file", data == NEW_FILE_MARKER))

    target = root / relpath if not Path(relpath).is_absolute() else Path(relpath)
    if new_file:
        if target.exists():
            target.unlink()
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return relpath
