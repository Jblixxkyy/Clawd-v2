"""Caching: mtime-based disk cache for tags + keyed in-memory map cache.

Disk cache: one JSON file per repo root under the cache dir, storing
{rel: mtime} for every source file plus the serialized tags. Any mtime
change, added file, or removed file invalidates the entry.

In-memory cache: keyed on (root, chat files, mentioned idents/files,
budget, mtime signature) so repeated get_map() calls in a session are
free.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .tags import Tag

CACHE_VERSION = 1


def default_cache_dir() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "clawd-code" / "repomap"


class RepoMapDiskCache:
    """mtime-validated on-disk cache of parsed tags, one file per repo."""

    def __init__(self, root: Path, cache_dir: Path | None = None):
        self.root = Path(root)
        self.cache_dir = Path(cache_dir) if cache_dir else default_cache_dir()
        digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:16]
        self.path = self.cache_dir / f"{digest}.json"

    # -- mtimes ----------------------------------------------------------

    def _mtimes(self, files: list[str]) -> dict[str, float] | None:
        mtimes: dict[str, float] = {}
        for rel in files:
            try:
                mtimes[rel] = (self.root / rel).stat().st_mtime
            except OSError:
                return None
        return mtimes

    def mtime_signature(self, files: list[str]) -> tuple:
        """Hashable signature of the repo's current mtimes."""
        m = self._mtimes(files)
        if m is None:
            return (time.time_ns(),)
        return tuple(sorted(m.items()))

    # -- persistence -----------------------------------------------------

    @staticmethod
    def _ser_tag(t: Tag) -> dict:
        return {
            "rel": t.rel, "name": t.name, "qualname": t.qualname,
            "kind": t.kind, "line": t.line, "signature": t.signature,
            "count": t.count,
        }

    @staticmethod
    def _deser_tag(d: dict) -> Tag:
        return Tag(
            d["rel"], d["name"], d["qualname"], d["kind"], d["line"],
            d.get("signature", ""), d.get("count", 1),
        )

    def save(self, files: list[str], tags: list[Tag]) -> None:
        mtimes = self._mtimes(files)
        if mtimes is None:
            return
        payload = {
            "version": CACHE_VERSION,
            "root": str(self.root),
            "files": mtimes,
            "tags": [self._ser_tag(t) for t in tags],
        }
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass

    def _read(self) -> dict | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if data.get("version") != CACHE_VERSION or data.get("root") != str(self.root):
            return None
        return data

    def load_stale(self, files: list[str]) -> list[Tag] | None:
        """Return cached tags without mtime validation (on_demand mode)."""
        data = self._read()
        if data is None:
            return None
        return [self._deser_tag(d) for d in data.get("tags", [])]

    def load_if_fresh(self, files: list[str]) -> list[Tag] | None:
        """Return cached tags only if every file's mtime still matches."""
        data = self._read()
        if data is None:
            return None
        cached_files = data.get("files", {})
        if set(cached_files.keys()) != set(files):
            return None
        current = self._mtimes(files)
        if current is None:
            return None
        for rel, mtime in current.items():
            if cached_files.get(rel) != mtime:
                return None
        return [self._deser_tag(d) for d in data.get("tags", [])]

    def clear(self) -> None:
        try:
            self.path.unlink()
        except OSError:
            pass


class MapCache:
    """Keyed in-memory cache for rendered maps."""

    def __init__(self):
        self._store: dict[tuple, str] = {}

    @staticmethod
    def key(root, chat_fnames, mentioned_idents, mentioned_fnames,
            budget, mtime_sig=None) -> tuple:
        return (
            str(root),
            tuple(sorted(chat_fnames)),
            tuple(sorted(mentioned_idents)),
            tuple(sorted(mentioned_fnames)),
            int(budget),
            mtime_sig,
        )

    def get(self, key: tuple) -> str | None:
        return self._store.get(key)

    def put(self, key: tuple, text: str) -> None:
        # bound the cache; drop oldest entries past 64
        if len(self._store) >= 64:
            for k in list(self._store)[: len(self._store) - 63]:
                del self._store[k]
        self._store[key] = text

    def clear(self) -> None:
        self._store.clear()
