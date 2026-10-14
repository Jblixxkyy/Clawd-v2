"""REPOMAP: PageRank repository map with token-budgeted rendering.

Near-1:1 port of aider's repomap.py pipeline, adapted for Clawd-Code:

    source files -> tags (defs + refs) -> definition/reference graph
    -> personalized PageRank -> ranked tags
    -> binary search over ranked tags to fill max_map_tokens
    -> rendered signature-only map (bodies elided)

Tag extraction uses a regex/heuristic tagger (tree-sitter is not installed
in this environment and pip is blocked by PEP 668), Python-focused.
PageRank is implemented with the stdlib (no networkx dependency).

Public API:
    RepoMap              - main entry point
    check_for_file_mentions
    extract_mentioned_idents
    build_repo_map       - one-shot convenience helper
"""

from __future__ import annotations

from .cache import MapCache, RepoMapDiskCache
from .mentions import check_for_file_mentions, extract_mentioned_idents
from .pagerank import rank_tags
from .render import render_map
from .tags import Tag, collect_source_files, extract_refs, extract_tags

__all__ = [
    "RepoMap",
    "Tag",
    "MapCache",
    "RepoMapDiskCache",
    "build_repo_map",
    "check_for_file_mentions",
    "collect_source_files",
    "extract_mentioned_idents",
    "extract_tags",
    "rank_tags",
    "render_map",
]


def build_repo_map(
    root,
    *,
    chat_fnames=(),
    mentioned_idents=(),
    mentioned_fnames=(),
    max_map_tokens=1024,
    refresh="auto",
):
    """One-shot helper: build and render a repo map for *root*."""
    rm = RepoMap(root, max_map_tokens=max_map_tokens, refresh=refresh)
    return rm.get_map(
        chat_fnames=chat_fnames,
        mentioned_idents=mentioned_idents,
        mentioned_fnames=mentioned_fnames,
    )


class RepoMap:
    """Build, cache, and render a PageRank-weighted repository map.

    Args:
        root: repository root directory.
        max_map_tokens: token budget for the rendered map.
        refresh: cache refresh mode — "auto" (mtime-based), "on_demand"
            (only recompute when refresh() is called), or "off" (no caching).
        cache_dir: override for the on-disk tag cache directory.
    """

    def __init__(self, root, *, max_map_tokens=1024, refresh="auto", cache_dir=None):
        from pathlib import Path

        self.root = Path(root).expanduser().resolve()
        self.max_map_tokens = int(max_map_tokens)
        if refresh not in ("auto", "on_demand", "off"):
            raise ValueError(f"unknown refresh mode: {refresh!r}")
        self.refresh_mode = refresh
        self._disk = RepoMapDiskCache(self.root, cache_dir=cache_dir)
        self._mem = MapCache()
        self._files = None  # cached source file listing

    # -- file discovery -------------------------------------------------

    def source_files(self):
        """List of repo-relative source files considered for the map."""
        if self._files is None:
            self._files = collect_source_files(self.root)
        return self._files

    def source_file_count(self):
        return len(self.source_files())

    # -- tags -----------------------------------------------------------

    def _parse_all(self, files):
        """Two-pass parse: defs first, then refs against repo-wide def names."""
        defs_per_file: dict[str, list[Tag]] = {}
        for rel in files:
            tags = extract_tags(self.root / rel, rel, repo_def_names=set())
            defs_per_file[rel] = [t for t in tags if t.kind != "ref"]
        repo_names = {t.name for ts in defs_per_file.values() for t in ts}
        out: list[Tag] = []
        for rel in files:
            out.extend(defs_per_file[rel])
            out.extend(extract_refs(self.root / rel, rel, repo_names))
        return out

    def get_tags(self):
        """All tags for the repo, honoring the refresh mode / caches."""
        files = self.source_files()
        if self.refresh_mode == "off":
            return self._parse_all(files)
        cached = self._disk.load_if_fresh(files)
        if cached is not None:
            return cached
        if self.refresh_mode == "on_demand":
            # on_demand: reuse any disk cache regardless of mtime
            stale = self._disk.load_stale(files)
            if stale is not None:
                return stale
        tags = self._parse_all(files)
        self._disk.save(files, tags)
        return tags

    def refresh(self):
        """Force a re-parse of the repo, clearing all caches."""
        self._files = None
        self._disk.clear()
        self._mem.clear()

    # -- ranking + rendering --------------------------------------------

    def ranked_tags(self, *, chat_fnames=(), mentioned_idents=(), mentioned_fnames=()):
        tags = self.get_tags()
        return rank_tags(
            tags,
            chat_fnames=chat_fnames,
            mentioned_idents=mentioned_idents,
            mentioned_fnames=mentioned_fnames,
        )

    def get_map(self, *, chat_fnames=(), mentioned_idents=(), mentioned_fnames=()):
        """Render the repo map, honoring the refresh mode / caches.

        chat_fnames: repo-relative files already in the conversation.
        mentioned_idents: identifiers the user mentioned (boosted x10).
        mentioned_fnames: files the user mentioned (seeded like chat files).
        """
        if self.refresh_mode == "on_demand":
            key = self._mem.key(
                self.root, chat_fnames, mentioned_idents, mentioned_fnames,
                self.max_map_tokens,
            )
            hit = self._mem.get(key)
            if hit is not None:
                return hit
            text = render_map(
                self.ranked_tags(
                    chat_fnames=chat_fnames,
                    mentioned_idents=mentioned_idents,
                    mentioned_fnames=mentioned_fnames,
                ),
                max_map_tokens=self.max_map_tokens,
            )
            self._mem.put(key, text)
            return text

        if self.refresh_mode == "off":
            return render_map(
                self.ranked_tags(
                    chat_fnames=chat_fnames,
                    mentioned_idents=mentioned_idents,
                    mentioned_fnames=mentioned_fnames,
                ),
                max_map_tokens=self.max_map_tokens,
            )

        # auto: mem-cache keyed on the current mtime signature of the repo
        sig = self._disk.mtime_signature(self.source_files())
        key = self._mem.key(
            self.root, chat_fnames, mentioned_idents, mentioned_fnames,
            self.max_map_tokens, sig,
        )
        hit = self._mem.get(key)
        if hit is not None:
            return hit
        text = render_map(
            self.ranked_tags(
                chat_fnames=chat_fnames,
                mentioned_idents=mentioned_idents,
                mentioned_fnames=mentioned_fnames,
            ),
            max_map_tokens=self.max_map_tokens,
        )
        self._mem.put(key, text)
        return text
