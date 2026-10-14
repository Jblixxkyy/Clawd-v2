# 021 — REPOMAP: PageRank repo map with token-budgeted rendering

## What
A repository map that tells the model the shape of the codebase: tree-sitter symbol extraction → definition/reference graph → personalized PageRank seeded from chat files and mentioned identifiers → token-budgeted rendering (binary-search the ranked tags to fill the budget, bodies elided), with mtime-based disk caching.

## Why
From aider `repomap.py` (https://github.com/Aider-AI/aider/blob/main/aider/repomap.py): the single biggest structural context-engineering gap between Clawd-Code and the best open-source harness. A repo map gives even a weak model a working mental model of the codebase — which functions exist, who calls whom, what's relevant to the current task — without burning context on full file reads. It is the canonical small-model lifter: context *shaping* rather than context *size*. Tuned edge weights (in-chat ×50, user-mentioned ×10, long snake-case idents ×10), binary search over ranked tags to fill `max_map_tokens` ±15%, mtime disk cache + keyed map cache with refresh modes. Related free signal: aider's `check_for_file_mentions()` auto-feeds user-mentioned filenames into the PageRank personalization. Near-1:1 Python port (tree-sitter + networkx or a stdlib PageRank). Fully model-agnostic.

## Suggested approach
- Port the pipeline: tree-sitter tags (defs + refs) → MultiDiGraph → personalized PageRank (seed: files already in context + identifiers mentioned by the user, with aider's weight tuning as the starting point) → rank tags → binary-search how many fit in `max_map_tokens` → render signatures only (bodies elided).
- Cache: mtime-based disk cache per repo + keyed in-memory map cache with explicit refresh modes (auto/on-demand/off).
- Feed `check_for_file_mentions()`: scan user messages for filenames and auto-boost them in the personalization vector.
- Inject the map into context at session start and refresh on file changes (or on demand via a `/repomap` command); keep it prompt-diet-friendly (001): budget-capped, terse signature lines, omitted when the repo is trivially small.
- Respect ignores (.gitignore etc.) and binary files.

## Expected tests
- Map generated for a sample repo: top-ranked symbols include the files in context and user-mentioned identifiers (personalization works).
- Rendered map stays within `max_map_tokens` ±15% on repos of varying size.
- Cache: unchanged repo → cache hit (no re-parse); touched file → refresh.
- User mentions a filename in chat → that file's symbols rank higher in the next map.
- Benchmark: retrieval quality on a "find the relevant function" task with and without the map (map should win, especially with a small model).
- Full test suite green before shipping.
