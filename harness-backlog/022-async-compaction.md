# 022 — Async/background compaction wired into the turn lifecycle

## What
Run compaction in the background at turn-lifecycle checkpoints (pre-turn, mid-turn, post-turn) instead of only as a blocking foreground event: launch it when eligible, adopt the ready result at the next checkpoint while preserving newer messages/tool outputs/world-state updates, and fall back to one foreground compaction if the adopted result still exceeds the limit. Never let a cancelled turn launch new background work (reuse already-running work later).

## Why
From codex "Wire background compaction into the turn lifecycle" (#53800, https://github.com/openai/codex/commit/2b2d7923055f3ad4827270edefe65e71293cf26c): this resolves the PARITY.md watch item "Async compaction configuration" with the full mechanism, not just a flag. Clawd already has real compaction (foreground); backgrounding it removes the worst UX wart of long sessions — the multi-minute freeze while the context compresses — and the checkpoint-adopt design (preserve newer messages, reject stale results, one foreground fallback) is the correct concurrency discipline. Also relevant: codex retains client-marked tool outputs across compaction (commit 6c0c6759) — summaries drop exactly the evidence weak models need, so marked outputs must be protected. And opencode's compaction discipline (`compaction.ts`, https://github.com/anomalyco/opencode/blob/dev/packages/core/src/session/compaction.ts): forced `SUMMARY_TEMPLATE` ("You MUST use this format"), a token reserve (`RESERVE_MIN = 16_000`), shrink steps `[0.7, 0.5, 0.35]`; codex's compact prompt forces handoff framing (progress/decisions/remaining/critical refs). Fold those in: structured summary template + reserve + evidence protection. Fully model-agnostic.

## Suggested approach
- Add an `async_compaction` config (default off initially, like codex's `AsyncCompaction` flag): when enabled, eligible pre/mid/post-turn checkpoints capture world state + client routing state and launch a background compaction task.
- Adoption protocol: at the next checkpoint, adopt the ready result — but preserve messages, tool outputs, and state updates that arrived after the snapshot; discard/reject stale results; if the adopted summary still exceeds the limit, run exactly one foreground compaction.
- Cancellation discipline: a cancelled turn never launches new background work; already-running compactions are kept for later reuse.
- Structured summary: forced template (progress / decisions / remaining / critical references), token reserve floor, and client-marked tool outputs exempt from summarization (evidence protection).
- Surface compaction state in the UI (compacting… / adopted / foreground fallback) so it is watchable, not mysterious.

## Expected tests
- Long session with async compaction on: no blocking pause at the compaction threshold; summary adopted at the next checkpoint.
- Messages arriving during background compaction are preserved, not summarized away.
- Marked tool outputs survive compaction verbatim.
- Cancelled turn → no new background compaction launched; in-flight one reused later.
- Adopted summary still over limit → exactly one foreground compaction, then continue.
- Full test suite green before shipping.
