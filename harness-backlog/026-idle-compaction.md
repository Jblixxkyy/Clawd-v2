# 026 — Idle compaction: auto-compact large idle conversations

## What
When a conversation exceeds a token threshold and then sits idle for a configured duration, compact it automatically in the background — no user action, no blocking freeze.

## Why
From Claude Code 2.1.299 (changelog, https://code.claude.com/docs/en/changelog — released Oct 14, 2026, newer than our 2.1.296 watermark): "idle compaction in the terminal (a 200K+ token conversation idle ~54 minutes auto-compacts, configurable via `CLAUDE_CODE_IDLE_COMPACTION_MIN_TOKENS` or off with `"idleCompaction": false`)". The insight is behavioral, not technical: users don't compact proactively — they hit the context wall mid-thought and lose flow. Compacting during idle time means the next interaction starts fresh instead of starting with a multi-minute freeze (or a wall). Closed-source, but the mechanism is fully portable and model-agnostic: it's scheduling around the existing compaction (022), not a model feature. File as "Clawd-native equivalent" per the standing rule for non-1:1-portable items.

## Suggested approach
- Config: `idle_compaction: {enabled, min_tokens (default 200_000), idle_minutes (default ~54), }` plus a hard off switch — mirroring Claude's shape so the semantics stay familiar.
- Idle detection: no user/model activity for `idle_minutes` while context ≥ `min_tokens` → run the async compaction path from 022 (background, checkpoint-adopt discipline); never run foreground from the idle trigger.
- Guardrails: never idle-compact while background work is running (Bash jobs, subagents); never compact twice for the same idle window; log the trigger and the before/after token counts.
- UI: a quiet "compacted while idle (X→Y tokens)" notice on next interaction — visible, not noisy.

## Expected tests
- Simulated idle past the threshold with a large context → background compaction runs; next interaction starts on the compacted context.
- Idle with small context → no compaction.
- Background Bash job running → idle compaction deferred until it finishes.
- Disabled via config → never triggers.
- Double idle window → compacts at most once per window.
- Full test suite green before shipping.
