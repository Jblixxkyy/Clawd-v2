# 012 — Instruction file live-reload

## What
Watch project instruction files (AGENTS.md, CLAUDE.md) and notify mid-session when they change, instead of requiring a restart.

## Why
opencode's instruction-discovery (`packages/core/src/instruction-discovery.ts` in ~/workspace/opencode-ref) watches instruction files and injects messages like "The instructions changed: <diff>" / "New instructions apply from: <file>" / "Previously loaded instructions no longer apply." We load project instructions at startup via `src/context_system/claude_md.py` — edits made mid-session (common when Josh iterates on rules) are invisible until restart, and worse, the model keeps following stale instructions silently.

## Suggested approach
- Record mtimes + hashes of loaded instruction files at session start.
- Check each turn in `run_agent_loop` (cheap stat call); on change, inject a system-reminder-style message: which file changed and a compact diff summary (not the full file).
- Debounce rapid saves (only notify once per change burst, ~2s window).

## Expected tests
- Unit test: temp instruction file modified mid-loop → notification injected with the filename.
- Test unchanged file → no notification, no extra prompt tokens.
- Test rapid double-save → single notification (debounce).
