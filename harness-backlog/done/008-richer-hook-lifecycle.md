# 008 — Richer hook lifecycle

## What
Expand hooks beyond PreToolUse/PostToolUse to cover the full agent lifecycle: session, agent-turn, model-call, and compaction events.

## Why
Our hooks (`src/tool_system/hooks/runner.py`, config `~/.clawd/hooks.json`) fire exactly 2 events: `PreToolUse` and `PostToolUse`, as shell commands, fail-open. Compare:
- gemini-cli (`packages/core/src/hooks/types.ts` in /tmp/gemini-cli): 12 events — BeforeTool, AfterTool, BeforeAgent, AfterAgent, SessionStart, SessionEnd, PreCompress, BeforeModel, AfterModel, BeforeToolSelection, Notification — with block/deny/ask decisions, trusted hooks, and a hook planner.
- opencode (`packages/core/src/plugin/hooks.ts` in ~/workspace/opencode-ref): 5 typed domains — session, permission, shell, tool, aisdk — with `execute.before` able to reject a tool call.
- crush (`internal/hooks/hooks.go` in /tmp/crush): PreToolUse with halt semantics (exit 49 halts the whole turn, distinct from error exits).

SessionStart/End alone unlocks real workflows (env setup, session logging); PreCompress lets hooks react to compaction; BeforeModel/AfterModel enables prompt/response inspection.

## Suggested approach
- Add events incrementally, highest value first: `SessionStart`, `SessionEnd`, `BeforeAgent`, `AfterAgent`, `PreCompress`, `BeforeModel`, `AfterModel`. Keep `PreToolUse`/`PostToolUse` working exactly as today (backward compat).
- Extend the hooks.json schema: per-event command lists; runner passes event-specific JSON on stdin; support decisions `allow` / `block` (halt turn with message) / `ask` — currently everything is fail-open allow.
- Fire the new events from `run_agent_loop` (agent/turn boundaries), session start/end paths, and the compaction trigger.

## Expected tests
- Unit test per new event: stub hook script receives the right JSON payload.
- Test `block` decision halts the turn with the hook's message.
- Test malformed hook output still fails open (never breaks the loop).
- Existing PreToolUse/PostToolUse tests unchanged and green.
