# 031 — Hook-lifecycle extension: BeforeModel / BeforeToolSelection / session.retry

## What
Extend the hook lifecycle built in 008 with the enforcement-grade hook points the research surfaced: `BeforeModel` (block/stop/inject/rewrite the outgoing model call), `BeforeToolSelection` (rewrite the tool list before the model sees it), `tool.execute.before` (mutate input or fail pre-execution), and `session.retry` veto.

## Why
- gemini-cli `hookSystem.ts` L270–408 (https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/hooks/hookSystem.ts): `BeforeModel` can block, stop, inject, or rewrite; `BeforeToolSelection` can rewrite the tool list the model chooses from. These are the enforcement points that make 016's plan-mode policy, 027's budget routing, and 030's catalog pruning *actually* enforceable — without them, those features are prompt suggestions the model can ignore.
- opencode `hooks.ts` L20–30 (https://github.com/anomalyco/opencode/blob/dev/packages/core/src/plugin/hooks.ts): `tool.execute.before` mutates input or fails pre-execution (the seam 007's input repair and 019's edit-corrector plug into); `session.retry` veto.
- 008 built the 2-event foundation (PreToolUse/PostToolUse); this is the follow-up that turns hooks from observability into control. The 015 done-gate and 019 verify loop both want hook points that don't exist yet. Fully model-agnostic.

## Suggested approach
- Add hook points: `BeforeModel` (receives the assembled prompt; may block, inject context, or rewrite), `BeforeToolSelection` (receives the candidate tool list; may filter/reorder — the enforcement seam for 016's plan-mode policy and 030's pruning), `ToolExecuteBefore` (may mutate input or fail fast with a reason), `SessionRetry` (may veto a retry).
- Keep the 008 hook API shape (same registration, same event envelope) so existing hooks keep working; document the ordering guarantees (selection → before-execute → execute → post).
- Guardrails: hook timeouts (a hanging hook must not hang the loop), recursion protection (hooks cannot trigger themselves), and audit logging of every block/rewrite decision.
- Migrate 016's plan-mode policy onto `BeforeToolSelection` as the reference implementation.

## Expected tests
- `BeforeToolSelection` hook filtering write tools in plan mode → model never sees them (016's policy enforced, not suggested).
- `BeforeModel` injection appears in the prompt the model receives.
- `ToolExecuteBefore` mutating input → tool executes with mutated input; failing → clean pre-execution error, no partial effects.
- Hanging hook → timeout fires, loop continues with a logged warning.
- Hook attempting to trigger itself → recursion guard stops it.
- Full test suite green before shipping.
