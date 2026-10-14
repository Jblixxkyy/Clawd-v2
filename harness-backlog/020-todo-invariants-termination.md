# 020 — Todo tracking with invariants + hard termination protocol

## What
A structured todo tool with harness-enforced invariants (whole-list overwrite, at most one `in_progress`, statuses include `blocked`, planning and execution tracking kept disjoint), plus a hard termination protocol: `complete_task` as the only way to finish (schema-validated), `ReportFailure` for typed failure exits, a forced wrap-up artifact when max steps is hit, and a two-clause completion check.

## Why
- gemini-cli `write-todos.ts` (https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/write-todos.ts): whole-list overwrite; validation **rejects more than one `in_progress`**; statuses include `blocked`.
- codex `plan_spec.rs` (https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/plan_spec.rs): "at most one step in_progress"; `update_plan` **rejected in Plan mode** (`plan.rs` L89) — planning and execution tracking deliberately disjoint.
- crush `todos.go`: each call submits the entire updated list; harness replaces persisted session state (survives restarts), validates the status enum server-side, returns deltas as response metadata, and nudges "keep using the todo list" — gives run→observe→reflect a durable backbone.
- gemini-cli `complete-task.ts` (https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/complete-task.ts): "the ONLY way to finish", zod-validated, so subagents can't ramble out.
- OpenHands SDK `FinishTool`: typed Pydantic model forces a structured final artifact (`outcome` fields); `parse_last_response` recovers it from the event log — forced artifacts constrain answer shape and lift weak models.
- opencode `max-steps.ts` (https://github.com/anomalyco/opencode/blob/dev/packages/core/src/session/runner/max-steps.ts): on the final step, append "CRITICAL - MAXIMUM STEPS REACHED… MUST summarize work done, remaining tasks, recommendations" and disable tools — a forced wrap-up artifact instead of a silent cutoff.
- continue termination protocol: terminal tools set `isComplete=true`; fallback: assistant message with no tool calls → complete. Two-clause completion check = clean loop hygiene.

Gives the agent a durable spine (todos) and the harness a clean end-of-run contract (termination). All model-agnostic. Pairs with 015 (done-gate) and 017 (Exit/ReportFailure tools).

## Suggested approach
- `Todo` tool: whole-list overwrite semantics; server-side validation (≤1 `in_progress`, valid status enum incl. `blocked`); persisted in session state (survive restarts); reject plan-tracking updates while in plan mode (disjoint from 016's plan artifact); return list deltas as response metadata.
- `complete_task` tool: the only terminal success path; schema-validated structured output (`outcome`, `summary`, `artifacts[]`); subagent loops treat its absence + no tool calls as stall, not success.
- Max-steps wrap-up: at the step cap, disable tools and inject the forced wrap-up prompt (work done / remaining / recommendations) — the run always ends with a usable artifact, never a truncation.
- Two-clause completion check in the loop: terminal tool set `isComplete`, OR final assistant message with no tool calls (latter flagged as "ended without explicit completion" in the log).

## Expected tests
- Two items `in_progress` → rejected with a clear error.
- Todo list survives a session restart.
- `update` to plan tracking while in plan mode → rejected.
- Run hitting max steps → wrap-up artifact produced, tools disabled.
- `complete_task` with invalid schema → rejected; valid → run ends with structured outcome in the log.
- Full test suite green before shipping.
