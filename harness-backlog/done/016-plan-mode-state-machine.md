# 016 — Plan mode as a permission-enforced state machine

## What
Rebuild plan mode as a real state machine, not a prompt suggestion: enter/exit-plan-mode tools, a declarative mode-scoped tool policy (deny-all + read-only re-allow in plan mode), a plan-file validation gate before approval is requested, plan preservation across compaction, and a read-only planner agent profile whose toolset physically excludes mutation tools.

## Why
Three harnesses converge on the same mechanism, each contributing a piece:
- gemini-cli: `enter-plan-mode.ts` / `exit-plan-mode.ts` (https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/tools/enter-plan-mode.ts) — exit validates the plan file exists and is non-empty *before* asking approval; rejection-with-feedback → revise loop. `plan.toml` L60–L76 (https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/policy/policies/plan.toml) — priority-tiered mode-scoped allow/deny (`toolName="*"` denied in plan mode, read-only tools re-allowed, writes only for `.md` in the plans dir). Approved-plan prompt injection (`snippets.ts` L981–997: "Treat this file as your single source of truth") and plan preserved across compaction (L1462–1476).
- opencode: `plan.ts` agent (`mode: "primary"`, read-only via permission ruleset, https://github.com/anomalyco/opencode/blob/dev/packages/core/src/plugin/plan.ts) — plus a `session.hook("context")` **self-healing reminder** that re-injects the plan-mode reminder after compaction strips it.
- crush: plan mode as a separate SessionAgent with its own system prompt (`internal/agent/templates/plan.md.tpl`); file-mutation tools and bash *physically absent* from its toolset (prompt says don't, toolset enforces — defense in depth); final response bracketed by `<!-- CRUSH_PLAN_START -->` / `<!-- CRUSH_PLAN_READY -->` markers, UI opens an approve/edit/feedback dialog on detection. The template forces a self-review workflow (synthesize → list gaps/contradictions/unverified assumptions → re-read to close them → list unknowns before drafting) — pure prompt text, doubles as a small-model lifter.
- OpenHands SDK `planning.py`: forced `PLAN_STRUCTURE` (OBJECTIVE, CONTEXT SUMMARY, APPROACH OVERVIEW, IMPLEMENTATION STEPS as goal+method+reference tuples, TESTING AND VALIDATION) rendered into the planner prompt; planner fenced to Glob/Grep + a PLAN.md-only editor.
- cline `/deep-planning`: slash command expanding to a four-step protocol (silent investigation → clarifying questions → forced `implementation_plan.md` → implementation with trackable steps).

Clawd already has `plan_mode.py` — this brief upgrades it from a flag to an enforced mechanism. Fully model-agnostic.

## Suggested approach
- Add `enter_plan_mode` / `exit_plan_mode` tools. Exit validates: plan file exists, non-empty, matches the plan schema (adopt OpenHands' PLAN_STRUCTURE sections); only then request user approval. Rejection with feedback → revise loop (stay in plan mode).
- Add a mode-scoped tool policy (gemini `plan.toml` pattern): in plan mode, deny all tools by default; re-allow read-only tools; allow writes only to `*.md` under the plans dir. Enforce in the tool dispatcher, not just the prompt.
- On approval, inject the approved plan as "single source of truth" context and make plan-mode reminders compaction-safe (re-inject post-compaction, opencode pattern).
- Ship a planner agent profile (for 005-style subagents): restricted toolset, plan template prompt with the crush self-review workflow baked in, marker-delimited plan output the UI can detect for an approve/edit dialog.
- Add `/deep-planning` slash command implementing cline's four-step protocol as the manual entry point.

## Expected tests
- In plan mode, a write tool call outside the plans dir is denied by the dispatcher (not just discouraged by prompt).
- `exit_plan_mode` with missing/empty plan → validation error, stays in plan mode; with valid plan → approval prompt.
- Plan survives a compaction cycle (reminder re-injected; plan file reference intact).
- Planner subagent cannot call mutation tools even if the model tries (toolset exclusion).
- Full test suite green before shipping.
