# 028 — Model tiers: weak-model sidecar for utility calls

## What
First-class model tiers in the harness: a cheap "small model" binding for utility calls (titles, summaries, commit messages, compaction, reviewers, judges) so the main model's budget is preserved for reasoning — plus per-mode model binding (frontier plans, cheap acts) and restricted toolsets for weak-model subagents.

## Why
Three harnesses converge on this textbook small-model lifter:
- cline: "different models for Plan and Act" setting — frontier model plans, cheap model executes.
- aider: `ChatSummary(weak_model, main_model)` — history summarization in a background thread on the weak model; weak model also writes commit messages.
- crush: `largeModel`/`smallModel` bindings; sub-agent tools run on the small model with a restricted toolset (web_fetch, web_search, glob, grep, view) and auto-approved permissions — weak models only ever run tightly-scaffolded subtasks.
- opencode: `small_model` routing for title/summary/compaction calls.

The thesis fit is direct: the harness, not the model, is the product — routing the *right* work to the *cheap* model is harness intelligence. It also makes 018's reviewer/judge loops and 022's compaction nearly free to run. Needs Clawd-native model-tier wiring (config + routing), but the concept is 1:1 and model-agnostic (works with any provider pair).

## Suggested approach
- Config: `models: {main, small, plan?, act?}` — small defaults to main if unset (no behavior change for single-model users).
- Route utility calls to the small model: session titles, compaction summaries (022), commit messages, reviewer/judge roles (018), edit-corrector (019), file-mention classification.
- Per-mode binding: plan mode may use a stronger model, act mode the cheaper one (user opt-in).
- Weak-model subagent profile (024's investigator): small model + restricted read-only toolset + auto-approved read permissions — tight scaffolding compensates for the weaker model.
- Cost accounting attributes spend per tier (feeds the cost tracking Josh already has).

## Expected tests
- With small model configured: compaction and title calls go to the small model; main-model token spend drops on a long session.
- Single-model config: everything routes to main, zero behavior change.
- Plan/act split: plan turns use the plan model, execution the act model.
- Weak-model subagent cannot call tools outside its restricted set.
- Cost report splits spend by tier.
- Full test suite green before shipping.
