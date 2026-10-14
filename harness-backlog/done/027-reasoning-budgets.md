# 027 — Clawd-native reasoning budgets: per-role, per-phase effort controls

## What
A model-agnostic reasoning-budget abstraction: one portable `reasoning{enabled, effort: low|medium|high|max, budget_tokens}` setting that resolves per model call, plus phase/role-aware budget policies (plan bigger than execute; investigator high, classifier low) — with a Clawd-native fallback (N forced reflection turns) where the provider exposes no thinking control.

## Why
Convergent finding across the research:
- cline `portable-reasoning.ts` (322 lines, https://github.com/cline/cline — `sdk/packages/llms/src/providers/routing/portable-reasoning.ts`): one portable `reasoning{enabled, effort, budgetTokens}` resolves per call — snap to the model's advertised effort ladder, fit-disable sends "none" only where advertised, native provider options win, `mayReason` skips non-reasoning models. Prevents whole classes of 400-errors across 200+ models.
- codex `plan_mode_reasoning_effort` (`config_toml.rs` L399): bigger think budget while planning than executing.
- gemini-cli `defaultModelConfigs.ts` L23–87: classifier 512 tokens, helpers 0, investigator HIGH, loop-double-check escalates to pro.
- Claude Code 2.1.292/2.1.299: `effort` parameter on the Agent tool, `/effort` and `/fast` agent-view defaults, `/usage` effort warnings — effort as a user-facing control.
- opencode `Message.effort` per-message markers.

Raw provider thinking knobs fail the fitness test (API-specific). But the *policy layer* — budget thinking by phase/role, one portable schema, graceful degradation — is pure harness and passes: it makes any model cheaper where thinking is waste and stronger where it matters. Filed as "needs a Clawd-native equivalent" for the provider-mapping half, per the standing rule.

## Suggested approach
- Define the portable schema: `reasoning{enabled: bool, effort: low|medium|high|max, budget_tokens: int|null}` per call site.
- Adapter layer: map to native provider options where they exist (Anthropic thinking budgets, OpenAI reasoning effort, Gemini thinking level/budget); where the model advertises no reasoning control, send nothing (never 400); native user-specified options always win over the portable value.
- Phase/role policies (defaults, user-overridable): planning > executing; investigator/reviewer roles high; classifiers/summarizers/titles low or off; compaction medium.
- Clawd-native fallback: where no provider knob exists, emulate effort with structure — `high` = +N forced `think` (023) / reflection (019) turns before acting; `low` = none. Effort becomes behavior, not just a parameter.
- User-facing: `/effort` command + per-agent config, surfaced in `/usage`-style cost views.

## Expected tests
- Portable `high` on a reasoning model → native budget param sent; on a non-reasoning model → nothing sent, no 400, `mayReason` skips cleanly.
- Native user option overrides the portable value.
- Planning phase gets a larger budget than execution under default policy.
- No-knob provider + `effort: high` → forced think turns appear in the transcript.
- Full test suite green before shipping.
