# 030 — Skill loading as a small-model lifter: triggers, markers, progressive disclosure

## What
Make skill discovery and loading context-shaped for weak models: keyword `triggers:` in skill frontmatter, marker-triggered auto-activation (e.g. `uv.lock` present → uv skill loads), shell commands in frontmatter for dynamic context, progressive-disclosure SKILL.md loading, and per-model-family system prompt variants.

## Why
From the OpenHands software-agent-sdk skill system (https://github.com/OpenHands/software-agent-sdk): skills are not just a catalog — the *loading mechanism* is a small-model lifter. A weak model with 50 skills dumped into context drowns; the same model with 2 skills auto-activated by triggers/markers (because the repo has `uv.lock`, because the task mentions "pdf") behaves far smarter. This is distinct from the briefed 004 (skills Q&A eval — that's an eval harness) and 009 (skill @-mentions — that's explicit user invocation): this brief is the *automatic, relevance-shaped* loading layer. opencode's `optimize.ts` (https://github.com/anomalyco/opencode/blob/dev/packages/core/src/plugin/optimize.ts) adds the companion idea: per-model-family system prompt variants + pruning tools from the advertised catalog that a given model family doesn't need (e.g. drop grep/glob for models that shell anyway). All model-agnostic context engineering.

## Suggested approach
- Extend skill frontmatter: `triggers:` keyword list; `markers:` (repo markers like `uv.lock`, `package.json`, `.ps1` files); optional `context_cmd:` shell command whose stdout is injected when the skill activates.
- Activation: on session start and on relevant events (file mentions, tool results), score skills by trigger/marker match; auto-load matches progressively — SKILL.md body only for activated skills, one-line index for the rest.
- Progressive disclosure: skill bodies load in tiers (name+description → full body on first use), keeping the base prompt lean per 001's prompt diet.
- Per-model-family prompt variants: small overrides file keyed by model family for skill-heavy prompts (wording tuned per family, same semantics).
- Catalog pruning: drop tools from the advertised set that the active model family demonstrably doesn't need (config-driven, conservative default: no pruning).

## Expected tests
- Repo with `uv.lock` → uv skill auto-activates; repo without → it doesn't.
- Keyword trigger in the user message → matching skill loads its body; non-matching skills stay as one-liners.
- `context_cmd` output appears in the skill's injected context.
- Token count of the skill block with 50 installed skills stays under budget (progressive disclosure works).
- Full test suite green before shipping.
