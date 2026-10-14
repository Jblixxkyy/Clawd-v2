# 024 — Forced scratchpad protocol for subagents

## What
Give delegated subagents (005's true agentic subagents) a mandatory scratchpad artifact: created on turn 1, updated every turn (checklist / Questions to Resolve / Key Findings / Irrelevant Paths), with termination defined as "the Questions to Resolve list is empty" — plus read-only scoping, turn/time caps, low temperature, and schema-forced JSON output.

## Why
From gemini-cli's `codebase-investigator` agent (https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/agents/codebase-investigator.ts#L172-L201): the clearest "weak model behaves smart" artifact in the research. The scratchpad externalizes working memory — a small model that would lose track mid-investigation instead maintains an explicit open-questions list, and the harness defines done as "no open questions" rather than trusting the model's judgment. Combined with read-only scoping, `maxTurns`/`maxTimeMinutes` caps, temp 0.1, and zod-forced JSON output, the subagent becomes a tightly-scaffolded research instrument instead of a loose cannon. This is the subagent-side answer to Josh's shallow-analysis complaint (the PC "is this repo good" chat was exactly an investigation task with no scaffold). 1:1 port, model-agnostic.

## Suggested approach
- Add a scratchpad artifact to the subagent loop: `{checklist[], questions_to_resolve[], key_findings[], irrelevant_paths[]}` — created turn 1, re-emitted/updated every turn (whole-overwrite, like 020's todos).
- Termination rule: the subagent may only finish when `questions_to_resolve` is empty AND the output validates against the task's JSON schema; otherwise the harness continues it (within caps).
- Standard investigator profile: read-only toolset, `maxTurns: 50`, `maxTimeMinutes: 10`, temperature 0.1, schema-forced JSON final output.
- Render the scratchpad live in the web UI (collapsible per-subagent panel) — watchable autonomous work, per Josh's "video game" directive.
- Reuse for the 018 reviewer subagents (their rubric verdict becomes the scratchpad's final state).

## Expected tests
- Investigation subagent maintains the scratchpad across turns; Questions to Resolve shrinks as it works.
- Subagent attempting to finish with open questions → continued by the harness.
- Caps enforced: 50 turns / 10 minutes → forced wrap-up, never hangs.
- Read-only profile: mutation tool call attempted → denied.
- Scratchpad visible in the UI per subagent.
- Full test suite green before shipping.
