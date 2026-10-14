# Clawd-Code Parity Checklist

Kept by the daily `clawd-harness-scout` job. Upstream watermarks live in `upstream-watch.json`.
Feature-level review starts on run 2 (2026-10-14) — this first run established baselines only.

Status key: `ours` = already in Josh's/Jeff's Clawd-Code | `briefed` = build brief filed in `harness-backlog/` | `watch` = under evaluation | `gap` = parity gap, needs brief

## Watch-list health (as of 2026-10-13)

| Harness | State | Last activity |
|---|---|---|
| opencode | active, v1.19.0 | today |
| codex | active, rust-v0.162.1 | today |
| gemini-cli | active, v0.63.0 | 2026-10-12 |
| cline | very active, SDK v0.0.93 | today |
| crush | active, v0.98.1 | 2026-10-12 |
| OpenHands | active, v1.27.1 | today |
| aider | dormant (main untouched since May 2026) | 2026-05-22 |
| continue | quiescent (login retired after acquisition) | 2026-07-21 |
| Roo-Code | ARCHIVED | 2026-05-15 (frozen reference) |
| GPT-AGI/Clawd-Code (origin) | dormant | 2026-09-22 |
| Claude Code | active, v2.1.296 (changelog only) | ~2026-10-09 |

Corrections from run 1: `sst/opencode` → `anomalyco/opencode` (default branch `dev`); `All-Hands-AI/OpenHands` → `OpenHands/OpenHands`; `Cline/cline` → canonical owner lowercase `cline`.

## Feature checklist (baseline audit 2026-10-13 — read their actual code)

Feature | Theirs | Ours | Status
---|---|---|---
Prompt diet (lean core prompt, conditional per-tool guidance, <env> block) | opencode | — | briefed → `001-prompt-diet.md`
Delta instruction system | — | — | briefed → `002-delta-instruction-system.md`
Quiet tool display | — | — | briefed → `003-quiet-tool-display.md` (also an OPEN ask from Josh)
Skills Q&A eval regression test | — | — | briefed → `004-skills-qa-eval-regression.md`
**True agentic subagents** (LLM-driven child loop, not fixed call list) | opencode (prompt+model+background+sessionID), gemini-cli (specialized agents), codex (multi-agent v2 + message board), Claude Code (nesting, effort) | `Agent` tool only executes a pre-declared call list | briefed → `005-true-agentic-subagents.md`
**Tool-call loop detection** (break stuck repetition early) | crush (`loop_detection.go`: 10-step window, >5 repeats = stuck) | only `max_turns=50` | briefed → `006-tool-call-loop-detection.md`
**Tool input repair** (coerce `{"limit":"20"}` → `{"limit":20}`) | opencode (`tool-input-repair.ts` on execute.before) | validate-only, rejects | briefed → `007-tool-input-repair.md`
**Richer hook lifecycle** (session/agent/model/compaction events) | gemini-cli (12 events), opencode (5 typed domains), crush (halt exit 49) | 2 events: PreToolUse/PostToolUse | briefed → `008-richer-hook-lifecycle.md`
**Skill @-mentions** (`$skill://pdf` explicit invocation) | codex (`skills/src/mentions.rs`) | none — model-mediated only | briefed → `009-skill-mentions.md`
**A2A client** (delegate to external agents) | gemini-cli (`a2a-client-manager.ts`) | none | briefed → `010-a2a-client.md`
**WebFetch offset + unread reporting** | Claude Code 2.1.290 | truncates at 100k, no resume | briefed → `011-webfetch-offset.md`
**Instruction file live-reload** (AGENTS.md changes mid-session) | opencode (`instruction-discovery.ts`) | loaded at startup only | briefed → `012-instruction-live-reload.md`
Async compaction configuration | codex (#53800: background compaction wired into turn lifecycle) | real compaction done, foreground only | briefed → `022-async-compaction.md` (supersedes this watch item)
Permission modes (session-level ask/allow policy) | gemini-cli (DEFAULT/AUTO_EDIT/YOLO/PLAN), opencode (rulesets) | `PermissionMode`: DEFAULT/ACCEPT_EDITS/DONT_ASK/BYPASS_PERMISSIONS | ours (rough parity)
Skills system (install/index/version/create) | codex, gemini-cli | `src/skills/` + `~/.clawd/skills/` index + skill-creator | ours (strong)
MCP tools + resources | opencode, gemini-cli (stdio/SSE/StreamableHTTP) | `mcp.py`, `mcp_resources.py` | ours (verify transports vs gemini-cli)
Background Bash + streaming | opencode, Claude Code | built | ours
Checkpoints / rewind | opencode (`revert.ts`), Claude Code | `checkpoints.py` + `/rewind` | ours
Plan mode | gemini-cli, opencode | `plan_mode.py` | ours
Cron/scheduled tasks (in-agent) | Claude Code routines (cloud) | `cron.py` tool | ours (ahead: local)
LSP integration | — | `lsp.py` | ours (unique)
Worktree support | codex | `worktree.py` | ours
Eval harness | gemini-cli (`evals/`), cline (`evals/`) | `src/eval/` | ours
Session persistence | codex (rollout: SQLite, search, index) | `session_store.py` (save/load/list) | ours (thinner — session search is a future gap)
Provider retry with backoff | all | `_call_provider_with_retry` | ours
Cost tracking | opencode (`usage.ts`), codex | `cost_tracker.py` | ours
Subagent model pinning (workflow-level) | Claude Code (`CLAUDE_CODE_WORKFLOW_SUBAGENT_MODEL`) | — | watch (relevant after 005 lands)

## Run 2 — thinking-scaffolds research (2026-10-13, briefs 014–032)

Standing directive `thinking-scaffolds-research.md` executed: two research agents read actual code across opencode, codex, gemini-cli, cline, crush, OpenHands (software-agent-sdk), aider, continue. All findings fitness-tested (model-agnostic). Briefs ordered by impact, builder works top-down.

Feature | Theirs | Ours | Status
---|---|---|---
Evidence-before-submit contract (completion tool wording) | cline (`createSubmitAndExitTool`) | — | briefed → `014-evidence-before-submit.md`
Stop-hook done-gate (harness vetoes premature "done") | codex (`stop.rs` block/continue) | — | briefed → `015-stop-hook-done-gate.md`
Plan mode as permission-enforced state machine (enter/exit tools, mode policy, validation gate, compaction-safe reminder, read-only planner profile) | gemini-cli (enter/exit + `plan.toml`), opencode (plan agent + self-healing reminder), crush (separate SessionAgent + markers), OpenHands (PLAN_STRUCTURE), cline (`/deep-planning`) | `plan_mode.py` (flag only) | briefed → `016-plan-mode-state-machine.md`
Scaffold tool tier (Checklist / AskQuestion / ReportFailure / Exit) | continue, opencode (`question.ts`: recommended-first, dismissal-as-defect) | — | briefed → `017-scaffold-tool-tier.md`
Reviewer loop: `/review` fresh-context critique + patch capsule + judge-driven goal loop + critic | codex (`review_request.rs`, rubric), continue (`cn check`), OpenHands SDK (goal/judge/critic) | — | briefed → `018-reviewer-and-judge-loops.md`
Lint/test reflection loop (cap 3) + cheap-model edit-corrector | aider (`base_coder.py`), gemini-cli (`editCorrector.ts`), codex (approval-aware test policy) | — | briefed → `019-verify-reflection-loops.md`
Todo tracking with invariants (≤1 in_progress, `blocked`, plan/execution disjoint) + hard termination protocol (`complete_task`, schema-bound finish, max-steps wrap-up) | gemini-cli, codex, crush, OpenHands, opencode, continue | — | briefed → `020-todo-invariants-termination.md`
REPOMAP (PageRank repo map, token-budgeted) | aider (`repomap.py`) | — | briefed → `021-repomap.md`
`think` as a first-class tool (inspectable reasoning event) | OpenHands SDK (`ThinkTool`) | — | briefed → `023-think-tool.md`
Forced scratchpad protocol for subagents (Questions-to-Resolve-empty = done) | gemini-cli (`codebase-investigator`) | — (extends 005) | briefed → `024-subagent-scratchpad.md`
Truncation recovery (continue after output-token limits) | opencode (dev, Oct 13) | — | briefed → `025-truncation-recovery.md`
Idle compaction (large idle conversations auto-compact) | Claude Code 2.1.299 | — | briefed → `026-idle-compaction.md`
Clawd-native reasoning budgets (portable effort schema, per-role/phase policies, behavior fallback) | cline (`portable-reasoning.ts`), codex, gemini-cli, Claude Code (`/effort`) | — | briefed → `027-reasoning-budgets.md`
Model tiers: weak-model sidecar for utility calls, per-mode binding | cline, aider, crush, opencode | — | briefed → `028-model-tiers-sidecar.md`
Loop-detection v2 (nudge-before-terminate, per-tool error counters, doom_loop adjudication) | OpenHands SDK, crush (proposal), opencode, gemini-cli | 006 built (signature only) | briefed → `029-loop-detection-v2.md` (extends 006)
Skill loading as small-model lifter (triggers, markers, progressive disclosure, per-family prompts) | OpenHands SDK, opencode (`optimize.ts`) | skills system (no auto-activation) | briefed → `030-skill-loading-lifter.md`
Hook-lifecycle v2 (`BeforeModel`, `BeforeToolSelection`, `tool.execute.before`, `session.retry`) | gemini-cli, opencode | 008 built (2 events) | briefed → `031-hook-lifecycle-v2.md` (extends 008)
Git-backed checkpoints (snapshot-before-write, private-refs storage) | gemini-cli (`checkpointUtils.ts`), cline (private refs) | `checkpoints.py` + `/rewind` | briefed → `032-git-checkpoints.md`
Tool input repair (schema-driven coercion) | opencode (`tool-input-repair.ts`) | 007 built | confirmed — already shipped, no new brief
Content provenance tracking through prompt assembly/compaction | codex (#53831, `ContentItemMetadata`) | — | watch (trust/infra; thinking-scaffold value unproven — revisit if a verified/unverified claim-tagging use lands)
Plugin-asserted permissions (plugins as first-class deniers) | opencode (#54930) | — | watch (extends 008's hook story; brief when 008's follow-up lands)
Agent-initiated secret request via in-conversation modal | OpenHands (#18408) | — | watch (structured secret-request interrupt vs chat-pasted secrets)
Ctrl-C session-poisoning guard | gemini-cli (#29728) | — | watch (robustness; brief if our interrupt path shows the same bug)
Read-before-edit freshness guard (edit rejects stale reads) | crush (unverified lead) | — | watch (verify in code before briefing)
Cline Cloud sessions (hosted sandbox) | cline (#13799) | — | skip (proprietary cloud product, not portable)
Guardian review threads / daemon updater / WinGet packaging | codex | — | skip (product-internal, not harness features)

## Run 3 — incremental upstream review (2026-10-14)

Run was in flight when Josh disabled all three upkeep jobs (~4:10 AM EDT). Research completed read-only; no src changes, no PC contact. 8 opencode + 6 codex + 4 cline commits reviewed (others: trivial or zero new). Two briefs filed; no new thinking-scaffold material beyond briefs 014–032.

Feature | Theirs | Ours | Status
---|---|---|---
Plugin/hook isolation (worker-thread execution, no in-process fallback, loud named failures) | cline (`789f79e4`: plugins only in workers; `PluginBootstrapMissingError`; docs: "crash isolation, not a security boundary") | hooks fail-open, no isolation or loud-failure contract | briefed → `033-plugin-hook-isolation.md`
Optimistic async session operations (hide-on-delete, inline progress, narrow input gating, selection preserved) | codex (`95273b84`: dashboard stays browsable during task removal) | session views block/don't handle this | briefed → `034-optimistic-session-dashboard.md`

Reviewed, no brief (dup / internal / not portable / premature):
- opencode `f576da50` (policies config stabilization — same policy system already on watch)
- codex `cb03f28a` (incremental-tool instruction refresh — no incremental tool windows in Clawd)
- codex `4906ddf5` (resume unloaded subagent threads on TUI selection — re-propose after 005 lands)
- codex `7eed30ad` (turn-recovery lineage bugfix — product-internal)
- opencode `165cd2e8` (desktop device pairing via proprietary remote-tunnel service — not portable; web app already covers phone access)
- cline `83ea9a41` (remote hubs packaging refactor), cline `adc7b155`/`868464e4`, crush UI fix, openhands settings-form fix, codex telemetry/test commits — minor fixes or catalog churn
- gemini-cli v0.64.0 (stable rollup, no new features); cline v4.1.24/desktop-v0.0.46/cli-v3.0.71 (Oct 13); codex rust-v0.163.0-alpha.11 (alpha only); Claude Code still 2.1.299

## Notes for the builder (run 2)

- Briefs 014–032 are ordered by impact; the builder works top-down. Cheapest tier first: 014 (prompt contract), 015 (done-gate), 017 (scaffold tools) — all near-zero implementation cost, highest small-model lift.
- 005's subagents are the assumed substrate for 018 (reviewer), 024 (scratchpad), 028 (weak-model profile). 008's hooks are the substrate for 015, 019, 031.
- Caveat from the research: a few findings rest partly on secondary sources (GitHub 429s mid-run) — flagged in the run scratch at `~/workspace/goals/clawd-code-coding-agent-build-out/hidden_files/scout-run-2-scratch.md`. The builder should verify code paths before implementing 020's crush-todos details and the 031 hook signatures.
- OpenHands agent-loop code has moved to `OpenHands/software-agent-sdk` (SDK v1.54.0); `OpenHands/OpenHands` is now frontend/runtime. Watch list updated accordingly.

## Notes for the builder

- The seed backlog (001–004) is ordered by impact; builder works top-down.
- Never remove items; mark superseded ones as such.
- Model-agnostic fitness test for every new find: "does this make Clawd-Code better no matter which AI model is plugged in?"
- Roo-Code is archived — mine its frozen code for ideas but expect no new commits.
- Watch the GPT-AGI origin repo for drift: if it wakes up with features we lack, that's a parity gap by definition.
