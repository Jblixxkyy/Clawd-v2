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
Async compaction configuration | codex (HEAD commit) | real compaction done, config not reviewed | watch (flagged at baseline; predates watermark)
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

## Notes for the builder

- The seed backlog (001–004) is ordered by impact; builder works top-down.
- Never remove items; mark superseded ones as such.
- Model-agnostic fitness test for every new find: "does this make Clawd-Code better no matter which AI model is plugged in?"
- Roo-Code is archived — mine its frozen code for ideas but expect no new commits.
- Watch the GPT-AGI origin repo for drift: if it wakes up with features we lack, that's a parity gap by definition.
