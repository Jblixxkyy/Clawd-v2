# 005 — True agentic subagents

## What
Give the model real LLM-driven subagents: delegate with a natural-language prompt, the child runs its own agent loop with its own context, and returns a summarized result. Parameters: description, prompt, model (optional), background (optional).

## Why
Our `Agent` tool (`src/tool_system/tools/agent.py`) is NOT a subagent — it executes a fixed, pre-declared list of tool calls with no agentic reasoning in the child. Every major harness has the real thing:
- opencode (`packages/core/src/tool/plugin/subagent.ts` in ~/workspace/opencode-ref): prompt-driven subagents with model selection, `background: true` mode ("You will be notified when it completes. DO NOT sleep, poll"), and `sessionID` to continue a previous subagent conversation.
- gemini-cli (`packages/core/src/agents/agent-tool.ts` in /tmp/gemini-cli): subagent tool plus specialized agents (codebase-investigator, cli-help-agent).
- codex: multi-agent v2 with an inter-agent message board (`codex-rs/core/src/agent_message_board.rs` in /tmp/codex).
- Claude Code: subagents with nesting, worktree isolation, `effort` parameter, per-subagent `autoCompactWindow`.

Without this, the model can only parallelize via fixed call lists — no autonomous explore-then-report delegation, which is the core multi-agent pattern.

## Suggested approach
- New `SubagentTool` (keep the existing `Agent` call-list tool untouched): input `{description, prompt, model?, background?}`.
- Child runs `run_agent_loop` with a fresh `Conversation`, the same tool registry (minus anything session-scoped), and a compact subagent system prompt. Permission prompts inside the child cannot reach the user — auto-deny and report (same rule as the current Agent tool).
- Background mode: return immediately with a task id; reuse the existing background-Bash notification pattern to report completion.
- Later (not this brief): sessionID continuation, worktree isolation.

## Expected tests
- Unit test with a stub provider: child loop runs N turns and returns its final text.
- Test that child permission prompts auto-deny without reaching the user.
- Background-mode test: returns immediately, completion notification fires.
- Full existing suite stays green.
