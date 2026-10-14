# 010 — A2A client (delegate to external agents)

## What
Speak the Agent-to-Agent (A2A) protocol so Clawd-Code can delegate tasks to external agents (and later, be delegated to).

## Why
gemini-cli ships a full A2A client (`packages/core/src/agents/a2a-client-manager.ts`, `a2aUtils.ts` in /tmp/gemini-cli): AgentCard discovery, JSON-RPC + gRPC transports, 30-minute timeouts for long tasks (e.g. deep research), streaming artifact reassembly. Model-agnostic interop is squarely "best harness no matter the model" — it lets Clawd-Code orchestrate specialist agents (a code reviewer, a research agent) without caring what model they run.

## Suggested approach
- New module `src/tool_system/a2a/` (client only for this brief): fetch + parse AgentCard from a URL, send a message via JSON-RPC, reassemble incremental status/artifact updates into a final result.
- Expose as a tool, e.g. `DelegateAgent {agent_card_url, task, timeout?}`. Reuse the background-task notification pattern for long runs.
- Use the `a2a-sdk` Python package if it installs cleanly; otherwise a minimal JSON-RPC client (~150 lines) — no heavy deps.
- Server mode (accepting delegation) is explicitly out of scope for this brief.

## Expected tests
- Mock A2A server (http.server in test): card fetch → message → artifact stream → reassembled result.
- Test timeout path returns a clean error, not a hang.
- Test malformed AgentCard → clear validation error.
