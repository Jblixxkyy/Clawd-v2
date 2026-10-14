# 035 — Empty-response retry: never show the user an empty bubble

## What
When a model call succeeds but returns no text and no tool calls (empty/whitespace content), retry the call up to 2 more times before accepting the turn. If all retries come back empty, end the turn with an explicit notice ("[The model returned an empty response after 3 attempts.]") instead of saving an empty assistant message.

## Why
- Observed in the wild (Oct 14, 2026): nemotron-3.5-lightning-30b-a3b via NVIDIA NIM returned a single whitespace character after a 1.6s "thought"; the web UI rendered an empty bubble and the session appeared to "just stop". The user's follow-up "?" worked fine — transient model flakiness, not a stuck run.
- The existing `_call_provider_with_retry` only retries on exceptions, not on empty-but-successful responses. Small models on third-party inference APIs return empty content often enough that this needs a harness-level guard.
- Model-agnostic: pure control flow around the provider call; no model-specific heuristics.

## Suggested approach
- In `_run_agent_loop_impl`, after `_call_provider_with_retry` returns, check `_is_empty_response(response)` (no `tool_uses` and blank/whitespace `content`).
- Retry the provider call (same messages) up to `_MAX_EMPTY_RETRIES = 2` more times. Count retries separately from exception retries; do not burn a full turn per retry (no `turn_count` increment, no tool dispatch).
- If still empty: `conversation.add_assistant_message("[The model returned an empty response after 3 attempts. Try again.]")` and return it as the turn result so the UI shows something honest instead of an empty bubble.
- Fail-open: any exception during the retry path falls back to current behavior.

## Expected tests
- Mock provider returns empty twice then content → turn succeeds, provider called 3 times, no empty message saved.
- Mock provider always empty → result text is the explicit notice, not "".
- Non-empty first response → exactly 1 provider call (no behavior change).
- Empty response WITH tool calls → not treated as empty (tool calls count as content).
- Full test suite green before shipping.
