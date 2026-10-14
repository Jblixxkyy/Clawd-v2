# 025 — Truncation recovery: continue responses killed at output-token limits

## What
When a model response is cut off by the output-token limit (finish reason `length` / truncated), the harness automatically sends a "continue" follow-up and stitches the parts together, instead of handing the model (and user) a silently truncated answer.

## Why
From opencode `feat(core): continue responses after output token limits` (https://github.com/anomalyco/opencode/commit/ — dev branch, Oct 13 2026; v2.0.28 also hardened reasoning streams: chunk timeout 5→15 min "for models that pause while reasoning", split Gemini responses auto-continued): truncated outputs are one of the most confusing failure modes for users ("the answer just stops") and for the agent loop (half a tool call, half a plan). The mechanism — detect the truncation signal, re-prompt to continue, concatenate — is provider-portable: every major API surfaces an equivalent of `finish_reason: length`. It passes the fitness test: it makes any model behave better by recovering work that would otherwise be lost. Small models hit output limits more often (verbose, inefficient), so this disproportionately helps them.

## Suggested approach
- In the provider adapter layer: detect truncation signals per provider (`finish_reason: length`, Gemini `finishReason: MAX_TOKENS`, Anthropic `stop_reason: max_tokens`, etc.) and normalize to one internal `truncated` flag.
- On `truncated`: automatically issue a continuation turn ("Continue exactly where you left off — do not repeat what you already wrote"), up to N continuations (default 3); stitch text parts in order; mark the stitched result in the log.
- If the truncation happened mid-tool-call (partial JSON), attempt the continuation before declaring a parse failure — this composes with 007's input repair (repair handles malformed input; this prevents the malformation).
- Surface it in the UI: a subtle "continued after truncation" marker so the user knows the answer was stitched, not natively complete.
- Config: `truncation_recovery: {enabled, max_continuations}`.

## Expected tests
- Mock provider returning `finish_reason: length` → harness continues automatically; final text is the stitched whole.
- 3 continuations still truncated → stops and reports truncation to the user, no infinite loop.
- Truncation mid-tool-call JSON → recovered via continuation before parse-failure handling.
- UI shows the continuation marker.
- Full test suite green before shipping.
