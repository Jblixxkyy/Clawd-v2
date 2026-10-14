# 037 — Error-class circuit breaker: stop letting the model re-run known-failing calls

## What
Track (tool name, normalized error signature) pairs across the loop. When the same tool fails with the same error signature 3 times, inject a plain-English redirect as a system reminder ("The Bash tool has failed 3× with 'bash executable not found'. It is not available on this machine — use PowerShell instead."). Further calls to that tool short-circuit: return the last error immediately without executing, until the model tries something else.

## Why
- Observed in the wild (Oct 14, 2026): 6 identical Bash failures ("bash executable not found") and 3 identical PowerShell failures ("grep is not recognized"). The tool told the model exactly what was wrong every time; the model could not adapt.
- Complements 006 (exact-signature loop detection — blind here because inputs varied) and 029 (per-tool error counters → stuck verdict). This brief is the *intervention* layer: redirect first, short-circuit second, verdict last.
- Error signature normalization (first line of the error, lowercased, paths stripped) keeps "same error" matching robust against trivial output differences.

## Suggested approach
- In `_run_agent_loop_impl`: `circuit: dict[tuple[str, str], int]` counting consecutive same-signature errors per tool; reset a tool's count on any success.
- On hitting 3: append a user-role reminder to `conversation` (and the OpenAI message list) with the redirect; mark the circuit open for that tool.
- While open: intercept before `tool_registry.dispatch`, synthesize the stored error as the result (no execution, no side effects), still record via `loop_detector` so 006/029 verdicts keep working.
- Circuit closes on: any successful call of another tool followed by a *different* approach, or session end. Keep it simple: close when the model goes 5 turns without calling the broken tool.
- Fail-open: breaker state never raises; worst case the model gets an extra reminder.

## Expected tests
- Same tool + same error ×3 → reminder injected with the redirect text.
- 4th identical call → short-circuited (dispatch not reached), same error returned.
- Success resets the count.
- Different error signature → separate count.
- Full test suite green before shipping.
