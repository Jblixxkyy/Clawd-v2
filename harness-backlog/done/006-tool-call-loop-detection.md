# 006 — Tool-call loop detection

## What
Detect when the model is stuck repeating identical tool calls and break the loop early with a clear message, instead of burning all 50 max turns.

## Why
crush does exactly this (`internal/agent/loop_detection.go` in /tmp/crush): it hashes tool name + input + output (SHA-256) over a 10-step sliding window; if any signature appears more than 5 times, the agent is stuck. Our `run_agent_loop` (`src/tool_system/agent_loop.py`) has `max_turns=50` but no repetition detection — a confused model re-issuing the same Read/Bash burns the whole budget and the user's tokens. This is pure model-agnostic reliability: every model loops sometimes.

## Suggested approach
- Port crush's signature scheme: `sha256(tool_name + "\x00" + canonical_json(input) + "\x00" + truncated_output)`.
- In `run_agent_loop`, keep the last 10 step signatures; each turn, count occurrences — if any exceeds 5, stop with `[Loop detected: <tool name> repeated 6x with identical input — stopping. Try a different approach.]`.
- Make window size (10) and threshold (5) configurable constants at the top of the module.

## Expected tests
- Unit test: feed 6 identical fake steps → detection fires with the right tool named.
- Unit test: varied calls (different inputs) → no detection.
- Unit test: same tool but different outputs → no detection (output is part of the signature).
- Integration: loop-break message appears in the result; existing max-turns test still passes.
