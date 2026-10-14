# 015 — Stop-hook done-gate: harness vetoes premature completion

## What
A harness-level gate on the agent's "I'm done" signal: when the model tries to finish, an external definition-of-done check runs (tests? artifacts exist? task criteria met?) and can veto with `{"decision": "block", "reason": "..."}` — the reason is fed back as a continuation prompt so the model retries instead of stopping.

## Why
From codex `codex-rs/hooks/src/events/stop.rs` (https://github.com/openai/codex/blob/main/codex-rs/hooks/src/events/stop.rs): the harness, not the model, decides when done is done. This is the biggest small-model behavioral change found in the opencode/codex/gemini-cli research wave — weak models stop early and declare victory; an external done-gate they cannot argue with fixes the whole class. Model-agnostic (it is control flow around the model, not a model capability). Complements 014 (evidence contract): 014 makes the model *try* to verify, 015 makes the harness *enforce* it.

## Suggested approach
- Add a `Stop`/`PreFinish` hook point (extends the hook lifecycle from 008): fires when the agent loop is about to terminate via the completion tool.
- Default gate implementation: configurable checklist — e.g. run the repo's test command if one is configured, verify claimed output files exist, optionally re-read the task for unmet criteria. Gate returns block/allow + reason.
- On block: inject the reason as a user-role continuation message ("Done-gate blocked: <reason>. Continue working.") and resume the loop; cap blocks per run (e.g. 3) to avoid infinite veto loops, then escalate to the user.
- Make the gate itself a hook so users can plug custom definitions of done (test-gate, lint-gate) without core changes.

## Expected tests
- Finishing a task with failing tests → blocked with the failure reason; model gets a continuation turn.
- Finishing a task with all checks green → allowed, no extra turn.
- Block cap: 3 consecutive blocks → run stops and reports to the user instead of looping forever.
- Custom gate via hook fires and its reason surfaces verbatim.
- Full test suite green before shipping.
