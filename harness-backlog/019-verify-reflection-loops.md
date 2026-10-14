# 019 — Verification loops: lint/test reflection + edit-corrector

## What
Make run → observe → reflect → fix a first-class loop: after each edit batch, auto-run the configured lint and test commands, feed raw failure text back as the next turn's input, with a hard reflection cap; plus a cheap-model edit-corrector that repairs failed edits (whitespace/escaping only) with a deterministic regex fallback.

## Why
- aider `base_coder.py` `run_one()` (L868–886) and post-edit pipeline (L1595–1637) (https://github.com/Aider-AI/aider): after each edit batch — run lint cmd → run test cmd; raw failure text becomes `reflected_message`, the next turn's input. Hard cap `max_reflections=3`. (Corrects the naive assumption: aider commits *first*, then reflects — commit is not test-gated.)
- gemini-cli `editCorrector.ts` (https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/utils/editCorrector.ts): a flash-lite model with thinking budget 0 fixes only whitespace/escaping ("Do NOT invent a completely new edit") + LRU cache + deterministic regex fallback — a second, cheaper repair stage before the main model ever sees the failure.
- codex "Validating your work" policy (`default.md` L149, https://github.com/openai/codex/blob/main/codex-rs/protocol/src/prompts/base_instructions/default.md): verify proactively in non-interactive mode, propose-first in interactive; test tasks may bypass the hold regardless of mode — a sensible approval-aware test policy to copy.

This is the loop Josh's shallow-analysis complaint needs: the agent currently has no built-in "prove it works" cycle. Fully model-agnostic.

## Suggested approach
- Post-edit hook (extends 008's lifecycle): after a batch of file edits, run configured `lint_cmd` and `test_cmd` (repo-detected, user-overridable); on failure, inject the raw failure output as the next message (`reflected_message` pattern) with outcome flags; cap at 3 reflections, then surface to the user instead of looping.
- Approval-aware policy (codex pattern): non-interactive/background runs verify proactively; interactive runs propose-first ("tests failed — fix and re-run?"); explicit test tasks bypass the hold.
- Edit-corrector stage: when an edit tool call fails on mechanical grounds (bad whitespace/anchor/escaping), try a cheap secondary model with a strict "fix only the mechanics, do not invent a new edit" prompt; LRU-cache corrections; fall back to deterministic regex repair; only then surface the failure to the main model. Never let the corrector change semantics.
- Record reflection outcomes (pass/fail, attempts) as run-log events (feeds the verdict-event idea in 018).

## Expected tests
- Edit introducing a lint error → lint runs automatically, failure text fed back, model fixes it within the cap.
- 3 consecutive failed reflections → stops and reports to the user, no infinite loop.
- Edit failing on whitespace-only grounds → corrector repairs without involving the main model; semantic failure → corrector refuses, main model sees it.
- Interactive mode: test failure proposes before re-running; non-interactive: re-runs proactively.
- Full test suite green before shipping.
