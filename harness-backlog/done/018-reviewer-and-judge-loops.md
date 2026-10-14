# 018 — Reviewer loop: /review critique + judge-driven goal loop

## What
Two complementary verification loops: (1) an on-demand `/review` command that spawns a fresh-context reviewer over the diff and returns a machine-readable verdict (P0–P3 findings, exact code ranges, overall correctness + confidence as strict JSON); (2) a post-run judge loop where a judge model renders the transcript and returns a `complete/missing` verdict that decides continue-vs-stop, plus a per-run critic with an iteration cap.

## Why
- codex `/review`: `review_request.rs` (https://github.com/openai/codex/blob/main/codex-rs/prompts/src/review_request.rs) resolves UncommittedChanges/BaseBranch/Commit into review prompts; `rubric.md` (https://github.com/openai/codex/blob/main/codex-rs/prompts/templates/review/rubric.md) demands P0–P3 findings with exact ranges and strict-JSON verdict; `review.rs` (https://github.com/openai/codex/blob/main/codex-rs/core/src/session/review.rs) runs it in a fresh turn context (review model swappable to a cheaper one) and injects results back as a `<user_action>` message so the worker iterates. gemini-cli has no automatic reviewer — codex's is the one to copy.
- continue `cn check`: forks one subprocess per reviewer agent, each runs the normal agent loop over a diff-context prompt, captures the worktree diff as a suggested patch, emits `ReviewResult{status: pass|fail|error, patch, output, duration}` + failure table + JSON capsule — the reviewer as a first-class command producing a patch artifact.
- OpenHands SDK `conversation.goal`: after each run a judge renders the transcript as plain `role: text` (system prompt deliberately excluded for cost) → `GoalVerdict{complete, missing}`; `GoalController` decides continue-vs-stop with an iteration cap; `GoalContinue.followup` feeds "what's missing" back as the next prompt. Per-run critic with `IterativeRefinementConfig(success_threshold=0.7, max_iterations=3)` and customizable follow-up prompt. The judge can be a cheaper model.
- Emit verdict events (OpenHands `SecurityAnalysisEvent` pattern) into the run log per review/judge pass — first-class audit evidence enabling shadow evaluation later.

All harness control flow, model-agnostic; the reviewer/judge models are swappable (cheap model fine — this is Josh's "harness makes the model smart" thesis in action).

## Suggested approach
- `/review` command: snapshot the worktree diff, spawn a fresh-context reviewer subagent with the codex-style rubric prompt (P0–P3, ranges, JSON verdict, strict schema in prompt text but tolerant parsing — accept prose-wrapped JSON, infer missing fields, per codex `assessment.rs`), inject the verdict as a user-role message; worker iterates on failures. Reviewer model configurable, default to the session's cheap/secondary model if configured.
- Reviewer patch capsule (continue pattern): reviewers may return a suggested patch (unified diff) alongside the verdict; surface it in the UI as an apply/discard widget.
- Judge-driven goal loop (post-run hook, opt-in per run or via config): after the run ends, judge model → `GoalVerdict{complete, missing[]}`; if missing and under the iteration cap, feed "what's missing" back as the next prompt and resume; else stop. Cap iterations (default 3).
- Log every verdict as a structured run-log event (verdict, model, confidence, findings count) for audit/shadow evals.

## Expected tests
- `/review` on a clean diff → pass verdict, no iteration.
- `/review` on a diff with an injected bug → P0/P1 finding with correct file range; worker receives the verdict and fixes it.
- Judge loop: run that stops early → judge returns missing items → run resumes; 3rd iteration still incomplete → stops and reports.
- Malformed reviewer JSON (prose-wrapped) → parsed tolerantly, no crash.
- Verdict events appear in the run log with model + confidence.
- Full test suite green before shipping.
