# 017 — Scaffold tool tier: Checklist, AskQuestion, ReportFailure, Exit

## What
Add four first-class scaffold tools that structure the agent's reasoning instead of leaving it to free text: `Checklist` (read-only plan-as-structured-tool-state), `AskQuestion` (ambiguity forced into a typed 2–5-option channel), `ReportFailure` (unrecoverable failure gets a first-class exit with reason), and `Exit`/`Status` (sets `isComplete` so the harness distinguishes done vs. failed).

## Why
From continue's CLI (`extensions/cli/src/tools/`: `allBuiltIns.ts`, `reportFailure.ts`, `exit.ts`, `AskQuestion.ts` definitions — https://github.com/continuedev/continue): the cheapest, most model-lifting port in the cline/crush/OpenHands/aider/continue wave. Forcing ambiguity into a typed question channel and failure into a typed exit removes the two places weak models most often go off the rails (asking vague follow-ups in prose; dying silently or hallucinating success). opencode's `question.ts` (https://github.com/anomalyco/opencode/blob/dev/packages/core/src/tool/plugin/question.ts) adds two refinements worth copying: recommended option first with an explicit "(Recommended)" suffix, and dismissal tunneled as a defect — dismissal-with-feedback becomes a `ToolFailure` the model continues from rather than a dead end. Cline's AskQuestion variant explicitly forbids jailbreak options (e.g. "toggle to Act mode") in the contract — portable and worth copying. All model-agnostic.

## Suggested approach
- `Checklist`: tool holding an ordered list of `{label, status}` items; read-only with respect to the world (no side effects), updated via whole-list overwrite (pairs with the todo invariants in 020 — or implement 020's todo tool as this Checklist with stricter validation).
- `AskQuestion`: `{question, options[2..5]}` each option `{label, description}`; support a `recommended` flag rendered as "(Recommended)"; contract forbids options that change agent mode/permissions; dismissal returns a typed `ToolFailure`, dismissal-with-feedback returns the feedback as the failure reason so the model continues.
- `ReportFailure`: `{reason, attempted}` — terminal, marks the run failed with a structured reason (feeds the run log and the 015 done-gate's records).
- `Exit`/`Status`: sets `isComplete` with an outcome enum (done/failed) so loop hygiene (continue's two-clause termination: terminal tool set it, or assistant message with no tool calls) can distinguish clean finish from silent stall.
- Wire into the web UI as compact structured widgets (consistent with the ChatGPT-parity UI pass), not raw JSON dumps.

## Expected tests
- AskQuestion with >5 options or a mode-changing option → rejected by validation.
- Dismissing a question surfaces a `ToolFailure` the model can continue from (no dead end).
- ReportFailure marks the run failed with the structured reason in the log.
- Exit with no tool calls in the final message → run completes cleanly; tool-call loop with no Exit → stall detected, not marked done.
- Full test suite green before shipping.
