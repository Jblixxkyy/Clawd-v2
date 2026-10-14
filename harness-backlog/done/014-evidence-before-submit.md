# 014 — Evidence-before-submit contract in the completion tool

## What
Rewrite the completion/finish tool's description as an enforceable evidence contract: the model may only call it when it has *verified* the work by execution — re-read the task, confirm from its own tool output that every file exists with matching format/types/values, run tests if they exist, otherwise run the solution end-to-end and read the output back as evidence. Never submit on assumption. Structured `summary` + `verified` fields on the tool.

## Why
From cline's `createSubmitAndExitTool` (`sdk/packages/core/src/extensions/tools/definitions.ts` L903–946, https://github.com/cline/cline): the tool description *is* the self-critique scaffold. It is pure prompt-contract engineering — ~zero implementation cost — and it disproportionately lifts weak models, which are the ones that hallucinate completion. This is the single cheapest item in the thinking-scaffolds research and directly addresses Josh's "shallow analysis" complaint: the model currently declares done with no forced verification step. Passes the fitness test trivially (prompt text, model-agnostic).

## Suggested approach
- Find Clawd's completion tool (the terminal/finish tool) and rewrite its description to the evidence contract: (1) re-read the original task, (2) confirm every claimed artifact from own tool output (file exists, format/types/values match), (3) tests run green if tests exist, else end-to-end run with output read back, (4) explicit "never submit on assumption / never claim what you did not execute" clause.
- Add structured fields: `summary` (what was done) and `verified` (how it was verified, citing tool outputs) — make them required so the shape is enforced, not suggested.
- Keep it prompt-diet-friendly (brief 001): tight, no boilerplate; the contract should be ~150 words.

## Expected tests
- Tool description contains the evidence requirements (grep test).
- `verified` field is required by the schema; submission without it is rejected.
- Eval: run a task where the model would previously declare done early (e.g. file write it never re-read) — with the contract, the model re-reads / runs before submitting. At minimum, a prompt-level regression test asserting the contract text ships in the tool description.
- Full test suite green before shipping.
