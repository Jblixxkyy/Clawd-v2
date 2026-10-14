# 023 — `think` as a first-class tool

## What
A `think` tool: an explicit reasoning channel that lands in the transcript as a real, inspectable, replayable event instead of invisible chain-of-thought. Calling it forces a pause; the content is recorded as a structured event the harness (and the user, and replays) can see.

## Why
From the OpenHands software-agent-sdk (`openhands/sdk/tool/builtins/__init__.py` L48: `BUILT_IN_TOOLS = [FinishTool, ThinkTool]` on every agent — https://github.com/OpenHands/software-agent-sdk): making reasoning a tool call turns it from a model-internal habit into harness infrastructure. The transcript shows *when* the model stopped to think and *what* it concluded; replays and evals can audit it; the 015 done-gate and 018 judge can read it. Small models especially benefit from the forced pause — it breaks the "act first, think never" failure mode without any model capability requirement. 1:1 port, trivial implementation, real behavioral effect. Fully model-agnostic.

## Suggested approach
- Add a `think` tool: `{thought: string}` (optionally `{topic}`); no side effects; appends a `ThinkEvent` to the run log / transcript (rendered distinctly in the UI, collapsible like the web UI's "Thought for Ns" rows).
- Prompt guidance (light, diet-friendly): suggest calling `think` before high-stakes actions (multi-file edits, finishing, answering ambiguous questions) — suggest, don't mandate; the 015 done-gate and subagent scratchpad (024) are the enforcement layers.
- Make think events first-class in replays, checkpoints, and the judge loop (018 can weigh them).
- Optionally: subagent profiles (005/024) can require a think call before their first mutation tool call — cheap discipline for delegated work.

## Expected tests
- `think` call appears in the transcript as a distinct, replayable event.
- Think events survive compaction (summarized, not dropped) and appear in checkpoints.
- Judge loop (018) can reference think contents.
- No side effects: world state unchanged by think calls.
- Full test suite green before shipping.
