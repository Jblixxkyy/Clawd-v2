# 029 — Loop-detection follow-up: nudge-before-terminate + per-tool error counters

## What
Upgrade the briefed loop detection (006, built): instead of aborting the moment a loop is detected, inject an environment-sourced nudge and give the model one recovery turn before declaring stuck — and add per-tool error counters (3 consecutive errors on one tool = stuck) to catch the reworded-input retries that signature matching misses.

## Why
- OpenHands SDK `conversation/stuck_detector.py` (https://github.com/OpenHands/software-agent-sdk): 4 detectors over the last 20 events (same action+observation, same action+repeated error, monologue, A-B-A-B alternation); PR #4332 injects an environment-sourced nudge and gives the model one more step before declaring STUCK. Nudge-before-terminate recovers runs that a hard abort would kill — strictly better than 006's break-early behavior.
- crush's shipped `loop_detection.go` (the basis of 006) has a known blind spot the research surfaced: SHA-256 over (tool, input, output) triples is defeated by reworded-input retries — the same failing tool with slightly different phrasing never trips the signature. The unmerged crush issue-proposal (per-tool-name error counter, threshold 3) is the strictly better detector for that case.
- opencode `doom_loop` guard: same tool call ×3 with identical input → `permission.ask("doom_loop")` pauses for user adjudication instead of burning tokens — a third response mode (ask the human) worth having alongside nudge and abort.
- gemini-cli `turn.ts` L50–L56: `LoopDetected` as a first-class turn event (alongside `MaxSessionTurns`); suspected loops escalated to a stronger judge model.

All model-agnostic control flow. This brief extends 006; it does not replace it.

## Suggested approach
- Keep 006's signature detector; add: (a) per-tool error counter — 3 consecutive errors from the same tool → stuck; (b) the OpenHands detector family (same action+observation, repeated error, monologue, A-B-A-B) over a 20-event window.
- On detection: first inject a nudge ("You appear stuck: <pattern>. Try a different approach: <suggestion>.") and allow exactly one recovery turn; if the pattern persists → `LoopDetected` turn event → stop with a structured report (or `permission.ask("doom_loop")` in interactive mode for human adjudication).
- Escalation option: route the recovery turn through the stronger/model-tier judge (028) when configured.
- Log detections with the pattern type for later analysis.

## Expected tests
- Identical tool call ×6 → detected via signature; nudge injected; one recovery turn granted.
- Same tool, reworded inputs, 3 consecutive errors → detected via error counter (the case 006 misses).
- A-B-A-B tool alternation → detected via alternation detector.
- Recovery turn breaks the pattern → run continues normally, no abort.
- Interactive mode → user adjudication prompt on doom_loop; non-interactive → structured stop report.
- Full test suite green before shipping.
