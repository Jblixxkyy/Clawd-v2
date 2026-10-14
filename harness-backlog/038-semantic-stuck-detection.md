# 038 — Semantic stuck detection: catch the flailing-search spiral

## What
Extend the LoopDetector with an outcome-based heuristic: over a sliding window of the last 10 tool turns, if ≥6 results were errors and zero were file modifications (Edit/Write), the run is semantically stuck — same failing *strategy*, even though no exact (tool, input, output) signature repeats. Break with the standard loop-break report naming the pattern.

## Why
- Observed in the wild (Oct 14, 2026): 34 turns of Read/Grep/Bash/PowerShell searching for a `chat` method, 9+ error results, 0 edits. 006's SHA-256 signature detector never fired (inputs varied every call); 029's per-tool counters and A-B-A-B detectors don't cover "many different searches, all failing, no writes".
- This is the signature failure mode of small models on big tasks: not repeating one call, but cycling through near-miss strategies without ever committing to an edit. The harness must recognize *lack of progress*, not just repetition.
- Thresholds (10-window, 6-error, 0-write) are heuristics, model-agnostic, and conservative: a healthy run that writes files never trips it.

## Suggested approach
- `LoopDetector` gains `record_outcome(tool_name, is_error, modified_file: bool)` maintaining a `deque` of the last 10 outcomes; `is_semantically_stuck()` returns True on (errors ≥ 6 and modifications == 0).
- Wire into `_run_agent_loop_impl` next to the existing `loop_detector.record(...)` call sites (both success and exception paths).
- On trip: return `_loop_break_result`-style verdict with a distinct message ("[Stuck: 6+ of the last 10 tool calls failed and no files were modified — stopping. Try a different approach.]").
- Keep 006's exact-signature path untouched; the two detectors are independent.

## Expected tests
- 10 turns: 6 errors + 4 failed searches, 0 writes → stuck verdict with the semantic message.
- Same window with 1 Write → not stuck.
- 10 clean successful reads → not stuck.
- Exact-signature loop still detected via the original path.
- Full test suite green before shipping.
