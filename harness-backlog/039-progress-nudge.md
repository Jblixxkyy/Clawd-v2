# 039 — Progress nudge: ask for a plan before the spiral starts

## What
Track tool turns since the last file modification (Edit/Write). At 10 modification-free tool turns, inject a planning nudge as a user-role message: "You've made 10 tool calls without modifying any files. Briefly state your plan (what you'll change, in which files) before continuing." Repeat every 15 turns thereafter if still no modifications. Nudge only — never blocks.

## Why
- Observed in the wild (Oct 14, 2026): the model made 34 search-only tool calls without ever articulating what it was looking for or where it would edit. A planning scaffold at turn 10 would have forced the model to commit to a hypothesis ("the chat method is in web_app.py's handler") instead of flailing.
- The scout's thinking-scaffolds research (planning/self-critique across opencode, codex, gemini-cli) consistently finds that forced plan-articulation is the cheapest small-model lifter: it converts implicit thrash into an explicit, checkable claim.
- Complements 038 (which *stops* the run): this *steers* it earlier. 029's nudge-before-terminate fires at the verdict; this fires at the first smell.

## Suggested approach
- In `_run_agent_loop_impl`: `turns_since_modification` counter, reset on any Edit/Write success (also reset on Subagent/Task completion — delegated work counts as progress).
- At 10, 25, 40…: append nudge to `conversation` (+ OpenAI list) as a user-role message. One short paragraph; no tool calls involved.
- The nudge text asks for: target files, intended change, and what it's still looking for. Keep it under 60 words.
- Fail-open: counter logic never raises; a missed nudge is just a missed nudge.

## Expected tests
- 10 search-only turns → nudge message present in conversation.
- Write on turn 5 → counter resets; no nudge at turn 10.
- Nudge repeats at turn 25 with still no modifications.
- Nudge never blocks: loop continues normally after injection.
- Full test suite green before shipping.
