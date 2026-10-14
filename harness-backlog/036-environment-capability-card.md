# 036 — Environment capability card: the harness introduces itself at session start

## What
At loop start, detect the machine's actual capabilities and inject a compact "environment card" into the turn-0 system message: OS, which shells exist (bash? powershell?), shell-specific gotchas (PowerShell has no `grep` — use `Select-String` or the Grep tool), workspace root, and which shell tools are expected to fail.

## Why
- Observed in the wild (Oct 14, 2026): a 30B model burned 6 turns calling Bash on a Windows machine with no bash installed, and 3+ turns using `grep` inside PowerShell. Every failure was reported by the tool — the model never adapted. One upfront paragraph would have prevented the entire spiral.
- The model only knows what the harness tells it. Static prompt text can't cover per-machine facts (bash exists on Josh's server, not on his PC). Detection must happen at runtime, per session.
- Complements 001 (prompt diet): the card is ~8 lines, appended to the turn-0 system message only — zero cost on later turns thanks to the 002 delta system.

## Suggested approach
- New module `src/tool_system/env_card.py`: `build_env_card(tool_context) -> str`. Detect via `platform.system()`, `shutil.which("bash")`, `shutil.which("powershell")`/`pwsh`.
- Card content (example, Windows/no-bash): OS, shell availability with explicit DO/DON'T lines per shell, workspace root, `cd` persistence note.
- In `_run_agent_loop_impl`, append the card to `full_system_prompt` after `instruction_tracker.full_prompt()` (so the 002 delta baseline is unaffected — the card ships once on turn 0).
- Never break the loop: detection wrapped in try/except, falls back to no card.

## Expected tests
- Mocked Windows/no-bash → card says Bash will fail and names PowerShell + Select-String/Grep.
- Mocked Linux/with-bash → card says bash available, no warning.
- Detection exception → loop still starts, no card, no crash.
- Card appears in turn-0 system message; absent from turn-1+ delta appends.
- Full test suite green before shipping.
