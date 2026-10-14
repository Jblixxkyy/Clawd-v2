# 009 — Skill @-mentions in user input

## What
Let the user explicitly invoke skills from chat with a `$` sigil (e.g. `$pdf`, `$skill://pdf`), instead of relying solely on the model to pick the right skill.

## Why
codex parses tool mentions in user input (`codex-rs/skills/src/mentions.rs` in /tmp/codex): `$skill://`, `$mcp://`, `$plugin://`, `$app://` sigils classified by `ToolMentionKind`. Our skill system is strong (install/index/create/skill-creator, `src/skills/`), and the inventory IS injected into the prompt — but the Oct 11 Q&A test proved the 30B model ignores it 3x and invents generic skills. A user-side explicit invocation bypasses model flakiness entirely: no guessing, no prompt real estate, deterministic.

## Suggested approach
- Pre-process user input in the REPL before the agent loop: regex `\$([a-zA-Z0-9_-]+)` and `\$skill://([a-zA-Z0-9_-]+)`.
- Resolve the name against the skill index (`~/.clawd/skills/`); on hit, load the SKILL.md and inject it as context for the turn (same path the Skill tool uses), and tell the model the user explicitly requested it.
- Unknown name → error listing available skills (we already do this for the Skill tool — reuse).
- Later (not this brief): `$mcp:server` scoping for MCP servers.

## Expected tests
- Unit tests: `$pdf` and `$skill://pdf` parse to skill `pdf`; plain `$` or `$ ` ignored; `$unknown` → helpful error.
- Test injection: mentioned skill content reaches the model context.
- Test no-mention input passes through byte-identical.
