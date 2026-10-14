# Hooks

Hooks are shell commands that run around agent lifecycle events. Modeled on
Claude Code's hooks, extended to the full agent lifecycle (brief 008).

## Config

`~/.clawd/hooks.json`. Each event name maps to a per-event list of
`{ "command": ... }` entries (a bare string command per event is also
accepted):

```json
{
  "SessionStart": [
    { "command": "python3 ~/hooks/on-session-start.py" }
  ],
  "BeforeAgent": [
    { "command": "python3 ~/hooks/check-env.py" }
  ],
  "PreToolUse": [
    { "command": "python3 ~/hooks/block-dangerous-bash.py" }
  ],
  "PostToolUse": [
    { "command": "python3 ~/hooks/log-tool-use.py" }
  ]
}
```

## Events

- `SessionStart` — fired when a session is created (`Session.create()`).
  Informational only; cannot block.
- `SessionEnd` — fired when a session is torn down. Informational only;
  cannot block. (Wiring lives in the session teardown path.)
- `BeforeAgent` — fired once before the agent run starts. **Can halt the
  run**: a block/ask decision stops before the first model call and the
  hook's message becomes the run's response text.
- `AfterAgent` — fired once when the agent run finishes (including
  cancellation and errors). Informational only; cannot block.
- `PreCompress` — fired before auto-compaction. **Can veto the
  compaction**: a block/ask decision skips this compaction and the loop
  continues with the uncompacted conversation.
- `BeforeModel` — fired before every model call. **Can halt the turn**: a
  block/ask decision skips the model call and the hook's message becomes
  the turn's response text.
- `AfterModel` — fired after every model call. Informational only; cannot
  block.
- `PreToolUse` — runs before the tool executes. **Can block the call**
  (existing behavior, unchanged).
- `PostToolUse` — runs after the tool returns. Cannot block; output is
  ignored (existing behavior, unchanged).

## Stdin/stdout protocol

Each hook command runs via `subprocess` with `shell=True` and receives the
payload as JSON on stdin. Tool events:

```json
{
  "event": "PreToolUse",
  "tool": "Bash",
  "input": { "command": "ls" },
  "output": { "...": "..." }
}
```

(`output` is only present for `PostToolUse`.)

Lifecycle events receive event-specific keys:

| Event | Extra stdin keys |
|---|---|
| `SessionStart` | `session_id`, `provider`, `model`, `created_at`, `resumed` |
| `SessionEnd` | `session_id`, `reason` |
| `BeforeAgent` | `session_id`, `model`, `max_turns` |
| `AfterAgent` | `session_id`, `response_text`, `num_turns`, `usage`, `cancelled` (or `error` when the run raised) |
| `PreCompress` | `session_id`, `model`, `estimated_tokens`, `trigger` |
| `BeforeModel` | `session_id`, `turn`, `model`, `message_count`, `messages` |
| `AfterModel` | `session_id`, `turn`, `model`, `content`, `tool_uses`, `usage` |

A hook answers with a decision object on stdout. New style:

```json
{ "decision": "block", "message": "dangerous command blocked" }
```

or the legacy `PreToolUse` style (still accepted, means `"block"`):

```json
{ "allow": false, "message": "dangerous command blocked" }
```

- `"decision": "allow"` (or `"allow": true`, or empty/no output) lets the
  run proceed.
- `"decision": "block"` halts what the event wraps (see Events above); the
  message is shown to the user.
- `"decision": "ask"` is currently enforced like `"block"`: no interactive
  approval UI is wired to hooks yet, so asking degrades to halting rather
  than silently allowing.
- On informational events (`SessionStart`, `SessionEnd`, `AfterAgent`,
  `AfterModel`, `PostToolUse`) hook output is ignored entirely.

## Timeout and fail-open behavior

- Each hook command has a **10-second timeout**.
- Missing config file, timeout, non-zero exit, or invalid JSON output all
  **fail open**: the run proceeds and a dim warning is printed. Malformed
  hook output never breaks the loop, for any event.
- Hooks never raise; a broken hook can never break a session.

## Examples

See `hooks/examples/`:

- `block-dangerous-bash.py` — `PreToolUse` hook that blocks Bash commands
  matching `rm -rf` / `mkfs` / fork-bomb / `dd` to a device.
- `log-tool-use.py` — `PostToolUse` hook that appends
  `{"time": ..., "tool": ...}` lines to `~/.clawd/hook-log.jsonl`.
- `verify-after-edit.py` — `PostToolUse` recipe below: lint/test Python
  files after every `Edit`/`Write`.

## Recipe: post-edit verify (lint + test after edits)

Zero new runtime code — just packaging. This is the harness-level answer
to "agents don't verify": after every `Edit`/`Write` that touches a `.py`
file, run `ruff check` on the file and `pytest -x -q` on it when it looks
like a test module. Fail-open, timeout-capped, and silent when ruff/pytest
aren't installed.

1. Copy `hooks/examples/verify-after-edit.py` somewhere stable and point
   `~/.clawd/hooks.json` at it (sample entry in
   `hooks/examples/verify-hooks.json`):

```json
{
  "PostToolUse": [
    { "command": "python3 /path/to/verify-after-edit.py" }
  ]
}
```

2. What the script does, per `Edit`/`Write` call:
   - Skips non-Python files and non-Edit/Write tools (exit 0).
   - `ruff check <file>` on any `.py` file.
   - `pytest -x -q <file>` only for test modules (`test_*.py`, `*_test.py`);
     other files record `pytest: skipped (not a test file)`.
   - Each check capped at ~60s; a missing ruff/pytest, a timeout, or any
     internal error is recorded as `skipped`, never raised.
   - **Always exits 0** — a broken or slow verify can never block the edit.
     (`PostToolUse` hooks can't block anyway; the script is defensive so it
     stays safe if reused elsewhere.)
   - Appends one JSON record per run to `~/.clawd/verify-log.jsonl`
     (`{"time", "tool", "file", "checks": [{check, status, ...}]}`) —
     check that log, or run the script manually, to see results.

3. Caveats:
   - The hook runner kills hook commands after **10s** by default, so keep
     per-file checks fast (ruff on one file is milliseconds; a single test
     file is usually <10s). For full-suite runs, background them
     (`nohup pytest -q &`) instead of blocking the hook.
   - Follow-up, not built here: wire post-edit **LSP diagnostics**
     (`src/tool_system/tools/lsp.py`) into this script so type errors
     surface the moment an edit lands, without shelling out.
