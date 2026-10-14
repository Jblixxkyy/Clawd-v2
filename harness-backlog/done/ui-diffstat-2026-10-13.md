# Done: compact diff stat on terminal completion lines (2026-10-13)

Source: ui-tracker run 2026-10-13. gemini-cli
`packages/cli/src/ui/components/messages/DenseToolMessage.tsx`
(`getFileOpData`): file-edit tool calls render a `(+added, -removed)` stat
next to the result summary, green for additions, red for removals.

Implemented in `src/repl/tool_cards.py`:
- `_unified_diff_lines()` — raw full unified diff (untruncated)
- `_diff_stat()` — `(added, removed)` counts excluding `---`/`+++`/`@@` headers
- `_diff_renderables()` now takes the raw diff list and only truncates/renders
- `_write_diff()` / `_edit_diff()` return `(renderables, (added, removed))`
- `render_result()` appends ` (+N, -M)` to the `✓` completion line for
  Write/Edit when a real diff was computed; `+N` green, `-M` red

Tests: `DiffStatTests` in `tests/test_task_ux.py` (6 tests: edit stat,
multi-line write stat, span colors, no-stat-without-diff, stat survives
truncation, header exclusion). Full suite: 642 passed, 0 failed.

Proof: `~/workspace/your_files/ui-diffstat-2026-10-13.svg`

Follow-ups for future ui-tracker runs (terminal-first):
- crush `internal/ui/chat/tools.go` `pendingTool()`: animated spinner
  (`anim.Render()`) next to the pending tool name — ours prints a static `⠋`.
  Needs Rich Live or prompt_toolkit-aware in-place update in the REPL.
- gemini-cli `ToolStatsDisplay.tsx`: per-tool avg-duration table (a `/stats`
  style command), not per-call.
- opencode `packages/tui/src/component/dialog-workspace-file-changes.tsx`:
  workspace-level file-changes dialog — bigger feature, needs design.
- cline `apps/vscode/webview-ui/src/components/chat/task-header/`: task header
  with context-window meter — web-app side, we have a status bar already.
