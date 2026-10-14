# 034 — Keep the session dashboard browsable during async lifecycle operations

## What
For async session operations (delete/archive/rename) in the session dashboard / web app session list: hide the affected session optimistically, render an inline progress line, keep the list browsable while the operation is pending, gate only the actions that actually conflict with the pending operation, and preserve selection after the removal completes.

## Why
From openai/codex@`95273b84` (https://github.com/openai/codex/commit/95273b84.patch): archiving/deleting a task previously blocked the agent dashboard until the server request finished. Now the dashboard stays interactive while the RPC is pending: the affected task is hidden immediately (optimistic), an inline progress message shows ("Archiving task… You can still browse"), selection movement and search keep working, and only conflicting actions (task switching, new tasks, leaving the dashboard) are suppressed until completion; dashboard selection is preserved after removal.

Qualifies as a single-harness UX gem, model-agnostic: our web app and TUI session views have the same class of async lifecycle ops, and the pattern (optimistic hide + inline progress + narrowly-scoped input gating) ports directly. Passes the fitness test — UI responsiveness that helps every model/user equally.

## Suggested approach
- In the web app (and TUI session views if they share the code path): on delete/archive/rename, immediately remove the session row from the rendered list and show an inline progress line ("Deleting session… you can keep browsing").
- During the pending operation, allow navigation/selection/search; suppress only conflicting actions (switch-to-pending-delete, create new session, leaving the view) with an inline explanation.
- On completion, keep the selection index stable — the item below the removed one becomes selected rather than resetting to the top.
- On failure, restore the row with an error state and an actionable message.

## Expected tests
- Deleting a session keeps the list browsable (navigation + search work) during the async operation.
- Conflicting actions (switch/create) are blocked with an inline message while the operation is pending.
- Selection index is preserved after the removal completes.
- Failed deletion restores the row with an error message.
- Full test suite green before shipping.
