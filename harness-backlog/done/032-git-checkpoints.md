# 032 — Git-backed checkpoints: snapshot-before-write + private-refs storage

## What
Upgrade Clawd's existing checkpoints (`checkpoints.py` + `/rewind`) with the two checkpoint techniques the research surfaced: snapshot-before-write (a git commit before each file-modifying tool call, enabling verify-and-rollback) and private-refs storage (checkpoints stored under private git refs, invisible to normal git workflows).

## Why
- gemini-cli `checkpointUtils.ts` L86–181 (https://github.com/google-gemini/gemini-cli/blob/main/packages/core/src/utils/checkpointUtils.ts): git commit before each file-modifying tool call; JSON stores `{history, clientHistory, toolCall, commitHash, messageId}`; restore = git restore + conversation rewind. This is the verify-and-rollback primitive Clawd lacks: 019's reflection loop can *detect* a bad edit, but only a pre-write snapshot lets it *undo* one.
- cline: `git stash create` + synthesized third parent for untracked files, stored under `refs/cline/checkpoints/{sessionId}/{runCount}` in the user's own repo — invisible to `git stash list`; incremental hashing with a persistent `GIT_INDEX_FILE`; transactional restore via `refs/cline/restore-transactions/{uuid}`. The storage technique keeps checkpoints out of the user's way: no stray commits on their branch, no stash-list pollution.

Both are 1:1-portable techniques (git plumbing, no model involvement) that compose with the existing checkpoint system rather than replacing it. Fully model-agnostic.

## Suggested approach
- Snapshot-before-write: when enabled, a file-modifying tool call first creates a lightweight git snapshot (commit or `stash create`) of the affected paths; store `{toolCall, commitHash, message}` in the checkpoint record; on rollback, `git restore` to the snapshot then rewind the conversation (gemini pattern).
- Private-refs storage: keep snapshots under `refs/clawd/checkpoints/{sessionId}/{n}` (never on a branch, never in `stash list`); transactional restore records under `refs/clawd/restore-transactions/{uuid}` so interrupted restores can resume or roll back cleanly.
- Untracked files: synthesized third parent (cline pattern) so new files are covered too.
- Performance: incremental hashing with a persistent index; snapshots are cheap path-scoped operations, not full-repo commits.
- UI: `/rewind` lists snapshot checkpoints with what changed; rewind restores both files and conversation.

## Expected tests
- File-modifying tool call → snapshot ref created under `refs/clawd/checkpoints/`, invisible in `git log` and `git stash list`.
- Rollback restores file contents (tracked and untracked) and rewinds the conversation to the snapshot point.
- Interrupted restore → transaction record allows resume or clean rollback, no half-restored state.
- Snapshot overhead on a large repo stays within budget (incremental hashing, path-scoped).
- Full test suite green before shipping.
