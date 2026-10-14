"""Git-backed checkpoints: snapshot-before-write + private-refs storage (brief 032).

Before a file-modifying tool call (Write/Edit), this module creates a
lightweight git snapshot of the affected paths and stores the resulting
commit under a *private ref*::

    refs/clawd/checkpoints/<session_id>/<n>

Private refs never touch any branch and never appear in ``git log`` /
``git stash list`` -- the snapshots stay out of the user's way.

Techniques ported 1:1 from the research brief:
- gemini-cli ``checkpointUtils.ts``: git snapshot before each file-modifying
  tool call; the checkpoint record stores ``{toolCall, commitHash, message}``;
  rollback = git restore to the snapshot + conversation rewind.
- cline: synthesized third parent for untracked files (``git stash create``
  pattern), incremental hashing via a persistent ``GIT_INDEX_FILE``, and
  transactional restore records under
  ``refs/clawd/restore-transactions/<uuid>`` so an interrupted restore can
  resume or roll back cleanly.

Everything operates on the TARGET repo (the repo the agent is working in),
never on the Clawd-Code repo itself.  All public functions are total: they
return ``None``/``[]`` instead of raising, except the restore entry points
which raise ``ValueError``/``KeyError`` for unknown checkpoints -- same
convention as ``checkpoints.py``.

Snapshot layout (mirrors ``git stash``):
- parent1 = HEAD at snapshot time (or no parent when the repo has no commits)
- parent2 = index state (when a real index exists)
- parent3 = untracked-file state (always created; empty tree when none)
- tree   = full-repo tree with the *scoped* paths updated to worktree state

Only the scoped paths are hashed per snapshot; the persistent per-session
``GIT_INDEX_FILE`` makes repeat snapshots incremental.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path

#: Private-refs roots.
REFS_PREFIX = "refs/clawd/checkpoints"
TXN_REFS_PREFIX = "refs/clawd/restore-transactions"

#: git's well-known empty tree hash.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

#: Identity used for snapshot commits (never touches the user's git config).
_GIT_IDENTITY = {
    "GIT_AUTHOR_NAME": "clawd-checkpoints",
    "GIT_AUTHOR_EMAIL": "clawd@localhost",
    "GIT_COMMITTER_NAME": "clawd-checkpoints",
    "GIT_COMMITTER_EMAIL": "clawd@localhost",
}

_TIMEOUT_S = 60


def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._-") or "default"


def _run(args: list[str], cwd: Path, env_extra: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update(_GIT_IDENTITY)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=_TIMEOUT_S,
    )


def snapshot_enabled() -> bool:
    """Whether git-backed snapshots are enabled.

    Env var ``CLAWD_GIT_CHECKPOINTS=1|0`` wins; otherwise the ``git_checkpoints``
    key in ``~/.clawd/config.json``; default OFF (opt-in plumbing). Never raises.
    """
    try:
        raw = os.environ.get("CLAWD_GIT_CHECKPOINTS")
        if raw is not None:
            return raw.strip().lower() not in ("", "0", "false", "no", "off")
        from src.config import git_checkpoints_enabled

        return bool(git_checkpoints_enabled())
    except Exception:
        return False


def repo_toplevel(start: str | Path) -> Path | None:
    """Return the git work-tree root containing ``start``, or ``None``."""
    try:
        p = Path(start)
        cwd = p if p.is_dir() else p.parent
        proc = _run(["rev-parse", "--show-toplevel"], cwd=cwd)
        if proc.returncode != 0:
            return None
        return Path(proc.stdout.strip())
    except Exception:
        return None


def _git_dir(repo: Path) -> Path | None:
    try:
        proc = _run(["rev-parse", "--absolute-git-dir"], cwd=repo)
        if proc.returncode != 0:
            return None
        return Path(proc.stdout.strip())
    except Exception:
        return None


def _head(repo: Path) -> str | None:
    proc = _run(["rev-parse", "--verify", "HEAD"], cwd=repo)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def _state_dir(repo: Path, session_id: str) -> Path | None:
    gitdir = _git_dir(repo)
    if gitdir is None:
        return None
    sdir = gitdir / "clawd-checkpoints" / _sanitize(session_id)
    return sdir


def _manifest_path(sdir: Path) -> Path:
    return sdir / "manifest.json"


def _load_manifest(sdir: Path) -> dict[str, dict]:
    try:
        raw = _manifest_path(sdir).read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    except Exception:
        pass
    return {}


def _save_manifest(sdir: Path, manifest: dict[str, dict]) -> None:
    try:
        sdir.mkdir(parents=True, exist_ok=True)
        tmp = _manifest_path(sdir).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        tmp.replace(_manifest_path(sdir))
    except Exception:
        pass


def _untracked_files(repo: Path, relpaths: list[str]) -> list[str]:
    """Untracked (non-ignored) files under ``relpaths`` (repo-relative)."""
    if not relpaths:
        return []
    proc = _run(
        ["ls-files", "--others", "--exclude-standard", "-z", "--", *relpaths],
        cwd=repo,
    )
    if proc.returncode != 0:
        return []
    return [p for p in proc.stdout.split("\0") if p]


def _update_index_stdin(
    repo: Path,
    index_file: Path,
    paths: list[str],
    *,
    extra_args: list[str] | None = None,
) -> bool:
    """Run ``git update-index`` with an explicit NUL-separated path list.

    Explicit paths via ``--stdin`` are used instead of pathspecs because this
    environment's git ignores bare ``.``/``--all`` pathspecs on a freshly
    ``read-tree``-seeded index ("Ignoring path"). Never raises.
    """
    try:
        env = dict(os.environ)
        env.update(_GIT_IDENTITY)
        env["GIT_INDEX_FILE"] = str(index_file)
        proc = subprocess.run(
            ["git", "update-index", *(extra_args or []), "-z", "--stdin"],
            cwd=str(repo),
            env=env,
            input="\0".join(paths),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _tree_of_paths(repo: Path, paths: list[str], index_file: Path) -> str | None:
    """Write a tree containing exactly ``paths`` (repo-relative) via a temp index."""
    try:
        index_file.parent.mkdir(parents=True, exist_ok=True)
        if index_file.exists():
            index_file.unlink()
        env = {"GIT_INDEX_FILE": str(index_file)}
        proc = _run(["read-tree", "--empty"], cwd=repo, env_extra=env)
        if proc.returncode != 0:
            return None
        if paths:
            if not _update_index_stdin(repo, index_file, paths, extra_args=["--add"]):
                return None
        proc = _run(["write-tree"], cwd=repo, env_extra=env)
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()
    except Exception:
        return None


def _seed_session_index(repo: Path, idx: Path, head: str | None) -> bool:
    """Seed the persistent per-session index once (incremental hashing base)."""
    try:
        idx.parent.mkdir(parents=True, exist_ok=True)
        if idx.exists():
            return True
        env = {"GIT_INDEX_FILE": str(idx)}
        if head:
            proc = _run(["read-tree", head], cwd=repo, env_extra=env)
        else:
            proc = _run(["read-tree", "--empty"], cwd=repo, env_extra=env)
        return proc.returncode == 0
    except Exception:
        return False


def _snapshot_tree(
    repo: Path, session_id: str, relpaths: list[str]
) -> tuple[str | None, str | None, list[str]]:
    """Build (main_tree, untracked_commit, untracked_files) for a snapshot.

    Only ``relpaths`` are (re-)hashed into the persistent session index; the
    rest of the tree comes from the cached index state (incremental).
    """
    sdir = _state_dir(repo, session_id)
    if sdir is None:
        return None, None, []
    head = _head(repo)
    idx = sdir / "index"
    if not _seed_session_index(repo, idx, head):
        return None, None, []

    env = {"GIT_INDEX_FILE": str(idx)}
    # Path-scoped update: hash only the affected paths into the cached index.
    if not _update_index_stdin(
        repo, idx, relpaths, extra_args=["--add", "--remove", "-q"]
    ):
        return None, None, []
    proc = _run(["write-tree"], cwd=repo, env_extra=env)
    if proc.returncode != 0:
        return None, None, []
    main_tree = proc.stdout.strip()

    # Untracked files -> synthesized third parent (cline pattern).
    untracked = _untracked_files(repo, relpaths)
    tmpidx = sdir / "untracked-index"
    tree_u = _tree_of_paths(repo, untracked, tmpidx) if untracked else EMPTY_TREE
    if tree_u is None:
        return None, None, []
    proc = _run(
        ["commit-tree", tree_u, "-m", "clawd checkpoint: untracked file state"],
        cwd=repo,
    )
    if proc.returncode != 0:
        return None, None, []
    untracked_commit = proc.stdout.strip()
    return main_tree, untracked_commit, untracked


def _index_tree_commit(repo: Path) -> str | None:
    """Commit of the real index state (parent2), or None when unavailable."""
    gitdir = _git_dir(repo)
    if gitdir is None:
        return None
    real_index = gitdir / "index"
    if not real_index.exists():
        return None
    env = {"GIT_INDEX_FILE": str(real_index)}
    proc = _run(["write-tree"], cwd=repo, env_extra=env)
    if proc.returncode != 0:
        return None
    tree = proc.stdout.strip()
    if tree == EMPTY_TREE:
        return None
    proc = _run(["commit-tree", tree, "-m", "clawd checkpoint: index state"], cwd=repo)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def snapshot_paths(
    session_id: str,
    repo_root: str | Path,
    paths: list[str | Path],
    *,
    tool_name: str | None = None,
    message: str | None = None,
    message_count: int | None = None,
) -> dict | None:
    """Create a lightweight git snapshot of ``paths`` before a file-modifying call.

    The snapshot commit is stored under ``refs/clawd/checkpoints/<session>/<n>``
    -- never on a branch, invisible in ``git log`` and ``git stash list``.

    Returns the checkpoint record ``{n, ref, commit, tool_call, message, ...}``
    or ``None`` when no snapshot could be made (not a repo, git missing, ...).
    Never raises.
    """
    try:
        repo = repo_toplevel(repo_root)
        if repo is None:
            return None
        relpaths: list[str] = []
        for p in paths:
            pp = Path(p)
            if not pp.is_absolute():
                pp = repo / pp
            try:
                rel = pp.resolve().relative_to(repo).as_posix()
            except ValueError:
                continue  # outside the repo -- skip
            relpaths.append(rel)
        if not relpaths:
            return None

        head = _head(repo)
        main_tree, untracked_commit, untracked = _snapshot_tree(repo, _sanitize(session_id), relpaths)
        if main_tree is None or untracked_commit is None:
            return None

        sdir = _state_dir(repo, session_id)
        assert sdir is not None
        manifest = _load_manifest(sdir)
        n = max((int(k) for k in manifest if str(k).isdigit()), default=0) + 1
        ref = f"{REFS_PREFIX}/{_sanitize(session_id)}/{n:04d}"

        parents: list[str] = []
        if head:
            parents += ["-p", head]
        index_commit = _index_tree_commit(repo)
        if index_commit:
            parents += ["-p", index_commit]
        parents += ["-p", untracked_commit]

        label = message or f"{tool_name or 'tool'} {' '.join(relpaths)}"
        proc = _run(
            ["commit-tree", main_tree, *parents, "-m", f"clawd checkpoint #{n:04d}: {label}"],
            cwd=repo,
        )
        if proc.returncode != 0:
            return None
        commit = proc.stdout.strip()

        proc = _run(["update-ref", ref, commit], cwd=repo)
        if proc.returncode != 0:
            return None

        record = {
            "n": n,
            "ref": ref,
            "commit": commit,
            "parent1": head,
            "parent2": index_commit,
            "parent3": untracked_commit,
            "paths": relpaths,
            "untracked": untracked,
            "tool_call": {"name": tool_name} if tool_name else {},
            "message": label,
            "message_count": message_count,
            "created": datetime.now().isoformat(timespec="seconds"),
        }
        manifest[str(n)] = record
        _save_manifest(sdir, manifest)
        return record
    except Exception:
        return None


def snapshot_for_tool(
    session_id: str,
    file_path: str | Path,
    workspace_root: str | Path,
    *,
    tool_name: str | None = None,
    tool_input: dict | None = None,
    conversation: object | None = None,
) -> dict | None:
    """Snapshot-before-write entry point for the tool dispatch.

    Snapshots ``file_path`` in the TARGET repo (resolved from ``workspace_root``;
    returns ``None`` when that is not inside a git repo).  ``tool_input`` is
    summarized (never stored whole -- Write inputs can be huge).  When a live
    ``conversation`` object is supplied, its message count is recorded so a
    later restore can also rewind the conversation.  Never raises.
    """
    try:
        root = Path(workspace_root).resolve()
        p = Path(file_path)
        if not p.is_absolute():
            p = root / p
        message_count = None
        if conversation is not None:
            msgs = getattr(conversation, "messages", None)
            if msgs is not None:
                try:
                    message_count = len(msgs)
                except Exception:
                    message_count = None
        input_summary = None
        if isinstance(tool_input, dict):
            fp = tool_input.get("file_path")
            if isinstance(fp, str):
                input_summary = fp[:200]
        return _snapshot_for_tool_inner(
            session_id, root, p, tool_name, input_summary, message_count
        )
    except Exception:
        return None


def _snapshot_for_tool_inner(
    session_id: str,
    root: Path,
    p: Path,
    tool_name: str | None,
    input_summary: str | None,
    message_count: int | None,
) -> dict | None:
    rec = snapshot_paths(
        session_id,
        root,
        [p],
        tool_name=tool_name,
        message=f"{tool_name or 'tool'} {p.name}" if tool_name else None,
        message_count=message_count,
    )
    if rec is not None and input_summary:
        # keep the full file_path out of the record when it is long
        rec["tool_call"]["file_path"] = input_summary
    return rec


def list_git_checkpoints(session_id: str, repo_root: str | Path) -> list[dict]:
    """List git checkpoints for a session, oldest first. Never raises."""
    try:
        repo = repo_toplevel(repo_root)
        if repo is None:
            return []
        sdir = _state_dir(repo, session_id)
        if sdir is None or not _manifest_path(sdir).exists():
            return []
        manifest = _load_manifest(sdir)
        records = [v for k, v in manifest.items() if str(k).isdigit()]
        records.sort(key=lambda r: int(r.get("n", 0)))
        return records
    except Exception:
        return []


def changed_files_summary(repo: Path, record: dict, limit: int = 5) -> dict:
    """What changed between the snapshot's parent1 (HEAD) and the snapshot.

    Returns ``{"files": [...], "total": N, "stat": "..."}``. Never raises.
    """
    try:
        parent1 = record.get("parent1") or EMPTY_TREE
        commit = record["commit"]
        proc = _run(["diff-tree", "--no-commit-id", "--name-only", "-r", parent1, commit], cwd=repo)
        files = [f for f in proc.stdout.splitlines() if f] if proc.returncode == 0 else []
        stat = ""
        proc = _run(["diff-tree", "--no-commit-id", "--stat", "-r", parent1, commit], cwd=repo)
        if proc.returncode == 0:
            stat = proc.stdout.strip()
        return {"files": files[:limit], "total": len(files), "stat": stat}
    except Exception:
        return {"files": [], "total": 0, "stat": ""}


# ---------------------------------------------------------------------------
# Transactional restore
# ---------------------------------------------------------------------------


def _txn_ref(record_uuid: str) -> str:
    return f"{TXN_REFS_PREFIX}/{record_uuid}"


def _read_txn(repo: Path, record_uuid: str) -> dict | None:
    try:
        ref = _txn_ref(record_uuid)
        proc = _run(["cat-file", "blob", ref], cwd=repo)
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _write_txn(repo: Path, record_uuid: str, txn: dict) -> bool:
    try:
        blob = json.dumps(txn, indent=2).encode("utf-8")
        env = dict(os.environ)
        env.update(_GIT_IDENTITY)
        proc = subprocess.run(
            ["git", "hash-object", "-w", "--stdin"],
            cwd=str(repo),
            env=env,
            input=blob,
            capture_output=True,
            timeout=_TIMEOUT_S,
        )
        if proc.returncode != 0:
            return False
        sha = proc.stdout.decode().strip()
        proc = _run(["update-ref", _txn_ref(record_uuid), sha], cwd=repo)
        return proc.returncode == 0
    except Exception:
        return False


def _delete_txn(repo: Path, record_uuid: str) -> None:
    try:
        _run(["update-ref", "-d", _txn_ref(record_uuid)], cwd=repo)
    except Exception:
        pass


def _all_worktree_files(repo: Path) -> list[str]:
    """All tracked + untracked (non-ignored) files, repo-relative."""
    files: list[str] = []
    try:
        proc = _run(["ls-files", "-z"], cwd=repo)
        if proc.returncode == 0:
            files.extend(p for p in proc.stdout.split("\0") if p)
        proc = _run(["ls-files", "--others", "--exclude-standard", "-z"], cwd=repo)
        if proc.returncode == 0:
            files.extend(p for p in proc.stdout.split("\0") if p)
    except Exception:
        pass
    return files


def _full_worktree_snapshot(repo: Path, session_id: str, label: str) -> tuple[str | None, list[str]]:
    """Snapshot the ENTIRE current worktree state (safety net for restores)."""
    sdir = _state_dir(repo, session_id)
    if sdir is None:
        return None, []
    head = _head(repo)
    tmpidx = sdir / "safety-index"
    try:
        if tmpidx.exists():
            tmpidx.unlink()
        env = {"GIT_INDEX_FILE": str(tmpidx)}
        if head:
            proc = _run(["read-tree", head], cwd=repo, env_extra=env)
        else:
            proc = _run(["read-tree", "--empty"], cwd=repo, env_extra=env)
        if proc.returncode != 0:
            return None, []
        # Explicit path list: bare "." pathspecs are ignored by update-index
        # on a freshly seeded index (see _update_index_stdin).
        if not _update_index_stdin(
            repo, tmpidx, _all_worktree_files(repo), extra_args=["--add", "--remove", "-q"]
        ):
            return None, []
        proc = _run(["write-tree"], cwd=repo, env_extra=env)
        if proc.returncode != 0:
            return None, []
        tree = proc.stdout.strip()
        untracked = _untracked_files(repo, ["."])
        parents = ["-p", head] if head else []
        proc = _run(["commit-tree", tree, *parents, "-m", f"clawd restore safety: {label}"], cwd=repo)
        if proc.returncode != 0:
            return None, []
        return proc.stdout.strip(), untracked
    except Exception:
        return None, []
    finally:
        try:
            if tmpidx.exists():
                tmpidx.unlink()
        except Exception:
            pass


def begin_restore_transaction(session_id: str, n: int, repo_root: str | Path) -> dict:
    """Start a transactional restore: safety-snapshot current state, then write
    the transaction record under ``refs/clawd/restore-transactions/<uuid>``.

    The record is written BEFORE any file is touched, so a crash between here
    and completion leaves a resumable/abortable transaction -- never a
    half-restored worktree with no way back. Raises ``ValueError`` when the
    checkpoint does not exist.
    """
    repo = repo_toplevel(repo_root)
    if repo is None:
        raise ValueError(f"not a git repo: {repo_root}")
    sdir = _state_dir(repo, session_id)
    if sdir is None:
        raise ValueError(f"cannot access git dir for {repo}")
    manifest = _load_manifest(sdir)
    record = manifest.get(str(int(n)))
    if not isinstance(record, dict):
        raise ValueError(f"no git checkpoint #{int(n):04d} for session {session_id!r}")

    record_uuid = uuid.uuid4().hex[:12]
    safety_commit, safety_untracked = _full_worktree_snapshot(
        repo, session_id, f"before restoring #{int(n):04d}"
    )
    if safety_commit is None:
        raise ValueError("could not snapshot current worktree state; restore aborted safely")

    # Files created after the checkpoint (untracked now, not untracked then,
    # within the snapshot's paths) must be deleted on restore.
    now_untracked = set(_untracked_files(repo, list(record.get("paths") or [])))
    then_untracked = set(record.get("untracked") or [])
    created_after = sorted(now_untracked - then_untracked)

    txn = {
        "uuid": record_uuid,
        "session_id": _sanitize(session_id),
        "n": int(n),
        "ref": record.get("ref"),
        "target_commit": record.get("commit"),
        "target_untracked": list(record.get("untracked") or []),
        "scoped_paths": list(record.get("paths") or []),
        "safety_commit": safety_commit,
        "safety_untracked": safety_untracked,
        "created_after": created_after,
        "status": "in_progress",
        "error": None,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    if not _write_txn(repo, record_uuid, txn):
        raise ValueError("could not write restore transaction record; nothing was changed")
    return txn


def _apply_tree(
    repo: Path,
    target_commit: str,
    target_untracked: list[str],
    scoped_paths: list[str],
    created_after: list[str],
    session_id: str,
) -> dict:
    """Apply a snapshot commit to the worktree+index (idempotent).

    - ``git read-tree --reset -u <commit>`` restores tracked files (and
      recreates snapshot-time untracked files as index entries).
    - Snapshot-time untracked files are unstaged again (``git rm --cached``).
    - Files created after the snapshot (``created_after``) are deleted.
    Returns ``{"restored": [...], "deleted": [...]}``.
    """
    restored: list[str] = []
    deleted: list[str] = []
    proc = _run(["read-tree", "--reset", "-u", target_commit], cwd=repo)
    if proc.returncode != 0:
        raise ValueError(f"git read-tree failed: {proc.stderr.strip()[:200]}")
    restored = list(scoped_paths)

    if target_untracked:
        proc = _run(["rm", "--cached", "-q", "--", *target_untracked], cwd=repo)
        # non-fatal if some are already absent

    for rel in created_after:
        p = repo / rel
        try:
            if p.is_file() or p.is_symlink():
                p.unlink()
                deleted.append(rel)
            elif p.is_dir():
                # only remove if it became empty (never nuke a populated dir)
                try:
                    p.rmdir()
                    deleted.append(rel + "/")
                except OSError:
                    pass
        except Exception:
            continue

    # Re-seed the persistent session index from the restored commit so future
    # snapshots hash against the restored state, not stale pre-restore state.
    sdir = _state_dir(repo, session_id)
    if sdir is not None:
        try:
            idx = sdir / "index"
            if idx.exists():
                idx.unlink()
            _run(
                ["read-tree", "--index-output", str(idx), target_commit],
                cwd=repo,
            )
        except Exception:
            pass
    return {"restored": restored, "deleted": deleted}


def _complete_txn(repo: Path, record_uuid: str) -> None:
    _delete_txn(repo, record_uuid)


def restore_git_checkpoint(
    session_id: str,
    n: int,
    repo_root: str | Path,
    *,
    resume_existing: bool = True,
) -> dict:
    """Restore checkpoint ``n`` transactionally, then return the restore report.

    Report: ``{"restored": [...], "deleted": [...], "message_count": ...,
    "txn": <uuid>}``.  ``message_count`` is the conversation length recorded at
    snapshot time (``None`` when not recorded) -- feed it to
    :func:`rewind_conversation`.  When ``resume_existing`` is true, a stale
    in-progress transaction for the same session+checkpoint is resumed instead
    of starting a new one.  Raises ``ValueError`` for unknown checkpoints.
    """
    repo = repo_toplevel(repo_root)
    if repo is None:
        raise ValueError(f"not a git repo: {repo_root}")

    if resume_existing:
        for txn in list_pending_transactions(repo):
            if (
                txn.get("session_id") == _sanitize(session_id)
                and int(txn.get("n", -1)) == int(n)
                and txn.get("status") in ("in_progress", "failed")
            ):
                return resume_pending_restore(repo, txn["uuid"])

    txn = begin_restore_transaction(session_id, n, repo)
    record_uuid = txn["uuid"]
    try:
        report = _apply_tree(
            repo,
            txn["target_commit"],
            txn["target_untracked"],
            txn["scoped_paths"],
            txn["created_after"],
            _sanitize(session_id),
        )
    except Exception as exc:
        txn["status"] = "failed"
        txn["error"] = str(exc)[:500]
        _write_txn(repo, record_uuid, txn)
        raise
    _complete_txn(repo, record_uuid)

    sdir = _state_dir(repo, session_id)
    manifest = _load_manifest(sdir) if sdir else {}
    record = manifest.get(str(int(n)), {})
    report["message_count"] = record.get("message_count")
    report["txn"] = record_uuid
    return report


def list_pending_transactions(repo_root: str | Path) -> list[dict]:
    """List restore-transaction records (resumable/abortable). Never raises."""
    try:
        repo = repo_toplevel(repo_root)
        if repo is None:
            return []
        proc = _run(["for-each-ref", "--format=%(refname)", TXN_REFS_PREFIX], cwd=repo)
        if proc.returncode != 0:
            return []
        out = []
        for refname in proc.stdout.splitlines():
            record_uuid = refname.strip().split("/")[-1]
            txn = _read_txn(repo, record_uuid)
            if txn:
                out.append(txn)
        out.sort(key=lambda t: str(t.get("created", "")))
        return out
    except Exception:
        return []


def resume_pending_restore(repo_root: str | Path, record_uuid: str) -> dict:
    """Resume an interrupted restore (idempotent apply), then complete it.

    Raises ``KeyError`` for an unknown transaction uuid.
    """
    repo = repo_toplevel(repo_root)
    if repo is None:
        raise ValueError(f"not a git repo: {repo_root}")
    txn = _read_txn(repo, record_uuid)
    if txn is None:
        raise KeyError(f"no restore transaction {record_uuid!r}")
    if txn.get("status") == "completed":
        _complete_txn(repo, record_uuid)
        return {"status": "already_completed", "txn": record_uuid}
    try:
        report = _apply_tree(
            repo,
            txn["target_commit"],
            txn.get("target_untracked") or [],
            txn.get("scoped_paths") or [],
            txn.get("created_after") or [],
            txn.get("session_id") or "default",
        )
    except Exception as exc:
        txn["status"] = "failed"
        txn["error"] = str(exc)[:500]
        _write_txn(repo, record_uuid, txn)
        raise
    _complete_txn(repo, record_uuid)
    report["txn"] = record_uuid
    report["resumed"] = True
    return report


def abort_pending_restore(repo_root: str | Path, record_uuid: str) -> dict:
    """Cleanly roll back an interrupted restore to the pre-restore state.

    Applies the transaction's safety snapshot, then deletes the transaction
    record. Raises ``KeyError`` for an unknown transaction uuid.
    """
    repo = repo_toplevel(repo_root)
    if repo is None:
        raise ValueError(f"not a git repo: {repo_root}")
    txn = _read_txn(repo, record_uuid)
    if txn is None:
        raise KeyError(f"no restore transaction {record_uuid!r}")
    safety = txn.get("safety_commit")
    if not safety:
        _complete_txn(repo, record_uuid)
        raise ValueError("transaction has no safety snapshot; record discarded")
    report = _apply_tree(
        repo,
        safety,
        txn.get("safety_untracked") or [],
        [],  # safety snapshot is whole-repo; nothing extra to delete
        [],
        txn.get("session_id") or "default",
    )
    _complete_txn(repo, record_uuid)
    report["txn"] = record_uuid
    report["aborted"] = True
    return report


def rewind_conversation(conversation: object | None, record: dict | None) -> int:
    """Truncate ``conversation.messages`` to the length recorded at snapshot time.

    Returns the number of messages dropped (0 when the record has no
    ``message_count`` or there is nothing to drop). Never raises.
    """
    try:
        if conversation is None or not isinstance(record, dict):
            return 0
        count = record.get("message_count")
        if count is None:
            return 0
        msgs = getattr(conversation, "messages", None)
        if msgs is None:
            return 0
        count = int(count)
        dropped = max(0, len(msgs) - count)
        if dropped:
            del msgs[count:]
        return dropped
    except Exception:
        return 0


def git_checkpoint_refs(session_id: str, repo_root: str | Path) -> list[str]:
    """List private checkpoint refs for a session (for tests/diagnostics)."""
    try:
        repo = repo_toplevel(repo_root)
        if repo is None:
            return []
        proc = _run(
            ["for-each-ref", "--format=%(refname)", f"{REFS_PREFIX}/{_sanitize(session_id)}"],
            cwd=repo,
        )
        if proc.returncode != 0:
            return []
        return [r for r in proc.stdout.splitlines() if r.strip()]
    except Exception:
        return []
