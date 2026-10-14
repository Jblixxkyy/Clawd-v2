"""Tests for brief 032: git-backed checkpoints (snapshot-before-write).

All tests run in throwaway tmp git repos -- the snapshot machinery must
operate on the TARGET session repo, never on the Clawd-Code repo itself.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from src.tool_system import git_checkpoints as gcp


def _git(*args: str, cwd: str | Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=60,
    )


def _make_repo() -> Path:
    d = Path(tempfile.mkdtemp(prefix="gcp-test-"))
    _git("init", "-q", "-b", "main", cwd=d)
    _git("config", "user.email", "test@test.test", cwd=d)
    _git("config", "user.name", "test", cwd=d)
    (d / "tracked.txt").write_text("v1\n", encoding="utf-8")
    _git("add", "-A", cwd=d)
    _git("commit", "-qm", "init", cwd=d)
    return d


class _FakeConversation:
    def __init__(self, n: int):
        self.messages = [f"m{i}" for i in range(n)]


class GitCheckpointsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo = _make_repo()
        self._saved_env = os.environ.get("CLAWD_GIT_CHECKPOINTS")
        os.environ.pop("CLAWD_GIT_CHECKPOINTS", None)

    def tearDown(self) -> None:
        if self._saved_env is None:
            os.environ.pop("CLAWD_GIT_CHECKPOINTS", None)
        else:
            os.environ["CLAWD_GIT_CHECKPOINTS"] = self._saved_env
        shutil.rmtree(self.repo, ignore_errors=True)

    # -- snapshot creation & private-refs storage -------------------------

    def test_snapshot_creates_private_ref(self) -> None:
        rec = gcp.snapshot_for_tool(
            "sess", "tracked.txt", self.repo,
            tool_name="Edit", tool_input={"file_path": "tracked.txt"},
        )
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["n"], 1)
        self.assertEqual(rec["ref"], "refs/clawd/checkpoints/sess/0001")
        self.assertRegex(rec["commit"], r"^[0-9a-f]{40}$")
        refs = gcp.git_checkpoint_refs("sess", self.repo)
        self.assertEqual(refs, ["refs/clawd/checkpoints/sess/0001"])

    def test_snapshot_record_fields(self) -> None:
        """Record stores {toolCall, commitHash, message} per the brief."""
        rec = gcp.snapshot_for_tool(
            "sess", "tracked.txt", self.repo,
            tool_name="Edit", tool_input={"file_path": "tracked.txt"},
        )
        assert rec is not None
        self.assertEqual(rec["tool_call"]["name"], "Edit")   # toolCall
        self.assertRegex(rec["commit"], r"^[0-9a-f]{40}$")   # commitHash
        self.assertTrue(rec["message"])                       # message

    def test_snapshot_invisible_in_git_log_and_stash(self) -> None:
        before = _git("log", "--branches", "--oneline", cwd=self.repo).stdout
        rec = gcp.snapshot_for_tool("sess", "tracked.txt", self.repo, tool_name="Edit")
        assert rec is not None
        after = _git("log", "--branches", "--oneline", cwd=self.repo).stdout
        self.assertEqual(before, after, "snapshot must not appear in git log")
        self.assertEqual(_git("stash", "list", cwd=self.repo).stdout.strip(), "")
        branches = _git("branch", "-a", "--format=%(refname)", cwd=self.repo).stdout
        self.assertNotIn("clawd", branches)

    def test_snapshot_does_not_touch_real_index(self) -> None:
        status_before = _git("status", "--porcelain", cwd=self.repo).stdout
        gcp.snapshot_for_tool("sess", "tracked.txt", self.repo, tool_name="Edit")
        status_after = _git("status", "--porcelain", cwd=self.repo).stdout
        self.assertEqual(status_before, status_after)

    def test_non_repo_returns_none(self) -> None:
        d = Path(tempfile.mkdtemp(prefix="gcp-norepo-"))
        try:
            rec = gcp.snapshot_for_tool("sess", "f.txt", d, tool_name="Edit")
            self.assertIsNone(rec)
            self.assertEqual(gcp.list_git_checkpoints("sess", d), [])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_snapshot_enabled_env_override(self) -> None:
        os.environ["CLAWD_GIT_CHECKPOINTS"] = "1"
        self.assertTrue(gcp.snapshot_enabled())
        os.environ["CLAWD_GIT_CHECKPOINTS"] = "0"
        self.assertFalse(gcp.snapshot_enabled())

    # -- rollback: tracked + untracked + conversation ---------------------

    def test_rollback_restores_tracked_and_untracked(self) -> None:
        (self.repo / "new.txt").write_text("new-v1\n", encoding="utf-8")
        rec = gcp.snapshot_paths(
            "sess", self.repo, ["tracked.txt", "new.txt"], tool_name="Edit",
        )
        assert rec is not None
        self.assertEqual(rec["untracked"], ["new.txt"])  # cline third-parent pattern

        # mutate everything after the snapshot
        (self.repo / "tracked.txt").write_text("MUTATED\n", encoding="utf-8")
        (self.repo / "new.txt").write_text("MUTATED\n", encoding="utf-8")

        report = gcp.restore_git_checkpoint("sess", 1, self.repo)
        self.assertEqual(
            (self.repo / "tracked.txt").read_text(encoding="utf-8"), "v1\n"
        )
        self.assertEqual(
            (self.repo / "new.txt").read_text(encoding="utf-8"), "new-v1\n"
        )
        # restored untracked file is untracked again, not staged
        status = _git("status", "--porcelain", cwd=self.repo).stdout
        self.assertIn("?? new.txt", status)
        self.assertEqual(gcp.list_pending_transactions(self.repo), [])

    def test_rollback_deletes_file_created_after_snapshot(self) -> None:
        # Write of a brand-new file: snapshot the path first, then create it.
        gcp.snapshot_for_tool("sess", "brand_new.txt", self.repo, tool_name="Write")
        (self.repo / "brand_new.txt").write_text("hello\n", encoding="utf-8")
        report = gcp.restore_git_checkpoint("sess", 1, self.repo)
        self.assertIn("brand_new.txt", report["deleted"])
        self.assertFalse((self.repo / "brand_new.txt").exists())

    def test_rollback_rewinds_conversation(self) -> None:
        conv = _FakeConversation(2)
        rec = gcp.snapshot_for_tool(
            "sess", "tracked.txt", self.repo,
            tool_name="Edit", conversation=conv,
        )
        assert rec is not None
        self.assertEqual(rec["message_count"], 2)
        conv.messages.extend(["m2", "m3", "m4"])  # tool call + results happen
        report = gcp.restore_git_checkpoint("sess", 1, self.repo)
        dropped = gcp.rewind_conversation(conv, {"message_count": report["message_count"]})
        self.assertEqual(dropped, 3)
        self.assertEqual(conv.messages, ["m0", "m1"])

    def test_rewind_conversation_no_count_is_noop(self) -> None:
        conv = _FakeConversation(3)
        self.assertEqual(gcp.rewind_conversation(conv, {"message_count": None}), 0)
        self.assertEqual(len(conv.messages), 3)
        self.assertEqual(gcp.rewind_conversation(None, {"message_count": 1}), 0)

    def test_restore_unknown_checkpoint_raises(self) -> None:
        with self.assertRaises(ValueError):
            gcp.restore_git_checkpoint("sess", 99, self.repo)

    def test_rewind_lists_what_changed(self) -> None:
        rec = gcp.snapshot_for_tool("sess", "tracked.txt", self.repo, tool_name="Edit")
        assert rec is not None
        cps = gcp.list_git_checkpoints("sess", self.repo)
        self.assertEqual(len(cps), 1)
        summary = gcp.changed_files_summary(self.repo, cps[0])
        # snapshot taken on a clean tree: nothing differs from HEAD
        self.assertEqual(summary["total"], 0)
        # dirty the tree, snapshot again, now the diff shows the file
        (self.repo / "tracked.txt").write_text("dirty\n", encoding="utf-8")
        rec2 = gcp.snapshot_for_tool("sess2", "tracked.txt", self.repo, tool_name="Edit")
        assert rec2 is not None
        summary2 = gcp.changed_files_summary(self.repo, rec2)
        self.assertIn("tracked.txt", summary2["files"])

    # -- interrupted restore: transactional resume / abort ----------------

    def test_interrupted_restore_resume(self) -> None:
        gcp.snapshot_for_tool("sess", "tracked.txt", self.repo, tool_name="Edit")
        (self.repo / "tracked.txt").write_text("BEFORE-RESTORE\n", encoding="utf-8")

        # begin the transaction but "crash" before applying: no file touched,
        # yet a resumable record exists under refs/clawd/restore-transactions/
        txn = gcp.begin_restore_transaction("sess", 1, self.repo)
        self.assertEqual(txn["status"], "in_progress")
        self.assertEqual(
            (self.repo / "tracked.txt").read_text(encoding="utf-8"), "BEFORE-RESTORE\n"
        )
        pending = gcp.list_pending_transactions(self.repo)
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0]["uuid"], txn["uuid"])

        report = gcp.resume_pending_restore(self.repo, txn["uuid"])
        self.assertTrue(report.get("resumed"))
        self.assertEqual(
            (self.repo / "tracked.txt").read_text(encoding="utf-8"), "v1\n"
        )
        self.assertEqual(gcp.list_pending_transactions(self.repo), [])

    def test_interrupted_restore_abort_rolls_back_cleanly(self) -> None:
        gcp.snapshot_for_tool("sess", "tracked.txt", self.repo, tool_name="Edit")
        (self.repo / "tracked.txt").write_text("PRE-RESTORE\n", encoding="utf-8")

        txn = gcp.begin_restore_transaction("sess", 1, self.repo)
        # simulate a half-applied restore: worktree clobbered mid-apply
        (self.repo / "tracked.txt").write_text("HALF-RESTORED\n", encoding="utf-8")

        report = gcp.abort_pending_restore(self.repo, txn["uuid"])
        self.assertTrue(report.get("aborted"))
        self.assertEqual(
            (self.repo / "tracked.txt").read_text(encoding="utf-8"), "PRE-RESTORE\n"
        )
        self.assertEqual(gcp.list_pending_transactions(self.repo), [])

    def test_resume_unknown_uuid_raises(self) -> None:
        with self.assertRaises(KeyError):
            gcp.resume_pending_restore(self.repo, "deadbeef1234")

    # -- performance: incremental hashing, path-scoped --------------------

    def test_snapshot_overhead_stays_in_budget(self) -> None:
        big = Path(tempfile.mkdtemp(prefix="gcp-big-"))
        try:
            _git("init", "-q", "-b", "main", cwd=big)
            _git("config", "user.email", "t@t.t", cwd=big)
            _git("config", "user.name", "t", cwd=big)
            for i in range(1500):
                (big / f"file_{i:04d}.txt").write_text(f"content {i}\n", encoding="utf-8")
            _git("add", "-A", cwd=big)
            _git("commit", "-qm", "big", cwd=big)

            t0 = time.monotonic()
            rec1 = gcp.snapshot_for_tool("perf", "file_0001.txt", big, tool_name="Edit")
            first = time.monotonic() - t0
            self.assertIsNotNone(rec1)

            # incremental: only one file changed since; persistent index reused
            (big / "file_0002.txt").write_text("changed\n", encoding="utf-8")
            t0 = time.monotonic()
            rec2 = gcp.snapshot_for_tool("perf", "file_0002.txt", big, tool_name="Edit")
            second = time.monotonic() - t0
            self.assertIsNotNone(rec2)

            gitdir = _git("rev-parse", "--absolute-git-dir", cwd=big).stdout.strip()
            self.assertTrue(
                Path(gitdir, "clawd-checkpoints", "perf", "index").exists(),
                "persistent per-session index must exist for incremental hashing",
            )
            self.assertLess(first, 15.0, f"first snapshot took {first:.1f}s")
            self.assertLess(second, 8.0, f"incremental snapshot took {second:.1f}s")
        finally:
            shutil.rmtree(big, ignore_errors=True)

    # -- dispatch integration: Write tool -> git snapshot -----------------

    def test_registry_write_creates_git_snapshot(self) -> None:
        os.environ["CLAWD_GIT_CHECKPOINTS"] = "1"
        from src.tool_system.context import ToolContext
        from src.tool_system.protocol import ToolCall
        from src.tool_system.registry import ToolRegistry
        from src.tool_system.tools.write import FileWriteTool

        registry = ToolRegistry([FileWriteTool()])
        context = ToolContext(workspace_root=self.repo)
        result = registry.dispatch(
            ToolCall(
                name="Write",
                input={"file_path": "dispatched.txt", "content": "via-dispatch\n"},
                tool_use_id="t1",
            ),
            context,
        )
        self.assertFalse(result.is_error, result.output)
        self.assertTrue((self.repo / "dispatched.txt").exists())

        cps = gcp.list_git_checkpoints("default", self.repo)
        self.assertEqual(len(cps), 1)
        self.assertEqual(cps[0]["tool_call"]["name"], "Write")
        # git snapshot took over, so the legacy byte snapshot stayed out
        from src.tool_system import checkpoints as legacy

        self.assertEqual(legacy.list_checkpoints("default", self.repo), [])

        # rollback removes the dispatched file
        report = gcp.restore_git_checkpoint("default", 1, self.repo)
        self.assertIn("dispatched.txt", report["deleted"])
        self.assertFalse((self.repo / "dispatched.txt").exists())


if __name__ == "__main__":
    unittest.main()
