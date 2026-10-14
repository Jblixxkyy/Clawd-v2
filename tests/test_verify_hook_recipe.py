"""Tests for Q8: the post-edit verify hook recipe.

Covers hooks/examples/verify-after-edit.py (fail-open behavior, ruff/pytest
invocation, logging) and the sample hooks.json entry.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "hooks" / "examples" / "verify-after-edit.py"
SAMPLE_HOOKS_JSON = Path(__file__).resolve().parent.parent / "hooks" / "examples" / "verify-hooks.json"


def _run_hook(payload: dict | str, env: dict | None = None) -> subprocess.CompletedProcess:
    import sys as _sys

    stdin = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.run(
        [_sys.executable, str(SCRIPT)],
        input=stdin,
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def _payload(tool: str, file_path: str, event: str = "PostToolUse") -> dict:
    return {"event": event, "tool": tool, "input": {"file_path": file_path}, "output": {}}


class TestVerifyHookFailOpen(unittest.TestCase):
    def test_garbage_stdin_exits_zero(self):
        proc = _run_hook("this is not json{{{")
        self.assertEqual(proc.returncode, 0)

    def test_wrong_event_exits_zero(self):
        proc = _run_hook(_payload("Edit", "/tmp/x.py", event="PreToolUse"))
        self.assertEqual(proc.returncode, 0)

    def test_non_edit_write_tool_skipped(self):
        proc = _run_hook(_payload("Bash", "/tmp/x.py"))
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout.strip(), "")

    def test_non_python_file_skipped(self):
        proc = _run_hook(_payload("Edit", "/tmp/notes.txt"))
        self.assertEqual(proc.returncode, 0)

    def test_missing_file_skipped(self):
        proc = _run_hook(_payload("Write", "/tmp/does-not-exist-xyz.py"))
        self.assertEqual(proc.returncode, 0)


class TestVerifyHookChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.home = self.root / "home"
        self.home.mkdir()
        # fake ruff/pytest on PATH
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self._make_exe("ruff", "#!/bin/sh\nexit 0\n")
        self._make_exe("pytest", "#!/bin/sh\nexit 0\n")
        self.env = {
            **os.environ,
            "PATH": str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
            "HOME": str(self.home),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def _make_exe(self, name: str, body: str) -> Path:
        p = self.bin / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return p

    def _log_records(self) -> list[dict]:
        log = self.home / ".clawd" / "verify-log.jsonl"
        if not log.exists():
            return []
        return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]

    def test_edit_py_runs_ruff_and_skips_pytest_for_non_test(self):
        target = self.root / "mod.py"
        target.write_text("x = 1\n")
        proc = _run_hook(_payload("Edit", str(target)), env=self.env)
        self.assertEqual(proc.returncode, 0)
        records = self._log_records()
        self.assertEqual(len(records), 1)
        checks = {Path(c["check"].split()[0]).name: c["status"] for c in records[0]["checks"]}
        self.assertEqual(checks.get("ruff"), "pass")
        self.assertEqual(records[0]["checks"][1]["status"], "skipped")  # pytest: not a test file

    def test_test_file_runs_pytest(self):
        target = self.root / "test_mod.py"
        target.write_text("def test_x():\n    assert True\n")
        proc = _run_hook(_payload("Write", str(target)), env=self.env)
        self.assertEqual(proc.returncode, 0)
        records = self._log_records()
        checks = {Path(c["check"].split()[0]).name: c["status"] for c in records[0]["checks"]}
        self.assertEqual(checks.get("pytest"), "pass")

    def test_failing_check_is_logged_but_hook_still_exits_zero(self):
        self._make_exe("pytest", "#!/bin/sh\necho 'FAILED stuff'\nexit 1\n")
        target = self.root / "test_mod.py"
        target.write_text("def test_x():\n    assert True\n")
        proc = _run_hook(_payload("Edit", str(target)), env=self.env)
        self.assertEqual(proc.returncode, 0)  # fail-open: hook never blocks
        records = self._log_records()
        pytest_checks = [c for c in records[0]["checks"] if Path(c["check"].split()[0]).name == "pytest"]
        self.assertEqual(pytest_checks[0]["status"], "fail")
        self.assertEqual(pytest_checks[0]["exit_code"], 1)

    def test_missing_ruff_pytest_recorded_as_skipped(self):
        env = {**self.env, "PATH": "/nonexistent-bin-dir"}
        target = self.root / "mod.py"
        target.write_text("x = 1\n")
        proc = _run_hook(_payload("Edit", str(target)), env=env)
        self.assertEqual(proc.returncode, 0)
        records = self._log_records()
        statuses = [c["status"] for c in records[0]["checks"]]
        self.assertTrue(all(s == "skipped" for s in statuses))


class TestVerifyHooksJsonSample(unittest.TestCase):
    def test_sample_is_valid_json_with_post_tool_use(self):
        data = json.loads(SAMPLE_HOOKS_JSON.read_text())
        self.assertIn("PostToolUse", data)
        commands = [
            e["command"] if isinstance(e, dict) else e
            for e in data["PostToolUse"]
        ]
        self.assertTrue(any("verify-after-edit.py" in c for c in commands))

    def test_docs_mention_recipe(self):
        docs = Path(__file__).resolve().parent.parent / "docs" / "hooks.md"
        text = docs.read_text(encoding="utf-8")
        self.assertIn("verify-after-edit", text)
        self.assertIn("fail", text.lower())  # fail-open documented


if __name__ == "__main__":
    unittest.main()
