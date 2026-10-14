"""Tests for brief 033: plugin/hook isolation and loud load failures.

- Hook code runs in an isolated execution context (dedicated worker
  thread, never the agent loop's thread); a crashing hook cannot poison
  loop state. Failures are audit-logged ("hook-failed"/"hook-timeout"),
  the plugin is marked failed for the event, and the turn continues
  (fail-open guard stays).
- A hook that cannot load (missing bootstrap script/interpreter, bad
  config) surfaces a NAMED, actionable error (PluginLoadError subclass
  naming the exact missing piece) in /doctor output and session-start
  warnings. Never a quiet skip.
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tool_system.hooks.isolation import (
    PluginBootstrapMissingError,
    PluginConfigError,
    PluginLoadError,
    plugin_load_warnings,
    run_command_isolated,
    run_in_worker,
    validate_hook_config,
    validate_hook_entry,
)
from src.tool_system.hooks.runner import run_hooks

CAPTURE_SCRIPT = """\
import json, sys
payload = json.load(sys.stdin)
with open(sys.argv[1], "w") as f:
    json.dump(payload, f)
"""


def _write_script(tmpdir: Path, name: str, body: str) -> Path:
    path = tmpdir / name
    path.write_text(body)
    return path


class IsolationBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.hooks_json = self.tmpdir / "hooks.json"
        self.audit_log = self.tmpdir / "hooks-audit.log"
        self._config_patcher = patch(
            "src.tool_system.hooks.runner._default_config_path",
            return_value=self.hooks_json,
        )
        self._audit_patcher = patch(
            "src.tool_system.hooks.runner._audit_log_path",
            return_value=self.audit_log,
        )
        self._config_patcher.start()
        self._audit_patcher.start()

    def tearDown(self):
        self._audit_patcher.stop()
        self._config_patcher.stop()
        self.tmp.cleanup()

    def _write_config(self, config: dict) -> None:
        self.hooks_json.write_text(json.dumps(config))

    def _read_audit(self) -> list[dict]:
        if not self.audit_log.exists():
            return []
        return [
            json.loads(line)
            for line in self.audit_log.read_text().splitlines()
            if line.strip()
        ]


class TestIsolatedExecution(IsolationBase):
    """Hook code never runs on the caller's (loop's) thread."""

    def test_worker_runs_off_calling_thread(self):
        seen = {}

        def probe():
            seen["thread"] = threading.current_thread()

        completed, payload = run_in_worker(probe, timeout=5.0, name="test-probe")
        self.assertTrue(completed)
        self.assertIsNone(payload)
        self.assertIsNot(seen["thread"], threading.current_thread())
        self.assertNotEqual(seen["thread"].name, threading.current_thread().name)

    def test_worker_captures_system_exit(self):
        def die():
            raise SystemExit(3)

        completed, payload = run_in_worker(die, timeout=5.0)
        self.assertFalse(completed)
        self.assertIsInstance(payload, SystemExit)  # captured, not propagated

    def test_worker_captures_keyboard_interrupt(self):
        def die():
            raise KeyboardInterrupt()

        completed, payload = run_in_worker(die, timeout=5.0)
        self.assertFalse(completed)
        self.assertIsInstance(payload, KeyboardInterrupt)

    def test_worker_timeout_abandons_stuck_thread(self):
        started = time.monotonic()
        completed, payload = run_in_worker(time.sleep, 60, timeout=0.2)
        elapsed = time.monotonic() - started
        self.assertFalse(completed)
        self.assertIsInstance(payload, TimeoutError)
        self.assertLess(elapsed, 10)  # caller unblocked; daemon thread abandoned

    def test_command_isolated_ok(self):
        result = run_command_isolated("echo hello", "", timeout=5.0)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.returncode, 0)
        self.assertIn("hello", result.stdout)


class TestCrashingHookCannotKillLoop(IsolationBase):
    """A crashing hook: loop continues, plugin marked failed, failure logged."""

    def test_segfaulting_hook_is_fail_open_and_logged(self):
        crash_cmd = "python3 -c \"import os, signal; os.kill(os.getpid(), signal.SIGSEGV)\""
        capture_py = _write_script(self.tmpdir, "capture.py", CAPTURE_SCRIPT)
        capture_file = self.tmpdir / "capture.json"
        # Crashing hook first, then a healthy one: the turn must continue.
        self._write_config({
            "PreToolUse": [
                {"command": crash_cmd},
                {"command": f"python3 {capture_py} {capture_file}"},
            ]
        })
        result = run_hooks("PreToolUse", tool_name="Bash", tool_input={"command": "ls"})
        # Fail-open: not blocked, decision allow.
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")
        # Loop state not poisoned: the second hook ran normally afterwards.
        self.assertTrue(capture_file.exists(), "subsequent hook never ran: loop poisoned")
        payload = json.loads(capture_file.read_text())
        self.assertEqual(payload["tool"], "Bash")
        # Failure recorded in the audit log with the crashing command named.
        audit = self._read_audit()
        failed = [e for e in audit if e["decision"] == "hook-failed"]
        self.assertTrue(failed, f"no hook-failed audit entry: {audit}")
        self.assertIn(crash_cmd, failed[0]["command"])
        self.assertIn("139", failed[0]["message"] or "")  # SIGSEGV exit code

    def test_nonzero_exit_hook_is_fail_open_and_logged(self):
        self._write_config({
            "PostToolUse": [{"command": "python3 -c \"import sys; sys.exit(42)\""}]
        })
        result = run_hooks("PostToolUse", tool_name="Bash", tool_input={})
        self.assertFalse(result["blocked"])
        audit = self._read_audit()
        failed = [e for e in audit if e["decision"] == "hook-failed"]
        self.assertTrue(failed)
        self.assertIn("42", failed[0]["message"] or "")

    def test_unlaunchable_hook_is_fail_open_and_logged(self):
        self._write_config({
            "PreToolUse": [{"command": "/nonexistent-dir-xyz/no-such-binary --flag"}]
        })
        result = run_hooks("PreToolUse", tool_name="Bash", tool_input={})
        self.assertFalse(result["blocked"])
        audit = self._read_audit()
        failed = [e for e in audit if e["decision"] == "hook-failed"]
        self.assertTrue(failed, f"no hook-failed audit entry: {audit}")

    def test_hook_timeout_kills_plugin_turn_continues(self):
        capture_py = _write_script(self.tmpdir, "capture.py", CAPTURE_SCRIPT)
        capture_file = self.tmpdir / "capture.json"
        self._write_config({
            "PreToolUse": [
                {"command": "python3 -c \"import time; time.sleep(30)\""},
                {"command": f"python3 {capture_py} {capture_file}"},
            ]
        })
        started = time.monotonic()
        result = run_hooks(
            "PreToolUse", tool_name="Bash", tool_input={}, timeout=0.3
        )
        elapsed = time.monotonic() - started
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")
        self.assertLess(elapsed, 20, "hook timeout blocked the loop too long")
        # The timed-out plugin was skipped; the turn continued to the next hook.
        self.assertTrue(capture_file.exists(), "turn did not continue after timeout")
        audit = self._read_audit()
        timeouts = [e for e in audit if e["decision"] == "hook-timeout"]
        self.assertTrue(timeouts, f"no hook-timeout audit entry: {audit}")
        self.assertIn("sleep(30)", timeouts[0]["command"])


class TestNamedLoadErrors(unittest.TestCase):
    """Plugin load failures are named, actionable, and never silent."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_script_names_exact_path(self):
        with self.assertRaises(PluginBootstrapMissingError) as ctx:
            validate_hook_entry(
                "PreToolUse", {"command": "python3 /nonexistent/hook-xyz.py"}, 0
            )
        err = ctx.exception
        self.assertIn("/nonexistent/hook-xyz.py", err.missing)
        self.assertTrue(err.fix)
        self.assertIn("PluginBootstrapMissingError", str(err))

    def test_missing_interpreter_names_it(self):
        with self.assertRaises(PluginBootstrapMissingError) as ctx:
            validate_hook_entry(
                "PreToolUse", {"command": "no-such-interp-xyz /tmp/hook.py"}, 0
            )
        self.assertIn("no-such-interp-xyz", ctx.exception.missing)

    def test_entry_without_command_is_named(self):
        with self.assertRaises(PluginConfigError) as ctx:
            validate_hook_entry("PreToolUse", {"foo": "bar"}, 0)
        self.assertIn("command", ctx.exception.missing)
        self.assertIn("PluginConfigError", str(ctx.exception))

    def test_bad_entry_shape_is_named(self):
        with self.assertRaises(PluginConfigError):
            validate_hook_entry("PreToolUse", 42, 0)

    def test_non_dict_config_is_named(self):
        with self.assertRaises(PluginConfigError):
            validate_hook_config(["not", "a", "dict"])

    def test_valid_config_passes(self):
        hook = self.tmpdir / "hook.py"
        hook.write_text("# hook\n")
        validate_hook_config({"PreToolUse": [{"command": f"python3 {hook}"}]})
        validate_hook_config({})  # empty config: no hooks, not a failure

    def test_plugin_load_warnings_collects_all_problems(self):
        path = self.tmpdir / "hooks.json"
        path.write_text(json.dumps({
            "PreToolUse": [{"command": "python3 /nonexistent/a.py"}],
            "PostToolUse": [{"foo": "bar"}],
        }))
        warnings = plugin_load_warnings(path)
        self.assertEqual(len(warnings), 2)
        self.assertTrue(all(isinstance(w, PluginLoadError) for w in warnings))
        self.assertIsInstance(warnings[0], PluginBootstrapMissingError)
        self.assertIsInstance(warnings[1], PluginConfigError)

    def test_plugin_load_warnings_absent_file_is_not_a_failure(self):
        self.assertEqual(plugin_load_warnings(self.tmpdir / "nope.json"), [])

    def test_plugin_load_warnings_invalid_json_is_named(self):
        path = self.tmpdir / "hooks.json"
        path.write_text("{ not valid json !!!")
        warnings = plugin_load_warnings(path)
        self.assertEqual(len(warnings), 1)
        err = warnings[0]
        self.assertIsInstance(err, PluginConfigError)
        self.assertIn("valid JSON", err.missing)
        self.assertIn("PluginConfigError", str(err))


class TestDoctorSurfacesNamedErrors(unittest.TestCase):
    """Missing bootstrap/config surfaces the exact cause in /doctor."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _clawd(self) -> Path:
        clawd = self.home / ".clawd"
        clawd.mkdir(exist_ok=True)
        return clawd

    def test_missing_bootstrap_named_in_doctor(self):
        from src.command_system.builtins import doctor_check_hook_scripts

        clawd = self._clawd()
        (clawd / "hooks.json").write_text(json.dumps({
            "PreToolUse": [{"command": "python3 /nonexistent/hook-xyz.py"}],
        }))
        result = doctor_check_hook_scripts(home=self.home)
        self.assertFalse(result["ok"])
        self.assertIn("PluginBootstrapMissingError", result["detail"])
        self.assertIn("/nonexistent/hook-xyz.py", result["detail"])
        self.assertIsNotNone(result["fix"])

    def test_invalid_json_named_in_doctor(self):
        from src.command_system.builtins import doctor_check_hook_scripts

        clawd = self._clawd()
        (clawd / "hooks.json").write_text("{ not valid json !!!")
        result = doctor_check_hook_scripts(home=self.home)
        self.assertFalse(result["ok"], "invalid hooks.json must not be a silent skip")
        self.assertIn("PluginConfigError", result["detail"])
        self.assertIn("valid JSON", result["detail"])
        self.assertIsNotNone(result["fix"])

    def test_entry_without_command_named_in_doctor(self):
        from src.command_system.builtins import doctor_check_hook_scripts

        clawd = self._clawd()
        (clawd / "hooks.json").write_text(json.dumps({
            "PostToolUse": [{"foo": "bar"}],
        }))
        result = doctor_check_hook_scripts(home=self.home)
        self.assertFalse(result["ok"])
        self.assertIn("PluginConfigError", result["detail"])

    def test_absent_hooks_json_still_ok(self):
        from src.command_system.builtins import doctor_check_hook_scripts

        result = doctor_check_hook_scripts(home=self.home)
        self.assertTrue(result["ok"])

    def test_problems_on_any_event_surface(self):
        from src.command_system.builtins import doctor_check_hook_scripts

        clawd = self._clawd()
        hook = clawd / "hook.py"
        hook.write_text("# hook\n")
        (clawd / "hooks.json").write_text(json.dumps({
            # previously only PreToolUse/PostToolUse were checked; a broken
            # BeforeModel hook must not slip through silently
            "BeforeModel": [{"command": f"python3 {clawd}/missing-before-model.py"}],
            "PreToolUse": [{"command": f"python3 {hook}"}],
        }))
        result = doctor_check_hook_scripts(home=self.home)
        self.assertFalse(result["ok"])
        self.assertIn("missing-before-model.py", result["detail"])


class TestSessionStartWarnings(unittest.TestCase):
    """Session start warns loudly (stderr) with the named error, fail-open."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_session_create_warns_on_broken_hooks_config(self):
        from src.agent.session import Session

        hooks_json = self.tmpdir / "hooks.json"
        hooks_json.write_text(json.dumps({
            "PreToolUse": [{"command": "python3 /nonexistent/session-warn-hook.py"}],
        }))
        with patch(
            "src.tool_system.hooks.isolation._default_config_path",
            return_value=hooks_json,
        ), patch(
            "src.tool_system.hooks.runner._default_config_path",
            return_value=hooks_json,
        ):
            import io
            from contextlib import redirect_stderr

            buf = io.StringIO()
            with redirect_stderr(buf):
                session = Session.create("anthropic", "test-model")
            warned = buf.getvalue()
        # Session was still created (fail-open) ...
        self.assertIsNotNone(session.session_id)
        # ... but the load failure was loud, naming the exact missing piece.
        self.assertIn("PluginBootstrapMissingError", warned)
        self.assertIn("/nonexistent/session-warn-hook.py", warned)

    def test_session_create_quiet_when_hooks_valid(self):
        from src.agent.session import Session

        hook = self.tmpdir / "hook.py"
        hook.write_text("import json,sys; json.load(sys.stdin)\n")
        hooks_json = self.tmpdir / "hooks.json"
        hooks_json.write_text(json.dumps({
            "SessionStart": [{"command": f"python3 {hook}"}],
        }))
        with patch(
            "src.tool_system.hooks.isolation._default_config_path",
            return_value=hooks_json,
        ), patch(
            "src.tool_system.hooks.runner._default_config_path",
            return_value=hooks_json,
        ):
            import io
            from contextlib import redirect_stderr

            buf = io.StringIO()
            with redirect_stderr(buf):
                Session.create("anthropic", "test-model")
            self.assertNotIn("PluginBootstrapMissingError", buf.getvalue())


class TestRuntimeWarnsNamedOnBadConfig(IsolationBase):
    """run_hooks stays fail-open on bad config but warns with the named error."""

    def test_invalid_json_warns_named_but_allows(self):
        self.hooks_json.write_text("{ not valid json !!!")
        with patch("src.tool_system.hooks.runner._warn") as mock_warn:
            result = run_hooks("PreToolUse", tool_name="Bash", tool_input={})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")
        warned_text = " ".join(str(c.args[0]) for c in mock_warn.call_args_list)
        self.assertIn("PluginConfigError", warned_text)


if __name__ == "__main__":
    unittest.main()
