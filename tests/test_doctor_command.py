"""Tests for Q7: the /doctor diagnostics command.

Covers registration (LocalCommand + execute_command_sync + help text) and
each individual check against a fake HOME directory.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.command_system.builtins import (
    DOCTOR_COMMAND,
    doctor_check_bash,
    doctor_check_clawd_dir_writable,
    doctor_check_config_valid,
    doctor_check_hook_scripts,
    doctor_check_mcp_servers,
    doctor_check_provider_keys,
    doctor_check_sessions_dir,
    doctor_run_checks,
    execute_command_sync,
    get_builtin_commands,
    help_command_call,
    register_builtin_commands,
)
from src.command_system.engine import create_command_context


def _healthy_home(tmp: Path) -> Path:
    clawd = tmp / ".clawd"
    clawd.mkdir()
    (clawd / "config.json").write_text(json.dumps({
        "default_provider": "anthropic",
        "providers": {"anthropic": {"api_key": "a2V5"}},
    }))
    (clawd / "sessions").mkdir()
    (clawd / "mcp.json").write_text(json.dumps({
        "mcpServers": {"s1": {"command": "python3", "args": []}},
    }))
    hook = tmp / "hook.py"
    hook.write_text("# hook\n")
    (clawd / "hooks.json").write_text(json.dumps({
        "PostToolUse": [{"command": f"python3 {hook}"}],
    }))
    return tmp


class TestDoctorRegistration(unittest.TestCase):
    def test_registered_as_builtin(self):
        self.assertIn(DOCTOR_COMMAND, get_builtin_commands())
        self.assertEqual(DOCTOR_COMMAND.name, "doctor")

    def test_sync_execution(self):
        with tempfile.TemporaryDirectory() as td:
            ctx = create_command_context(workspace_root=td)
            ok, value, err = execute_command_sync("doctor", "", ctx)
        self.assertTrue(ok, err)
        self.assertIsNone(err)
        self.assertIn("diagnostics", value)
        self.assertIn("checks passed", value)

    def test_help_lists_doctor(self):
        register_builtin_commands()
        with tempfile.TemporaryDirectory() as td:
            ctx = create_command_context(workspace_root=td)
            value = help_command_call("", ctx).value
        self.assertIn("/doctor", value)


class TestDoctorChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_checks_pass_on_healthy_home(self):
        _healthy_home(self.home)
        results = doctor_run_checks(home=self.home)
        self.assertEqual(len(results), 7)
        failures = [r for r in results if not r["ok"]]
        self.assertEqual(failures, [], f"unexpected failures: {failures}")

    def test_missing_config_fails_with_fix(self):
        r = doctor_check_config_valid(home=self.home)
        self.assertFalse(r["ok"])
        self.assertIsNotNone(r["fix"])

    def test_corrupt_config_fails_with_fix(self):
        clawd = self.home / ".clawd"
        clawd.mkdir()
        (clawd / "config.json").write_text("{not json")
        r = doctor_check_config_valid(home=self.home)
        self.assertFalse(r["ok"])
        self.assertIn("JSON", r["detail"])

    def test_missing_provider_key_fails_with_fix(self):
        clawd = self.home / ".clawd"
        clawd.mkdir()
        (clawd / "config.json").write_text(json.dumps({
            "default_provider": "anthropic",
            "providers": {"anthropic": {"api_key": ""}},
        }))
        with patch.dict("os.environ", {}, clear=False):
            # ensure no stray env key leaks in (test envs rarely set these)
            import os
            for var in ("ANTHROPIC_API_KEY",):
                os.environ.pop(var, None)
            r = doctor_check_provider_keys(home=self.home)
        self.assertFalse(r["ok"])
        self.assertIn("API key", r["fix"])

    def test_provider_key_via_env_passes(self):
        _healthy_home(self.home)
        (self.home / ".clawd" / "config.json").write_text(json.dumps({
            "default_provider": "openai",
            "providers": {"openai": {"api_key": ""}},
        }))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            r = doctor_check_provider_keys(home=self.home)
        self.assertTrue(r["ok"])

    def test_bash_missing_suggests_git_bash(self):
        with patch("shutil.which", return_value=None):
            r = doctor_check_bash()
        self.assertFalse(r["ok"])
        self.assertIn("Git Bash", r["fix"])
        self.assertIn("bash -lc", r["fix"])

    def test_bash_present_passes(self):
        with patch("shutil.which", return_value="/usr/bin/bash"):
            r = doctor_check_bash()
        self.assertTrue(r["ok"])

    def test_unreachable_mcp_server_fails(self):
        clawd = self.home / ".clawd"
        clawd.mkdir()
        (clawd / "mcp.json").write_text(json.dumps({
            "mcpServers": {"bad": {"command": "definitely-not-a-real-binary-xyz"}},
        }))
        r = doctor_check_mcp_servers(home=self.home)
        self.assertFalse(r["ok"])
        self.assertIn("bad", r["detail"])
        self.assertIsNotNone(r["fix"])

    def test_missing_mcp_config_is_ok(self):
        r = doctor_check_mcp_servers(home=self.home)
        self.assertTrue(r["ok"])

    def test_missing_hook_script_fails(self):
        clawd = self.home / ".clawd"
        clawd.mkdir()
        (clawd / "hooks.json").write_text(json.dumps({
            "PostToolUse": [{"command": "python3 /nonexistent/hook-xyz.py"}],
        }))
        r = doctor_check_hook_scripts(home=self.home)
        self.assertFalse(r["ok"])
        self.assertIsNotNone(r["fix"])

    def test_missing_hooks_config_is_ok(self):
        r = doctor_check_hook_scripts(home=self.home)
        self.assertTrue(r["ok"])

    def test_missing_sessions_dir_fails_with_fix(self):
        r = doctor_check_sessions_dir(home=self.home)
        self.assertFalse(r["ok"])
        self.assertIn("mkdir", r["fix"])

    def test_unwritable_clawd_dir_fails(self):
        with patch("pathlib.Path.mkdir", side_effect=OSError("denied")):
            r = doctor_check_clawd_dir_writable(home=self.home)
        self.assertFalse(r["ok"])
        self.assertIsNotNone(r["fix"])


if __name__ == "__main__":
    unittest.main()
