"""Tests for background Bash jobs (run_in_background / poll / kill / forget)
and the streaming-default change in the REPL / CLI.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from rich.console import Console

from src.repl.tool_cards import ToolCardRenderer
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolInputError, ToolPermissionError
from src.tool_system.protocol import ToolCall
from src.tool_system.tools import bash as bash_module
from src.tool_system.tools.bash import BashTool


# BashTool shells out to `bash -lc`; stock Windows has no bash. Execution
# tests skip there (the tool itself returns a clean error instead).
_HAVE_BASH = shutil.which("bash") is not None
_needs_bash = unittest.skipUnless(_HAVE_BASH, "bash not available on this system")


def _wait_done(tool: BashTool, ctx: ToolContext, job_id: str, timeout: float = 10.0):
    """Poll until the job finishes; return the final poll result."""
    deadline = time.time() + timeout
    while True:
        res = tool.run({"action": "poll", "job_id": job_id}, ctx)
        if not res.output["running"]:
            return res
        if time.time() > deadline:
            raise AssertionError(f"job {job_id} did not finish in {timeout}s")
        time.sleep(0.05)


class TestBackgroundBash(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        self.tool = BashTool()

    def tearDown(self) -> None:
        # Never leak background processes from tests.
        for job_id in list(bash_module._BG_JOBS):
            try:
                self.tool.run({"action": "forget", "job_id": job_id}, self.ctx)
            except Exception:
                pass
        self.tmp.cleanup()

    @_needs_bash
    def test_start_returns_job_immediately(self) -> None:
        res = self.tool.run(
            {"command": "sleep 30", "run_in_background": True}, self.ctx
        )
        self.assertFalse(res.is_error)
        out = res.output
        self.assertTrue(out["job_id"].startswith("job_"))
        self.assertTrue(out["running"])
        self.assertTrue(out["started"])

    @_needs_bash
    def test_poll_captures_output_when_done(self) -> None:
        res = self.tool.run(
            {"command": "echo hello-bg && echo err-line 1>&2",
             "run_in_background": True},
            self.ctx,
        )
        job_id = res.output["job_id"]
        final = _wait_done(self.tool, self.ctx, job_id)
        self.assertFalse(final.is_error)
        self.assertIn("hello-bg", final.output["stdout"])
        self.assertIn("err-line", final.output["stderr"])
        self.assertEqual(final.output["exit_code"], 0)

    @_needs_bash
    def test_poll_while_running_has_no_exit_code(self) -> None:
        res = self.tool.run(
            {"command": "sleep 30", "run_in_background": True}, self.ctx
        )
        job_id = res.output["job_id"]
        polled = self.tool.run({"action": "poll", "job_id": job_id}, self.ctx)
        self.assertTrue(polled.output["running"])
        self.assertNotIn("exit_code", polled.output)

    @_needs_bash
    def test_poll_nonzero_exit_is_error(self) -> None:
        res = self.tool.run(
            {"command": "exit 3", "run_in_background": True}, self.ctx
        )
        final = _wait_done(self.tool, self.ctx, res.output["job_id"])
        self.assertTrue(final.is_error)
        self.assertEqual(final.output["exit_code"], 3)

    @_needs_bash
    def test_kill_stops_running_job(self) -> None:
        res = self.tool.run(
            {"command": "sleep 60", "run_in_background": True}, self.ctx
        )
        job_id = res.output["job_id"]
        killed = self.tool.run({"action": "kill", "job_id": job_id}, self.ctx)
        self.assertTrue(killed.output["killed"])
        self.assertTrue(killed.output["was_running"])
        polled = self.tool.run({"action": "poll", "job_id": job_id}, self.ctx)
        self.assertFalse(polled.output["running"])

    @_needs_bash
    def test_forget_removes_job(self) -> None:
        res = self.tool.run(
            {"command": "sleep 60", "run_in_background": True}, self.ctx
        )
        job_id = res.output["job_id"]
        forgotten = self.tool.run({"action": "forget", "job_id": job_id}, self.ctx)
        self.assertTrue(forgotten.output["forgotten"])
        with self.assertRaises(ToolInputError):
            self.tool.run({"action": "poll", "job_id": job_id}, self.ctx)
        with self.assertRaises(ToolInputError):
            self.tool.run({"action": "forget", "job_id": job_id}, self.ctx)

    def test_poll_unknown_job_raises(self) -> None:
        with self.assertRaises(ToolInputError):
            self.tool.run({"action": "poll", "job_id": "job_nope1234"}, self.ctx)

    def test_kill_unknown_job_raises(self) -> None:
        with self.assertRaises(ToolInputError):
            self.tool.run({"action": "kill", "job_id": "job_nope1234"}, self.ctx)

    def test_invalid_action_raises(self) -> None:
        with self.assertRaises(ToolInputError):
            self.tool.run({"action": "explode", "job_id": "job_x"}, self.ctx)

    def test_dangerous_command_blocked_in_background(self) -> None:
        with self.assertRaises(ToolPermissionError):
            self.tool.run(
                {"command": "sudo echo nope", "run_in_background": True}, self.ctx
            )

    def test_background_requires_command(self) -> None:
        with self.assertRaises(ToolInputError):
            self.tool.run({"run_in_background": True}, self.ctx)

    def test_background_permissions_match_foreground(self) -> None:
        """Same check_permissions path for foreground and background calls.

        BashTool defines no check_permissions of its own; the registry falls
        back to PermissionResult.allow() for both call shapes. Replicates the
        registry's fallback expression (registry.py) so the test breaks if
        the two paths ever diverge.
        """
        from src.tool_system.permission_handler import (
            PermissionBehavior,
            PermissionResult,
        )

        def _registry_check(tool, tool_input, context):
            return (
                tool.check_permissions(tool_input, context)
                if hasattr(tool, "check_permissions")
                else PermissionResult.allow()
            )

        fg = _registry_check(self.tool, {"command": "echo hi"}, self.ctx)
        bg = _registry_check(
            self.tool, {"command": "echo hi", "run_in_background": True}, self.ctx
        )
        self.assertEqual(fg.behavior, PermissionBehavior.ALLOW)
        self.assertEqual(bg.behavior, PermissionBehavior.ALLOW)

    def test_plan_mode_blocks_background_bash(self) -> None:
        """Plan-mode gate applies to background Bash exactly like foreground."""
        registry = build_default_registry()
        self.ctx.plan_mode = True
        res = registry.dispatch(
            ToolCall(name="Bash",
                     input={"command": "echo hi", "run_in_background": True}),
            self.ctx,
        )
        self.assertTrue(res.is_error)
        self.assertIn("plan mode", str(res.output.get("error", "")))
        # No job should have been started.
        self.assertEqual(len(bash_module._BG_JOBS), 0)

    @_needs_bash
    def test_foreground_still_works(self) -> None:
        res = self.tool.run({"command": "echo fg-ok"}, self.ctx)
        self.assertFalse(res.is_error)
        self.assertEqual(res.output["exit_code"], 0)
        self.assertIn("fg-ok", res.output["stdout"])


class TestBackgroundBashCards(unittest.TestCase):
    def setUp(self) -> None:
        self.renderer = ToolCardRenderer(Console(), shorten=lambda s: s)

    @staticmethod
    def _unwrap(rendered):
        """render_* now returns a list; background payloads are [Panel]."""
        if isinstance(rendered, list):
            assert len(rendered) == 1, rendered
            return rendered[0]
        return rendered

    def _panel_text(self, rendered) -> str:
        panel = self._unwrap(rendered)
        return panel.title + " | " + panel.subtitle + " | " + str(panel.renderable)

    def _line_text(self, rendered) -> str:
        from rich.console import Console as _Console

        buf = io.StringIO()
        _Console(file=buf, width=100, legacy_windows=False).print(*rendered)
        return buf.getvalue()

    def test_render_use_background_start_variant(self) -> None:
        panel = self.renderer.render_use(
            "Bash", {"command": "sleep 60", "run_in_background": True}
        )
        text = self._panel_text(panel)
        self.assertIn("background job", text)

    def test_render_use_foreground_is_quiet_line(self) -> None:
        rendered = self.renderer.render_use("Bash", {"command": "echo hi"})
        text = self._line_text(rendered)
        self.assertNotIn("background job", text)
        # One dim line, no bordered panel.
        self.assertIn("Bash", text)
        self.assertIn("echo hi", text)
        self.assertNotIn("╭", text)

    def test_render_use_poll_action_not_start_variant(self) -> None:
        rendered = self.renderer.render_use(
            "Bash", {"action": "poll", "job_id": "job_x"}
        )
        text = self._line_text(rendered)
        self.assertNotIn("background job · starting", text)
        self.assertIn("Bash", text)

    def test_render_result_background_start(self) -> None:
        panel = self.renderer.render_result(
            "Bash",
            {"job_id": "job_abc123", "started": True, "running": True,
             "command": "sleep 60"},
            is_error=False,
        )
        text = self._panel_text(panel)
        self.assertIn("job_abc123", text)
        self.assertIn("background", text)

    def test_render_result_poll_running(self) -> None:
        panel = self.renderer.render_result(
            "Bash",
            {"job_id": "job_abc123", "running": True, "stdout": "partial\n"},
            is_error=False,
        )
        self.assertIn("running", self._panel_text(panel))

    def test_render_result_poll_done(self) -> None:
        panel = self.renderer.render_result(
            "Bash",
            {"job_id": "job_abc123", "running": False, "exit_code": 0,
             "stdout": "done\n"},
            is_error=False,
        )
        text = self._panel_text(panel)
        self.assertIn("done", text)
        self.assertIn("exit 0", text)

    def test_render_result_foreground_is_quiet_line(self) -> None:
        rendered = self.renderer.render_result(
            "Bash",
            {"cwd": "/tmp", "exit_code": 0, "stdout": "hi\n", "stderr": ""},
            is_error=False,
        )
        text = self._line_text(rendered)
        self.assertNotIn("background", text)
        self.assertIn("✓", text)
        self.assertIn("Bash", text)
        self.assertNotIn("╭", text)


class TestStreamDefault(unittest.TestCase):
    """Streaming is on by default in the interactive REPL; script mode stays
    non-streamed unless explicitly asked."""

    def _make_repl(self, **kwargs):
        import json as _json

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        config_dir = Path(tmp.name) / "clawd"
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / "config.json", "w") as f:
            _json.dump(
                {"providers": {"glm": {"api_key": "test_api_key_12345678",
                                        "base_url": "https://x",
                                        "default_model": "glm-4.5"}}},
                f,
            )
        from src.repl.core import ClawdREPL

        with patch("src.config.get_config_path",
                   return_value=config_dir / "config.json"):
            with patch("src.repl.core.Session.create"):
                with patch("src.repl.core.get_provider_class") as mock_cls:
                    mock_provider = Mock()
                    mock_provider.model = "glm-4.5"
                    mock_cls.return_value = mock_provider
                    return ClawdREPL(provider_name="glm", **kwargs)

    def test_stream_defaults_to_true(self) -> None:
        repl = self._make_repl()
        self.assertTrue(repl.stream)

    def test_stream_can_still_be_disabled_explicitly(self) -> None:
        repl = self._make_repl(stream=False)
        self.assertFalse(repl.stream)

    def test_stream_toggle_still_works(self) -> None:
        repl = self._make_repl()
        repl.handle_command("/stream off")
        self.assertFalse(repl.stream)
        repl.handle_command("/stream toggle")
        self.assertTrue(repl.stream)

    def test_print_mode_defaults_to_non_streamed(self) -> None:
        import inspect

        from src.repl.print_mode import run_print

        sig = inspect.signature(run_print)
        self.assertEqual(sig.parameters["stream"].default, False)


if __name__ == "__main__":
    unittest.main()
