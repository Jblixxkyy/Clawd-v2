"""Tests for the permission system."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.tool_system.context import ToolContext
from src.tool_system.permission_handler import (
    PermissionBehavior,
    PermissionResult,
    apply_permission_mode,
)
from src.tool_system.permissions import PermissionMode, ToolPermissionContext
from src.tool_system.protocol import ToolCall
from src.tool_system.registry import ToolRegistry
from src.tool_system.tools.write import FileWriteTool
from src.tool_system.tools.edit import FileEditTool


class TestPermissionResult(unittest.TestCase):
    def test_allow_returns_correct_behavior(self) -> None:
        result = PermissionResult.allow()
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)
        self.assertIsNone(result.message)
        self.assertIsNone(result.updated_input)

    def test_allow_with_updated_input(self) -> None:
        updated = {"key": "value"}
        result = PermissionResult.allow(updated_input=updated)
        self.assertEqual(result.updated_input, updated)

    def test_deny_returns_correct_behavior(self) -> None:
        result = PermissionResult.deny("test message")
        self.assertEqual(result.behavior, PermissionBehavior.DENY)
        self.assertEqual(result.message, "test message")

    def test_ask_returns_correct_behavior(self) -> None:
        result = PermissionResult.ask("test message", "try this")
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertEqual(result.message, "test message")
        self.assertEqual(result.suggestion, "try this")


class TestWriteToolPermissions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        # Default context has allow_docs=False
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_check_permissions_allow_for_regular_file(self) -> None:
        """Regular files should be allowed without any permission check."""
        tool = FileWriteTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.txt"), "content": "hello"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)

    def test_check_permissions_ask_for_md_file_when_docs_disallowed(self) -> None:
        """MD files should trigger 'ask' when allow_docs is False."""
        tool = FileWriteTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.md"), "content": "hello"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertIn("allow_docs", result.message.lower())

    def test_check_permissions_allow_for_md_file_when_docs_allowed(self) -> None:
        """MD files should be allowed when allow_docs is True."""
        self.ctx.permission_context.allow_docs = True
        tool = FileWriteTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.md"), "content": "hello"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)

    def test_check_permissions_allow_for_markdown_file(self) -> None:
        """Markdown files should also trigger 'ask' when allow_docs is False."""
        tool = FileWriteTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.markdown"), "content": "hello"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ASK)


class TestEditToolPermissions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        # Default context has allow_docs=False
        self.ctx = ToolContext(workspace_root=self.root)

        # Create a test file to edit
        self.test_file = self.root / "test.md"
        self.test_file.write_text("original content", encoding="utf-8")
        self.ctx.mark_file_read(self.test_file)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_check_permissions_allow_for_regular_file(self) -> None:
        """Regular files should be allowed without any permission check."""
        tool = FileEditTool()
        result = tool.check_permissions(
            {"file_path": str(self.root / "test.txt"), "old_string": "a", "new_string": "b"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)

    def test_check_permissions_ask_for_md_file_when_docs_disallowed(self) -> None:
        """MD files should trigger 'ask' when allow_docs is False."""
        tool = FileEditTool()
        result = tool.check_permissions(
            {"file_path": str(self.test_file), "old_string": "original", "new_string": "modified"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ASK)
        self.assertIn("allow_docs", result.message.lower())

    def test_check_permissions_allow_for_md_file_when_docs_allowed(self) -> None:
        """MD files should be allowed when allow_docs is True."""
        self.ctx.permission_context.allow_docs = True
        tool = FileEditTool()
        result = tool.check_permissions(
            {"file_path": str(self.test_file), "old_string": "original", "new_string": "modified"},
            self.ctx,
        )
        self.assertEqual(result.behavior, PermissionBehavior.ALLOW)


class TestToolRegistryDispatchPermissions(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        self.registry = ToolRegistry([FileWriteTool()])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_dispatch_allows_regular_file(self) -> None:
        """Regular files should work without permission handler."""
        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.txt"), "content": "hello"}),
            self.ctx,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.output.get("type"), "create")

    def test_dispatch_denies_md_file_without_handler(self) -> None:
        """MD files should be denied when no permission handler is set."""
        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.md"), "content": "hello"}),
            self.ctx,
        )
        self.assertTrue(result.is_error)
        error_msg = result.output.get("error", "").lower()
        self.assertTrue(
            "permission" in error_msg or "allow_docs" in error_msg or "blocked" in error_msg,
            f"Expected 'permission', 'allow_docs', or 'blocked' in error, got: {error_msg}",
        )

    def test_dispatch_calls_permission_handler_for_ask(self) -> None:
        """When behavior is 'ask', permission_handler should be called."""
        call_count = 0
        captured_args = None

        def mock_handler(tool_name: str, message: str, suggestion: str | None):
            nonlocal call_count, captured_args
            call_count += 1
            captured_args = (tool_name, message, suggestion)
            return True, False  # Allow

        self.ctx.permission_handler = mock_handler

        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.md"), "content": "hello"}),
            self.ctx,
        )

        self.assertEqual(call_count, 1)
        self.assertEqual(captured_args[0], "Write")
        self.assertIn("allow_docs", captured_args[1].lower())
        self.assertFalse(result.is_error)

    def test_dispatch_respects_handler_deny(self) -> None:
        """When permission handler denies, tool should be blocked."""
        def mock_handler(tool_name: str, message: str, suggestion: str | None):
            return False, False  # Deny

        self.ctx.permission_handler = mock_handler

        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.md"), "content": "hello"}),
            self.ctx,
        )

        self.assertTrue(result.is_error)
        self.assertIn("denied", result.output.get("error", "").lower())

    def test_dispatch_allows_after_handler_enables_setting(self) -> None:
        """When handler enables setting, tool should proceed."""
        def mock_handler(tool_name: str, message: str, suggestion: str | None):
            # Enable allow_docs
            self.ctx.permission_context.allow_docs = True
            return True, False

        self.ctx.permission_handler = mock_handler

        result = self.registry.dispatch(
            ToolCall(name="Write", input={"file_path": str(self.root / "test.md"), "content": "hello"}),
            self.ctx,
        )

        self.assertFalse(result.is_error)
        self.assertEqual(result.output.get("type"), "create")


class TestPermissionContext(unittest.TestCase):
    def test_default_allow_docs_is_false(self) -> None:
        """By default, allow_docs should be False."""
        pc = ToolPermissionContext()
        self.assertFalse(pc.allow_docs)

    def test_allow_docs_can_be_set_true(self) -> None:
        """allow_docs can be set to True."""
        pc = ToolPermissionContext(allow_docs=True)
        self.assertTrue(pc.allow_docs)

    def test_permission_context_is_mutable(self) -> None:
        """ToolPermissionContext should be mutable to allow runtime changes."""
        pc = ToolPermissionContext()
        self.assertFalse(pc.allow_docs)
        pc.allow_docs = True
        self.assertTrue(pc.allow_docs)


class TestPermissionModeEnum(unittest.TestCase):
    def test_values(self) -> None:
        self.assertEqual(PermissionMode.DEFAULT.value, "default")
        self.assertEqual(PermissionMode.ACCEPT_EDITS.value, "acceptEdits")
        self.assertEqual(PermissionMode.DONT_ASK.value, "dontAsk")
        self.assertEqual(PermissionMode.BYPASS_PERMISSIONS.value, "bypassPermissions")

    def test_from_string_canonical(self) -> None:
        self.assertIs(PermissionMode.from_string("default"), PermissionMode.DEFAULT)
        self.assertIs(PermissionMode.from_string("acceptEdits"), PermissionMode.ACCEPT_EDITS)
        self.assertIs(PermissionMode.from_string("dontAsk"), PermissionMode.DONT_ASK)
        self.assertIs(
            PermissionMode.from_string("bypassPermissions"),
            PermissionMode.BYPASS_PERMISSIONS,
        )

    def test_from_string_case_insensitive(self) -> None:
        self.assertIs(PermissionMode.from_string("acceptedits"), PermissionMode.ACCEPT_EDITS)
        self.assertIs(PermissionMode.from_string("DONTASK"), PermissionMode.DONT_ASK)

    def test_from_string_passthrough(self) -> None:
        self.assertIs(
            PermissionMode.from_string(PermissionMode.DEFAULT), PermissionMode.DEFAULT
        )

    def test_from_string_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            PermissionMode.from_string("turbo")


class TestApplyPermissionMode(unittest.TestCase):
    def setUp(self) -> None:
        self.ask = PermissionResult.ask("need approval")
        self.allow = PermissionResult.allow()
        self.deny = PermissionResult.deny("no")
        self.pc = ToolPermissionContext.from_iterables(deny_names=["DangerTool"])

    def test_default_leaves_result_unchanged(self) -> None:
        self.assertIs(
            apply_permission_mode("Bash", self.ask, PermissionMode.DEFAULT).behavior,
            PermissionBehavior.ASK,
        )
        self.assertIs(
            apply_permission_mode("Bash", self.allow, PermissionMode.DEFAULT).behavior,
            PermissionBehavior.ALLOW,
        )
        self.assertIs(
            apply_permission_mode("Bash", self.deny, PermissionMode.DEFAULT).behavior,
            PermissionBehavior.DENY,
        )

    def test_deny_rules_win_in_every_mode(self) -> None:
        for mode in PermissionMode:
            result = apply_permission_mode(
                "DangerTool", self.ask, mode, permission_context=self.pc
            )
            self.assertIs(
                result.behavior, PermissionBehavior.DENY, f"mode={mode}"
            )
            self.assertIn("blocked by permission context", result.message)

    def test_deny_rules_win_even_when_preapproved(self) -> None:
        result = apply_permission_mode(
            "DangerTool",
            self.ask,
            PermissionMode.DONT_ASK,
            permission_context=self.pc,
            always_allowed=True,
        )
        self.assertIs(result.behavior, PermissionBehavior.DENY)

    def test_accept_edits_auto_allows_edit_write(self) -> None:
        for name in ("Edit", "Write", "edit", "WRITE"):
            result = apply_permission_mode(name, self.ask, PermissionMode.ACCEPT_EDITS)
            self.assertIs(result.behavior, PermissionBehavior.ALLOW, f"tool={name}")

    def test_accept_edits_still_asks_for_bash_and_others(self) -> None:
        for name in ("Bash", "Read", "Grep"):
            result = apply_permission_mode(name, self.ask, PermissionMode.ACCEPT_EDITS)
            self.assertIs(result.behavior, PermissionBehavior.ASK, f"tool={name}")

    def test_dont_ask_denies_without_prompt(self) -> None:
        result = apply_permission_mode("Bash", self.ask, PermissionMode.DONT_ASK)
        self.assertIs(result.behavior, PermissionBehavior.DENY)
        self.assertIn("dontAsk", result.message)

    def test_dont_ask_allows_preapproved(self) -> None:
        result = apply_permission_mode(
            "Bash", self.ask, PermissionMode.DONT_ASK, always_allowed=True
        )
        self.assertIs(result.behavior, PermissionBehavior.ALLOW)

    def test_bypass_allows_any_ask(self) -> None:
        for name in ("Bash", "Edit", "Read"):
            result = apply_permission_mode(
                name, self.ask, PermissionMode.BYPASS_PERMISSIONS
            )
            self.assertIs(result.behavior, PermissionBehavior.ALLOW, f"tool={name}")

    def test_modes_never_upgrade_tool_level_deny(self) -> None:
        for mode in PermissionMode:
            result = apply_permission_mode("Bash", self.deny, mode)
            self.assertIs(result.behavior, PermissionBehavior.DENY, f"mode={mode}")

    def test_updated_input_preserved_through_conversion(self) -> None:
        ask = PermissionResult(
            behavior=PermissionBehavior.ASK,
            message="need approval",
            updated_input={"a": 1},
        )
        result = apply_permission_mode("Edit", ask, PermissionMode.ACCEPT_EDITS)
        self.assertIs(result.behavior, PermissionBehavior.ALLOW)
        self.assertEqual(result.updated_input, {"a": 1})


class TestRegistryPermissionModes(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        self.registry = ToolRegistry([FileWriteTool()])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _md_write(self):
        return ToolCall(
            name="Write",
            input={"file_path": str(self.root / "test.md"), "content": "hello"},
        )

    def test_accept_edits_auto_allows_write_without_handler(self) -> None:
        self.ctx.permission_mode = PermissionMode.ACCEPT_EDITS

        def boom(tool_name, message, suggestion):
            raise AssertionError("handler must not be called in acceptEdits mode")

        self.ctx.permission_handler = boom
        result = self.registry.dispatch(self._md_write(), self.ctx)
        self.assertFalse(result.is_error)
        self.assertTrue((self.root / "test.md").exists())

    def test_accept_edits_still_prompts_for_bash(self) -> None:
        # A Bash-like tool whose own check says ASK: acceptEdits must NOT
        # auto-allow it -- the prompt path (handler) still runs.
        from src.tool_system.protocol import ToolResult as _ToolResult
        from src.tool_system.registry import ToolSpec as _ToolSpec

        class _AskBashTool:
            def spec(self):
                return _ToolSpec(
                    name="Bash",
                    description="fake asking bash",
                    input_schema={"type": "object", "additionalProperties": True},
                )

            def check_permissions(self, tool_input, context):
                return PermissionResult.ask("bash needs approval")

            def run(self, tool_input, context):
                return _ToolResult(name="Bash", output={"ok": True})

        registry = ToolRegistry([_AskBashTool()])
        self.ctx.permission_mode = PermissionMode.ACCEPT_EDITS
        calls = []

        def handler(tool_name, message, suggestion):
            calls.append(tool_name)
            return False, False

        self.ctx.permission_handler = handler
        result = registry.dispatch(
            ToolCall(name="Bash", input={"command": "echo hi"}), self.ctx
        )
        self.assertTrue(result.is_error)
        self.assertEqual(calls, ["Bash"])

    def test_dont_ask_denies_without_prompt(self) -> None:
        self.ctx.permission_mode = PermissionMode.DONT_ASK

        def boom(tool_name, message, suggestion):
            raise AssertionError("handler must not be called in dontAsk mode")

        self.ctx.permission_handler = boom
        with unittest.mock.patch(
            "src.tool_system.registry.is_always_allowed", return_value=False
        ):
            result = self.registry.dispatch(self._md_write(), self.ctx)
        self.assertTrue(result.is_error)
        self.assertIn("dontAsk", result.output.get("error", ""))

    def test_bypass_skips_prompt_but_keeps_deny_rules(self) -> None:
        self.ctx.permission_mode = PermissionMode.BYPASS_PERMISSIONS

        def boom(tool_name, message, suggestion):
            raise AssertionError("handler must not be called in bypassPermissions mode")

        self.ctx.permission_handler = boom
        result = self.registry.dispatch(self._md_write(), self.ctx)
        self.assertFalse(result.is_error)

        # deny rules still win: dispatch raises ToolPermissionError
        # (pre-existing deny-rule behavior) before any mode logic runs.
        from src.tool_system.errors import ToolPermissionError

        self.ctx.permission_context = ToolPermissionContext.from_iterables(
            deny_names=["Write"],
            workspace_root=self.root,
        )
        with self.assertRaises(ToolPermissionError):
            self.registry.dispatch(self._md_write(), self.ctx)


class TestMaxTurnsSummary(unittest.TestCase):
    def test_full_summary(self) -> None:
        from src.tool_system.agent_loop import _format_max_turns_summary

        summary = _format_max_turns_summary(
            20,
            20,
            {"Bash": 12, "Read": 8},
            ["/a.py", "/b.py"],
            [
                ("Bash", False, "Bash · exit=0"),
                ("Edit", True, "Edit · /a.py · boom"),
            ],
        )
        self.assertIn("[Max tool turns reached: 20/20]", summary)
        self.assertIn("Turns used: 20 of 20", summary)
        self.assertIn("Tool calls (20): Bash x12, Read x8", summary)
        self.assertIn("Files touched (2): /a.py, /b.py", summary)
        self.assertIn("Last tool results (1/2 succeeded):", summary)
        self.assertIn("- Bash (ok): Bash · exit=0", summary)
        self.assertIn("- Edit (error): Edit · /a.py · boom", summary)

    def test_empty_state(self) -> None:
        from src.tool_system.agent_loop import _format_max_turns_summary

        summary = _format_max_turns_summary(5, 20, {}, [], [])
        self.assertIn("Turns used: 5 of 20", summary)
        self.assertIn("Tool calls: none", summary)
        self.assertIn("Files touched: none", summary)
        self.assertNotIn("Last tool results", summary)

    def test_long_result_truncated(self) -> None:
        from src.tool_system.agent_loop import _format_max_turns_summary

        long_text = "x" * 500
        summary = _format_max_turns_summary(
            1, 20, {"Read": 1}, [], [("Read", False, long_text)]
        )
        self.assertNotIn(long_text, summary)
        self.assertIn("...", summary)

    def test_many_files_capped(self) -> None:
        from src.tool_system.agent_loop import _format_max_turns_summary

        files = [f"/f{i}.py" for i in range(15)]
        summary = _format_max_turns_summary(3, 20, {"Read": 15}, files, [])
        self.assertIn("Files touched (15):", summary)
        self.assertIn("(+5 more)", summary)


class TestBashMissingError(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_sync_missing_bash_is_actionable(self) -> None:
        from src.tool_system.tools.bash import BashTool

        tool = BashTool()
        with unittest.mock.patch(
            "src.tool_system.tools.bash.subprocess.run",
            side_effect=FileNotFoundError(),
        ):
            result = tool.run({"command": "echo hi"}, self.ctx)
        self.assertTrue(result.is_error)
        error = result.output.get("error", "")
        self.assertIn("Git Bash", error)
        self.assertIn("SETUP_GUIDE", error)

    def test_background_missing_bash_is_actionable(self) -> None:
        from src.tool_system.tools.bash import BashTool

        tool = BashTool()
        with unittest.mock.patch(
            "src.tool_system.tools.bash.subprocess.Popen",
            side_effect=FileNotFoundError(),
        ):
            result = tool.run(
                {"command": "echo hi", "run_in_background": True}, self.ctx
            )
        self.assertTrue(result.is_error)
        error = result.output.get("error", "")
        self.assertIn("Git Bash", error)
        self.assertIn("SETUP_GUIDE", error)


class TestPowerShellOptIn(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_non_windows_keeps_platform_error(self) -> None:
        from src.tool_system.tools.misc import PowerShellTool

        tool = PowerShellTool()
        # default context: enable_powershell=False
        result = tool.run({"command": "echo hi"}, self.ctx)
        self.assertTrue(result.is_error)
        self.assertIn("only supported on Windows", result.output.get("error", ""))

    def test_non_windows_platform_error_even_when_opted_in(self) -> None:
        from src.tool_system.tools.misc import PowerShellTool

        tool = PowerShellTool()
        self.ctx.enable_powershell = True
        result = tool.run({"command": "echo hi"}, self.ctx)
        self.assertTrue(result.is_error)
        self.assertIn("only supported on Windows", result.output.get("error", ""))

    def test_windows_without_opt_in_is_gated(self) -> None:
        from src.tool_system.tools import misc as misc_module
        from src.tool_system.tools.misc import PowerShellTool

        tool = PowerShellTool()
        with unittest.mock.patch.object(
            misc_module.platform, "system", return_value="Windows"
        ):
            result = tool.run({"command": "echo hi"}, self.ctx)
        self.assertTrue(result.is_error)
        error = result.output.get("error", "")
        self.assertIn("enable_powershell", error)
        self.assertIn("not enabled", error)

    def test_windows_with_opt_in_executes(self) -> None:
        from src.tool_system.tools import misc as misc_module
        from src.tool_system.tools.misc import PowerShellTool

        tool = PowerShellTool()
        self.ctx.enable_powershell = True
        fake_completed = unittest.mock.Mock(returncode=0, stdout="hi\n", stderr="")
        with unittest.mock.patch.object(
            misc_module.platform, "system", return_value="Windows"
        ), unittest.mock.patch.object(
            misc_module.subprocess, "run", return_value=fake_completed
        ) as mock_run:
            result = tool.run({"command": "Write-Host hi"}, self.ctx)
        self.assertFalse(result.is_error)
        self.assertEqual(result.output.get("exit_code"), 0)
        self.assertEqual(result.output.get("stdout"), "hi\n")
        args = mock_run.call_args[0][0]
        self.assertEqual(args[0], "powershell")
        self.assertIn("Write-Host hi", args)

    def test_powershell_opt_in_config_defaults_off(self) -> None:
        from src.config import powershell_opt_in

        with unittest.mock.patch(
            "src.config.load_config", return_value={}
        ):
            self.assertFalse(powershell_opt_in())
        with unittest.mock.patch(
            "src.config.load_config", return_value={"enable_powershell": True}
        ):
            self.assertTrue(powershell_opt_in())
        with unittest.mock.patch(
            "src.config.load_config", side_effect=OSError("nope")
        ):
            self.assertFalse(powershell_opt_in())


class TestStatusBarPermissionMode(unittest.TestCase):
    def _repl(self, mode=None):
        from src.repl.status_bar import render_status

        repl = unittest.mock.Mock()
        repl.provider_name = "test"
        repl.provider = None
        repl.cost_tracker = None
        if mode is None:
            repl.tool_context = None
        else:
            ctx = unittest.mock.Mock()
            ctx.plan_mode = False
            ctx.permission_mode = mode
            repl.tool_context = ctx
        return render_status(repl)

    def test_mode_shown_when_not_default(self) -> None:
        out = self._repl(PermissionMode.ACCEPT_EDITS)
        self.assertIn("\\[acceptEdits]", out)

    def test_mode_hidden_when_default(self) -> None:
        out = self._repl(PermissionMode.DEFAULT)
        self.assertNotIn("acceptEdits", out)
        self.assertNotIn("default", out.replace("test", ""))

    def test_no_tool_context_does_not_raise(self) -> None:
        out = self._repl(None)
        self.assertIsInstance(out, str)


if __name__ == "__main__":
    unittest.main()
