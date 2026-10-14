"""Tests for brief 016: plan mode as a permission-enforced state machine.

Covers:
- dispatcher-level plan-mode policy: writes outside the plans dir denied,
  *.md writes under the plans dir allowed, read-only tools allowed.
- exit_plan_mode validation gate: missing/empty/schema-invalid plan ->
  denial, stays in plan mode; valid plan -> approval prompt.
- plan survives a compaction cycle (compaction-safe reminder re-injected).
- planner subagent profile physically excludes mutation tools.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.command_system.builtins import get_builtin_commands
from src.tool_system.context import ToolContext
from src.tool_system.permission_handler import PermissionBehavior
from src.tool_system.plan_policy import (
    build_approved_plan_context,
    build_plan_reminder,
    is_plan_path,
    plan_context_snippet,
    register_plan_mode_hooks,
    resolve_plans_dir,
    tool_allowed_in_plan_mode,
    validate_plan_schema,
)
from src.tool_system.planner_profile import (
    PLANNER_PROFILE,
    build_planner_registry,
    detect_plan_markers,
)
from src.tool_system.protocol import ToolCall
from src.tool_system.registry import ToolRegistry
from src.tool_system.tools import (
    BashTool,
    EnterPlanModeTool,
    ExitPlanModeTool,
    FileReadTool,
    FileWriteTool,
)
from src.tool_system.tools.plan_mode import EnterPlanModeV2, ExitPlanModeV2

VALID_PLAN = """\
## OBJECTIVE
Rebuild plan mode as a permission-enforced state machine.

## CONTEXT SUMMARY
Clawd already has plan_mode.py; this upgrades it from a flag to an enforced
mechanism with a validation gate and a mode-scoped tool policy.

## APPROACH OVERVIEW
Add enter/exit plan-mode tools, a declarative tiered policy enforced in the
dispatcher, OpenHands-style plan schema validation, and a planner profile.

## IMPLEMENTATION STEPS
1. Goal: enforce the mode-scoped tool policy in the dispatcher
   Method: consult plan_policy.tool_allowed_in_plan_mode in registry.dispatch
   Reference: src/tool_system/registry.py, src/tool_system/plan_policy.py
2. Goal: validate plans before approval is requested
   Method: exit_plan_mode runs the schema gate in check_permissions
   Reference: src/tool_system/tools/plan_mode.py (ExitPlanModeV2)

## TESTING AND VALIDATION
Run the full unittest suite; plan-mode write outside the plans dir must be
denied by the dispatcher; exit with a missing plan must stay in plan mode.
"""

INVALID_PLAN = """\
## OBJECTIVE
Do the thing.

## CONTEXT SUMMARY
Some context.
"""


class PlanModeStateMachineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        # Auto-approve permission prompts so dispatch tests reach run().
        self.ctx.permission_handler = lambda *a: (True, False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # -- policy unit tests -------------------------------------------------
    def test_policy_allows_read_only_in_plan_mode(self) -> None:
        allowed, _ = tool_allowed_in_plan_mode("Read", True, {}, self.ctx)
        self.assertTrue(allowed)

    def test_policy_denies_bash_in_plan_mode(self) -> None:
        allowed, reason = tool_allowed_in_plan_mode("Bash", False, {}, self.ctx)
        self.assertFalse(allowed)
        self.assertIn("plan mode", reason or "")

    def test_policy_allows_write_to_plans_dir_md(self) -> None:
        plans = resolve_plans_dir(self.ctx)
        target = plans / "implementation_plan.md"
        allowed, _ = tool_allowed_in_plan_mode(
            "Write", False, {"file_path": str(target)}, self.ctx
        )
        self.assertTrue(allowed)

    def test_policy_denies_write_outside_plans_dir(self) -> None:
        target = self.root / "src" / "main.py"
        allowed, reason = tool_allowed_in_plan_mode(
            "Write", False, {"file_path": str(target)}, self.ctx
        )
        self.assertFalse(allowed)
        self.assertIn("plans directory", reason or "")

    def test_policy_denies_write_to_plans_dir_non_md(self) -> None:
        plans = resolve_plans_dir(self.ctx)
        target = plans / "script.py"
        allowed, _ = tool_allowed_in_plan_mode(
            "Write", False, {"file_path": str(target)}, self.ctx
        )
        self.assertFalse(allowed)

    def test_policy_denies_path_traversal_out_of_plans_dir(self) -> None:
        plans = resolve_plans_dir(self.ctx)
        sneaky = plans / ".." / "evil.md"
        self.assertFalse(is_plan_path(sneaky, plans))

    # -- dispatcher enforcement --------------------------------------------
    def _registry(self) -> ToolRegistry:
        reg = ToolRegistry()
        reg.register(FileReadTool())
        reg.register(FileWriteTool())
        reg.register(BashTool())
        reg.register(EnterPlanModeV2())
        reg.register(ExitPlanModeV2())
        return reg

    def test_dispatch_denies_write_outside_plans_dir_in_plan_mode(self) -> None:
        reg = self._registry()
        self.ctx.plan_mode = True
        result = reg.dispatch(
            ToolCall(
                name="Write",
                input={"file_path": str(self.root / "notes.md"), "content": "x"},
            ),
            self.ctx,
        )
        self.assertTrue(result.is_error)
        self.assertIn("plan mode", str(result.output.get("error", "")))
        self.assertTrue(self.ctx.plan_mode)  # still planning
        self.assertFalse((self.root / "notes.md").exists())

    def test_dispatch_allows_write_to_plans_dir_md_in_plan_mode(self) -> None:
        reg = self._registry()
        self.ctx.plan_mode = True
        plans = resolve_plans_dir(self.ctx)
        target = plans / "implementation_plan.md"
        result = reg.dispatch(
            ToolCall(
                name="Write",
                input={"file_path": str(target), "content": "# plan\n"},
            ),
            self.ctx,
        )
        self.assertFalse(result.is_error, result.output)
        self.assertTrue(target.exists())

    def test_dispatch_denies_bash_in_plan_mode(self) -> None:
        reg = self._registry()
        self.ctx.plan_mode = True
        result = reg.dispatch(
            ToolCall(name="Bash", input={"command": "echo hi"}), self.ctx
        )
        self.assertTrue(result.is_error)
        self.assertIn("plan mode", str(result.output.get("error", "")))

    def test_dispatch_allows_read_in_plan_mode(self) -> None:
        reg = self._registry()
        self.ctx.plan_mode = True
        probe = self.root / "probe.txt"
        probe.write_text("hello", encoding="utf-8")
        result = reg.dispatch(
            ToolCall(name="Read", input={"file_path": str(probe)}), self.ctx
        )
        self.assertFalse(result.is_error, result.output)

    def test_dispatch_allows_exit_tool_in_plan_mode(self) -> None:
        reg = self._registry()
        self.ctx.plan_mode = True
        plans = resolve_plans_dir(self.ctx)
        (plans / "implementation_plan.md").write_text(VALID_PLAN, encoding="utf-8")
        result = reg.dispatch(
            ToolCall(
                name="exit_plan_mode",
                input={"planFilePath": str(plans / "implementation_plan.md")},
            ),
            self.ctx,
        )
        # permission_handler auto-approves -> run() executes -> exits plan mode
        self.assertFalse(result.is_error, result.output)
        self.assertFalse(self.ctx.plan_mode)
        self.assertIn("single source of truth", result.output["approved_plan_context"])

    # -- enter_plan_mode ----------------------------------------------------
    def test_enter_plan_mode_sets_state(self) -> None:
        tool = EnterPlanModeV2()
        out = tool.run({"reason": "test"}, self.ctx).output
        self.assertTrue(self.ctx.plan_mode)
        self.assertTrue(out["plan_mode"])
        self.assertTrue(Path(out["plans_dir"]).is_dir())
        self.assertIn("PLAN MODE", out["reminder"])

    # -- exit_plan_mode validation gate --------------------------------------
    def _exit_check(self, tool_input: dict) -> tuple[PermissionBehavior, str]:
        tool = ExitPlanModeV2()
        res = tool.check_permissions(tool_input, self.ctx)
        return res.behavior, res.message or ""

    def test_exit_missing_plan_file_stays_in_plan_mode(self) -> None:
        self.ctx.plan_mode = True
        plans = resolve_plans_dir(self.ctx)
        behavior, message = self._exit_check(
            {"planFilePath": str(plans / "nope.md")}
        )
        self.assertEqual(behavior, PermissionBehavior.DENY)
        self.assertIn("does not exist", message)
        self.assertTrue(self.ctx.plan_mode)

    def test_exit_empty_plan_file_stays_in_plan_mode(self) -> None:
        self.ctx.plan_mode = True
        plans = resolve_plans_dir(self.ctx)
        empty = plans / "empty.md"
        empty.write_text("   \n", encoding="utf-8")
        behavior, message = self._exit_check({"planFilePath": str(empty)})
        self.assertEqual(behavior, PermissionBehavior.DENY)
        self.assertIn("empty", message)
        self.assertTrue(self.ctx.plan_mode)

    def test_exit_schema_invalid_plan_stays_in_plan_mode(self) -> None:
        self.ctx.plan_mode = True
        plans = resolve_plans_dir(self.ctx)
        bad = plans / "bad.md"
        bad.write_text(INVALID_PLAN, encoding="utf-8")
        behavior, message = self._exit_check({"planFilePath": str(bad)})
        self.assertEqual(behavior, PermissionBehavior.DENY)
        self.assertIn("APPROACH OVERVIEW", message)
        self.assertIn("IMPLEMENTATION STEPS", message)
        self.assertTrue(self.ctx.plan_mode)

    def test_exit_plan_outside_plans_dir_denied(self) -> None:
        self.ctx.plan_mode = True
        outsider = self.root / "plan.md"
        outsider.write_text(VALID_PLAN, encoding="utf-8")
        behavior, message = self._exit_check({"planFilePath": str(outsider)})
        self.assertEqual(behavior, PermissionBehavior.DENY)
        self.assertIn("plans directory", message)
        self.assertTrue(self.ctx.plan_mode)

    def test_exit_valid_plan_requests_approval(self) -> None:
        self.ctx.plan_mode = True
        plans = resolve_plans_dir(self.ctx)
        good = plans / "implementation_plan.md"
        good.write_text(VALID_PLAN, encoding="utf-8")
        behavior, message = self._exit_check({"planFilePath": str(good)})
        self.assertEqual(behavior, PermissionBehavior.ASK)
        self.assertIn("Approve this plan", message)
        self.assertTrue(self.ctx.plan_mode)  # not exited until approved

    def test_exit_inline_plan_written_then_validated(self) -> None:
        self.ctx.plan_mode = True
        plans = resolve_plans_dir(self.ctx)
        target = plans / "inline.md"
        behavior, _ = self._exit_check(
            {"planFilePath": str(target), "plan": VALID_PLAN}
        )
        self.assertEqual(behavior, PermissionBehavior.ASK)
        self.assertTrue(target.exists())

    def test_exit_run_after_approval_exits_and_injects_plan(self) -> None:
        self.ctx.plan_mode = True
        plans = resolve_plans_dir(self.ctx)
        good = plans / "implementation_plan.md"
        good.write_text(VALID_PLAN, encoding="utf-8")
        tool = ExitPlanModeV2()
        # check_permissions stashes the validated plan; run() finalizes.
        asked = tool.check_permissions({"planFilePath": str(good)}, self.ctx)
        self.assertEqual(asked.behavior, PermissionBehavior.ASK)
        out = tool.run({"planFilePath": str(good)}, self.ctx).output
        self.assertFalse(self.ctx.plan_mode)
        self.assertEqual(self.ctx.plan_file_path, str(good))
        self.assertIn("OBJECTIVE", self.ctx.approved_plan or "")
        self.assertIn("single source of truth", out["approved_plan_context"])

    def test_exit_run_rejects_when_not_in_plan_mode(self) -> None:
        tool = ExitPlanModeV2()
        with self.assertRaises(Exception):
            tool.run({"planFilePath": "x.md"}, self.ctx)

    # -- schema validation ---------------------------------------------------
    def test_schema_valid_plan(self) -> None:
        self.assertEqual(validate_plan_schema(VALID_PLAN), [])

    def test_schema_reports_all_missing_sections(self) -> None:
        errors = validate_plan_schema(INVALID_PLAN)
        self.assertTrue(any("APPROACH OVERVIEW" in e for e in errors))
        self.assertTrue(any("IMPLEMENTATION STEPS" in e for e in errors))
        self.assertTrue(any("TESTING AND VALIDATION" in e for e in errors))

    def test_schema_requires_goal_method_reference_tuples(self) -> None:
        no_tuples = VALID_PLAN.replace("Method:", "How:").replace(
            "Reference:", "See:"
        )
        errors = validate_plan_schema(no_tuples)
        self.assertTrue(any("goal+method+reference" in e for e in errors))

    # -- compaction safety ----------------------------------------------------
    def test_plan_reminder_reinjected_post_compaction(self) -> None:
        self.ctx.plan_mode = True
        plans = resolve_plans_dir(self.ctx)
        draft = plans / "implementation_plan.md"
        draft.write_text(VALID_PLAN, encoding="utf-8")
        self.ctx.plan_file_path = str(draft)
        # Simulate a compaction cycle: transcript is rebuilt from scratch;
        # the loop re-injects plan_context_snippet(context).
        snippet = plan_context_snippet(self.ctx)
        self.assertIsNotNone(snippet)
        assert snippet is not None
        self.assertIn("PLAN MODE", snippet)
        self.assertIn(str(draft), snippet)
        # The snippet is derived from context state, not transcript history.
        self.assertIn("read-only", snippet)

    def test_approved_plan_survives_compaction(self) -> None:
        self.ctx.plan_mode = False
        self.ctx.plan_file_path = "/tmp/plans/implementation_plan.md"
        self.ctx.approved_plan = VALID_PLAN
        snippet = plan_context_snippet(self.ctx)
        self.assertIsNotNone(snippet)
        assert snippet is not None
        self.assertIn("single source of truth", snippet)
        self.assertIn("OBJECTIVE", snippet)

    def test_no_plan_state_no_snippet(self) -> None:
        self.assertIsNone(plan_context_snippet(self.ctx))

    def test_reminder_builder_mentions_policy(self) -> None:
        reminder = build_plan_reminder(self.ctx)
        self.assertIn("PLAN MODE", reminder)
        self.assertIn("exit_plan_mode", reminder)

    def test_approved_context_builder(self) -> None:
        ctx_text = build_approved_plan_context(VALID_PLAN, "/p/plan.md")
        self.assertIn("single source of truth", ctx_text)
        self.assertIn("/p/plan.md", ctx_text)

    # -- planner profile -------------------------------------------------------
    def test_planner_profile_excludes_mutation_tools(self) -> None:
        parent = ToolRegistry()
        parent.register(FileReadTool())
        parent.register(FileWriteTool())
        parent.register(BashTool())
        parent.register(ExitPlanModeTool())
        planner_reg = build_planner_registry(parent)
        names = {t.spec().name for t in planner_reg.list_tools()}
        self.assertIn("Read", names)
        self.assertIn("ExitPlanMode", names)  # the way out is kept
        self.assertNotIn("Write", names)
        self.assertNotIn("Bash", names)
        # Physical exclusion: dispatch fails even if the model tries.
        result = planner_reg.dispatch(
            ToolCall(name="Write", input={"file_path": "x", "content": "y"}),
            self.ctx,
        )
        self.assertTrue(result.is_error)
        self.assertIn("unknown tool", str(result.output.get("error", "")))

    def test_planner_profile_metadata(self) -> None:
        self.assertEqual(PLANNER_PROFILE["name"], "planner")
        self.assertIn("write", PLANNER_PROFILE["forbidden_tools"])
        self.assertIn("bash", PLANNER_PROFILE["forbidden_tools"])
        prompt = PLANNER_PROFILE["system_prompt"]
        self.assertIn("read-only PLANNER", prompt)
        self.assertIn("Self-review workflow", prompt)
        self.assertIn("<!-- PLAN_START -->", prompt)
        self.assertIn("<!-- PLAN_END -->", prompt)

    def test_plan_marker_detection(self) -> None:
        text = "thinking...\n<!-- PLAN_START -->\n## OBJECTIVE\nx\n<!-- PLAN_END -->\n"
        found = detect_plan_markers(text)
        self.assertIsNotNone(found)
        assert found is not None
        self.assertIn("## OBJECTIVE", found[1])
        self.assertIsNone(detect_plan_markers("no markers here"))

    # -- hook registration entry point ------------------------------------------
    def test_register_plan_mode_hooks_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as home:
            hooks_json = Path(home) / "hooks.json"
            first = register_plan_mode_hooks(hooks_json)
            second = register_plan_mode_hooks(hooks_json)
            self.assertTrue(first["registered"])
            config = json.loads(hooks_json.read_text(encoding="utf-8"))
            entries = config["BeforeToolSelection"]
            self.assertEqual(len(entries), 1)
            self.assertIn("plan_mode_tool_filter.py", entries[0]["command"])
            self.assertEqual(first, second)

    # -- /deep-planning slash command ---------------------------------------------
    def test_deep_planning_command_registered(self) -> None:
        commands = {c.name: c for c in get_builtin_commands()}
        self.assertIn("deep-planning", commands)
        cmd = commands["deep-planning"]
        content = cmd.markdown_content
        self.assertIn("Step 1", content)
        self.assertIn("Step 2", content)
        self.assertIn("Step 3", content)
        self.assertIn("Step 4", content)
        self.assertIn("implementation_plan.md", content)
        self.assertIn("OBJECTIVE", content)

    # -- legacy tools untouched ------------------------------------------------------
    def test_legacy_plan_mode_tools_still_work(self) -> None:
        EnterPlanModeTool().run({}, self.ctx)
        self.assertTrue(self.ctx.plan_mode)
        ExitPlanModeTool().run({}, self.ctx)
        self.assertFalse(self.ctx.plan_mode)


if __name__ == "__main__":
    unittest.main()
