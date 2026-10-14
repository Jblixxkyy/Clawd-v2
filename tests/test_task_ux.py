"""Tests for the "wire everything up" task UX build (worker A).

Covers:
- deny-and-report permission behavior inside subagent Task/Agent runs
  (never blocks on stdin; denials reported back to the parent context),
- Agent tool description/task_id/enriched output,
- task start/finish cards in the REPL (new card types only; existing
  card rendering must be untouched),
- the "## Task delegation" system-prompt section.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

from rich.console import Console

from src.repl.tool_cards import ToolCardRenderer, is_task_tool
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall
from src.tool_system.system_prompt import build_agent_guidelines


def _render_text(panel) -> str:
    buf = io.StringIO()
    con = Console(file=buf, width=100, legacy_windows=False)
    con.print(panel)
    return buf.getvalue()


def _render_all(rendered) -> str:
    """Render a render_* list (quiet mode) to plain text."""
    buf = io.StringIO()
    con = Console(file=buf, width=100, legacy_windows=False)
    con.print(*rendered)
    return buf.getvalue()


class TaskUxTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.registry = build_default_registry(include_user_tools=False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    # ------------------------------------------------------------------
    # permission deny-and-report inside subagent runs
    # ------------------------------------------------------------------
    def _md_write_call(self) -> ToolCall:
        # Writing a .md file triggers 'ask' (allow_docs=False by default).
        return ToolCall(
            name="Write",
            input={"file_path": str(self.root / "notes.md"), "content": "hi"},
        )

    def test_non_interactive_ask_denied_without_consulting_handler(self) -> None:
        """'ask' under non_interactive=True must deny without calling the handler."""
        handler_calls: list[str] = []

        def handler(tool_name: str, message: str, suggestion: str | None):
            handler_calls.append(tool_name)
            return True, False  # would allow — must never be reached

        ctx = ToolContext(workspace_root=self.root, non_interactive=True)
        ctx.permission_handler = handler
        result = self.registry.dispatch(self._md_write_call(), ctx)

        self.assertEqual(handler_calls, [])
        self.assertTrue(result.is_error)
        self.assertTrue(result.output.get("subagent_permission_denied"))
        self.assertIn("subagent", result.output.get("error", "").lower())

    def test_interactive_path_still_consults_handler(self) -> None:
        """Without the flag, 'ask' still goes to the permission handler."""
        handler_calls: list[str] = []

        def handler(tool_name: str, message: str, suggestion: str | None):
            handler_calls.append(tool_name)
            return True, False

        ctx = ToolContext(workspace_root=self.root)
        ctx.permission_handler = handler
        result = self.registry.dispatch(self._md_write_call(), ctx)

        self.assertEqual(handler_calls, ["Write"])
        self.assertFalse(result.is_error)

    def test_agent_tool_denies_and_reports_nested_permission_asks(self) -> None:
        """Nested 'ask' calls inside Agent are denied and reported to the parent."""
        handler_calls: list[str] = []

        def handler(tool_name: str, message: str, suggestion: str | None):
            handler_calls.append(tool_name)
            return True, False

        ctx = ToolContext(workspace_root=self.root)
        ctx.permission_handler = handler
        agent = self.registry.get("Agent")
        assert agent is not None
        out = agent.run(
            {
                "description": "write some docs",
                "stop_on_error": False,
                "calls": [
                    {"name": "Write", "input": {"file_path": str(self.root / "n.md"), "content": "x"}},
                    {"name": "Read", "input": {"file_path": str(self.root / "n.md")}},
                ],
            },
            ctx,
        ).output

        # Handler never consulted for the nested ask...
        self.assertEqual(handler_calls, [])
        # ...denial recorded and reported in the parent-facing output.
        self.assertEqual(len(out["permission_denials"]), 1)
        self.assertEqual(out["permission_denials"][0]["name"], "Write")
        self.assertIn("subagent", out["permission_denials"][0]["error"].lower())
        self.assertIn("note", out)
        self.assertEqual(out["num_failed"], 2)  # Write denied, Read then fails (no file)
        self.assertTrue(out["results"][0]["is_error"])
        # Parent context is untouched by the subagent flag.
        self.assertFalse(ctx.non_interactive)

    def test_agent_tool_shares_task_state_with_parent(self) -> None:
        """Nested calls see (and mutate) the parent's task list."""
        ctx = ToolContext(workspace_root=self.root)
        agent = self.registry.get("Agent")
        assert agent is not None
        out = agent.run(
            {"calls": [
                {"name": "TaskCreate", "input": {"subject": "s", "description": "d"}},
                {"name": "TaskList", "input": {}},
            ]},
            ctx,
        ).output
        created_id = out["results"][0]["output"]["task"]["id"]
        self.assertIn(created_id, ctx.tasks)
        listed_ids = [t["id"] for t in out["results"][1]["output"]["tasks"]]
        self.assertIn(created_id, listed_ids)

    def test_agent_tool_ask_user_disabled_inside_subagent(self) -> None:
        """AskUserQuestion inside Agent must fail cleanly, not block on stdin."""
        from src.tool_system.errors import ToolPermissionError

        ctx = ToolContext(workspace_root=self.root)
        ctx.ask_user = lambda questions: (_ for _ in ()).throw(AssertionError("must not prompt"))
        agent = self.registry.get("Agent")
        assert agent is not None
        with self.assertRaises(ToolPermissionError):
            agent.run(
                {"calls": [{
                    "name": "AskUserQuestion",
                    "input": {"questions": [{
                        "question": "q?", "header": "h",
                        "options": [
                            {"label": "a", "description": "d"},
                            {"label": "b", "description": "d"},
                        ],
                    }]},
                }]},
                ctx,
            )

    def test_agent_tool_description_and_task_id(self) -> None:
        """Optional description is accepted (strict schema) and echoed with a task_id."""
        ctx = ToolContext(workspace_root=self.root)
        result = self.registry.dispatch(
            ToolCall(name="Task", input={
                "description": "list the workspace",
                "calls": [{"name": "TaskList", "input": {}}],
            }),
            ctx,
        )
        self.assertFalse(result.is_error)
        out = result.output
        self.assertEqual(out["description"], "list the workspace")
        self.assertTrue(isinstance(out["task_id"], str) and len(out["task_id"]) > 0)
        self.assertEqual(out["num_calls"], 1)
        self.assertEqual(out["num_succeeded"], 1)
        self.assertEqual(out["num_failed"], 0)
        self.assertEqual(out["permission_denials"], [])

    def test_agent_tool_without_description(self) -> None:
        """description remains optional."""
        ctx = ToolContext(workspace_root=self.root)
        result = self.registry.dispatch(
            ToolCall(name="Agent", input={"calls": [{"name": "TaskList", "input": {}}]}),
            ctx,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["description"], "")

    def test_non_interactive_is_a_context_field(self) -> None:
        field_names = {f.name for f in fields(ToolContext)}
        self.assertIn("non_interactive", field_names)
        self.assertFalse(ToolContext(workspace_root=self.root).non_interactive)

    # ------------------------------------------------------------------
    # task cards
    # ------------------------------------------------------------------
    def test_is_task_tool_matches_only_delegation_tool(self) -> None:
        for name in ("Agent", "Task", "task", "AGENT"):
            self.assertTrue(is_task_tool(name), name)
        for name in ("TaskCreate", "TaskGet", "TaskList", "TaskUpdate",
                     "TaskOutput", "TaskStop", "Read", "Bash"):
            self.assertFalse(is_task_tool(name), name)

    def test_render_task_start_card(self) -> None:
        cards = ToolCardRenderer(Console(file=io.StringIO(), width=80))
        panel = cards.render_task_start(
            "Task",
            {"description": "explore the repo",
             "calls": [{"name": "Read", "input": {}}, {"name": "Glob", "input": {}}]},
            task_id="toolu_abc123",
        )
        self.assertEqual(panel.border_style, "magenta")
        text = _render_text(panel)
        self.assertIn("toolu_abc123", text)
        self.assertIn("explore the repo", text)
        self.assertIn("2 calls", text)
        self.assertIn("Read", text)

    def test_render_task_finish_success_card(self) -> None:
        cards = ToolCardRenderer(Console(file=io.StringIO(), width=80))
        panel = cards.render_task_finish(
            "Task",
            {"num_succeeded": 2, "num_failed": 0, "permission_denials": []},
            task_id="toolu_abc123",
            is_error=False,
        )
        self.assertEqual(panel.border_style, "green")
        text = _render_text(panel)
        self.assertIn("toolu_abc123", text)
        self.assertIn("2/2", text)
        self.assertIn("tool result", text)

    def test_render_task_finish_failure_card_reports_denials(self) -> None:
        cards = ToolCardRenderer(Console(file=io.StringIO(), width=80))
        panel = cards.render_task_finish(
            "Agent",
            {"num_succeeded": 1, "num_failed": 1,
             "permission_denials": [{"name": "Write", "error": "denied"}]},
            task_id="toolu_xyz",
            is_error=True,
        )
        self.assertEqual(panel.border_style, "red")
        text = _render_text(panel)
        self.assertIn("toolu_xyz", text)
        self.assertIn("auto-denied", text)
        self.assertIn("tool result", text)

    def test_quiet_rendering_one_dim_line_per_call(self) -> None:
        """Brief 003: no bordered panels per call -- one dim line each."""
        cards = ToolCardRenderer(Console(file=io.StringIO(), width=80))
        use_lines = cards.render_use("Read", {"file_path": "a.txt", "limit": 5})
        use_text = _render_all(use_lines)
        self.assertEqual(use_text.count("\n"), 1)
        self.assertIn("Read", use_text)
        self.assertIn("a.txt", use_text)
        self.assertNotIn("╭", use_text)

        result_lines = cards.render_result("Read", {"some": "output"}, is_error=False)
        result_text = _render_all(result_lines)
        self.assertEqual(result_text.count("\n"), 1)
        self.assertIn("✓", result_text)
        self.assertIn("Read", result_text)
        self.assertNotIn("╭", result_text)

    def test_quiet_error_is_red_line(self) -> None:
        """Brief 003: red only on errors -- one red line, no panel."""
        cards = ToolCardRenderer(Console(file=io.StringIO(), width=80))
        err_lines = cards.render_error("Bash", "boom")
        self.assertEqual(len(err_lines), 1)
        text = _render_all(err_lines)
        self.assertIn("✗", text)
        self.assertIn("boom", text)
        self.assertNotIn("╭", text)

    def test_edit_diff_is_single_spaced(self) -> None:
        """Regression: diff lines must not render double-spaced.

        difflib yields lines with their original terminators; the renderer
        strips them so joined lines don't double-space.
        """
        cards = ToolCardRenderer(Console(file=io.StringIO(), width=80))
        rendered = cards.render_result(
            "Edit",
            {
                "file_path": "a.py",
                "originalFile": "x = 1\ny = 2\n",
                "oldString": "x = 1",
                "newString": "x = 10",
            },
            is_error=False,
        )
        # First line is the dim ✓ summary, the rest are red-green diff lines.
        self.assertGreater(len(rendered), 1)
        text = _render_all(rendered)
        self.assertIn("✓", text)
        self.assertIn("-x = 1", text)
        self.assertIn("+x = 10", text)
        # No blank line between adjacent diff lines.
        self.assertNotIn("-x = 1\n\n", text)
        self.assertNotIn("+x = 10\n\n", text)

    def test_write_diff_shown_without_panel(self) -> None:
        cards = ToolCardRenderer(Console(file=io.StringIO(), width=80))
        rendered = cards.render_result(
            "Write",
            {"file_path": "b.py", "originalFile": "a = 1\n", "content": "a = 2\n"},
            is_error=False,
        )
        text = _render_all(rendered)
        self.assertIn("✓", text)
        self.assertIn("-a = 1", text)
        self.assertIn("+a = 2", text)
        self.assertNotIn("╭", text)

    # ------------------------------------------------------------------
    # system prompt delegation section (prompt diet: conditional guidance)
    # ------------------------------------------------------------------
    def test_system_prompt_has_task_delegation_section(self) -> None:
        from src.outputStyles.base_prompt import BASE_SYSTEM_PROMPT
        from src.tool_system.system_prompt import build_tool_guidance

        # Delegation guidance renders only when the tools are registered.
        guidance = build_tool_guidance({"agent", "taskcreate"})
        self.assertIn("TaskCreate", guidance)
        self.assertIn("parallel", guidance.lower())
        # Unregistered tools contribute nothing.
        self.assertNotIn("TaskCreate", build_tool_guidance({"read"}))
        # Core still carries the TodoWrite discipline line.
        self.assertIn("TodoWrite", BASE_SYSTEM_PROMPT)

    def test_agent_guidelines_now_windows_shell_only(self) -> None:
        guidelines = build_agent_guidelines()
        # On non-Windows this is empty: the old sections merged into the core.
        self.assertNotIn("## Agent Capabilities", guidelines)
        self.assertNotIn("## Tool-Use Judgment", guidelines)


class DiffStatTests(unittest.TestCase):
    """Compact (+N, -M) diff stats on Write/Edit completion lines.

    Ported from gemini-cli's DenseToolMessage diff-stat summary: the change
    size is visible at a glance without reading the full diff below.
    """

    def _cards(self) -> ToolCardRenderer:
        return ToolCardRenderer(Console(file=io.StringIO(), width=100))

    def test_edit_completion_line_carries_stat(self) -> None:
        rendered = self._cards().render_result(
            "Edit",
            {
                "file_path": "a.py",
                "originalFile": "x = 1\ny = 2\n",
                "oldString": "x = 1",
                "newString": "x = 10",
            },
            is_error=False,
        )
        text = _render_all(rendered)
        self.assertIn("✓", text)
        self.assertIn("(+1, -1)", text)

    def test_write_stat_counts_multiple_lines(self) -> None:
        rendered = self._cards().render_result(
            "Write",
            {
                "file_path": "b.py",
                "originalFile": "a\nb\nc\n",
                "content": "a\nB\nC\nd\n",
            },
            is_error=False,
        )
        text = _render_all(rendered)
        self.assertIn("(+3, -2)", text)

    def test_stat_spans_are_colored(self) -> None:
        """The +N span is green and the -M span is red on the Text itself."""
        rendered = self._cards().render_result(
            "Edit",
            {
                "file_path": "a.py",
                "originalFile": "x = 1\n",
                "oldString": "x = 1",
                "newString": "x = 2",
            },
            is_error=False,
        )
        first = rendered[0]
        plain = first.plain
        green_hit = any(
            plain[s.start : s.end] == "+1" for s in first.spans if s.style == "green"
        )
        red_hit = any(
            plain[s.start : s.end] == "-1" for s in first.spans if s.style == "red"
        )
        self.assertTrue(green_hit, f"no green '+1' span in {first.spans!r}")
        self.assertTrue(red_hit, f"no red '-1' span in {first.spans!r}")

    def test_no_stat_when_no_diff(self) -> None:
        """Edit that can't compute a diff gets a plain ✓ line, no stat."""
        rendered = self._cards().render_result(
            "Edit",
            {
                "file_path": "a.py",
                "originalFile": "x = 1\n",
                "oldString": "nope, not here",
                "newString": "x = 2",
            },
            is_error=False,
        )
        text = _render_all(rendered)
        self.assertIn("✓", text)
        self.assertNotIn("(+", text)

    def test_stat_reflects_full_diff_not_truncated_view(self) -> None:
        """Stat counts the whole change even when diff lines are truncated."""
        before = "".join(f"line{i}\n" for i in range(60))
        after = "".join(f"LINE{i}\n" for i in range(60))
        rendered = self._cards().render_result(
            "Write",
            {"file_path": "big.py", "originalFile": before, "content": after},
            is_error=False,
        )
        text = _render_all(rendered)
        self.assertIn("(+60, -60)", text)
        self.assertIn("more lines", text)  # truncation overflow note still shown

    def test_diff_stat_helper_excludes_headers(self) -> None:
        from src.repl.tool_cards import _diff_stat

        diff = [
            "--- a.py",
            "+++ b.py",
            "@@ -1,2 +1,2 @@",
            "-old",
            "+new",
            "+added",
            " context",
        ]
        self.assertEqual(_diff_stat(diff), (2, 1))


if __name__ == "__main__":
    unittest.main()
