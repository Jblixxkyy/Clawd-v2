"""Tests for brief 020: todo invariants + hard termination protocol."""

import tempfile
import unittest
import uuid
from pathlib import Path

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import completion_record, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError
from src.tool_system.protocol import ToolCall
from src.tool_system.registry import ToolRegistry, ToolSpec
from src.tool_system.tools.complete_task import CompleteTaskTool
from src.tool_system.tools.scaffold import (
    ExitTool,
    ReportFailureTool,
    ToolFailure,
)
from src.tool_system.tools.subagent import SubagentTool
from src.tool_system.tools.todo_write import (
    TodoWriteTool,
    load_session_todos,
    session_todos_path,
)


def _response(text, tool_uses=None):
    return ChatResponse(
        content=text,
        model="stub-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="tool_use" if tool_uses else "stop",
        tool_uses=tool_uses or [],
    )


def _tool_use(name, tool_input, tool_id="toolu_1"):
    return {"id": tool_id, "name": name, "input": tool_input}


class StubProvider:
    """Scripted provider; records (messages, kwargs) per chat call."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if not self._script:
            return _response("script exhausted")
        return self._script.pop(0)

    def chat_stream_response(self, *args, **kwargs):
        raise NotImplementedError()


class FlakyTool:
    """Test tool that raises a continuable ToolFailure."""

    def spec(self):
        return ToolSpec(
            name="Flaky",
            description="Always raises a continuable ToolFailure.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
        )

    def run(self, tool_input, context):
        raise ToolFailure("boom: try a different approach", kind="demo")


def _todo(content, status="pending"):
    return {"content": content, "status": status, "activeForm": f"Doing {content}"}


class TestTodoInvariants(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.context = ToolContext(workspace_root=self.workspace)
        self.tool = TodoWriteTool()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_two_in_progress_rejected(self):
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run(
                {"todos": [_todo("a", "in_progress"), _todo("b", "in_progress")]},
                self.context,
            )
        self.assertIn("at most one", str(cm.exception))
        self.assertIn("in_progress", str(cm.exception))

    def test_blocked_status_accepted(self):
        result = self.tool.run(
            {"todos": [_todo("a", "blocked"), _todo("b", "pending")]},
            self.context,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(self.context.todos[0]["status"], "blocked")

    def test_deltas_returned_as_response_metadata(self):
        self.tool.run({"todos": [_todo("a"), _todo("b")]}, self.context)
        result = self.tool.run(
            {"todos": [_todo("a", "in_progress"), _todo("c")]}, self.context
        )
        delta = result.output["delta"]
        self.assertEqual(delta["added"], ["c"])
        self.assertEqual(delta["removed"], ["b"])
        self.assertEqual(
            delta["status_changed"],
            [{"content": "a", "from": "pending", "to": "in_progress"}],
        )

    def test_plan_mode_rejects_todo_updates(self):
        self.context.plan_mode = True
        with self.assertRaises(ToolInputError) as cm:
            self.tool.run({"todos": [_todo("a")]}, self.context)
        self.assertIn("plan mode", str(cm.exception))

    def test_todos_survive_session_restart(self):
        session_id = f"test-020-{uuid.uuid4().hex}"
        try:
            self.context.session_id = session_id
            self.tool.run(
                {"todos": [_todo("a", "in_progress"), _todo("b", "blocked")]},
                self.context,
            )
            # A fresh context for the same session sees the persisted list.
            fresh = ToolContext(workspace_root=self.workspace)
            fresh.session_id = session_id
            loaded = load_session_todos(session_id)
            self.assertEqual(loaded, self.context.todos)
            self.assertTrue(session_todos_path(session_id).exists())
        finally:
            path = session_todos_path(session_id)
            if path.exists():
                path.unlink()


class TestCompleteTaskTool(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.context = ToolContext(workspace_root=Path(self.temp_dir.name))
        self.registry = ToolRegistry([CompleteTaskTool()])

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_invalid_schema_rejected(self):
        # Direct run(): schema-level validation.
        with self.assertRaises(ToolInputError):
            CompleteTaskTool().run({"outcome": "bogus", "summary": "x"}, self.context)
        with self.assertRaises(ToolInputError):
            CompleteTaskTool().run({"outcome": "success"}, self.context)  # no summary
        # Via dispatch: JSON-schema validation rejects before run().
        with self.assertRaises(ToolInputError):
            self.registry.dispatch(
                ToolCall(name="complete_task", input={"outcome": "bogus", "summary": "x"}),
                self.context,
            )

    def test_valid_input_returns_terminal_record(self):
        result = CompleteTaskTool().run(
            {
                "outcome": "success",
                "summary": "All done.",
                "artifacts": [{"name": "report", "path": "out.md"}],
            },
            self.context,
        )
        self.assertFalse(result.is_error)
        self.assertTrue(result.output["isComplete"])
        self.assertEqual(result.output["outcome"], "success")
        self.assertEqual(result.output["log_record"]["tool"], "complete_task")


class TestLoopTermination(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = ToolRegistry(
            [CompleteTaskTool(), ReportFailureTool(), ExitTool(), TodoWriteTool(), FlakyTool()]
        )
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, provider, max_turns=5):
        conversation = Conversation()
        conversation.add_user_message("do the thing")
        return run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            max_turns=max_turns,
        )

    def test_complete_task_valid_ends_run_with_structured_outcome(self):
        provider = StubProvider(
            [
                _response(
                    "finishing",
                    [_tool_use("complete_task", {
                        "outcome": "success",
                        "summary": "Implemented the feature.",
                        "artifacts": [{"name": "patch", "path": "a.py"}],
                    })],
                ),
            ]
        )
        result = self._run(provider)
        self.assertEqual(provider.calls.__len__(), 1)
        self.assertIn("Task complete (success)", result.response_text)
        terminal = next(
            r for r in (result.run_log or []) if r.get("event") == "terminal_tool"
        )
        self.assertEqual(terminal["tool"], "complete_task")
        self.assertEqual(terminal["outcome"], "success")
        self.assertEqual(terminal["log_record"]["summary"], "Implemented the feature.")
        record = completion_record(result)
        self.assertEqual(record["outcome"], "success")
        self.assertEqual(record["terminal_tool"], "complete_task")

    def test_report_failure_ends_run_as_failure(self):
        provider = StubProvider(
            [
                _response(
                    "giving up",
                    [_tool_use("ReportFailure", {
                        "reason": "API is down",
                        "attempted": ["retry", "fallback"],
                    })],
                ),
            ]
        )
        result = self._run(provider)
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("Task failed", result.response_text)
        record = completion_record(result)
        self.assertEqual(record["outcome"], "failed")
        self.assertEqual(record["terminal_tool"], "ReportFailure")

    def test_exit_tool_ends_run(self):
        provider = StubProvider(
            [_response("bye", [_tool_use("Exit", {"outcome": "done", "summary": "clean"})])]
        )
        result = self._run(provider)
        record = completion_record(result)
        self.assertEqual(record["outcome"], "done")
        self.assertEqual(record["terminal_tool"], "Exit")

    def test_no_tool_calls_logged_as_ended_without_explicit_completion(self):
        provider = StubProvider([_response("here is your answer")])
        result = self._run(provider)
        self.assertEqual(result.response_text, "here is your answer")
        events = [r.get("event") for r in (result.run_log or [])]
        self.assertIn("ended_without_explicit_completion", events)
        record = completion_record(result)
        # A main-loop run with tool calls but no terminal tool keeps its
        # text; only the pure no-call case is a stall (see subagent tests).
        self.assertEqual(record["outcome"], "stall")

    def test_max_steps_wrapup_disables_tools_and_produces_artifact(self):
        provider = StubProvider(
            [
                _response("working", [_tool_use("TodoWrite", {"todos": [_todo("a", "in_progress")]})]),
                _response("WRAP-UP: did X, remaining Y, recommend Z."),
            ]
        )
        result = self._run(provider, max_turns=2)
        self.assertEqual(len(provider.calls), 2)
        # Final turn: tools disabled, forced wrap-up prompt injected.
        _, final_kwargs = provider.calls[1]
        self.assertEqual(final_kwargs["tools"], [])
        final_messages = provider.calls[1][0]
        self.assertTrue(
            any(
                isinstance(m, dict)
                and "MAXIMUM STEPS REACHED" in str(m.get("content", ""))
                for m in final_messages
            ),
            "wrap-up prompt not injected on the final turn",
        )
        # The run ends with the model's wrap-up artifact, not a truncation.
        self.assertIn("WRAP-UP", result.response_text)
        events = [r.get("event") for r in (result.run_log or [])]
        self.assertIn("max_steps_wrapup", events)
        wrapup = next(r for r in result.run_log if r.get("event") == "max_steps_wrapup")
        self.assertTrue(wrapup["tools_disabled"])

    def test_toolfailure_is_continuable(self):
        provider = StubProvider(
            [
                _response("trying", [_tool_use("Flaky", {})]),
                _response("recovered after the typed failure"),
            ]
        )
        result = self._run(provider, max_turns=5)
        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(result.response_text, "recovered after the typed failure")
        failures = [
            r for r in (result.run_log or []) if r.get("event") == "tool_failure"
        ]
        self.assertEqual(len(failures), 1)
        self.assertTrue(failures[0]["continuable"])
        self.assertIn("boom", failures[0]["reason"])
        # The model saw the failure reason as the tool error result.
        _, second_kwargs = provider.calls[1]
        _ = second_kwargs  # kwargs not asserted; message history carries the error


class TestSubagentCompletionClause(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        from src.tool_system.defaults import build_default_registry

        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _dispatch(self, tool_input, provider):
        self.context.provider = provider
        return self.registry.dispatch(ToolCall(name="Subagent", input=tool_input), self.context)

    def test_stall_without_tool_calls_or_terminal_tool(self):
        provider = StubProvider([_response("hello, nothing to do")])
        result = self._dispatch(
            {"description": "say hi", "prompt": "Say hi."}, provider
        )
        out = result.output
        self.assertEqual(out["status"], "stalled")
        self.assertTrue(result.is_error)
        self.assertEqual(out["completion"]["outcome"], "stall")
        self.assertEqual(out["completion"]["tool_calls"], 0)

    def test_done_with_tool_calls_but_no_terminal_tool(self):
        target = self.workspace / "notes.txt"
        target.write_text("hello subagent")
        provider = StubProvider(
            [
                _response(
                    "reading",
                    [_tool_use("Read", {"file_path": str(target)}, tool_id="t1")],
                ),
                _response("done reading notes"),
            ]
        )
        result = self._dispatch(
            {"description": "read a file", "prompt": "Read notes.txt and report."},
            provider,
        )
        out = result.output
        self.assertEqual(out["status"], "done")
        self.assertFalse(result.is_error)
        self.assertEqual(out["completion"]["outcome"], "ended_without_explicit_completion")

    def test_child_report_failure_is_failed(self):
        provider = StubProvider(
            [
                _response(
                    "giving up",
                    [_tool_use("ReportFailure", {
                        "reason": "cannot proceed",
                        "attempted": ["tried once"],
                    })],
                ),
            ]
        )
        result = self._dispatch(
            {"description": "fail fast", "prompt": "Try and report failure."}, provider
        )
        out = result.output
        self.assertEqual(out["status"], "failed")
        self.assertTrue(result.is_error)
        self.assertEqual(out["completion"]["outcome"], "failed")


if __name__ == "__main__":
    unittest.main()
