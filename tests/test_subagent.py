"""Tests for true agentic subagents (brief 005): Subagent + SubagentOutput."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.permission_handler import PermissionResult
from src.tool_system.protocol import ToolCall, ToolResult
from src.tool_system.registry import ToolRegistry, ToolSpec
from src.tool_system.tools.subagent import (
    SubagentOutputTool,
    SubagentTool,
    build_child_registry,
)


def _response(text: str, tool_uses: list[dict] | None = None) -> ChatResponse:
    return ChatResponse(
        content=text,
        model="stub-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="tool_use" if tool_uses else "stop",
        tool_uses=tool_uses or [],
    )


class RecordingProvider:
    """Minimal stub provider that records the model it was called with."""

    def __init__(self, model: str, script: list[ChatResponse]):
        self.model = model
        self._script = list(script)
        self.calls: list[tuple[str, dict]] = []

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        self.calls.append((self.model, kwargs))
        if not self._script:
            return _response("script exhausted")
        return self._script.pop(0)

    def chat_stream_response(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()


class AskTool:
    """Test-only tool whose permission check always asks."""

    def __init__(self):
        self.handler_invocations = 0

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="AskTool",
            description="Tool that requires approval.",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
        )

    def check_permissions(self, tool_input: dict, context: ToolContext) -> PermissionResult:
        self.handler_invocations += 1
        return PermissionResult.ask("needs approval")

    def run(self, tool_input: dict, context: ToolContext) -> ToolResult:
        return ToolResult(name="AskTool", output={"ok": True})


class TestSubagentTool(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _dispatch(self, tool_input: dict, provider: Any = None) -> ToolResult:
        if provider is not None:
            self.context.provider = provider
        return self.registry.dispatch(
            ToolCall(name="Subagent", input=tool_input), self.context
        )

    def test_sync_child_runs_loop_and_returns_text(self) -> None:
        target = self.workspace / "notes.txt"
        target.write_text("hello subagent")
        provider = RecordingProvider(
            "parent-model",
            [
                _response(
                    "reading",
                    [{"id": "t1", "name": "Read", "input": {"file_path": str(target)}}],
                ),
                _response("done reading notes"),
            ],
        )
        result = self._dispatch(
            {"description": "read a file", "prompt": "Read notes.txt and report."},
            provider,
        )
        self.assertFalse(result.is_error)
        out = result.output
        self.assertEqual(out["status"], "done")
        self.assertIn("done reading notes", out["response_text"])
        self.assertEqual(out["num_turns"], 2)
        # Parent provider object itself is untouched; the child gets a copy.
        self.assertEqual(provider.model, "parent-model")
        self.assertEqual(out["model"], "parent-model")

    def test_model_override_reaches_child_provider(self) -> None:
        provider = RecordingProvider("parent-model", [_response("child report")])
        result = self._dispatch(
            {
                "description": "quick task",
                "prompt": "Say hi.",
                "model": "child-model",
            },
            provider,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["model"], "child-model")
        # The recorded chat call saw the overridden model.
        self.assertTrue(provider.calls)
        self.assertEqual(provider.calls[0][0], "child-model")

    def test_permission_prompts_auto_deny_in_child(self) -> None:
        ask_tool = AskTool()
        registry = ToolRegistry()
        registry.register(ask_tool)
        registry.register(SubagentTool(registry))

        def _boom(tool_name: str, message: str, suggestion):
            raise AssertionError("interactive permission handler must never run in a subagent")

        context = ToolContext(workspace_root=self.workspace, permission_handler=_boom)
        provider = RecordingProvider(
            "parent-model",
            [
                _response(
                    "trying",
                    [{"id": "t1", "name": "AskTool", "input": {}}],
                ),
                _response("AskTool was denied: needs approval unavailable"),
            ],
        )
        context.provider = provider
        result = registry.dispatch(
            ToolCall(
                name="Subagent",
                input={"description": "approval test", "prompt": "Use AskTool."},
            ),
            context,
        )
        self.assertFalse(result.is_error)
        # The child saw an auto-denial (never reached the parent's handler).
        self.assertIn("denied", result.output["response_text"].lower())
        self.assertEqual(ask_tool.handler_invocations, 1)

    def test_background_mode_returns_task_id_and_output_tool_fetches(self) -> None:
        provider = RecordingProvider("parent-model", [_response("background report")])
        result = self._dispatch(
            {
                "description": "bg task",
                "prompt": "Report back.",
                "background": True,
            },
            provider,
        )
        self.assertFalse(result.is_error)
        task_id = result.output["task_id"]
        self.assertTrue(task_id.startswith("subagent_"))
        self.assertEqual(result.output["status"], "running")

        out_tool = SubagentOutputTool()
        fetched = out_tool.run(
            {"task_id": task_id, "block": True, "timeout": 15.0}, self.context
        )
        self.assertFalse(fetched.is_error)
        self.assertEqual(fetched.output["retrieval_status"], "success")
        task = fetched.output["task"]
        self.assertEqual(task["status"], "done")
        self.assertIn("background report", task["response_text"])

    def test_no_provider_is_clean_error(self) -> None:
        result = self.registry.dispatch(
            ToolCall(
                name="Subagent",
                input={"description": "x", "prompt": "y"},
            ),
            self.context,
        )
        self.assertTrue(result.is_error)
        self.assertIn("no provider", result.output["error"])

    def test_input_validation(self) -> None:
        from src.tool_system.errors import ToolInputError

        for bad in (
            {},
            {"description": "x"},
            {"prompt": "y"},
            {"description": "", "prompt": "y"},
            {"description": "x", "prompt": "  "},
            {"description": "x", "prompt": "y", "max_turns": "many"},
            {"description": "x", "prompt": "y", "model": 42},
        ):
            with self.assertRaises((ToolInputError, Exception), msg=str(bad)):
                self.registry.dispatch(ToolCall(name="Subagent", input=bad), self.context)

    def test_child_registry_filters_session_scoped_tools(self) -> None:
        child = build_child_registry(self.registry)
        names = {spec.name.lower() for spec in child.list_specs()}
        for kept in ("read", "bash", "write", "grep", "glob", "skill", "agent", "todowrite"):
            self.assertIn(kept, names, f"{kept} should survive into the child")
        for dropped in (
            "sendusermessage",
            "askuserquestion",
            "sendmessage",
            "structuredoutput",
            "croncreate",
            "cronlist",
            "crondelete",
            "teamcreate",
            "teamdelete",
            "enterplanmode",
            "exitplanmode",
            "enterworktree",
            "exitworktree",
            "config",
            "taskstop",
            "subagent",
        ):
            self.assertNotIn(dropped, names, f"{dropped} must not leak into the child")
        # Registry-capturing tools are re-bound to the child registry.
        agent_tool = child.get("agent")
        self.assertIsNotNone(agent_tool)
        self.assertIs(agent_tool._registry, child)
        search_tool = child.get("toolsearch") or child.get("tool_search")
        if search_tool is not None:
            self.assertIs(search_tool._registry, child)

    def test_child_uses_subagent_system_prompt(self) -> None:
        from src.outputStyles import resolve_output_style

        style = resolve_output_style("subagent")
        self.assertEqual(style.name, "subagent")
        self.assertIn("subagent", style.prompt.lower())
        self.assertIn("cannot interact with the user", style.prompt.lower())


class TestSubagentOutputTool(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.context = ToolContext(workspace_root=Path(self.temp_dir.name))
        self.tool = SubagentOutputTool()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_unknown_task_id(self) -> None:
        result = self.tool.run({"task_id": "subagent_nope"}, self.context)
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["retrieval_status"], "success")
        self.assertIsNone(result.output["task"])

    def test_not_ready_then_done(self) -> None:
        from src.tool_system.tools import subagent as subagent_mod

        task_id = "subagent_test123"
        event = threading.Event()
        with subagent_mod._BG_LOCK:
            subagent_mod._BG_SUBAGENTS[task_id] = {
                "task_id": task_id,
                "description": "slow",
                "model": "m",
                "status": "running",
                "started_at": 0.0,
                "done_event": event,
                "response_text": "",
                "num_turns": 0,
                "cancelled": False,
                "usage": None,
                "error": None,
            }
        try:
            first = self.tool.run({"task_id": task_id}, self.context)
            self.assertEqual(first.output["retrieval_status"], "not_ready")

            def _finish():
                time.sleep(0.2)
                with subagent_mod._BG_LOCK:
                    rec = subagent_mod._BG_SUBAGENTS[task_id]
                    rec["status"] = "done"
                    rec["response_text"] = "finished"
                    event.set()

            t = threading.Thread(target=_finish, daemon=True)
            t.start()
            second = self.tool.run(
                {"task_id": task_id, "block": True, "timeout": 10.0}, self.context
            )
            self.assertEqual(second.output["retrieval_status"], "success")
            self.assertEqual(second.output["task"]["response_text"], "finished")
        finally:
            with subagent_mod._BG_LOCK:
                subagent_mod._BG_SUBAGENTS.pop(task_id, None)


if __name__ == "__main__":
    unittest.main()
