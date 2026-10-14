"""Brief 037: error-class circuit breaker — agent-loop wiring tests.

Unit behavior of CircuitBreaker itself is covered in
tests/test_harness_guards.py; this file tests the loop integration:
the 3x trip redirect, the 4th-call short-circuit (dispatch not reached,
same error returned), success resetting the count, separate counts per
error signature, and the 5-turn cooldown that closes an open circuit.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.harness_guards import CircuitBreaker
from src.tool_system.protocol import ToolResult
from src.tool_system.registry import ToolRegistry, ToolSpec


BASH_ERR = "bash executable not found"


def _tool_response(name="Bash", tool_input=None, uid="toolu_1"):
    return ChatResponse(
        content="",
        model="test-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="tool_use",
        tool_uses=[{"id": uid, "name": name, "input": tool_input or {}}],
    )


def _done_response(text="all done"):
    return ChatResponse(
        content=text,
        model="test-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="stop",
        tool_uses=None,
    )


class _ScriptedTool:
    """A test tool with scripted outcomes; counts real executions."""

    def __init__(self, name, script):
        # script: list of ("error", text) | ("ok", text), consumed per call.
        self._name = name
        self._script = list(script)
        self.executions = 0

    def spec(self):
        return ToolSpec(
            name=self._name,
            description="scripted test tool",
            input_schema={"type": "object"},
        )

    def run(self, tool_input, context):
        self.executions += 1
        kind, text = self._script.pop(0) if self._script else ("ok", "done")
        if kind == "error":
            return ToolResult(name=self._name, output={"error": text}, is_error=True)
        return ToolResult(name=self._name, output={"result": text}, is_error=False)


class TestCircuitBreakerLoopWiring(unittest.TestCase):
    """End-to-end through run_agent_loop with a scripted failing tool."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.bash = _ScriptedTool("Bash", [("error", BASH_ERR)] * 10)
        self.registry = ToolRegistry([self.bash])
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _provider(self, plan):
        """plan: list of ("tool", name, input) | ("done",)."""
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        state = {"n": 0}

        def chat_side_effect(messages, **kwargs):
            idx = state["n"]
            state["n"] += 1
            step = plan[idx] if idx < len(plan) else ("done",)
            if step[0] == "done":
                return _done_response()
            _, name, tool_input = step
            return _tool_response(name=name, tool_input=tool_input, uid=f"toolu_{idx}")

        mock_provider.chat.side_effect = chat_side_effect
        return mock_provider

    def _run(self, provider):
        events = []
        conversation = Conversation()
        conversation.add_user_message("do the thing")
        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
            on_event=events.append,
        )
        return result, events

    def _user_texts(self, provider):
        texts = []
        for call in provider.chat.call_args_list:
            messages = call.args[0] if call.args else call.kwargs.get("messages", [])
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "user":
                    content = m.get("content")
                    if isinstance(content, str):
                        texts.append(content)
        return texts

    def test_three_identical_errors_inject_redirect(self):
        """Same tool + same error x3 -> plain-English redirect reminder."""
        plan = [("tool", "Bash", {"command": f"try {i}"}) for i in range(4)]
        provider = self._provider(plan)
        result, _events = self._run(provider)

        redirects = [
            t for t in self._user_texts(provider) if "has failed 3x" in t
        ]
        self.assertTrue(redirects, "no 037 redirect reminder injected")
        self.assertIn("Bash", redirects[0])
        self.assertIn("PowerShell", redirects[0])  # tailored bash hint
        # The 029 verdict still fires last on the persisting pattern.
        self.assertTrue(
            result.response_text.startswith("[LoopDetected: tool_error"),
            result.response_text,
        )

    def test_fourth_identical_call_short_circuits(self):
        """4th call: dispatch not reached; the stored error is returned."""
        plan = [("tool", "Bash", {"command": f"try {i}"}) for i in range(4)]
        provider = self._provider(plan)
        _result, events = self._run(provider)

        self.assertEqual(
            self.bash.executions, 3, "dispatch must not run on the 4th call"
        )
        tool_results = [
            e for e in events if e.kind == "tool_result" and e.tool_name == "Bash"
        ]
        self.assertEqual(len(tool_results), 4)
        fourth = tool_results[3]
        self.assertTrue(fourth.is_error)
        self.assertEqual(fourth.tool_output, {"error": BASH_ERR})

    def test_success_resets_count(self):
        """Fail, fail, success, fail, fail (new signature) -> no trip."""
        err_b = "timeout after 30s"
        self.bash = _ScriptedTool(
            "Bash",
            [("error", BASH_ERR), ("error", BASH_ERR), ("ok", "fine"),
             ("error", err_b), ("error", err_b)],
        )
        self.registry = ToolRegistry([self.bash])
        plan = [("tool", "Bash", {"command": f"c{i}"}) for i in range(5)]
        plan.append(("done",))
        provider = self._provider(plan)
        result, _events = self._run(provider)

        self.assertEqual(self.bash.executions, 5)
        self.assertFalse(
            [t for t in self._user_texts(provider) if "has failed 3x" in t],
            "breaker must not trip after a success reset the count",
        )
        self.assertFalse(result.response_text.startswith("[LoopDetected"))

    def test_cooldown_closes_circuit(self):
        """Trip, then 5 turns of other-tool calls -> Bash dispatches again."""
        read = _ScriptedTool("Read", [("ok", "contents")] * 6)
        self.registry = ToolRegistry([self.bash, read])
        plan = (
            [("tool", "Bash", {"command": f"x{i}"}) for i in range(3)]
            + [("tool", "Read", {"file_path": "f"})] * 5
            + [("tool", "Bash", {"command": "y"}), ("done",)]
        )
        provider = self._provider(plan)
        result, _events = self._run(provider)

        # 3 real executions to trip, 1 more after the 5-turn cooldown closed
        # the circuit (the 4th Bash call was NOT short-circuited).
        self.assertEqual(self.bash.executions, 4)
        self.assertFalse(result.response_text.startswith("[LoopDetected"))


class TestCircuitBreakerSignatures(unittest.TestCase):
    """Unit tests for per-signature counting and turn-end aging."""

    def test_different_error_signature_separate_count(self):
        cb = CircuitBreaker()
        a = {"error": "bash executable not found"}
        b = {"error": "timeout after 30s"}
        self.assertIsNone(cb.record("Bash", a, True))
        self.assertIsNone(cb.record("Bash", a, True))
        # A different signature starts its own consecutive chain.
        self.assertIsNone(cb.record("Bash", b, True))
        self.assertIsNone(cb.record("Bash", b, True))
        redirect = cb.record("Bash", b, True)
        self.assertIsNotNone(redirect)
        self.assertIsNotNone(cb.should_short_circuit("Bash"))

    def test_normalization_first_line_lowercased_paths_stripped(self):
        cb = CircuitBreaker()
        e1 = {"error": "No such file C:\\a\\b\\c.txt: 123\nsecond line differs"}
        e2 = {"error": "no such file c:\\x\\y\\z.txt: 456\nanother second line"}
        self.assertIsNone(cb.record("Bash", e1, True))
        self.assertIsNone(cb.record("Bash", e1, True))
        # Same normalized signature despite path/number/case differences.
        self.assertIsNotNone(cb.record("Bash", e2, True))

    def test_note_turn_end_closes_after_five_quiet_turns(self):
        cb = CircuitBreaker()
        err = {"error": "bash executable not found"}
        for _ in range(3):
            cb.record("Bash", err, True)
        self.assertIsNotNone(cb.should_short_circuit("Bash"))
        for _ in range(4):
            cb.note_turn_end({"Read"})
            self.assertIsNotNone(cb.should_short_circuit("Bash"))
        cb.note_turn_end({"Read"})
        self.assertIsNone(cb.should_short_circuit("Bash"))

    def test_note_turn_end_resets_quiet_counter_when_tool_called(self):
        cb = CircuitBreaker()
        err = {"error": "bash executable not found"}
        for _ in range(3):
            cb.record("Bash", err, True)
        for _ in range(4):
            cb.note_turn_end({"Read"})
        cb.note_turn_end({"Bash"})  # called again -> quiet counter resets
        for _ in range(4):
            cb.note_turn_end({"Read"})
        self.assertIsNotNone(cb.should_short_circuit("Bash"))

    def test_note_turn_end_never_raises(self):
        cb = CircuitBreaker()
        cb.note_turn_end(None)
        cb.note_turn_end(set())
        cb.note_turn_end({"Bash"})


if __name__ == "__main__":
    unittest.main()
