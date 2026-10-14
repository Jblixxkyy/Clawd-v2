"""Tests for harness self-awareness guards (briefs 035-039).

035: empty-response retry — never show the user an empty bubble.
036: environment capability card — the harness introduces itself.
037: error-class circuit breaker — same tool + same error 3x redirects, then short-circuits.
038: semantic stuck detection — failing strategy trips even without exact repeats.
039: progress nudge — planning nudge after N modification-free tool turns.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system import agent_loop
from src.tool_system.agent_loop import LoopDetector, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.harness_guards import (
    CircuitBreaker,
    build_env_card,
    build_progress_nudge,
    detect_environment,
    is_progress_tool,
    normalize_error_signature,
    should_nudge,
)


def _resp(content="", tool_uses=None):
    return ChatResponse(
        content=content,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 5},
        finish_reason="stop",
        tool_uses=tool_uses,
    )


class TestEmptyResponseRetry(unittest.TestCase):
    def _run(self, responses):
        conv = Conversation()
        conv.add_user_message("hello")
        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.side_effect = responses
        ctx = ToolContext(workspace_root=Path(tempfile.mkdtemp()))
        result = run_agent_loop(
            conversation=conv, provider=provider,
            tool_registry=build_default_registry(), tool_context=ctx,
            verbose=False,
        )
        return result, provider, conv

    def test_empty_then_content_retries_and_succeeds(self):
        result, provider, _ = self._run([_resp("  "), _resp(""), _resp("hi there")])
        self.assertEqual(result.response_text, "hi there")
        self.assertEqual(provider.chat.call_count, 3)

    def test_always_empty_returns_notice_not_blank(self):
        result, provider, _ = self._run([_resp(" "), _resp(""), _resp("\n")])
        self.assertEqual(provider.chat.call_count, 3)
        self.assertIn("empty response", result.response_text)
        self.assertTrue(result.response_text.strip())

    def test_non_empty_first_response_single_call(self):
        result, provider, _ = self._run([_resp("hello!")])
        self.assertEqual(provider.chat.call_count, 1)
        self.assertEqual(result.response_text, "hello!")

    def test_empty_text_with_tool_calls_is_not_empty(self):
        tool_use = {"id": "t1", "name": "Read",
                    "input": {"file_path": "/tmp/x"}}
        # tool call with blank text still dispatches the tool
        result, provider, _ = self._run([
            _resp(" ", tool_uses=[tool_use]),
            _resp("done"),
        ])
        self.assertEqual(provider.chat.call_count, 2)
        self.assertEqual(result.response_text, "done")

    def test_is_empty_response_helper(self):
        self.assertTrue(agent_loop._is_empty_response(_resp("   ")))
        self.assertTrue(agent_loop._is_empty_response(_resp("")))
        self.assertFalse(agent_loop._is_empty_response(_resp("x")))
        self.assertFalse(agent_loop._is_empty_response(
            _resp("", tool_uses=[{"id": "t", "name": "Read", "input": {}}])))


class TestEnvCard(unittest.TestCase):
    def test_detect_environment_shape(self):
        env = detect_environment()
        self.assertIn("os", env)
        self.assertIn("bash_available", env)
        self.assertIn("powershell_available", env)

    def test_card_mentions_missing_bash_on_windows(self):
        fake_env = {"os": "Windows", "is_windows": True,
                    "bash_available": False, "powershell_available": True}
        with patch("src.tool_system.harness_guards.detect_environment",
                   return_value=fake_env):
            card = build_env_card()
        self.assertIn("NOT installed", card)
        self.assertIn("PowerShell", card)
        self.assertIn("Select-String", card)

    def test_card_no_warning_when_bash_present(self):
        fake_env = {"os": "Linux", "is_windows": False,
                    "bash_available": True, "powershell_available": False}
        with patch("src.tool_system.harness_guards.detect_environment",
                   return_value=fake_env):
            card = build_env_card()
        self.assertNotIn("NOT installed", card)
        self.assertIn("Bash IS installed", card)

    def test_card_never_raises(self):
        with patch("src.tool_system.harness_guards.detect_environment",
                   side_effect=RuntimeError("boom")):
            self.assertEqual(build_env_card(), "")


class TestCircuitBreaker(unittest.TestCase):
    def test_trips_after_three_same_errors(self):
        cb = CircuitBreaker()
        err = {"error": "bash executable not found on this system"}
        self.assertIsNone(cb.record("Bash", err, True))
        self.assertIsNone(cb.record("Bash", err, True))
        redirect = cb.record("Bash", err, True)
        self.assertIsNotNone(redirect)
        self.assertIn("Bash", redirect)
        self.assertIn("PowerShell", redirect)  # tailored hint

    def test_short_circuits_after_trip(self):
        cb = CircuitBreaker()
        err = {"error": "bash executable not found"}
        for _ in range(3):
            cb.record("Bash", err, True)
        self.assertIsNotNone(cb.should_short_circuit("Bash"))
        self.assertIsNone(cb.should_short_circuit("PowerShell"))

    def test_success_resets_counts(self):
        cb = CircuitBreaker()
        err = {"error": "bash executable not found"}
        cb.record("Bash", err, True)
        cb.record("Bash", err, True)
        cb.record("Bash", {"ok": 1}, False)  # success
        self.assertIsNone(cb.record("Bash", err, True))
        self.assertIsNone(cb.record("Bash", err, True))
        # third consecutive failure after reset trips again
        self.assertIsNotNone(cb.record("Bash", err, True))

    def test_different_error_signature_separate_count(self):
        cb = CircuitBreaker()
        cb.record("Bash", {"error": "bash executable not found"}, True)
        cb.record("Bash", {"error": "bash executable not found"}, True)
        # different error resets the consecutive chain for the old signature
        self.assertIsNone(cb.record("Bash", {"error": "timeout after 30s"}, True))
        self.assertIsNone(cb.record("Bash", {"error": "timeout after 30s"}, True))

    def test_cooldown_closes_circuit(self):
        cb = CircuitBreaker()
        err = {"error": "bash executable not found"}
        for _ in range(3):
            cb.record("Bash", err, True)
        self.assertIsNotNone(cb.should_short_circuit("Bash"))
        for _ in range(CircuitBreaker.COOLDOWN_TURNS):
            cb.note_turn_without_tool("PowerShell")
        self.assertIsNone(cb.should_short_circuit("Bash"))

    def test_normalize_error_signature_strips_paths(self):
        a = normalize_error_signature({"error": "No such file C:\\a\\b\\c.txt: 123"})
        b = normalize_error_signature({"error": "No such file C:\\x\\y\\z.txt: 456"})
        self.assertEqual(a, b)

    def test_breaker_never_raises(self):
        cb = CircuitBreaker()
        self.assertIsNone(cb.record(None, None, True))
        self.assertIsNone(cb.should_short_circuit(None))


class TestSemanticStuckDetection(unittest.TestCase):
    def _detector_with(self, outcomes):
        d = LoopDetector()
        for name, is_err, mod in outcomes:
            d.record_outcome(name, is_err, mod)
        return d

    def test_flailing_searches_trips(self):
        # Josh's session pattern: searches + shell errors, zero writes.
        outcomes = ([("Grep", False, False)] * 2 + [("Bash", True, False)] * 6
                    + [("PowerShell", True, False)] * 2)
        self.assertTrue(self._detector_with(outcomes).is_semantically_stuck())

    def test_one_write_prevents_trip(self):
        outcomes = [("Bash", True, False)] * 6 + [("Write", False, True)] \
            + [("Read", False, False)] * 3
        self.assertFalse(self._detector_with(outcomes).is_semantically_stuck())

    def test_clean_reads_do_not_trip(self):
        outcomes = [("Read", False, False)] * 10
        self.assertFalse(self._detector_with(outcomes).is_semantically_stuck())

    def test_short_window_does_not_trip(self):
        outcomes = [("Bash", True, False)] * 9
        self.assertFalse(self._detector_with(outcomes).is_semantically_stuck())

    def test_exact_signature_path_still_works(self):
        d = LoopDetector()
        n = 0
        for _ in range(8):
            n = d.record("Read", {"file_path": "/a"}, {"type": "text"})
        self.assertTrue(d.is_stuck(n))


class TestProgressNudge(unittest.TestCase):
    def test_schedule(self):
        self.assertFalse(should_nudge(9))
        self.assertTrue(should_nudge(10))
        self.assertFalse(should_nudge(11))
        self.assertTrue(should_nudge(25))
        self.assertTrue(should_nudge(40))

    def test_nudge_text(self):
        text = build_progress_nudge(10)
        self.assertIn("10 tool calls", text)
        self.assertIn("plan", text)

    def test_is_progress_tool(self):
        self.assertTrue(is_progress_tool("Edit"))
        self.assertTrue(is_progress_tool("Write"))
        self.assertTrue(is_progress_tool("Subagent"))
        self.assertFalse(is_progress_tool("Bash"))
        self.assertFalse(is_progress_tool("Grep"))


class TestLoopIntegration(unittest.TestCase):
    def test_circuit_breaker_redirect_reaches_conversation(self):
        conv = Conversation()
        conv.add_user_message("find the chat method")
        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        # Model keeps calling Bash with failing commands; then gives up with text.
        bad_bash = {"id": "t1", "name": "Bash", "input": {"command": "grep x"}}
        provider.chat.side_effect = [
            _resp("", tool_uses=[dict(bad_bash, id="t1")]),
            _resp("", tool_uses=[dict(bad_bash, id="t2")]),
            _resp("", tool_uses=[dict(bad_bash, id="t3")]),
            _resp("I see bash is unavailable, I'll use another approach."),
        ]
        ctx = ToolContext(workspace_root=Path(tempfile.mkdtemp()))
        with patch("shutil.which", return_value=None):  # no bash anywhere
            result = run_agent_loop(
                conversation=conv, provider=provider,
                tool_registry=build_default_registry(), tool_context=ctx,
                verbose=False,
            )
        texts = [m.content for m in conv.messages
                 if m.role == "user" and isinstance(m.content, str)]
        self.assertTrue(any("[Harness]" in t for t in texts),
                        "circuit-breaker redirect not injected")
        self.assertIn("bash is unavailable", result.response_text)


if __name__ == "__main__":
    unittest.main()
