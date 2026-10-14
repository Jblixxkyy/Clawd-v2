"""Tests for bounded provider-call retry in the agent loop.

Covers: retry on 429 / 5xx / timeouts / connection errors (max 3 attempts,
exponential backoff ~1s/2s/4s), no retry on other 4xx, clean error result
on final failure (no raw traceback), and correct transient classification.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system import agent_loop
from src.tool_system.agent_loop import (
    AgentLoopCancelled,
    AgentLoopResult,
    _is_transient_provider_error,
    run_agent_loop,
)
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


class StatusError(Exception):
    """SDK-style error carrying an HTTP status code."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


class HttpxStyleError(Exception):
    """httpx.HTTPStatusError-style: status lives on .response."""

    def __init__(self, message, status_code):
        super().__init__(message)
        self.response = type("Resp", (), {"status_code": status_code})()


class FlakyProvider:
    """Fake provider whose chat fails a fixed number of times, then succeeds."""

    def __init__(self, failures, error_factory):
        self.failures = list(failures)
        self.error_factory = error_factory
        self.chat_calls = 0
        self.success_response = ChatResponse(
            content="done",
            model="fake-model",
            usage={"input_tokens": 1, "output_tokens": 1},
            finish_reason="stop",
            tool_uses=None,
        )

    def chat_stream_response(self, *args, **kwargs):
        raise NotImplementedError()

    def chat(self, *args, **kwargs):
        self.chat_calls += 1
        if self.failures:
            self.failures.pop(0)
            raise self.error_factory()
        return self.success_response


class TestIsTransientProviderError(unittest.TestCase):
    def test_retryable_statuses(self):
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status):
                self.assertTrue(
                    _is_transient_provider_error(StatusError("boom", status))
                )

    def test_non_retryable_4xx(self):
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertFalse(
                    _is_transient_provider_error(StatusError("bad", status))
                )

    def test_httpx_style_status(self):
        self.assertTrue(_is_transient_provider_error(HttpxStyleError("x", 429)))
        self.assertTrue(_is_transient_provider_error(HttpxStyleError("x", 503)))
        self.assertFalse(_is_transient_provider_error(HttpxStyleError("x", 400)))

    def test_builtin_timeout_and_connection(self):
        self.assertTrue(_is_transient_provider_error(TimeoutError("t")))
        self.assertTrue(_is_transient_provider_error(ConnectionError("c")))

    def test_non_transient_errors(self):
        self.assertFalse(_is_transient_provider_error(ValueError("v")))
        self.assertFalse(_is_transient_provider_error(RuntimeError("r")))

    def test_cancellation_never_retried(self):
        self.assertFalse(_is_transient_provider_error(AgentLoopCancelled("stop")))


class TestProviderRetryInLoop(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, provider):
        conversation = Conversation()
        conversation.add_user_message("hello")
        with patch("src.tool_system.agent_loop.time") as mock_time:
            result = run_agent_loop(
                conversation=conversation,
                provider=provider,
                tool_registry=self.registry,
                tool_context=self.context,
                verbose=False,
            )
        return result, mock_time

    def test_retries_429_then_succeeds(self):
        provider = FlakyProvider(
            failures=[True, True],
            error_factory=lambda: StatusError("rate limited", 429),
        )
        result, mock_time = self._run(provider)
        self.assertIsInstance(result, AgentLoopResult)
        self.assertEqual(result.response_text, "done")
        self.assertEqual(provider.chat_calls, 3)
        delays = [c.args[0] for c in mock_time.sleep.call_args_list]
        self.assertEqual(delays, [1.0, 2.0])

    def test_retries_timeout_then_succeeds(self):
        provider = FlakyProvider(
            failures=[True],
            error_factory=lambda: TimeoutError("timed out"),
        )
        result, _ = self._run(provider)
        self.assertEqual(result.response_text, "done")
        self.assertEqual(provider.chat_calls, 2)

    def test_gives_up_after_max_attempts_with_clean_error(self):
        provider = FlakyProvider(
            failures=[True, True, True, True],
            error_factory=lambda: StatusError("server exploded", 500),
        )
        result, _ = self._run(provider)
        self.assertIsInstance(result, AgentLoopResult)
        self.assertEqual(provider.chat_calls, 3)
        self.assertIn("Provider error after 3 attempts", result.response_text)
        self.assertIn("server exploded", result.response_text)
        # No raw traceback leak in the surfaced message.
        self.assertNotIn("Traceback", result.response_text)

    def test_400_is_not_retried(self):
        provider = FlakyProvider(
            failures=[True, True, True],
            error_factory=lambda: StatusError("bad request", 400),
        )
        result, mock_time = self._run(provider)
        self.assertEqual(provider.chat_calls, 1)
        mock_time.sleep.assert_not_called()
        self.assertIn("Provider error", result.response_text)
        self.assertIn("bad request", result.response_text)

    def test_non_transient_error_is_not_retried(self):
        provider = FlakyProvider(
            failures=[True, True],
            error_factory=lambda: ValueError("caller bug"),
        )
        result, mock_time = self._run(provider)
        self.assertEqual(provider.chat_calls, 1)
        mock_time.sleep.assert_not_called()
        self.assertIn("Provider error", result.response_text)

    def test_cancellation_is_not_retried_and_propagates(self):
        provider = FlakyProvider(
            failures=[True],
            error_factory=lambda: AgentLoopCancelled("stop now"),
        )
        conversation = Conversation()
        conversation.add_user_message("hello")
        with patch("src.tool_system.agent_loop.time"):
            result = run_agent_loop(
                conversation=conversation,
                provider=provider,
                tool_registry=self.registry,
                tool_context=self.context,
                verbose=False,
            )
        self.assertTrue(result.cancelled)
        self.assertEqual(provider.chat_calls, 1)


if __name__ == "__main__":
    unittest.main()
