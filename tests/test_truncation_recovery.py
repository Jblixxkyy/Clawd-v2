"""Tests for brief 025: truncation recovery (continue responses killed at output-token limits).

Covers the helpers in src/providers/truncation.py, the normalized truncated
flag on ChatResponse, and the per-provider truncation-signal adapters.
"""

from __future__ import annotations

import json
import unittest
from typing import Any, Optional

from src.providers.base import BaseProvider, ChatResponse, MessageInput
from src.providers.truncation import (
    CONTINUED_MARKER,
    CONTINUATION_PROMPT,
    LOG_PREFIX,
    STILL_TRUNCATED_MARKER,
    is_truncated_finish_reason,
    mark_truncated,
    maybe_continue_truncated,
)


class ScriptedProvider(BaseProvider):
    """Provider returning scripted (content, finish_reason) pairs per call."""

    def __init__(self, script: list[tuple[str, str]]):
        super().__init__(api_key="test")
        self.script = script
        self.calls: list[list[MessageInput]] = []

    def chat(
        self,
        messages: list[MessageInput],
        tools: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> ChatResponse:
        self.calls.append(list(messages))
        content, finish_reason = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        response = ChatResponse(
            content=content,
            model="test-model",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason=finish_reason,
        )
        return mark_truncated(response)

    def chat_stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError

    def get_available_models(self) -> list[str]:
        return ["test-model"]


class TestTruncationSignals(unittest.TestCase):
    def test_openai_length_is_truncated(self):
        self.assertTrue(is_truncated_finish_reason("length"))

    def test_anthropic_max_tokens_is_truncated(self):
        self.assertTrue(is_truncated_finish_reason("max_tokens"))

    def test_gemini_max_tokens_case_insensitive(self):
        self.assertTrue(is_truncated_finish_reason("MAX_TOKENS"))

    def test_stop_is_not_truncated(self):
        self.assertFalse(is_truncated_finish_reason("stop"))

    def test_content_filter_is_not_truncated(self):
        self.assertFalse(is_truncated_finish_reason("content_filter"))

    def test_none_and_empty_are_not_truncated(self):
        self.assertFalse(is_truncated_finish_reason(None))
        self.assertFalse(is_truncated_finish_reason(""))

    def test_mark_truncated_sets_flag(self):
        response = ChatResponse(
            content="x", model="m", usage={}, finish_reason="length"
        )
        self.assertFalse(response.truncated)
        mark_truncated(response)
        self.assertTrue(response.truncated)

    def test_chatresponse_defaults(self):
        response = ChatResponse(content="x", model="m", usage={}, finish_reason="stop")
        self.assertFalse(response.truncated)
        self.assertEqual(response.continuations, 0)


class TestAutoContinue(unittest.TestCase):
    def _initial(self, content: str = "Hello ", finish_reason: str = "length") -> ChatResponse:
        response = ChatResponse(
            content=content, model="test-model", usage={}, finish_reason=finish_reason
        )
        return mark_truncated(response)

    def test_auto_continue_and_stitch(self):
        provider = ScriptedProvider([("world", "stop")])
        messages = [{"role": "user", "content": "Say hello"}]
        stitched, log = maybe_continue_truncated(
            self._initial(), provider, messages, config={"truncation_recovery": {"enabled": True, "max_continuations": 3}}
        )
        self.assertEqual(stitched.content, "Hello world")
        self.assertEqual(stitched.continuations, 1)
        self.assertFalse(stitched.truncated)
        self.assertEqual(stitched.finish_reason, "stop")
        # initial response was provided directly; one continuation call
        self.assertEqual(len(provider.calls), 1)

    def test_continuation_turn_appends_partial_and_prompt(self):
        provider = ScriptedProvider([("b", "stop")])
        messages = [{"role": "user", "content": "Go"}]
        maybe_continue_truncated(self._initial("a"), provider, messages, config={})
        continuation_messages = provider.calls[0]
        roles = [m["role"] for m in continuation_messages]
        self.assertEqual(roles[-2:], ["assistant", "user"])
        self.assertEqual(continuation_messages[-2]["content"], "a")
        self.assertEqual(continuation_messages[-1]["content"], CONTINUATION_PROMPT)

    def test_multiple_continuations_stitch_in_order(self):
        provider = ScriptedProvider([("two-", "length"), ("three", "stop")])
        stitched, _ = maybe_continue_truncated(self._initial("one-"), provider, [{"role": "user", "content": "x"}], config={})
        self.assertEqual(stitched.content, "one-two-three")
        self.assertEqual(stitched.continuations, 2)
        self.assertFalse(stitched.truncated)

    def test_stops_after_max_continuations(self):
        provider = ScriptedProvider([("part", "length")] * 10)
        config = {"truncation_recovery": {"enabled": True, "max_continuations": 3}}
        stitched, log = maybe_continue_truncated(
            self._initial(), provider, [{"role": "user", "content": "x"}], config=config
        )
        # initial response is provided directly; exactly 3 continuation calls: no infinite loop
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(stitched.continuations, 3)
        self.assertTrue(stitched.truncated)
        joined = "\n".join(log)
        self.assertIn(STILL_TRUNCATED_MARKER, joined)

    def test_disabled_config_does_not_continue(self):
        provider = ScriptedProvider([("world", "stop")])
        config = {"truncation_recovery": {"enabled": False}}
        response = self._initial()
        stitched, _ = maybe_continue_truncated(response, provider, [{"role": "user", "content": "x"}], config=config)
        self.assertEqual(len(provider.calls), 0)
        self.assertIs(stitched, response)
        self.assertEqual(stitched.content, "Hello ")

    def test_not_truncated_is_noop(self):
        provider = ScriptedProvider([("ignored", "stop")])
        response = ChatResponse(content="done", model="m", usage={}, finish_reason="stop")
        stitched, log = maybe_continue_truncated(response, provider, [{"role": "user", "content": "x"}], config={})
        self.assertIs(stitched, response)
        self.assertEqual(log, [])
        self.assertEqual(provider.calls, [])

    def test_continuation_error_keeps_partial(self):
        class FailingProvider(ScriptedProvider):
            def chat(self, messages, tools=None, **kwargs):
                raise RuntimeError("boom")

        provider = FailingProvider([("Hello ", "length")])
        stitched, log = maybe_continue_truncated(
            self._initial(), provider, [{"role": "user", "content": "x"}], config={}
        )
        self.assertEqual(stitched.content, "Hello ")
        self.assertIn("failed", "\n".join(log).lower())


class TestContinuationMarker(unittest.TestCase):
    def test_continued_marker_present_in_log(self):
        provider = ScriptedProvider([("world", "stop")])
        response = ChatResponse(content="Hello ", model="m", usage={}, finish_reason="length")
        mark_truncated(response)
        _, log = maybe_continue_truncated(response, provider, [{"role": "user", "content": "x"}], config={})
        joined = "\n".join(log)
        self.assertIn(LOG_PREFIX, joined)
        self.assertIn(CONTINUED_MARKER, joined)

    def test_marker_detectable_from_response(self):
        provider = ScriptedProvider([("world", "stop")])
        response = ChatResponse(content="Hello ", model="m", usage={}, finish_reason="length")
        mark_truncated(response)
        stitched, _ = maybe_continue_truncated(response, provider, [{"role": "user", "content": "x"}], config={})
        # UI can render the "continued after truncation" marker from this:
        self.assertGreater(stitched.continuations, 0)
        self.assertFalse(stitched.truncated)

    def test_usage_merged_across_continuations(self):
        provider = ScriptedProvider([("b", "stop")])
        response = ChatResponse(
            content="a", model="m",
            usage={"input_tokens": 10, "output_tokens": 5},
            finish_reason="length",
        )
        mark_truncated(response)
        stitched, _ = maybe_continue_truncated(response, provider, [{"role": "user", "content": "x"}], config={})
        self.assertEqual(stitched.usage["output_tokens"], 10)
        self.assertEqual(stitched.usage["input_tokens"], 20)


class TestMidToolCallRecovery(unittest.TestCase):
    """Truncation mid-tool-call JSON is recovered via continuation before any
    parse-failure handling (composes with 007 input repair)."""

    def test_stitched_json_parses_after_continuation(self):
        # A tool call whose JSON arguments were cut off at the output limit.
        part1 = '{"name": "bash", "input": {"command": "ls /tm'
        part2 = 'p"}}'
        provider = ScriptedProvider([(part2, "stop")])
        response = ChatResponse(content=part1, model="m", usage={}, finish_reason="length")
        mark_truncated(response)
        stitched, log = maybe_continue_truncated(
            response, provider, [{"role": "user", "content": "list tmp"}], config={}
        )
        # The stitched whole parses cleanly: no parse failure ever fires.
        parsed = json.loads(stitched.content)
        self.assertEqual(parsed["name"], "bash")
        self.assertEqual(parsed["input"]["command"], "ls /tmp")
        self.assertIn(CONTINUED_MARKER, "\n".join(log))

    def test_partial_tool_use_arguments_completed(self):
        # Simulates the OpenAI adapter: truncated turn yields an entry with
        # empty input; the continuation turn carries the full version (same id).
        first = ChatResponse(
            content="",
            model="m",
            usage={},
            finish_reason="length",
            tool_uses=[{"id": "tu1", "name": "bash", "input": {}}],
        )
        mark_truncated(first)

        class ToolUseProvider(ScriptedProvider):
            def chat(self, messages, tools=None, **kwargs):
                self.calls.append(list(messages))
                continuation = ChatResponse(
                    content="",
                    model="m",
                    usage={},
                    finish_reason="stop",
                    tool_uses=[{"id": "tu1", "name": "bash", "input": {"command": "ls /tmp"}}],
                )
                return mark_truncated(continuation)

        provider = ToolUseProvider([("", "length")])
        stitched, _ = maybe_continue_truncated(
            first, provider, [{"role": "user", "content": "x"}], config={}
        )
        self.assertFalse(stitched.truncated)
        self.assertEqual(len(stitched.tool_uses), 1)
        self.assertEqual(stitched.tool_uses[0]["input"], {"command": "ls /tmp"})


class TestConfigDefaults(unittest.TestCase):
    def test_missing_config_section_uses_defaults(self):
        provider = ScriptedProvider([("b", "length"), ("c", "length"), ("d", "stop")])
        response = ChatResponse(content="a", model="m", usage={}, finish_reason="length")
        mark_truncated(response)
        stitched, _ = maybe_continue_truncated(response, provider, [{"role": "user", "content": "x"}], config=None)
        # default max_continuations = 3
        self.assertEqual(stitched.continuations, 3)

    def test_zero_max_continuations(self):
        provider = ScriptedProvider([("b", "stop")])
        response = ChatResponse(content="a", model="m", usage={}, finish_reason="length")
        mark_truncated(response)
        stitched, _ = maybe_continue_truncated(
            response,
            provider,
            [{"role": "user", "content": "x"}],
            config={"truncation_recovery": {"max_continuations": 0}},
        )
        self.assertEqual(provider.calls, [])
        self.assertEqual(stitched.continuations, 0)


if __name__ == "__main__":
    unittest.main()
