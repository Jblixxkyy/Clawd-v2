"""Tests for prompt caching + cost tracking (Q5/Q6)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.cost_tracker import CostTracker
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import ChatResponse
from src.providers.openai_compatible import OpenAICompatibleProvider
from src.repl.status_bar import render_status
from src.tool_system.agent_loop import _provider_pricing_key, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry

from pathlib import Path


# ----------------------------------------------------------------------
# Anthropic prompt caching
# ----------------------------------------------------------------------

def _anthropic_client():
    """Mocked anthropic.Anthropic client with a canned messages.create."""
    client = MagicMock()
    message = MagicMock()
    block = MagicMock()
    block.type = "text"
    block.text = "hello"
    message.content = [block]
    message.model = "claude-sonnet-4-6"
    message.stop_reason = "end_turn"
    usage = MagicMock()
    usage.input_tokens = 100
    usage.output_tokens = 10
    usage.cache_creation_input_tokens = 40
    usage.cache_read_input_tokens = 60
    message.usage = usage
    client.messages.create.return_value = message
    return client


class TestAnthropicCacheControl(unittest.TestCase):
    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_adds_cache_breakpoints(self, mock_anthropic):
        client = _anthropic_client()
        mock_anthropic.return_value = client
        provider = AnthropicProvider(api_key="k")
        tools = [
            {"name": "a", "description": "a", "input_schema": {"type": "object"}},
            {"name": "b", "description": "b", "input_schema": {"type": "object"}},
        ]
        provider.chat(
            [{"role": "user", "content": "hi"}],
            tools=tools,
            system="static system prompt",
        )
        _, kwargs = client.messages.create.call_args
        # system block carries the breakpoint
        self.assertEqual(
            kwargs["system"],
            [{
                "type": "text",
                "text": "static system prompt",
                "cache_control": {"type": "ephemeral"},
            }],
        )
        # last tool definition carries the breakpoint
        sent_tools = kwargs["tools"]
        self.assertEqual(len(sent_tools), 2)
        self.assertEqual(
            sent_tools[-1]["cache_control"], {"type": "ephemeral"}
        )
        self.assertNotIn("cache_control", sent_tools[0])
        # caller-owned tools not mutated
        self.assertNotIn("cache_control", tools[-1])
        # stable prefix: system/tools before messages
        keys = list(kwargs.keys())
        self.assertLess(keys.index("system"), keys.index("messages"))
        self.assertLess(keys.index("tools"), keys.index("messages"))

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_no_system_no_tools(self, mock_anthropic):
        client = _anthropic_client()
        mock_anthropic.return_value = client
        provider = AnthropicProvider(api_key="k")
        provider.chat([{"role": "user", "content": "hi"}])
        _, kwargs = client.messages.create.call_args
        self.assertNotIn("system", kwargs)
        self.assertNotIn("tools", kwargs)

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_usage_captures_cache_fields(self, mock_anthropic):
        client = _anthropic_client()
        mock_anthropic.return_value = client
        provider = AnthropicProvider(api_key="k")
        resp = provider.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(resp.usage["cache_creation_input_tokens"], 40)
        self.assertEqual(resp.usage["cache_read_input_tokens"], 60)

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_stream_adds_cache_breakpoints(self, mock_anthropic):
        client = _anthropic_client()
        stream = MagicMock()
        stream.text_stream = ["hi"]
        client.messages.stream.return_value.__enter__.return_value = stream
        mock_anthropic.return_value = client
        provider = AnthropicProvider(api_key="k")
        tools = [{"name": "a", "description": "a", "input_schema": {"type": "object"}}]
        chunks = list(provider.chat_stream(
            [{"role": "user", "content": "hi"}], tools=tools, system="sys"
        ))
        self.assertEqual(chunks, ["hi"])
        _, kwargs = client.messages.stream.call_args
        self.assertEqual(
            kwargs["system"][0]["cache_control"], {"type": "ephemeral"}
        )
        self.assertEqual(
            kwargs["tools"][-1]["cache_control"], {"type": "ephemeral"}
        )
        keys = list(kwargs.keys())
        self.assertLess(keys.index("system"), keys.index("messages"))

    @patch("src.providers.anthropic_provider.anthropic.Anthropic")
    def test_chat_stream_response_cache_and_usage(self, mock_anthropic):
        client = _anthropic_client()
        stream = MagicMock()
        stream.text_stream = ["hel", "lo"]
        final_message = MagicMock()
        block = MagicMock()
        block.type = "text"
        block.text = "hello"
        final_message.content = [block]
        final_message.model = "claude-sonnet-4-6"
        final_message.stop_reason = "end_turn"
        usage = MagicMock()
        usage.input_tokens = 50
        usage.output_tokens = 5
        usage.cache_creation_input_tokens = 10
        usage.cache_read_input_tokens = 20
        final_message.usage = usage
        stream.get_final_message.return_value = final_message
        client.messages.stream.return_value.__enter__.return_value = stream
        mock_anthropic.return_value = client
        provider = AnthropicProvider(api_key="k")
        resp = provider.chat_stream_response(
            [{"role": "user", "content": "hi"}], system="sys"
        )
        self.assertEqual(resp.content, "hello")
        self.assertEqual(resp.usage["cache_read_input_tokens"], 20)
        _, kwargs = client.messages.stream.call_args
        self.assertEqual(
            kwargs["system"][0]["cache_control"], {"type": "ephemeral"}
        )


# ----------------------------------------------------------------------
# OpenAI-compatible: cached tokens + prompt_cache_key
# ----------------------------------------------------------------------

class _ConcreteProvider(OpenAICompatibleProvider):
    def __init__(self, client):
        super().__init__(api_key="k", model="test-model")
        self._client = client

    def _create_client(self):
        return self._client

    def get_available_models(self):
        return ["test-model"]


def _openai_response(prompt_tokens=100, completion_tokens=10, cached=None,
                     deepseek_hit=None):
    response = MagicMock()
    choice = MagicMock()
    choice.message.content = "done"
    choice.message.reasoning_content = None
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    response.choices = [choice]
    response.model = "test-model"
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    if cached is not None:
        details = MagicMock()
        details.cached_tokens = cached
        usage.prompt_tokens_details = details
    else:
        usage.prompt_tokens_details = None
    if deepseek_hit is not None:
        usage.prompt_cache_hit_tokens = deepseek_hit
    response.usage = usage
    return response


class TestOpenAICompatibleCaching(unittest.TestCase):
    def test_usage_captures_openai_cached_tokens(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _openai_response(cached=77)
        provider = _ConcreteProvider(client)
        resp = provider.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(resp.usage["cached_tokens"], 77)
        self.assertEqual(resp.usage["input_tokens"], 100)

    def test_usage_captures_deepseek_style_cache_hits(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _openai_response(
            deepseek_hit=33
        )
        provider = _ConcreteProvider(client)
        resp = provider.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(resp.usage["cached_tokens"], 33)

    def test_usage_backward_compatible_without_cache_fields(self):
        client = MagicMock()
        usage = MagicMock(spec=["prompt_tokens", "completion_tokens", "total_tokens"])
        usage.prompt_tokens = 5
        usage.completion_tokens = 6
        usage.total_tokens = 11
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = "x"
        choice.message.reasoning_content = None
        choice.message.tool_calls = None
        choice.finish_reason = "stop"
        response.choices = [choice]
        response.model = "test-model"
        response.usage = usage
        client.chat.completions.create.return_value = response
        provider = _ConcreteProvider(client)
        resp = provider.chat([{"role": "user", "content": "hi"}])
        self.assertEqual(resp.usage["cached_tokens"], 0)
        self.assertEqual(resp.usage["input_tokens"], 5)

    def test_prompt_cache_key_forwarded_when_given(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _openai_response()
        provider = _ConcreteProvider(client)
        provider.chat(
            [{"role": "user", "content": "hi"}], prompt_cache_key="session-abc"
        )
        _, kwargs = client.chat.completions.create.call_args
        self.assertEqual(kwargs["prompt_cache_key"], "session-abc")

    def test_prompt_cache_key_absent_by_default(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _openai_response()
        provider = _ConcreteProvider(client)
        provider.chat([{"role": "user", "content": "hi"}])
        _, kwargs = client.chat.completions.create.call_args
        self.assertNotIn("prompt_cache_key", kwargs)


# ----------------------------------------------------------------------
# CostTracker pricing math
# ----------------------------------------------------------------------

class TestCostTracker(unittest.TestCase):
    def test_legacy_record_api(self):
        t = CostTracker()
        t.record("turn_1_tokens", 100)
        t.record("turn_2_tokens", 50)
        self.assertEqual(t.total_units, 150)
        self.assertEqual(t.events, ["turn_1_tokens:100", "turn_2_tokens:50"])

    def test_record_turn_basic_pricing(self):
        t = CostTracker()
        # anthropic sonnet default: $3 in / $15 out per MTok
        cost = t.record_turn("anthropic", "claude-sonnet-4-6", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 18.0)
        self.assertAlmostEqual(t.cost_usd, 18.0)
        self.assertAlmostEqual(t.session_cost_usd, 18.0)
        self.assertAlmostEqual(t.last_turn_cost_usd, 18.0)
        self.assertEqual(t.total_units, 2_000_000)
        self.assertEqual(len(t.turns), 1)
        self.assertEqual(t.turns[0]["provider"], "anthropic")

    def test_record_turn_cached_input_pricing(self):
        t = CostTracker()
        # 1M input, all cache-read: billed at cached rate only
        cost = t.record_turn("anthropic", "claude-sonnet-4-6", 1_000_000, 0,
                             cached_tokens=1_000_000)
        self.assertAlmostEqual(cost, 0.30)
        self.assertAlmostEqual(t.cost_usd, 0.30)

    def test_record_turn_cache_creation_pricing(self):
        t = CostTracker()
        # cache writes bill at 1.25x the input rate
        cost = t.record_turn("anthropic", "claude-sonnet-4-6", 1_000_000, 0,
                             cache_creation_tokens=1_000_000)
        self.assertAlmostEqual(cost, 3.75)

    def test_record_turn_accumulates(self):
        t = CostTracker()
        t.record_turn("anthropic", "claude-sonnet-4-6", 1_000_000, 0)
        t.record_turn("anthropic", "claude-sonnet-4-6", 0, 1_000_000)
        self.assertAlmostEqual(t.cost_usd, 18.0)
        self.assertAlmostEqual(t.last_turn_cost_usd, 15.0)
        self.assertEqual(len(t.turns), 2)

    def test_model_prefix_override_beats_provider_default(self):
        t = CostTracker()
        self.assertEqual(
            t.price_for("anthropic", "claude-opus-4-6"), (15.0, 75.0, 1.5)
        )
        self.assertEqual(
            t.price_for("openai", "nemotron-3.5-lightning-30b"), (0.6, 1.2, 0.06)
        )

    def test_unknown_provider_falls_back_to_default(self):
        t = CostTracker()
        self.assertEqual(t.price_for("weird", "weird-model"), (1.0, 2.0, 0.1))

    def test_set_prices_override(self):
        t = CostTracker()
        t.set_prices({
            "anthropic:claude-opus-4-6": {"input": 15.0, "output": 75.0, "cached": 1.5},
            "glm": [0.1, 0.2, 0.05],
        })
        self.assertEqual(
            t.price_for("anthropic", "claude-opus-4-6"), (15.0, 75.0, 1.5)
        )
        self.assertEqual(t.price_for("glm", "glm-4"), (0.1, 0.2, 0.05))
        # other providers unaffected
        self.assertEqual(t.price_for("openai", "gpt-4o"), (2.5, 10.0, 1.25))

    def test_config_file_override(self):
        with tempfile.TemporaryDirectory() as home:
            cfg = Path(home) / ".clawd"
            cfg.mkdir()
            (cfg / "config.json").write_text(json.dumps({
                "pricing": {
                    "anthropic": {"input": 9.0, "output": 9.0, "cached": 0.9},
                }
            }))
            with patch.dict(os.environ, {"HOME": home}):
                t = CostTracker()
                # provider-level override applies to a model without a
                # built-in prefix entry
                self.assertEqual(
                    t.price_for("anthropic", "claude-3-opus-20240229"),
                    (9.0, 9.0, 0.9),
                )

    def test_never_raises_on_junk(self):
        t = CostTracker()
        self.assertEqual(t.cost_for(None, None, None, "junk"), 0.0)
        self.assertEqual(
            t.record_turn(None, None, "x", -5, cached_tokens=None), 0.0
        )
        self.assertEqual(t.price_for(None, None), (1.0, 2.0, 0.1))


# ----------------------------------------------------------------------
# Status bar cost rendering
# ----------------------------------------------------------------------

class TestStatusBarCost(unittest.TestCase):
    def test_cost_lights_up(self):
        tracker = CostTracker()
        tracker.record_turn("anthropic", "claude-sonnet-4-6", 1_000_000, 0)
        repl = SimpleNamespace(
            provider_name="anthropic",
            provider=SimpleNamespace(model="claude-sonnet-4-6"),
            cost_tracker=tracker,
            tool_context=None,
        )
        out = render_status(repl)
        self.assertIn("$3.0000", out)
        self.assertIn("1.0M tokens", out)

    def test_no_cost_without_numeric(self):
        repl = SimpleNamespace(
            provider_name="anthropic",
            provider=SimpleNamespace(model="m"),
            cost_tracker=SimpleNamespace(total_units=10),
            tool_context=None,
        )
        out = render_status(repl)
        self.assertNotIn("$", out)

    def test_never_raises(self):
        self.assertEqual(render_status(None), "❯ · ? tokens")
        self.assertEqual(render_status(object()), "❯ · ? tokens")
        bad = SimpleNamespace(
            provider_name="x",
            provider=SimpleNamespace(model="y"),
            cost_tracker=SimpleNamespace(total_units="junk", cost_usd="junk"),
            tool_context=None,
        )
        self.assertIsInstance(render_status(bad), str)


# ----------------------------------------------------------------------
# Agent-loop wiring
# ----------------------------------------------------------------------

class TestAgentLoopCostWiring(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, cost_tracker, usage):
        conversation = Conversation()
        conversation.add_user_message("Say hi.")
        mock_provider = MagicMock()
        mock_provider.chat_stream_response.side_effect = NotImplementedError()
        mock_provider.chat.return_value = ChatResponse(
            content="hi",
            model="test-model",
            usage=usage,
            finish_reason="stop",
            tool_uses=None,
        )
        result = run_agent_loop(
            conversation=conversation,
            provider=mock_provider,
            tool_registry=self.registry,
            tool_context=self.context,
            max_turns=1,
            verbose=False,
            cost_tracker=cost_tracker,
        )
        return result

    def test_per_turn_recording_with_provider_and_model(self):
        tracker = CostTracker()
        result = self._run(tracker, {
            "input_tokens": 1000,
            "output_tokens": 100,
            "cache_read_input_tokens": 400,
            "cache_creation_input_tokens": 200,
        })
        self.assertEqual(len(tracker.turns), 1)
        turn = tracker.turns[0]
        self.assertEqual(turn["input_tokens"], 1000)
        self.assertEqual(turn["output_tokens"], 100)
        self.assertEqual(turn["cached_tokens"], 400)
        self.assertEqual(turn["cache_creation_tokens"], 200)
        self.assertGreater(tracker.cost_usd, 0)
        # accumulated usage carries cached tokens to the caller
        self.assertEqual(result.usage["cached_tokens"], 400)

    def test_openai_style_cached_tokens(self):
        tracker = CostTracker()
        self._run(tracker, {
            "input_tokens": 1000,
            "output_tokens": 100,
            "cached_tokens": 250,
        })
        self.assertEqual(tracker.turns[0]["cached_tokens"], 250)

    def test_loop_survives_broken_tracker(self):
        class Broken:
            def record_turn(self, **kwargs):
                raise RuntimeError("boom")

        result = self._run(Broken(), {"input_tokens": 10, "output_tokens": 5})
        self.assertEqual(result.response_text, "hi")

    def test_loop_survives_tracker_without_record_turn(self):
        result = self._run(object(), {"input_tokens": 10, "output_tokens": 5})
        self.assertEqual(result.response_text, "hi")

    def test_no_tracker_still_accumulates_cached_usage(self):
        result = self._run(None, {
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_tokens": 3,
        })
        self.assertEqual(result.usage["cached_tokens"], 3)


class TestProviderPricingKey(unittest.TestCase):
    def test_known_classes(self):
        self.assertEqual(
            _provider_pricing_key(AnthropicProvider(api_key="x")), "anthropic"
        )

    def test_openai_compatible_subclasses_map_to_openai(self):
        from src.providers.glm_provider import GLMProvider
        from src.providers.openai_provider import OpenAIProvider

        self.assertEqual(_provider_pricing_key(GLMProvider(api_key="x")), "glm")
        self.assertEqual(
            _provider_pricing_key(OpenAIProvider(api_key="x")), "openai"
        )

    def test_unknown_never_raises_and_prices_at_default_tier(self):
        key = _provider_pricing_key(None)
        self.assertIsInstance(key, str)
        self.assertEqual(
            CostTracker().price_for(key, "some-model"), (1.0, 2.0, 0.1)
        )


if __name__ == "__main__":
    unittest.main()
