"""Tests for reasoning budgets (brief 027).

Portable reasoning{enabled, effort, budget_tokens} resolved per call site,
mapped to native provider options where they exist, nothing sent where the
model advertises no reasoning control, native user options win, and a
Clawd-native forced-reflection fallback for no-knob providers.

Uses stub providers and config dicts -- no network, no real config file.
"""

from __future__ import annotations

import unittest
from typing import Any, Optional
from unittest.mock import patch

from src.providers.base import BaseProvider, ChatResponse
from src.providers.reasoning import (
    apply_reasoning_kwargs,
    may_reason,
    native_options,
    register_reasoning_capable,
)
from src.reasoning_budgets import (
    CONFIG_KEY,
    EFFORT_HIGH,
    EFFORT_LOW,
    EFFORT_MAX,
    EFFORT_MEDIUM,
    LOG_PREFIX,
    PHASE_EXECUTION,
    PHASE_PLANNING,
    REFLECTION_PROMPT,
    ROLE_CLASSIFIER,
    ROLE_INVESTIGATOR,
    ROLE_SESSION_TITLE,
    ReasoningBudget,
    effort_command_call,
    fallback_turn_count,
    resolve_budget,
    run_fallback_reflection,
    should_emulate,
)


class StubProvider(BaseProvider):
    """Records chat kwargs; returns a canned response."""

    def __init__(self, **kw):
        super().__init__(api_key="x", **kw)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": messages, "tools": tools, **kwargs})
        return ChatResponse(
            content="stub reply",
            model=kwargs.get("model", "stub"),
            usage={},
            finish_reason="stop",
        )

    def chat_stream(self, messages, tools=None, **kwargs):
        yield "stub"

    def get_available_models(self):
        return ["stub"]


class TestMayReason(unittest.TestCase):
    def test_reasoning_model_advertises(self):
        self.assertTrue(may_reason("claude-sonnet-4-6", "anthropic"))
        self.assertTrue(may_reason("claude-opus-4-5", "anthropic"))
        self.assertTrue(may_reason("gpt-5.4", "openai"))
        self.assertTrue(may_reason("glm-4.5", "glm"))

    def test_non_reasoning_model_skips_cleanly(self):
        self.assertFalse(may_reason("gpt-4o", "openai"))
        self.assertFalse(may_reason("gpt-3.5-turbo", "openai"))
        self.assertFalse(may_reason("claude-3-5-sonnet-20241022", "anthropic"))
        self.assertFalse(may_reason("some-unknown-model", "minimax"))
        self.assertFalse(may_reason(None, "anthropic"))
        self.assertFalse(may_reason("gpt-5.4", None))
        self.assertFalse(may_reason("gpt-5.4", "definitely-not-a-provider"))


class TestNativeAdapter(unittest.TestCase):
    def test_portable_high_on_reasoning_model_sends_native_budget(self):
        budget = ReasoningBudget(enabled=True, effort=EFFORT_HIGH)
        kwargs = apply_reasoning_kwargs(
            {"model": "claude-sonnet-4-6"},
            "anthropic",
            "claude-sonnet-4-6",
            budget,
        )
        self.assertIn("thinking", kwargs)
        self.assertEqual(kwargs["thinking"]["type"], "enabled")
        self.assertGreaterEqual(kwargs["thinking"]["budget_tokens"], 1024)
        # input kwargs not mutated
        self.assertNotIn("thinking", {"model": "claude-sonnet-4-6"})

    def test_openai_maps_effort(self):
        budget = ReasoningBudget(enabled=True, effort=EFFORT_MEDIUM)
        kwargs = apply_reasoning_kwargs({"model": "gpt-5.4"}, "openai", "gpt-5.4", budget)
        self.assertEqual(kwargs.get("reasoning"), {"effort": "medium"})

    def test_non_reasoning_model_sends_nothing_no_400(self):
        budget = ReasoningBudget(enabled=True, effort=EFFORT_HIGH)
        kwargs = apply_reasoning_kwargs(
            {"model": "gpt-4o", "temperature": 0.5},
            "openai",
            "gpt-4o",
            budget,
        )
        # No reasoning keys injected; unrelated kwargs untouched.
        self.assertEqual(kwargs, {"model": "gpt-4o", "temperature": 0.5})

    def test_disabled_budget_sends_nothing(self):
        budget = ReasoningBudget(enabled=False, effort=EFFORT_MAX)
        kwargs = apply_reasoning_kwargs(
            {"model": "claude-sonnet-4-6"}, "anthropic", "claude-sonnet-4-6", budget
        )
        self.assertNotIn("thinking", kwargs)

    def test_native_user_option_wins_over_portable(self):
        budget = ReasoningBudget(enabled=True, effort=EFFORT_LOW)
        user = {
            "model": "claude-sonnet-4-6",
            "thinking": {"type": "enabled", "budget_tokens": 9999},
        }
        kwargs = apply_reasoning_kwargs(
            dict(user), "anthropic", "claude-sonnet-4-6", budget
        )
        # Portable low would have been 1024; the native 9999 survives.
        self.assertEqual(kwargs["thinking"]["budget_tokens"], 9999)

    def test_native_options_empty_when_user_set_reasoning(self):
        budget = ReasoningBudget(enabled=True, effort=EFFORT_HIGH)
        opts = native_options(
            "openai",
            "gpt-5.4",
            budget,
            user_kwargs={"reasoning": {"effort": "low"}},
        )
        self.assertEqual(opts, {})

    def test_budget_tokens_override_effort_default(self):
        budget = ReasoningBudget(
            enabled=True, effort=EFFORT_LOW, budget_tokens=8000
        )
        kwargs = apply_reasoning_kwargs(
            {"model": "claude-sonnet-4-6"},
            "anthropic",
            "claude-sonnet-4-6",
            budget,
        )
        self.assertEqual(kwargs["thinking"]["budget_tokens"], 8000)

    def test_never_raises(self):
        # Garbage in -> {} out, never an exception.
        self.assertEqual(native_options("anthropic", "claude-sonnet-4-6", object()), {})
        self.assertEqual(apply_reasoning_kwargs(None, None, None, None), {})


class TestPolicy(unittest.TestCase):
    def test_planning_budget_larger_than_execution(self):
        plan = resolve_budget({}, phase=PHASE_PLANNING)
        execute = resolve_budget({}, phase=PHASE_EXECUTION)
        self.assertTrue(plan.enabled)
        self.assertTrue(execute.enabled)
        self.assertGreater(plan.rank(), execute.rank())
        self.assertEqual(plan.effort, EFFORT_HIGH)
        self.assertEqual(execute.effort, EFFORT_MEDIUM)

    def test_role_defaults(self):
        self.assertEqual(resolve_budget({}, role=ROLE_INVESTIGATOR).effort, EFFORT_HIGH)
        self.assertEqual(resolve_budget({}, role=ROLE_CLASSIFIER).effort, EFFORT_LOW)
        title = resolve_budget({}, role=ROLE_SESSION_TITLE)
        self.assertFalse(title.enabled)

    def test_role_beats_phase(self):
        budget = resolve_budget({}, phase=PHASE_EXECUTION, role=ROLE_INVESTIGATOR)
        self.assertEqual(budget.effort, EFFORT_HIGH)
        self.assertEqual(budget.source, "role")

    def test_agent_override_wins(self):
        config = {CONFIG_KEY: {"agents": {"my-agent": "low"}}}
        budget = resolve_budget(config, phase=PHASE_PLANNING, agent="my-agent")
        self.assertEqual(budget.effort, EFFORT_LOW)
        self.assertEqual(budget.source, "agent")

    def test_agent_off_disables(self):
        config = {CONFIG_KEY: {"agents": {"quiet": "off"}}}
        budget = resolve_budget(config, phase=PHASE_PLANNING, agent="quiet")
        self.assertFalse(budget.enabled)

    def test_user_phase_override(self):
        config = {CONFIG_KEY: {"phases": {"execution": "high"}}}
        budget = resolve_budget(config, phase=PHASE_EXECUTION)
        self.assertEqual(budget.effort, EFFORT_HIGH)

    def test_global_disable(self):
        config = {CONFIG_KEY: {"enabled": False}}
        self.assertFalse(resolve_budget(config, phase=PHASE_PLANNING).enabled)

    def test_unknown_effort_falls_back_to_medium(self):
        config = {CONFIG_KEY: {"default_effort": "turbo"}}
        budget = resolve_budget(config)
        self.assertEqual(budget.effort, EFFORT_MEDIUM)


class TestFallback(unittest.TestCase):
    def test_fallback_turn_counts(self):
        self.assertEqual(fallback_turn_count(ReasoningBudget(True, EFFORT_LOW)), 0)
        self.assertEqual(fallback_turn_count(ReasoningBudget(True, EFFORT_MEDIUM)), 1)
        self.assertEqual(fallback_turn_count(ReasoningBudget(True, EFFORT_HIGH)), 2)
        self.assertEqual(fallback_turn_count(ReasoningBudget(True, EFFORT_MAX)), 3)
        self.assertEqual(fallback_turn_count(ReasoningBudget(False, EFFORT_MAX)), 0)

    def test_should_emulate_only_no_knob(self):
        high = ReasoningBudget(True, EFFORT_HIGH)
        # Reasoning model -> native knob exists -> no emulation.
        self.assertFalse(should_emulate("anthropic", "claude-sonnet-4-6", high))
        # Non-reasoning model -> emulate.
        self.assertFalse(should_emulate("openai", "gpt-5.4", high))  # has knob
        self.assertTrue(should_emulate("openai", "gpt-4o", high))
        self.assertTrue(should_emulate("minimax", "minimax-m2", high))
        # Low effort -> no turns -> no emulation even without a knob.
        self.assertFalse(
            should_emulate("minimax", "minimax-m2", ReasoningBudget(True, EFFORT_LOW))
        )

    def test_no_knob_high_effort_forces_think_turns(self):
        provider = StubProvider()
        budget = ReasoningBudget(enabled=True, effort=EFFORT_HIGH)
        entries, log = run_fallback_reflection(
            provider,
            [{"role": "user", "content": "do the thing"}],
            budget,
            config={CONFIG_KEY: {"use_think_tool": False}},
        )
        # 2 forced turns = 4 transcript entries (prompt + reply each).
        self.assertEqual(len(entries), 4)
        self.assertTrue(all(e["role"] == "user" for e in entries[::2]))
        self.assertTrue(all(REFLECTION_PROMPT in e["content"] for e in entries[::2]))
        self.assertTrue(all(e["role"] == "assistant" for e in entries[1::2]))
        self.assertEqual(len(provider.calls), 2)
        self.assertTrue(any("forced think turn" in line for line in log))
        self.assertTrue(all(line.startswith(LOG_PREFIX) for line in log))

    def test_low_effort_no_turns(self):
        provider = StubProvider()
        entries, log = run_fallback_reflection(
            provider,
            [{"role": "user", "content": "hi"}],
            ReasoningBudget(True, EFFORT_LOW),
        )
        self.assertEqual(entries, [])
        self.assertEqual(provider.calls, [])

    def test_provider_failure_returns_partial(self):
        class Failing(BaseProvider):
            def chat(self, messages, tools=None, **kwargs):
                raise RuntimeError("boom")

            def chat_stream(self, messages, tools=None, **kwargs):
                raise RuntimeError("boom")

            def get_available_models(self):
                return []

        entries, log = run_fallback_reflection(
            Failing(api_key="x"),
            [{"role": "user", "content": "hi"}],
            ReasoningBudget(True, EFFORT_HIGH),
            config={CONFIG_KEY: {"use_think_tool": False}},
        )
        self.assertEqual(entries, [])
        self.assertTrue(any("failed" in line for line in log))

    def test_think_tool_hook_is_defensive(self):
        # 023 has not landed: the hook finds nothing and the prompt-based
        # path is used instead.
        from src import reasoning_budgets as rb

        with patch.object(rb, "_think_tool_callable", return_value=None):
            provider = StubProvider()
            entries, _ = run_fallback_reflection(
                provider,
                [{"role": "user", "content": "hi"}],
                ReasoningBudget(True, EFFORT_MEDIUM),
            )
            self.assertEqual(len(entries), 2)


class TestEffortCommand(unittest.TestCase):
    def _context(self):
        from types import SimpleNamespace

        return SimpleNamespace(config=None, cost_tracker=None)

    def test_show_reports_policy(self):
        result = effort_command_call("", self._context())
        self.assertIn("Reasoning budgets", result.value)
        self.assertIn("planning", result.value)

    def test_set_default_effort_persists(self):
        seen: dict[str, Any] = {}

        def fake_load():
            return dict(seen)

        def fake_save(config):
            seen.update(config)

        with (
            patch("src.config.load_config", side_effect=fake_load),
            patch("src.config.save_config", side_effect=fake_save),
        ):
            result = effort_command_call("high", self._context())
            self.assertIn("high", result.value)
            self.assertEqual(seen["reasoning"]["default_effort"], "high")

            result = effort_command_call("off", self._context())
            self.assertFalse(seen["reasoning"]["enabled"])

            result = effort_command_call("agent bob low", self._context())
            self.assertEqual(seen["reasoning"]["agents"]["bob"], "low")

            result = effort_command_call("agent bob clear", self._context())
            self.assertNotIn("bob", seen["reasoning"]["agents"])

    def test_bad_usage_returns_help(self):
        result = effort_command_call("phase", self._context())
        self.assertIn("Usage", result.value)


class TestCapabilityExtension(unittest.TestCase):
    def test_register_reasoning_capable(self):
        register_reasoning_capable("fakeco", r"^fake-reasoner")
        self.assertTrue(may_reason("fake-reasoner-1", "fakeco"))
        budget = ReasoningBudget(True, EFFORT_HIGH)
        # Capability table says yes but no native key mapping -> send nothing.
        self.assertEqual(
            native_options("fakeco", "fake-reasoner-1", budget), {}
        )


if __name__ == "__main__":
    unittest.main()
