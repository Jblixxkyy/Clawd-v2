"""Tests for model tiers (brief 028): weak-model sidecar for utility calls.

Uses stub providers and patched config -- no network, no real config file.
"""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.conversation import Conversation, Message
from src.cost_tracker import CostTracker
from src.model_tiers import (
    TIER_ACT,
    TIER_MAIN,
    TIER_PLAN,
    TIER_SMALL,
    UTILITY_TASKS,
    attribute_tier,
    get_tier_snapshot,
    is_weak_profile_tool_allowed,
    model_for_mode,
    model_for_tier,
    reset_tier_snapshot,
    resolve_models,
    tier_for_model,
    utility_model,
    weak_profile_allowed_tools,
    weak_profile_model,
    weak_read_permission_handler,
)
from src.providers.base import ChatResponse
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.protocol import ToolCall
from src.tool_system.tools.subagent import (
    _run_child_loop,
    _sub_context,
    build_weak_child_registry,
)


MAIN = "claude-main"
SMALL = "tiny-cheap"


def _config(models: dict[str, str], default_model: str = MAIN) -> dict[str, Any]:
    return {
        "default_provider": "anthropic",
        "providers": {
            "anthropic": {
                "api_key": "",
                "base_url": "https://api.anthropic.com",
                "default_model": default_model,
            }
        },
        "models": dict(models),
    }


def _patched_load_config(models: dict[str, str], default_model: str = MAIN):
    return patch("src.model_tiers.load_config", return_value=_config(models, default_model))


def _fake_snapshot(models: dict[str, str]):
    full = {"main": MAIN, "small": "", "plan": "", "act": ""}
    full.update(models)
    if not full["small"]:
        full["small"] = full["main"]
    if not full["plan"]:
        full["plan"] = full["main"]
    if not full["act"]:
        full["act"] = full["main"]
    return patch("src.model_tiers.get_tier_snapshot", return_value=full)


class TestTierResolution(unittest.TestCase):
    def test_single_model_all_tiers_resolve_to_main(self):
        """Single-model config: every tier is main -- zero behavior change."""
        models = resolve_models(_config({}))
        self.assertEqual(models[TIER_MAIN], MAIN)
        self.assertEqual(models[TIER_SMALL], MAIN)
        self.assertEqual(models[TIER_PLAN], MAIN)
        self.assertEqual(models[TIER_ACT], MAIN)

    def test_utility_model_single_model_returns_passed_model(self):
        """utility_model() passes the model through untouched when no small."""
        self.assertEqual(utility_model("claude-main", _config({})), "claude-main")
        self.assertEqual(utility_model(None, _config({})), MAIN)

    def test_small_configured_utility_routes_to_small(self):
        models = resolve_models(_config({"small": SMALL}))
        self.assertEqual(models[TIER_MAIN], MAIN)
        self.assertEqual(models[TIER_SMALL], SMALL)
        self.assertEqual(utility_model(MAIN, _config({"small": SMALL})), SMALL)
        # explicit main override still respected for non-utility use
        self.assertEqual(model_for_tier("main", _config({"small": SMALL})), MAIN)

    def test_main_defaults_to_provider_default_model(self):
        models = resolve_models(_config({"small": SMALL}, default_model="provider-model"))
        self.assertEqual(models[TIER_MAIN], "provider-model")
        self.assertEqual(models[TIER_SMALL], SMALL)

    def test_unknown_tier_resolves_to_main(self):
        self.assertEqual(model_for_tier("nonsense", _config({"small": SMALL})), MAIN)

    def test_utility_tasks_known(self):
        for task in ("compaction_summary", "session_title", "commit_message",
                     "reviewer", "judge", "edit_corrector", "file_mention_classification"):
            self.assertIn(task, UTILITY_TASKS)


class TestPlanActModes(unittest.TestCase):
    def test_plan_act_split(self):
        cfg = _config({"plan": "claude-opus", "act": SMALL})
        self.assertEqual(model_for_mode("plan", cfg), "claude-opus")
        self.assertEqual(model_for_mode("act", cfg), SMALL)
        self.assertEqual(model_for_mode("PLAN", cfg), "claude-opus")  # case-insensitive

    def test_plan_act_unset_fall_back_to_main(self):
        """Opt-in: unset plan/act keys keep the main model (zero change)."""
        cfg = _config({})
        self.assertEqual(model_for_mode("plan", cfg), MAIN)
        self.assertEqual(model_for_mode("act", cfg), MAIN)

    def test_unknown_mode_resolves_to_main(self):
        self.assertEqual(model_for_mode("weird", _config({"plan": "x"})), MAIN)


class TestTierAttribution(unittest.TestCase):
    def test_tier_for_model_small(self):
        self.assertEqual(tier_for_model(SMALL, _config({"small": SMALL})), TIER_SMALL)

    def test_tier_for_model_main(self):
        self.assertEqual(tier_for_model(MAIN, _config({"small": SMALL})), TIER_MAIN)

    def test_tier_for_model_single_model_is_main(self):
        """Single-model config attributes everything to main, not small."""
        self.assertEqual(tier_for_model(MAIN, _config({})), TIER_MAIN)

    def test_tier_for_model_unknown(self):
        self.assertIsNone(tier_for_model("some-other-model", _config({"small": SMALL})))
        self.assertIsNone(tier_for_model(None, _config({})))
        self.assertIsNone(tier_for_model("", _config({})))


def _make_conversation() -> Conversation:
    conv = Conversation()
    conv.messages = [
        Message(role="user", content="Hello world " * 50),
        Message(role="assistant", content="Hi there! " * 50),
        Message(role="user", content="What about the code? " * 50),
    ]
    return conv


class TestCompactRouting(unittest.TestCase):
    """Compaction summaries are utility calls: they ride the small tier."""

    def test_compact_routes_summary_to_small_model(self):
        from src.compact_service.service import compact_conversation
        mock_provider = MagicMock()
        mock_provider.chat_async = AsyncMock(
            return_value=ChatResponse(content="Summary", model="test", usage={}, finish_reason="stop")
        )
        with _patched_load_config({"small": SMALL}):
            asyncio.run(compact_conversation(_make_conversation(), mock_provider, MAIN))
        _, kwargs = mock_provider.chat_async.call_args
        self.assertEqual(kwargs["model"], SMALL)

    def test_compact_single_model_zero_change(self):
        """No small configured: compaction uses the passed model, as before."""
        from src.compact_service.service import compact_conversation
        mock_provider = MagicMock()
        mock_provider.chat_async = AsyncMock(
            return_value=ChatResponse(content="Summary", model="test", usage={}, finish_reason="stop")
        )
        with _patched_load_config({}):
            asyncio.run(compact_conversation(_make_conversation(), mock_provider, MAIN))
        _, kwargs = mock_provider.chat_async.call_args
        self.assertEqual(kwargs["model"], MAIN)


class TestWeakSubagentProfile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tmp.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.tmp.cleanup()

    def test_weak_registry_is_read_only_subset(self):
        weak = build_weak_child_registry(self.registry)
        names = {t.spec().name.lower() for t in weak.list_tools()}
        allowed = weak_profile_allowed_tools()
        self.assertTrue(names <= allowed, f"weak registry leaked tools: {names - allowed}")
        for tool in ("read", "glob", "grep", "webfetch", "websearch"):
            self.assertIn(tool, names, f"expected read-only tool {tool!r} in weak registry")

    def test_weak_registry_excludes_write_edit_bash(self):
        weak = build_weak_child_registry(self.registry)
        names = {t.spec().name.lower() for t in weak.list_tools()}
        for tool in ("write", "edit", "bash", "subagent", "agent", "todowrite"):
            self.assertNotIn(tool, names)

    def test_weak_subagent_cannot_call_outside_restricted_set(self):
        """Dispatch of a non-read tool fails: the child can never run it."""
        weak = build_weak_child_registry(self.registry)
        result = weak.dispatch(
            ToolCall(name="Write", input={"file_path": "x.txt", "content": "y"}),
            self.context,
        )
        self.assertTrue(result.is_error)
        self.assertIn("unknown tool", str(result.output.get("error", "")).lower())

    def test_weak_read_permission_handler(self):
        """Read-only tools auto-approve; everything else is denied."""
        self.assertEqual(weak_read_permission_handler("Read", "msg", None), (True, False))
        self.assertEqual(weak_read_permission_handler("WebSearch", "msg", None), (True, False))
        self.assertEqual(weak_read_permission_handler("Write", "msg", None), (False, False))
        self.assertEqual(weak_read_permission_handler("Bash", "msg", None), (False, False))
        self.assertTrue(is_weak_profile_tool_allowed("grep"))
        self.assertFalse(is_weak_profile_tool_allowed("edit"))

    def test_weak_profile_model_defaults_to_small_tier(self):
        with _patched_load_config({"small": SMALL}):
            self.assertEqual(weak_profile_model(), SMALL)
        with _patched_load_config({}):
            self.assertEqual(weak_profile_model(), MAIN)

    def test_weak_child_loop_wires_model_registry_permissions(self):
        """profile='weak' + no explicit model -> small model, weak registry,
        auto-approving read permissions."""
        captured: dict[str, Any] = {}

        def fake_run_agent_loop(*, conversation, provider, tool_registry, tool_context, max_turns):
            captured["model"] = getattr(provider, "model", None)
            captured["tools"] = {t.spec().name.lower() for t in tool_registry.list_tools()}
            captured["handler"] = tool_context.permission_handler
            return SimpleNamespace(
                response_text="ok", num_turns=1, cancelled=False, usage=None,
                # A tool call happened: completion is "ended without explicit
                # completion", not a stall (brief 020).
                run_log=[{"event": "tool_result", "tool": "Read", "is_error": False}],
            )

        parent_provider = SimpleNamespace(model=MAIN)
        with _patched_load_config({"small": SMALL}), \
                patch("src.tool_system.tools.subagent.run_agent_loop", fake_run_agent_loop):
            record = _run_child_loop(
                task_id="t1",
                description="d",
                prompt="p",
                provider=parent_provider,
                parent_registry=self.registry,
                parent_context=self.context,
                max_turns=2,
                model=None,
                profile="weak",
            )
        self.assertEqual(record["status"], "done")
        self.assertEqual(captured["model"], SMALL)
        self.assertTrue(captured["tools"] <= weak_profile_allowed_tools())
        self.assertNotIn("write", captured["tools"])
        handler = captured["handler"]
        self.assertTrue(handler("Read", "m", None)[0])
        self.assertFalse(handler("Write", "m", None)[0])

    def test_weak_profile_explicit_model_wins(self):
        """An explicit model override beats the profile's small default."""
        captured: dict[str, Any] = {}

        def fake_run_agent_loop(*, conversation, provider, tool_registry, tool_context, max_turns):
            captured["model"] = getattr(provider, "model", None)
            return SimpleNamespace(
                response_text="ok", num_turns=1, cancelled=False, usage=None
            )

        parent_provider = SimpleNamespace(model=MAIN)
        with _patched_load_config({"small": SMALL}), \
                patch("src.tool_system.tools.subagent.run_agent_loop", fake_run_agent_loop):
            _run_child_loop(
                task_id="t1",
                description="d",
                prompt="p",
                provider=parent_provider,
                parent_registry=self.registry,
                parent_context=self.context,
                max_turns=2,
                model="explicit-override",
                profile="weak",
            )
        self.assertEqual(captured["model"], "explicit-override")

    def test_standard_profile_unchanged(self):
        """Standard subagents keep the parent's model and full registry."""
        captured: dict[str, Any] = {}

        def fake_run_agent_loop(*, conversation, provider, tool_registry, tool_context, max_turns):
            captured["model"] = getattr(provider, "model", None)
            captured["tools"] = {t.spec().name.lower() for t in tool_registry.list_tools()}
            captured["handler"] = tool_context.permission_handler
            return SimpleNamespace(
                response_text="ok", num_turns=1, cancelled=False, usage=None
            )

        parent_provider = SimpleNamespace(model=MAIN)
        with _patched_load_config({"small": SMALL}), \
                patch("src.tool_system.tools.subagent.run_agent_loop", fake_run_agent_loop):
            _run_child_loop(
                task_id="t1",
                description="d",
                prompt="p",
                provider=parent_provider,
                parent_registry=self.registry,
                parent_context=self.context,
                max_turns=2,
                model=None,
            )
        self.assertEqual(captured["model"], MAIN)
        self.assertIn("write", captured["tools"])  # full filtered registry
        self.assertIsNone(captured["handler"])  # no auto-approve handler

    def test_invalid_profile_rejected(self):
        from src.tool_system.errors import ToolInputError
        tool = self.registry.get("Subagent")
        self.context.provider = SimpleNamespace(model=MAIN)
        with self.assertRaises(ToolInputError):
            tool.run({"description": "d", "prompt": "p", "profile": "bogus"}, self.context)


class TestCostByTier(unittest.TestCase):
    def setUp(self):
        reset_tier_snapshot()

    def tearDown(self):
        reset_tier_snapshot()

    def test_spend_by_tier_splits(self):
        t = CostTracker()
        with _fake_snapshot({"small": SMALL}):
            t.record_turn("anthropic", SMALL, 1_000_000, 0)          # small tier
            t.record_turn("anthropic", MAIN, 1_000_000, 0)           # main tier
            t.record_turn("anthropic", SMALL, 1_000_000, 0, tier="judge")  # explicit
        by_tier = t.spend_by_tier()
        self.assertGreater(by_tier.get("small", 0.0), 0.0)
        self.assertGreater(by_tier.get("main", 0.0), 0.0)
        self.assertGreater(by_tier.get("judge", 0.0), 0.0)
        self.assertEqual(t.turns[0]["tier"], "small")
        self.assertEqual(t.turns[1]["tier"], "main")
        self.assertEqual(t.turns[2]["tier"], "judge")
        self.assertAlmostEqual(sum(by_tier.values()), t.cost_usd)

    def test_single_model_all_spend_is_main(self):
        t = CostTracker()
        with _fake_snapshot({}):
            t.record_turn("anthropic", MAIN, 1_000_000, 0)
        by_tier = t.spend_by_tier()
        self.assertEqual(set(by_tier), {"main"})

    def test_explicit_tier_param(self):
        t = CostTracker()
        with _fake_snapshot({}):
            t.record_turn("anthropic", MAIN, 1_000_000, 0, tier="small")
        self.assertEqual(t.turns[0]["tier"], "small")
        self.assertGreater(t.spend_by_tier().get("small", 0.0), 0.0)

    def test_spend_by_tier_returns_copy(self):
        t = CostTracker()
        with _fake_snapshot({}):
            t.record_turn("anthropic", MAIN, 1000, 0)
        by_tier = t.spend_by_tier()
        by_tier["main"] = 0.0
        self.assertGreater(t.spend_by_tier()["main"], 0.0)


class TestConfigModelTiers(unittest.TestCase):
    def test_default_config_has_models_key(self):
        from src.config import get_default_config
        config = get_default_config()
        self.assertIn("models", config)
        self.assertEqual(
            config["models"], {"main": "", "small": "", "plan": "", "act": ""}
        )

    def test_set_and_get_model_tier(self):
        from src.config import get_model_tier, set_model_tier
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"
            with patch("src.config.get_config_path", return_value=config_path):
                set_model_tier("small", SMALL)
                self.assertEqual(get_model_tier("small"), SMALL)
                self.assertEqual(get_model_tier("main"), "")
                set_model_tier("small", "")  # clear back to fallback
                self.assertEqual(get_model_tier("small"), "")

    def test_unknown_tier_raises(self):
        from src.config import get_model_tier, set_model_tier
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / ".clawd" / "config.json"
            with patch("src.config.get_config_path", return_value=config_path):
                with self.assertRaises(ValueError):
                    set_model_tier("huge", "x")
                with self.assertRaises(ValueError):
                    get_model_tier("huge")


if __name__ == "__main__":
    unittest.main()
