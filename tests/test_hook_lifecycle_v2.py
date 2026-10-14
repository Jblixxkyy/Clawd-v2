"""Tests for the hook-lifecycle extension (brief 031).

Covers the new enforcement-grade hook points: BeforeToolSelection
(filter/reorder the candidate tool list), BeforeModel inject/rewrite
(block/stop/inject/rewrite the outgoing model call), ToolExecuteBefore
(mutate input or fail pre-execution), and SessionRetry (veto a retry) —
plus the brief-031 guardrails: hook timeouts, recursion protection, and
audit logging of every block/rewrite decision.
"""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.hooks.runner import (
    BLOCKING_EVENTS,
    HOOK_EVENTS,
    REWRITE_EVENTS,
    run_hooks,
)
from src.tool_system.protocol import ToolCall, ToolResult
from src.tool_system.registry import ToolRegistry, ToolSpec


CAPTURE_SCRIPT = """\
import json, sys
payload = json.load(sys.stdin)
with open(sys.argv[1], "w") as f:
    json.dump(payload, f)
"""

DECIDE_SCRIPT = """\
import json, sys
json.load(sys.stdin)  # consume the payload
message = sys.argv[2] if len(sys.argv) > 2 else "halted by test hook"
print(json.dumps({"decision": sys.argv[1], "message": message}))
"""

PRINT_JSON_SCRIPT = """\
import json, sys
json.load(sys.stdin)  # consume the payload
with open(sys.argv[1]) as f:
    print(f.read())
"""

INJECT_SCRIPT = """\
import json, sys
json.load(sys.stdin)  # consume the payload
print(json.dumps({"decision": "inject", "role": sys.argv[1], "content": sys.argv[2]}))
"""

RECURSIVE_SCRIPT = """\
import json, sys
from pathlib import Path
from src.tool_system.hooks.runner import run_hooks
json.load(sys.stdin)  # consume the payload
inner = run_hooks("ToolExecuteBefore", "Bash", {"command": "x"},
                  config_path=Path(sys.argv[2]))
with open(sys.argv[1], "w") as f:
    json.dump(inner, f)
"""


def _write_script(tmpdir: Path, name: str, body: str) -> Path:
    path = tmpdir / name
    path.write_text(body)
    return path


class HookLifecycleV2TestCase(unittest.TestCase):
    """Tmp hooks.json + stub scripts, runner pointed at the tmp config."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.hooks_json = self.tmpdir / "hooks.json"
        self.capture_py = _write_script(self.tmpdir, "capture.py", CAPTURE_SCRIPT)
        self.decide_py = _write_script(self.tmpdir, "decide.py", DECIDE_SCRIPT)
        self.print_json_py = _write_script(self.tmpdir, "print_json.py", PRINT_JSON_SCRIPT)
        self.inject_py = _write_script(self.tmpdir, "inject.py", INJECT_SCRIPT)
        self._config_patcher = patch(
            "src.tool_system.hooks.runner._default_config_path",
            return_value=self.hooks_json,
        )
        self._config_patcher.start()
        self.audit_log = self.tmpdir / "hooks-audit.log"
        self._audit_patcher = patch(
            "src.tool_system.hooks.runner._audit_log_path",
            return_value=self.audit_log,
        )
        self._audit_patcher.start()

    def tearDown(self):
        self._audit_patcher.stop()
        self._config_patcher.stop()
        self.tmp.cleanup()

    def _write_config(self, config: dict) -> None:
        self.hooks_json.write_text(json.dumps(config))

    def _decide_hook(self, event: str, decision: str, message: str = "halted by test hook"):
        self._write_config(
            {event: [{"command": f"python3 {self.decide_py} {decision} {message!r}"}]}
        )

    def _print_json_hook(self, event: str, payload: dict):
        payload_file = self.tmpdir / f"rewrite-{event}.json"
        payload_file.write_text(json.dumps(payload))
        self._write_config(
            {event: [{"command": f"python3 {self.print_json_py} {payload_file}"}]}
        )

    def _read_audit(self) -> list[dict]:
        if not self.audit_log.exists():
            return []
        return [json.loads(line) for line in self.audit_log.read_text().splitlines() if line.strip()]


class TestNewEventsKnown(HookLifecycleV2TestCase):
    def test_new_events_are_known(self):
        for event in ("BeforeToolSelection", "ToolExecuteBefore", "SessionRetry"):
            self.assertIn(event, HOOK_EVENTS)

    def test_before_model_already_known_not_duplicated(self):
        self.assertIn("BeforeModel", HOOK_EVENTS)
        self.assertEqual(HOOK_EVENTS.count("BeforeModel"), 1)

    def test_blocking_events_extended(self):
        self.assertIn("ToolExecuteBefore", BLOCKING_EVENTS)
        self.assertIn("SessionRetry", BLOCKING_EVENTS)
        # BeforeToolSelection is rewrite-only, not blocking: a "block" there
        # must not halt anything (see TestBeforeToolSelection).
        self.assertNotIn("BeforeToolSelection", BLOCKING_EVENTS)

    def test_rewrite_events(self):
        self.assertEqual(
            REWRITE_EVENTS,
            frozenset({"BeforeModel", "BeforeToolSelection", "ToolExecuteBefore"}),
        )


class TestBeforeToolSelection(HookLifecycleV2TestCase):
    def test_payload_carries_candidate_tools_and_policy_context(self):
        capture_file = self.tmpdir / "capture-bts.json"
        self._write_config(
            {"BeforeToolSelection": [{"command": f"python3 {self.capture_py} {capture_file}"}]}
        )
        tools = [{"name": "Read"}, {"name": "Write"}]
        run_hooks(
            "BeforeToolSelection",
            extra={"session_id": "s1", "turn": 0, "model": "m",
                   "plan_mode": True, "read_only_tools": ["Read"], "tools": tools},
        )
        payload = json.loads(capture_file.read_text())
        self.assertEqual(payload["event"], "BeforeToolSelection")
        self.assertEqual(payload["tools"], tools)
        self.assertTrue(payload["plan_mode"])
        self.assertEqual(payload["read_only_tools"], ["Read"])

    def test_rewrite_filters_tool_list(self):
        self._print_json_hook(
            "BeforeToolSelection",
            {"decision": "rewrite", "tools": [{"name": "Read"}]},
        )
        result = run_hooks(
            "BeforeToolSelection",
            extra={"tools": [{"name": "Read"}, {"name": "Write"}]},
        )
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "rewrite")
        self.assertEqual(result["rewrite"], [{"name": "Read"}])

    def test_rewrite_allows_empty_tool_list(self):
        # A hook may legitimately hide every tool from the model.
        self._print_json_hook(
            "BeforeToolSelection", {"decision": "rewrite", "tools": []}
        )
        result = run_hooks("BeforeToolSelection", extra={"tools": [{"name": "Bash"}]})
        self.assertEqual(result["decision"], "rewrite")
        self.assertEqual(result["rewrite"], [])

    def test_malformed_rewrite_fails_open(self):
        self._print_json_hook(
            "BeforeToolSelection", {"decision": "rewrite", "tools": "not-a-list"}
        )
        result = run_hooks("BeforeToolSelection", extra={"tools": [{"name": "Bash"}]})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")
        self.assertIsNone(result["rewrite"])

    def test_block_decision_ignored(self):
        # BeforeToolSelection is rewrite-only; block degrades to allow so a
        # misconfigured hook cannot wedge the loop with a tool-less turn.
        self._decide_hook("BeforeToolSelection", "block", "no tools for you")
        result = run_hooks("BeforeToolSelection", extra={"tools": [{"name": "Bash"}]})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")


class TestBeforeModelControl(HookLifecycleV2TestCase):
    def test_inject_appends_message(self):
        self._write_config(
            {"BeforeModel": [{"command": f"python3 {self.inject_py} system CTX-123"}]}
        )
        result = run_hooks("BeforeModel", extra={"turn": 0, "messages": []})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "inject")
        self.assertEqual(result["rewrite"], {"role": "system", "content": "CTX-123"})

    def test_inject_defaults_role_to_user(self):
        script = _write_script(
            self.tmpdir, "inject_norole.py",
            'import json, sys\njson.load(sys.stdin)\n'
            'print(json.dumps({"decision": "inject", "content": "hi"}))\n',
        )
        self._write_config({"BeforeModel": [{"command": f"python3 {script}"}]})
        result = run_hooks("BeforeModel", extra={})
        self.assertEqual(result["rewrite"], {"role": "user", "content": "hi"})

    def test_rewrite_replaces_messages(self):
        messages = [{"role": "user", "content": "REPLACED"}]
        self._print_json_hook("BeforeModel", {"decision": "rewrite", "messages": messages})
        result = run_hooks("BeforeModel", extra={"messages": [{"role": "user", "content": "orig"}]})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "rewrite")
        self.assertEqual(result["rewrite"], messages)

    def test_malformed_rewrite_fails_open(self):
        self._print_json_hook(
            "BeforeModel", {"decision": "rewrite", "messages": [{"no_role": True}]}
        )
        result = run_hooks("BeforeModel", extra={})
        self.assertEqual(result["decision"], "allow")
        self.assertIsNone(result["rewrite"])

    def test_block_still_halts(self):
        # 008 behavior preserved alongside the new control decisions.
        self._decide_hook("BeforeModel", "block", "stop")
        result = run_hooks("BeforeModel", extra={})
        self.assertTrue(result["blocked"])
        self.assertEqual(result["decision"], "block")


class _RecordingTool:
    """Minimal tool for registry.dispatch tests."""

    def __init__(self):
        self.calls: list[dict] = []

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="EchoTool",
            description="echo",
            input_schema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
            is_read_only=True,
        )

    def check_permissions(self, tool_input, context):
        from src.tool_system.permission_handler import PermissionResult

        return PermissionResult.allow()

    def run(self, tool_input: dict, context) -> ToolResult:
        self.calls.append(dict(tool_input))
        return ToolResult(name="EchoTool", output={"echo": tool_input.get("text")})


class TestToolExecuteBefore(HookLifecycleV2TestCase):
    def _registry(self) -> tuple[ToolRegistry, _RecordingTool]:
        tool = _RecordingTool()
        return ToolRegistry([tool]), tool

    def _ctx(self) -> ToolContext:
        return ToolContext(workspace_root=self.tmpdir)

    def test_rewrite_mutates_input(self):
        self._print_json_hook(
            "ToolExecuteBefore",
            {"decision": "rewrite", "input": {"text": "MUTATED"}},
        )
        registry, tool = self._registry()
        result = registry.dispatch(
            ToolCall(name="EchoTool", input={"text": "original"}), self._ctx()
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["echo"], "MUTATED")
        self.assertEqual(tool.calls, [{"text": "MUTATED"}])

    def test_fail_blocks_pre_execution_with_no_partial_effects(self):
        self._decide_hook("ToolExecuteBefore", "fail", "forbidden by policy")
        registry, tool = self._registry()
        result = registry.dispatch(
            ToolCall(name="EchoTool", input={"text": "original"}), self._ctx()
        )
        self.assertTrue(result.is_error)
        self.assertIn("forbidden by policy", str(result.output))
        # The tool itself never ran: no partial effects.
        self.assertEqual(tool.calls, [])

    def test_block_also_fails_pre_execution(self):
        self._decide_hook("ToolExecuteBefore", "block", "nope")
        registry, tool = self._registry()
        result = registry.dispatch(
            ToolCall(name="EchoTool", input={"text": "original"}), self._ctx()
        )
        self.assertTrue(result.is_error)
        self.assertEqual(tool.calls, [])

    def test_allow_passes_through_unchanged(self):
        self._decide_hook("ToolExecuteBefore", "allow")
        registry, tool = self._registry()
        result = registry.dispatch(
            ToolCall(name="EchoTool", input={"text": "original"}), self._ctx()
        )
        self.assertFalse(result.is_error)
        self.assertEqual(tool.calls, [{"text": "original"}])

    def test_broken_hook_fails_open(self):
        self._write_config({"ToolExecuteBefore": [{"command": "exit 3"}]})
        registry, tool = self._registry()
        result = registry.dispatch(
            ToolCall(name="EchoTool", input={"text": "original"}), self._ctx()
        )
        self.assertFalse(result.is_error)
        self.assertEqual(tool.calls, [{"text": "original"}])


class TestSessionRetry(HookLifecycleV2TestCase):
    def test_block_vetoes_retry(self):
        self._decide_hook("SessionRetry", "block", "do not retry")
        result = run_hooks(
            "SessionRetry",
            extra={"attempt": 1, "max_attempts": 3, "error": "boom", "error_type": "ConnectionError"},
        )
        self.assertTrue(result["blocked"])
        self.assertEqual(result["decision"], "block")
        self.assertEqual(result["message"], "do not retry")

    def test_allow_retries(self):
        self._decide_hook("SessionRetry", "allow")
        result = run_hooks("SessionRetry", extra={"attempt": 1})
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")


class AgentLoopV2TestCase(HookLifecycleV2TestCase):
    def setUp(self):
        super().setUp()
        self.registry = build_default_registry()
        self.ctx = ToolContext(workspace_root=self.tmpdir)
        self.ctx.session_id = "test-session"

    def _provider(self, final="Done."):
        p = MagicMock()
        p.model = "test-model"
        p.chat = MagicMock(
            return_value=ChatResponse(
                content=final, model="test", usage={}, finish_reason="stop"
            )
        )
        return p

    def _run(self, provider, **kwargs):
        conv = Conversation()
        conv.add_user_message("hello")
        return run_agent_loop(conv, provider, self.registry, self.ctx, max_turns=5, **kwargs)


class TestSessionRetryInLoop(AgentLoopV2TestCase):
    def test_veto_stops_retry_and_surfaces_error(self):
        self._decide_hook("SessionRetry", "block", "no retries allowed")
        provider = self._provider()
        provider.chat.side_effect = [
            ConnectionError("boom"),
            ChatResponse(content="Done.", model="test", usage={}, finish_reason="stop"),
        ]
        with patch("time.sleep") as sleep_mock:
            result = self._run(provider)
        # Retry vetoed: exactly one attempt and no backoff sleep. (The
        # global time.sleep patch also observes subprocess's own 1ms poll
        # sleeps, so only backoff-length (>=1s) sleeps count here.)
        self.assertEqual(provider.chat.call_count, 1)
        backoff_sleeps = [
            c for c in sleep_mock.call_args_list if c.args and c.args[0] >= 1.0
        ]
        self.assertEqual(backoff_sleeps, [])
        self.assertIn("boom", result.response_text)

    def test_no_veto_retries_and_succeeds(self):
        provider = self._provider()
        provider.chat.side_effect = [
            ConnectionError("boom"),
            ChatResponse(content="Done.", model="test", usage={}, finish_reason="stop"),
        ]
        with patch("time.sleep"):
            result = self._run(provider)
        self.assertEqual(provider.chat.call_count, 2)
        self.assertEqual(result.response_text, "Done.")


class TestBeforeToolSelectionInLoop(AgentLoopV2TestCase):
    def test_plan_mode_policy_filters_write_tools(self):
        # 016's plan-mode policy as the reference BeforeToolSelection hook:
        # the model never sees write tools while planning.
        example = (
            Path(__file__).resolve().parent.parent
            / "src" / "tool_system" / "hooks" / "examples" / "plan_mode_tool_filter.py"
        )
        self.assertTrue(example.exists(), f"reference hook missing: {example}")
        self._write_config({"BeforeToolSelection": [{"command": f"python3 {example}"}]})
        self.ctx.plan_mode = True
        provider = self._provider()
        self._run(provider)
        tools_kwarg = provider.chat.call_args.kwargs["tools"]
        names = {t["name"] for t in tools_kwarg}
        for write_tool in ("Write", "Edit", "Bash", "PowerShell", "NotebookEdit"):
            self.assertNotIn(write_tool, names, f"{write_tool} leaked to the model in plan mode")
        self.assertIn("Read", names)
        self.assertIn("ExitPlanMode", names)

    def test_no_hook_leaves_tool_list_untouched(self):
        provider = self._provider()
        self._run(provider)
        tools_kwarg = provider.chat.call_args.kwargs["tools"]
        names = {t["name"] for t in tools_kwarg}
        self.assertIn("Write", names)
        self.assertIn("Bash", names)

    def test_registry_tool_schemas_not_mutated_by_rewrite(self):
        self._print_json_hook(
            "BeforeToolSelection", {"decision": "rewrite", "tools": [{"name": "Read"}]}
        )
        before = [s.name for s in self.registry.list_specs()]
        provider = self._provider()
        self._run(provider)
        after = [s.name for s in self.registry.list_specs()]
        self.assertEqual(before, after)


class TestBeforeModelControlInLoop(AgentLoopV2TestCase):
    def test_injection_appears_in_prompt(self):
        self._write_config(
            {"BeforeModel": [{"command": f"python3 {self.inject_py} system CTX-123"}]}
        )
        provider = self._provider()
        self._run(provider)
        messages = provider.chat.call_args.args[0]
        self.assertEqual(messages[-1], {"role": "system", "content": "CTX-123"})

    def test_rewrite_replaces_outgoing_messages(self):
        self._print_json_hook(
            "BeforeModel",
            {"decision": "rewrite", "messages": [{"role": "user", "content": "REPLACED"}]},
        )
        provider = self._provider()
        self._run(provider)
        messages = provider.chat.call_args.args[0]
        self.assertEqual(messages, [{"role": "user", "content": "REPLACED"}])


class TestGuardrails(HookLifecycleV2TestCase):
    def test_hanging_hook_timeout_fails_open(self):
        self._write_config({"BeforeModel": [{"command": "sleep 30"}]})
        start = time.monotonic()
        result = run_hooks("BeforeModel", extra={"turn": 0}, timeout=0.3)
        elapsed = time.monotonic() - start
        self.assertFalse(result["blocked"])
        self.assertEqual(result["decision"], "allow")
        self.assertLess(elapsed, 10, "hanging hook must not hang the loop")

    def test_recursion_guard_blocks_self_triggering_hook(self):
        script = _write_script(self.tmpdir, "recursive.py", RECURSIVE_SCRIPT)
        inner_result_file = self.tmpdir / "inner.json"
        # NOTE: PYTHONPATH is set inline in the shell command; run_hooks
        # itself adds CLAWD_HOOK_EVENTS to the hook subprocess env.
        self._write_config(
            {
                "ToolExecuteBefore": [
                    {
                        "command": (
                            f"PYTHONPATH=/home/hatch/workspace/Clawd-Code "
                            f"python3 {script} {inner_result_file} {self.hooks_json}"
                        )
                    },
                    {"command": f"python3 {self.decide_py} block outer-block"},
                ]
            }
        )
        result = run_hooks("ToolExecuteBefore", "Bash", {"command": "x"})
        # The nested run_hooks call for the same event was stopped by the
        # recursion guard instead of spawning hooks again.
        inner = json.loads(inner_result_file.read_text())
        self.assertTrue(inner.get("recursion_blocked"))
        self.assertEqual(inner["decision"], "allow")
        # The outer run still processed its own hook list normally.
        self.assertTrue(result["blocked"])
        self.assertEqual(result["decision"], "block")

    def test_recursion_guard_multi_event_cycle(self):
        # A -> B -> A cycle is caught: the repeated event is skipped, while
        # the intermediate (non-repeated) event still runs its hooks.
        script = _write_script(
            self.tmpdir, "cycle.py",
            "import json, sys\n"
            "from pathlib import Path\n"
            "from src.tool_system.hooks.runner import run_hooks\n"
            "payload = json.load(sys.stdin)\n"
            'inner = run_hooks("PreToolUse", "Bash", {"command": "y"},\n'
            "                  config_path=Path(sys.argv[2]))\n"
            "with open(sys.argv[1], 'w') as f:\n"
            "    json.dump(inner, f)\n",
        )
        inner_file = self.tmpdir / "inner-cycle.json"
        deep_file = self.tmpdir / "deep-cycle.json"
        self._write_config(
            {
                "BeforeModel": [
                    {
                        "command": (
                            "PYTHONPATH=/home/hatch/workspace/Clawd-Code "
                            f"python3 {script} {inner_file} {self.hooks_json}"
                        )
                    }
                ],
                # PreToolUse's own hook re-enters BeforeModel -> cycle.
                # It records the re-entrant BeforeModel result for assertion.
                "PreToolUse": [
                    {
                        "command": (
                            "PYTHONPATH=/home/hatch/workspace/Clawd-Code "
                            "python3 -c \"import json,sys;from pathlib import Path;"
                            "from src.tool_system.hooks.runner import run_hooks;"
                            "json.load(sys.stdin);"
                            "r=run_hooks('BeforeModel', extra={}, config_path=Path(sys.argv[1]));"
                            "open(sys.argv[2],'w').write(json.dumps(r))\" "
                            f"{self.hooks_json} {deep_file}"
                        )
                    }
                ],
            }
        )
        start = time.monotonic()
        result = run_hooks("BeforeModel", extra={})
        elapsed = time.monotonic() - start
        inner = json.loads(inner_file.read_text())
        deep = json.loads(deep_file.read_text())
        # The cycle terminates (no infinite hook spawning).
        self.assertLess(elapsed, 30)
        self.assertEqual(result["decision"], "allow")
        # The intermediate PreToolUse event ran normally (not a repeat).
        self.assertEqual(inner["decision"], "allow")
        self.assertIsNone(inner.get("recursion_blocked"))
        # The re-entrant BeforeModel was recursion-blocked.
        self.assertTrue(deep.get("recursion_blocked"))
        self.assertEqual(deep["decision"], "allow")

    def test_audit_log_records_block_decision(self):
        self._decide_hook("PreToolUse", "block", "audit me")
        run_hooks("PreToolUse", "Bash", {"command": "ls"})
        entries = self._read_audit()
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["event"], "PreToolUse")
        self.assertEqual(entry["decision"], "block")
        self.assertEqual(entry["message"], "audit me")
        self.assertIn("decide.py", entry["command"])
        self.assertIn("ts", entry)

    def test_audit_log_records_rewrite_and_inject(self):
        self._print_json_hook(
            "BeforeToolSelection", {"decision": "rewrite", "tools": []}
        )
        run_hooks("BeforeToolSelection", extra={})
        self._write_config(
            {"BeforeModel": [{"command": f"python3 {self.inject_py} user hi"}]}
        )
        run_hooks("BeforeModel", extra={})
        entries = self._read_audit()
        decisions = {(e["event"], e["decision"]) for e in entries}
        self.assertIn(("BeforeToolSelection", "rewrite"), decisions)
        self.assertIn(("BeforeModel", "inject"), decisions)

    def test_first_control_decision_wins(self):
        self._write_config(
            {
                "BeforeModel": [
                    {"command": f"python3 {self.decide_py} block first"},
                    {"command": f"python3 {self.decide_py} block second"},
                ]
            }
        )
        result = run_hooks("BeforeModel", extra={})
        self.assertTrue(result["blocked"])
        self.assertEqual(result["message"], "first")


if __name__ == "__main__":
    unittest.main()
