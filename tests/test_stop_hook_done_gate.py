"""Tests for brief 015: stop-hook done-gate.

The Stop hook event fires when the agent loop is about to terminate via a
terminal tool (complete_task / ReportFailure / Exit). A "block" decision
vetoes the completion, feeds the reason back as a continuation prompt, and
resumes the loop; consecutive blocks are capped at DONE_GATE_MAX_BLOCKS
(3), then the run stops and escalates to the user.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system.agent_loop import DONE_GATE_MAX_BLOCKS, run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.hooks.runner import BLOCKING_EVENTS, HOOK_EVENTS, run_hooks


REPO_ROOT = Path(__file__).resolve().parents[1]
GATE_SCRIPT = (
    REPO_ROOT / "src" / "tool_system" / "hooks" / "examples" / "stop_done_gate.py"
)


BLOCK_SCRIPT = """\
import json, sys
json.load(sys.stdin)  # consume the payload
reason = sys.argv[1] if len(sys.argv) > 1 else "blocked by test hook"
print(json.dumps({"decision": "block", "message": reason}))
"""

ALLOW_SCRIPT = """\
import json, sys
json.load(sys.stdin)  # consume the payload
print(json.dumps({"decision": "allow"}))
"""

# Blocks the first N invocations, then allows. argv: counter_file, n, reason.
STATEFUL_BLOCK_SCRIPT = """\
import json, sys
from pathlib import Path
json.load(sys.stdin)  # consume the payload
counter = Path(sys.argv[1])
count = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(count + 1))
n = int(sys.argv[2])
reason = sys.argv[3] if len(sys.argv) > 3 else "blocked by test hook"
if count < n:
    print(json.dumps({"decision": "block", "message": reason}))
else:
    print(json.dumps({"decision": "allow"}))
"""

# Writes the received payload to a file (proves the hook fired + its shape).
CAPTURE_SCRIPT = """\
import json, sys
payload = json.load(sys.stdin)
with open(sys.argv[1], "w") as f:
    json.dump(payload, f)
print(json.dumps({"decision": "allow"}))
"""


def _write_script(tmpdir: Path, name: str, body: str) -> Path:
    path = tmpdir / name
    path.write_text(body)
    return path


def _complete_task_use(tool_id: str = "toolu_1") -> dict:
    return {
        "id": tool_id,
        "name": "complete_task",
        "input": {"outcome": "success", "summary": "task done"},
    }


def _tool_response(tool_use: dict) -> ChatResponse:
    return ChatResponse(
        content="Calling the completion tool.",
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 20},
        finish_reason="tool_use",
        tool_uses=[tool_use],
    )


def _final_text_response(text: str = "All done, no tools needed.") -> ChatResponse:
    return ChatResponse(
        content=text,
        model="test-model",
        usage={"input_tokens": 30, "output_tokens": 10},
        finish_reason="stop",
        tool_uses=None,
    )


class DoneGateLoopTestCase(unittest.TestCase):
    """End-to-end: Stop hook vetoes completion inside the real agent loop."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.hooks_json = self.tmpdir / "hooks.json"
        self.audit_log = self.tmpdir / "hooks-audit.log"
        self._config_patcher = patch(
            "src.tool_system.hooks.runner._default_config_path",
            return_value=self.hooks_json,
        )
        self._audit_patcher = patch(
            "src.tool_system.hooks.runner._audit_log_path",
            return_value=self.audit_log,
        )
        self._config_patcher.start()
        self._audit_patcher.start()
        self.workspace = self.tmpdir / "workspace"
        self.workspace.mkdir()
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self._audit_patcher.stop()
        self._config_patcher.stop()
        self.tmp.cleanup()

    def _write_config(self, config: dict) -> None:
        self.hooks_json.write_text(json.dumps(config))

    def _run(self, responses: list[ChatResponse]):
        conversation = Conversation()
        conversation.add_user_message("Do the thing, then finish.")
        provider = MagicMock()
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.side_effect = list(responses)
        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            verbose=False,
        )
        return result, provider, conversation

    def _events(self, result) -> list[str]:
        return [r.get("event") for r in (result.run_log or [])]

    def test_stop_event_is_registered_and_blocking(self):
        self.assertIn("Stop", HOOK_EVENTS)
        self.assertIn("Stop", BLOCKING_EVENTS)

    def test_failing_checks_block_completion_and_model_continues(self):
        """Finishing with failing checks -> blocked with the failure reason;
        the model gets a continuation turn."""
        counter = self.tmpdir / "counter.txt"
        script = _write_script(self.tmpdir, "block_once.py", STATEFUL_BLOCK_SCRIPT)
        self._write_config(
            {
                "Stop": [
                    {
                        "command": (
                            f"{sys.executable} {script} {counter} 1 "
                            "'tests failing: 2 failed'"
                        )
                    }
                ]
            }
        )
        result, provider, conversation = self._run(
            [_tool_response(_complete_task_use("toolu_1")),
             _tool_response(_complete_task_use("toolu_2"))]
        )
        # The model got a continuation turn instead of stopping.
        self.assertEqual(provider.chat.call_count, 2)
        # Second attempt passed the gate, so the run completed normally.
        self.assertIn("[Task complete", result.response_text)
        blocks = [
            r for r in (result.run_log or []) if r.get("event") == "done_gate_block"
        ]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["reason"], "tests failing: 2 failed")
        # The block reason reached the model verbatim as a user message.
        texts = [
            m.get("content") for m in conversation.get_messages()
            if m.get("role") == "user"
        ]
        self.assertTrue(
            any(
                "Done-gate blocked: tests failing: 2 failed. Continue working." in t
                for t in texts
                if isinstance(t, str)
            ),
            f"continuation notice missing from user messages: {texts!r}",
        )

    def test_all_checks_green_allows_with_no_extra_turn(self):
        """All checks green -> allowed, no extra turn, no continuation."""
        script = _write_script(self.tmpdir, "allow.py", ALLOW_SCRIPT)
        self._write_config({"Stop": [{"command": f"{sys.executable} {script}"}]})
        result, provider, _ = self._run([_tool_response(_complete_task_use())])
        self.assertEqual(provider.chat.call_count, 1)
        self.assertIn("[Task complete", result.response_text)
        self.assertNotIn("done_gate_block", self._events(result))

    def test_three_consecutive_blocks_stop_and_escalate(self):
        """3 consecutive blocks -> the run stops and reports to the user."""
        self.assertEqual(DONE_GATE_MAX_BLOCKS, 3)
        script = _write_script(self.tmpdir, "block.py", BLOCK_SCRIPT)
        self._write_config(
            {"Stop": [{"command": f"{sys.executable} {script} 'still failing'"}]}
        )
        result, provider, _ = self._run(
            [
                _tool_response(_complete_task_use("toolu_1")),
                _tool_response(_complete_task_use("toolu_2")),
                _tool_response(_complete_task_use("toolu_3")),
            ]
        )
        # Three blocks happened, then the run stopped instead of a 4th turn.
        self.assertEqual(provider.chat.call_count, 3)
        blocks = [
            r for r in (result.run_log or []) if r.get("event") == "done_gate_block"
        ]
        self.assertEqual([b["blocks"] for b in blocks], [1, 2, 3])
        escalations = [
            r for r in (result.run_log or []) if r.get("event") == "done_gate_escalation"
        ]
        self.assertEqual(len(escalations), 1)
        self.assertEqual(escalations[0]["blocks"], 3)
        self.assertIn("Done-gate: completion vetoed 3x", result.response_text)
        self.assertIn("still failing", result.response_text)

    def test_custom_gate_reason_surfaces_verbatim(self):
        """A custom gate's reason reaches the model word-for-word."""
        counter = self.tmpdir / "counter.txt"
        script = _write_script(self.tmpdir, "block_once.py", STATEFUL_BLOCK_SCRIPT)
        reason = "custom veto: CHANGELOG.md not updated"
        self._write_config(
            {
                "Stop": [
                    {
                        "command": (
                            f"{sys.executable} {script} {counter} 1 '{reason}'"
                        )
                    }
                ]
            }
        )
        result, provider, conversation = self._run(
            [_tool_response(_complete_task_use("toolu_1")),
             _tool_response(_complete_task_use("toolu_2"))]
        )
        self.assertEqual(provider.chat.call_count, 2)
        texts = [
            m.get("content") for m in conversation.get_messages()
            if m.get("role") == "user"
        ]
        expected = f"Done-gate blocked: {reason}. Continue working."
        self.assertTrue(
            any(expected == t for t in texts if isinstance(t, str)),
            f"verbatim reason missing: {texts!r}",
        )

    def test_soft_ending_clause_is_not_gated(self):
        """A final assistant message with no tool calls ends the run without
        firing Stop (the done-gate only covers the terminal-tool path)."""
        marker = self.tmpdir / "stop-fired.json"
        script = _write_script(self.tmpdir, "capture.py", CAPTURE_SCRIPT)
        self._write_config({"Stop": [{"command": f"{sys.executable} {script} {marker}"}]})
        result, provider, _ = self._run([_final_text_response()])
        self.assertEqual(provider.chat.call_count, 1)
        self.assertFalse(marker.exists(), "Stop hook must not fire on the soft ending")
        self.assertIn("ended_without_explicit_completion", self._events(result))

    def test_failing_hook_fails_open(self):
        """A hook that crashes/exits non-zero never blocks the run."""
        self._write_config({"Stop": [{"command": "exit 42"}]})
        result, provider, _ = self._run([_tool_response(_complete_task_use())])
        self.assertEqual(provider.chat.call_count, 1)
        self.assertIn("[Task complete", result.response_text)
        self.assertNotIn("done_gate_block", self._events(result))

    def test_stop_hook_receives_completion_payload(self):
        """The Stop payload carries the terminal tool's outcome/summary."""
        marker = self.tmpdir / "stop-payload.json"
        script = _write_script(self.tmpdir, "capture.py", CAPTURE_SCRIPT)
        self._write_config({"Stop": [{"command": f"{sys.executable} {script} {marker}"}]})
        self._run([_tool_response(_complete_task_use())])
        self.assertTrue(marker.exists())
        payload = json.loads(marker.read_text())
        self.assertEqual(payload["event"], "Stop")
        self.assertEqual(payload["tool"], "complete_task")
        self.assertEqual(payload["outcome"], "success")
        self.assertEqual(payload["summary"], "task done")


class StopHookRunnerTestCase(unittest.TestCase):
    """run_hooks-level: Stop is a blocking-capable event."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.hooks_json = self.tmpdir / "hooks.json"
        self.audit_log = self.tmpdir / "hooks-audit.log"
        self._config_patcher = patch(
            "src.tool_system.hooks.runner._default_config_path",
            return_value=self.hooks_json,
        )
        self._audit_patcher = patch(
            "src.tool_system.hooks.runner._audit_log_path",
            return_value=self.audit_log,
        )
        self._config_patcher.start()
        self._audit_patcher.start()

    def tearDown(self):
        self._audit_patcher.stop()
        self._config_patcher.stop()
        self.tmp.cleanup()

    def test_block_decision_blocks(self):
        script = _write_script(self.tmpdir, "block.py", BLOCK_SCRIPT)
        self.hooks_json.write_text(
            json.dumps({"Stop": [{"command": f"{sys.executable} {script} 'nope'"}]})
        )
        out = run_hooks("Stop", extra={"tool": "complete_task", "outcome": "success"})
        self.assertTrue(out["blocked"])
        self.assertEqual(out["decision"], "block")
        self.assertEqual(out["message"], "nope")

    def test_allow_decision_allows(self):
        script = _write_script(self.tmpdir, "allow.py", ALLOW_SCRIPT)
        self.hooks_json.write_text(
            json.dumps({"Stop": [{"command": f"{sys.executable} {script}"}]})
        )
        out = run_hooks("Stop", extra={"tool": "complete_task"})
        self.assertFalse(out["blocked"])
        self.assertEqual(out["decision"], "allow")


class DefaultChecklistGateTestCase(unittest.TestCase):
    """The reference checklist gate (examples/stop_done_gate.py)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmpdir = Path(self.tmp.name)
        self.home = self.tmpdir / "home"
        self.home.mkdir()
        self._home_patcher = patch.dict(os.environ, {"HOME": str(self.home)})
        self._home_patcher.start()
        self.workdir = self.tmpdir / "repo"
        self.workdir.mkdir()

    def tearDown(self):
        self._home_patcher.stop()
        self.tmp.cleanup()

    def _gate(self, payload: dict, config: dict | None) -> dict:
        if config is not None:
            (self.home / ".clawd").mkdir(exist_ok=True)
            (self.home / ".clawd" / "done-gate.json").write_text(json.dumps(config))
        proc = subprocess.run(
            [sys.executable, str(GATE_SCRIPT)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(self.workdir),
        )
        self.assertEqual(proc.returncode, 0)
        return json.loads(proc.stdout)

    def _payload(self, **overrides) -> dict:
        base = {
            "event": "Stop",
            "tool": "complete_task",
            "outcome": "success",
            "summary": "done",
            "artifacts": [],
        }
        base.update(overrides)
        return base

    def test_no_config_allows_fail_open(self):
        out = self._gate(self._payload(), None)
        self.assertEqual(out["decision"], "allow")

    def test_disabled_allows(self):
        out = self._gate(self._payload(), {"enabled": False})
        self.assertEqual(out["decision"], "allow")

    def test_failing_test_command_blocks_with_reason(self):
        out = self._gate(
            self._payload(),
            {"enabled": True, "test_command": "false", "check_artifacts": False},
        )
        self.assertEqual(out["decision"], "block")
        self.assertIn("test command failed", out["reason"])
        self.assertIn("exit 1", out["reason"])

    def test_passing_test_command_allows(self):
        out = self._gate(
            self._payload(),
            {"enabled": True, "test_command": "true", "check_artifacts": False},
        )
        self.assertEqual(out["decision"], "allow")

    def test_missing_artifact_blocks(self):
        out = self._gate(
            self._payload(artifacts=["nope/missing.txt"]),
            {"enabled": True, "test_command": "true"},
        )
        self.assertEqual(out["decision"], "block")
        self.assertIn("nope/missing.txt", out["reason"])

    def test_artifact_dict_form_blocks_when_missing(self):
        out = self._gate(
            self._payload(artifacts=[{"name": "report", "path": "nope/out.md"}]),
            {"enabled": True, "test_command": "true"},
        )
        self.assertEqual(out["decision"], "block")
        self.assertIn("nope/out.md", out["reason"])

    def test_existing_artifact_allows(self):
        (self.workdir / "out.md").write_text("x")
        out = self._gate(
            self._payload(artifacts=[{"name": "report", "path": "out.md"}]),
            {"enabled": True, "test_command": "true"},
        )
        self.assertEqual(out["decision"], "allow")

    def test_failure_outcome_is_not_gated(self):
        out = self._gate(
            self._payload(tool="ReportFailure", outcome="failed"),
            {"enabled": True, "test_command": "false"},
        )
        self.assertEqual(out["decision"], "allow")

    def test_malformed_input_fails_open(self):
        proc = subprocess.run(
            [sys.executable, str(GATE_SCRIPT)],
            input="this is not json",
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(self.workdir),
        )
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(json.loads(proc.stdout)["decision"], "allow")


if __name__ == "__main__":
    unittest.main()
