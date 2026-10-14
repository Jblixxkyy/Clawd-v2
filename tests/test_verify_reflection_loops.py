"""Brief 019: verification loops — post-edit lint/test reflection + edit-corrector.

Covers the expected tests from harness-backlog/019-verify-reflection-loops.md:
- edit introducing a lint error -> lint runs automatically, failure text fed
  back, model fixes it within the cap;
- 3 consecutive failed reflections -> stops and reports, no infinite loop;
- whitespace-only edit failure -> corrector repairs without the main model
  (mocked cheap model); semantic failure -> corrector refuses;
- interactive mode proposes before re-running; non-interactive re-runs
  proactively; explicit test tasks bypass the hold.
"""

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.agent.conversation import Conversation
from src.providers.base import ChatResponse
from src.tool_system import verify as verify_mod
from src.tool_system.agent_loop import run_agent_loop
from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry


LINT_SH = """\
#!/bin/sh
# Fails when app.py contains the BROKEN marker.
if grep -q BROKEN app.py 2>/dev/null; then
  echo "lint: BROKEN marker found in app.py"
  exit 1
fi
exit 0
"""

ALWAYS_FAIL_SH = """\
#!/bin/sh
echo "lint: always fails"
exit 1
"""


def _resp(content="", tool_uses=None):
    return ChatResponse(
        content=content,
        model="test-model",
        usage={"input_tokens": 10, "output_tokens": 10},
        finish_reason="tool_use" if tool_uses else "stop",
        tool_uses=tool_uses,
    )


def _tool_use(tool_id, name, tool_input):
    return {"id": tool_id, "name": name, "input": tool_input}


class VerifyReflectionTestBase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = build_default_registry()
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_script(self, name, body):
        p = self.workspace / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        return p

    def _mock_provider(self, responses):
        provider = MagicMock()
        provider.model = "test-model"
        provider.chat_stream_response.side_effect = NotImplementedError()
        provider.chat.side_effect = list(responses)
        return provider

    def _run(self, prompt, provider, script_mode=False):
        conversation = Conversation()
        conversation.add_user_message(prompt)
        result = run_agent_loop(
            conversation=conversation,
            provider=provider,
            tool_registry=self.registry,
            tool_context=self.context,
            script_mode=script_mode,
            verbose=False,
        )
        return conversation, result

    def _reflection_events(self, result):
        return [e for e in result.run_log if e.get("event") == "reflection"]

    def _user_texts(self, conversation):
        texts = []
        for msg in conversation.messages:
            if msg.role == "user" and isinstance(msg.content, str):
                texts.append(msg.content)
        return texts


class TestPostEditReflection(VerifyReflectionTestBase):
    def test_lint_failure_reflected_and_fixed_within_cap(self):
        """Edit introducing a lint error -> lint runs, failure fed back,
        model fixes it within the cap (non-interactive: proactive)."""
        self._write_script("lint.sh", LINT_SH)
        self.context.verify_config = {
            "enabled": True,
            "lint_cmd": "./lint.sh",
            "test_cmd": None,
        }
        app = str(self.workspace / "app.py")
        provider = self._mock_provider([
            _resp("writing", [_tool_use("t1", "Write", {
                "file_path": app, "content": "x = BROKEN\n"})]),
            _resp("fixing", [_tool_use("t2", "Edit", {
                "file_path": app,
                "old_string": "x = BROKEN",
                "new_string": "x = 1",
            })]),
            _resp("done"),
        ])
        conversation, result = self._run(
            "write app.py", provider, script_mode=True)

        self.assertEqual(Path(app).read_text(), "x = 1\n")
        self.assertEqual(provider.chat.call_count, 3)
        events = self._reflection_events(result)
        self.assertEqual([e["passed"] for e in events], [False, True])
        self.assertEqual(events[0]["kind"], "lint")
        # Raw failure output became the next user message (reflected_message).
        user_texts = self._user_texts(conversation)
        self.assertTrue(
            any("lint: BROKEN marker found in app.py" in t for t in user_texts),
            "raw lint failure should be injected as a user message",
        )
        self.assertTrue(
            any("Fix the failures above" in t for t in user_texts),
            "non-interactive runs verify proactively",
        )

    def test_three_failed_reflections_stop_and_report(self):
        """3 consecutive failed reflections -> stops and reports to the user."""
        self._write_script("lint.sh", ALWAYS_FAIL_SH)
        self.context.verify_config = {
            "enabled": True,
            "lint_cmd": "./lint.sh",
            "test_cmd": None,
        }
        app = str(self.workspace / "app.py")
        responses = [
            _resp("writing", [_tool_use("t1", "Write", {
                "file_path": app, "content": "x = 1\n"})]),
        ]
        for i, val in enumerate(["2", "3", "4"]):
            responses.append(_resp("tweak", [_tool_use(
                f"t{i + 2}", "Edit", {
                    "file_path": app,
                    "old_string": f"x = {i + 1}",
                    "new_string": f"x = {val}",
                })]))
        provider = self._mock_provider(responses)
        conversation, result = self._run(
            "write app.py", provider, script_mode=True)

        # 4 model turns: write + 3 reflected edits, then the cap escalates.
        self.assertEqual(provider.chat.call_count, 4)
        self.assertIn("stopping instead of looping", result.response_text)
        self.assertIn("lint: always fails", result.response_text)
        cap_events = [e for e in result.run_log
                      if e.get("event") == "reflection_cap_reached"]
        self.assertEqual(len(cap_events), 1)
        self.assertEqual(cap_events[0]["attempts"], 4)
        fails = [e for e in self._reflection_events(result) if not e["passed"]]
        self.assertEqual(len(fails), 4)

    def test_interactive_mode_proposes_before_rerun(self):
        """Interactive: test failure proposes instead of re-running."""
        self._write_script("lint.sh", LINT_SH)
        self.context.verify_config = {
            "enabled": True,
            "lint_cmd": "./lint.sh",
            "test_cmd": None,
        }
        app = str(self.workspace / "app.py")
        provider = self._mock_provider([
            _resp("writing", [_tool_use("t1", "Write", {
                "file_path": app, "content": "x = BROKEN\n"})]),
            _resp("The lint check failed. Shall I fix and re-run the checks?"),
        ])
        conversation, result = self._run("write app.py", provider,
                                         script_mode=False)

        self.assertEqual(provider.chat.call_count, 2)
        self.assertIn("fix and re-run", result.response_text)
        user_texts = self._user_texts(conversation)
        self.assertTrue(
            any("do NOT keep editing on your own" in t for t in user_texts),
            "interactive runs must propose-first, not auto-fix",
        )
        self.assertFalse(
            any("Fix the failures above and continue" in t for t in user_texts),
        )

    def test_explicit_test_task_bypasses_hold_in_interactive(self):
        """Explicit test tasks bypass the interactive propose-first hold."""
        self._write_script("lint.sh", LINT_SH)
        self.context.verify_config = {
            "enabled": True,
            "lint_cmd": "./lint.sh",
            "test_cmd": None,
        }
        app = str(self.workspace / "app.py")
        provider = self._mock_provider([
            _resp("writing", [_tool_use("t1", "Write", {
                "file_path": app, "content": "x = BROKEN\n"})]),
            _resp("fixing", [_tool_use("t2", "Edit", {
                "file_path": app,
                "old_string": "x = BROKEN",
                "new_string": "x = 1",
            })]),
            _resp("tests pass now"),
        ])
        # Interactive mode, but the prompt explicitly asks for tests.
        conversation, result = self._run(
            "Write app.py and make sure the tests pass", provider,
            script_mode=False)

        self.assertEqual(Path(app).read_text(), "x = 1\n")
        self.assertEqual(provider.chat.call_count, 3)
        user_texts = self._user_texts(conversation)
        self.assertTrue(
            any("Fix the failures above and continue" in t for t in user_texts),
            "explicit test tasks verify proactively even when interactive",
        )

    def test_no_edits_no_verification(self):
        """Turns without file edits never run the verify commands."""
        self._write_script("lint.sh", ALWAYS_FAIL_SH)
        self.context.verify_config = {
            "enabled": True,
            "lint_cmd": "./lint.sh",
            "test_cmd": None,
        }
        provider = self._mock_provider([_resp("just chatting")])
        conversation, result = self._run(
            "hello", provider, script_mode=True)

        self.assertEqual(result.response_text, "just chatting")
        self.assertEqual(self._reflection_events(result), [])


class TestEditCorrector(VerifyReflectionTestBase):
    def _setup_edit_scenario(self, content):
        app = self.workspace / "app.py"
        app.write_text(content)
        return str(app)

    def _corrector_response(self, old_string):
        return _resp(json.dumps({"old_string": old_string}))

    def test_whitespace_failure_repaired_by_mocked_cheap_model(self):
        """Whitespace-only failure -> corrector repairs via the cheap model;
        the main model never sees the failure."""
        app = self._setup_edit_scenario("def foo():\n    return 1\n")
        self.context.verify_config = {"enabled": True}  # no commands
        provider = self._mock_provider([
            _resp("editing", [
                _tool_use("t0", "Read", {"file_path": app}),
                _tool_use("t1", "Edit", {
                    "file_path": app,
                    "old_string": "def foo():\n  return 1\n",  # 2-space indent: miss
                    "new_string": "def foo():\n    return 2\n",
                }),
            ]),
            self._corrector_response("def foo():\n    return 1\n"),
            _resp("fixed"),
        ])
        conversation, result = self._run(
            "bump the return value", provider, script_mode=True)

        # The correction path ran (cheap model), and the edit landed.
        corrections = [e for e in result.run_log
                       if e.get("event") == "edit_correction"]
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["method"], "cheap_model")
        self.assertEqual(Path(app).read_text(), "def foo():\n    return 2\n")
        # Main model turns: 2. The cheap-model call consumed one chat call.
        self.assertEqual(provider.chat.call_count, 3)
        # The strict mechanics-only prompt was used for the corrector call.
        corrector_call = provider.chat.call_args_list[1]
        messages = corrector_call.args[0] if corrector_call.args else \
            corrector_call.kwargs["messages"]
        system_text = messages[0]["content"]
        self.assertIn("Do NOT invent", system_text)
        self.assertIn("fix only whitespace", system_text.lower())
        # The failure never reached the main model's context.
        user_texts = self._user_texts(conversation)
        self.assertFalse(any("old_string not found" in t for t in user_texts))

    def test_regex_fallback_when_cheap_model_refuses(self):
        """Cheap model returns null -> deterministic regex repair still fixes."""
        app = self._setup_edit_scenario("def foo():\n    return 1\n")
        self.context.verify_config = {"enabled": True}
        provider = self._mock_provider([
            _resp("editing", [
                _tool_use("t0", "Read", {"file_path": app}),
                _tool_use("t1", "Edit", {
                    "file_path": app,
                    "old_string": "def foo():\n  return 1\n",
                    "new_string": "def foo():\n    return 2\n",
                }),
            ]),
            self._corrector_response(None),  # refuses: not safely fixable
            _resp("fixed"),
        ])
        conversation, result = self._run(
            "bump the return value", provider, script_mode=True)

        corrections = [e for e in result.run_log
                       if e.get("event") == "edit_correction"]
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0]["method"], "regex")
        self.assertEqual(Path(app).read_text(), "def foo():\n    return 2\n")

    def test_semantic_failure_corrector_refuses(self):
        """Anchor that doesn't exist at all -> corrector refuses; the main
        model sees the original failure."""
        app = self._setup_edit_scenario("def foo():\n    return 1\n")
        self.context.verify_config = {"enabled": True}
        provider = self._mock_provider([
            _resp("editing", [
                _tool_use("t0", "Read", {"file_path": app}),
                _tool_use("t1", "Edit", {
                    "file_path": app,
                    "old_string": "def bar():\n    return 99\n",  # not in file
                    "new_string": "def bar():\n    return 100\n",
                }),
            ]),
            self._corrector_response(None),
            _resp("I see the edit failed; the function does not exist."),
        ])
        conversation, result = self._run(
            "change bar", provider, script_mode=True)

        corrections = [e for e in result.run_log
                       if e.get("event") == "edit_correction"]
        self.assertEqual(corrections, [])
        # File untouched, failure surfaced to the main model.
        self.assertEqual(Path(app).read_text(), "def foo():\n    return 1\n")
        self.assertIn("does not exist", result.response_text)
        # The original failure reached the main model as the tool result in
        # the turn-2 request (chat calls: turn1, corrector, turn2).
        turn2_call = provider.chat.call_args_list[2]
        turn2_messages = (turn2_call.args[0] if turn2_call.args
                          else turn2_call.kwargs["messages"])
        tool_texts = [m.get("content", "") for m in turn2_messages
                      if m.get("role") == "tool"]
        self.assertTrue(
            any("old_string not found" in t for t in tool_texts),
            "original failure should reach the main model",
        )

    def test_corrector_never_changes_new_string(self):
        """Even a creative cheap model cannot change edit semantics: a
        candidate that differs by more than whitespace is refused, and the
        regex fallback cannot repair a non-whitespace miss either."""
        corrector = verify_mod.EditCorrector()
        tool_input = {
            "file_path": str(self.workspace / "app.py"),
            # "return 2" vs the file's "return 1": a semantic miss, not a
            # whitespace mechanic.
            "old_string": "def foo():\n    return 2\n",
            "new_string": "def foo():\n    return 3\n",
        }
        (self.workspace / "app.py").write_text("def foo():\n    return 1\n")

        class SneakyProvider:
            model = "test-model"

            def chat(self, messages, model=None):
                # Tries to retarget the edit at a different anchor.
                return _resp(json.dumps(
                    {"old_string": "def foo():\n    return 1\n"}))

        out = corrector.correct(
            tool_input, "old_string not found in 'app.py' (0 occurrences).",
            self.context, SneakyProvider())
        self.assertIsNone(out, "non-whitespace anchor change must be rejected")

    def test_corrector_lru_cache(self):
        """Repeated identical failures reuse the cached correction."""
        corrector = verify_mod.EditCorrector(cache_size=2)
        (self.workspace / "app.py").write_text("def foo():\n    return 1\n")
        tool_input = {
            "file_path": str(self.workspace / "app.py"),
            "old_string": "def foo():\n  return 1\n",
            "new_string": "def foo():\n    return 2\n",
        }
        err = "old_string not found in 'app.py' (0 occurrences)."
        first = corrector.correct(tool_input, err, self.context, None)
        self.assertIsNotNone(first)
        self.assertEqual(first[1], "regex")
        # Second identical call: no provider, still repaired from cache.
        second = corrector.correct(tool_input, err, self.context, None)
        self.assertIsNotNone(second)
        self.assertEqual(second[1], "cache")
        self.assertEqual(first[0]["old_string"], second[0]["old_string"])
        # new_string untouched in both.
        self.assertEqual(second[0]["new_string"], "def foo():\n    return 2\n")


class TestVerifyHelpers(unittest.TestCase):
    def test_resolve_config_repo_detection_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "package.json").write_text(json.dumps({
                "scripts": {"test": "jest"},
            }))
            (root / "eslint.config.js").write_text("export default [];")
            cfg = verify_mod.resolve_verify_config(root, config={})
            self.assertEqual(cfg["test_cmd"], "npm test")
            self.assertEqual(cfg["lint_cmd"], "npx eslint .")
            self.assertTrue(cfg["enabled"])
            # User overrides win; explicit null disables a stage.
            cfg = verify_mod.resolve_verify_config(root, config={
                "verify": {"lint_cmd": "my-lint", "test_cmd": None,
                           "enabled": False},
            })
            self.assertEqual(cfg["lint_cmd"], "my-lint")
            self.assertIsNone(cfg["test_cmd"])
            self.assertFalse(cfg["enabled"])

    def test_resolve_config_never_creates_config_file(self):
        # Regression: the agent loop resolves verify config on every run;
        # a load_config()-style mkdir+write side effect would mutate the
        # filesystem (it broke test_delta_instructions' workspace snapshot).
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"HOME": tmp}):
                cfg = verify_mod.resolve_verify_config(tmp, config=None)
                self.assertTrue(cfg["enabled"])
                self.assertFalse((Path(tmp) / ".clawd").exists())

    def test_run_verify_command_statuses(self):
        status, out = verify_mod.run_verify_command("exit 0", ".")
        self.assertEqual(status, "pass")
        status, out = verify_mod.run_verify_command("echo oops >&2; exit 3", ".")
        self.assertEqual(status, "fail")
        self.assertIn("oops", out)
        status, out = verify_mod.run_verify_command("exit 0", "/nonexistent-dir")
        self.assertEqual(status, "skip")

    def test_is_explicit_test_task(self):
        conv = Conversation()
        conv.add_user_message("please run the tests and fix failures")
        self.assertTrue(verify_mod.is_explicit_test_task(conv))
        conv2 = Conversation()
        conv2.add_user_message("write me a poem")
        self.assertFalse(verify_mod.is_explicit_test_task(conv2))

    def test_is_mechanical_edit_error(self):
        self.assertTrue(verify_mod.is_mechanical_edit_error(
            "old_string not found in 'x.py' (0 occurrences)."))
        self.assertFalse(verify_mod.is_mechanical_edit_error(
            "file does not exist: x.py"))
        self.assertFalse(verify_mod.is_mechanical_edit_error(
            "expected_occurrences=2 but found 3 of old_string"))
        self.assertFalse(verify_mod.is_mechanical_edit_error(
            "old_string not found in 'x.py' (0 occurrences). "
            "refusing to edit: file must be read first"))

    def test_regex_repair_anchor(self):
        content = "def foo():\n    return 1\n"
        # Tabs vs spaces (trailing newline aligned from the intended anchor).
        repaired = verify_mod.regex_repair_anchor(
            "def foo():\n\treturn 1\n", content)
        self.assertEqual(repaired, "def foo():\n    return 1\n")
        # Trailing whitespace.
        repaired = verify_mod.regex_repair_anchor(
            "def foo():   \n    return 1\n", content)
        self.assertEqual(repaired, "def foo():\n    return 1\n")
        # Ambiguous match -> refuse.
        dup = "x = 1\nx = 1\n"
        self.assertIsNone(verify_mod.regex_repair_anchor("x  =  1", dup))
        # No match -> None.
        self.assertIsNone(verify_mod.regex_repair_anchor("zzz", content))


if __name__ == "__main__":
    unittest.main()
