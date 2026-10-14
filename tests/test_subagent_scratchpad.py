"""Tests for the forced scratchpad protocol (brief 024).

Covers the scratchpad artifact (whole-overwrite updates, validation,
termination gate, forced wrap-up), the Scratchpad tool, and the
investigator subagent profile wired through the Subagent tool and the
``run_investigator_subagent`` entry point (the clean API the 018
reviewer builder calls).
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from src.agent.scratchpad import (
    INVESTIGATOR_MAX_TURNS,
    INVESTIGATOR_TEMPERATURE,
    Scratchpad,
    ScratchpadError,
    ScratchpadTool,
    extract_json_payload,
    forced_wrapup,
    investigator_system_preamble,
    record_verdict,
    termination_check,
    validate_final_output,
)
from src.model_tiers import weak_profile_allowed_tools
from src.providers.base import ChatResponse
from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError
from src.tool_system.protocol import ToolCall, ToolResult
from src.tool_system.registry import ToolRegistry, ToolSpec
from src.tool_system.tools.subagent import (
    SubagentTool,
    _run_child_loop,
    build_weak_child_registry,
    run_investigator_subagent,
)


# ---------------------------------------------------------------------------
# Stubs (mirroring tests/test_subagent.py)
# ---------------------------------------------------------------------------

def _response(text: str, tool_uses: list[dict] | None = None) -> ChatResponse:
    return ChatResponse(
        content=text,
        model="stub-model",
        usage={"input_tokens": 5, "output_tokens": 5},
        finish_reason="tool_use" if tool_uses else "stop",
        tool_uses=tool_uses or [],
    )


class RecordingProvider:
    """Minimal stub provider that records kwargs (incl. temperature)."""

    def __init__(self, model: str, script: list[ChatResponse]):
        self.model = model
        self._script = list(script)
        self.calls: list[tuple[str, dict]] = []

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        self.calls.append((self.model, kwargs))
        if not self._script:
            return _response("script exhausted")
        return self._script.pop(0)

    def chat_stream_response(self, *args: Any, **kwargs: Any):
        raise NotImplementedError()


class LoopingScratchpadProvider(RecordingProvider):
    """Always emits a Scratchpad update with one open question, forever."""

    def __init__(self, model: str = "stub-model"):
        super().__init__(model, [])
        self._n = 0

    def chat(self, messages: list[dict], **kwargs: Any) -> ChatResponse:
        self.calls.append((self.model, kwargs))
        self._n += 1
        return _response(
            "updating",
            [
                {
                    "id": f"t{self._n}",
                    "name": "Scratchpad",
                    "input": {
                        "checklist": ["keep going"],
                        "questions_to_resolve": ["still open"],
                        "key_findings": [],
                        "irrelevant_paths": [],
                    },
                }
            ],
        )


class ReadStub:
    """Read-only stub tool (inside 028's weak profile set)."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Read",
            description="stub read",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
            is_read_only=True,
        )

    def run(self, tool_input: dict, context: ToolContext) -> ToolResult:
        return ToolResult(name="Read", output={"ok": True})


class WriteStub:
    """Mutation stub tool (outside the weak profile's read-only set)."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Write",
            description="stub write",
            input_schema={"type": "object", "additionalProperties": False, "properties": {}},
        )

    def run(self, tool_input: dict, context: ToolContext) -> ToolResult:
        return ToolResult(name="Write", output={"ok": True})


def _scratchpad_use(
    questions: list[str],
    findings: list[str] | None = None,
    checklist: list[str] | None = None,
    irrelevant: list[str] | None = None,
    tool_id: str = "t1",
) -> dict:
    return {
        "id": tool_id,
        "name": "Scratchpad",
        "input": {
            "checklist": checklist or ["investigate"],
            "questions_to_resolve": questions,
            "key_findings": findings or [],
            "irrelevant_paths": irrelevant or [],
        },
    }


VALID_FINAL = json.dumps({"summary": "done", "findings": ["f1"]})


# ---------------------------------------------------------------------------
# Artifact unit tests
# ---------------------------------------------------------------------------

class TestScratchpadArtifact(unittest.TestCase):
    def test_initial_seeds_questions(self) -> None:
        sp = Scratchpad.initial("survey the repo", ["Q1", "Q2"])
        self.assertEqual(sp.questions_to_resolve, ["Q1", "Q2"])
        self.assertFalse(sp.is_complete())

    def test_initial_without_seeds_derives_one(self) -> None:
        sp = Scratchpad.initial("survey the repo")
        self.assertEqual(len(sp.questions_to_resolve), 1)
        self.assertIn("survey the repo", sp.questions_to_resolve[0])

    def test_validate_payload_rejects_bad_shapes(self) -> None:
        self.assertTrue(Scratchpad.validate_payload("nope"))
        self.assertTrue(Scratchpad.validate_payload({"checklist": []}))
        self.assertTrue(
            Scratchpad.validate_payload(
                {
                    "checklist": "not-a-list",
                    "questions_to_resolve": [],
                    "key_findings": [],
                    "irrelevant_paths": [],
                }
            )
        )
        self.assertTrue(
            Scratchpad.validate_payload(
                {
                    "checklist": [],
                    "questions_to_resolve": [42],
                    "key_findings": [],
                    "irrelevant_paths": [],
                }
            )
        )
        good = {
            "checklist": ["a"],
            "questions_to_resolve": [],
            "key_findings": ["f"],
            "irrelevant_paths": ["x"],
        }
        self.assertEqual(Scratchpad.validate_payload(good), [])

    def test_update_is_whole_overwrite(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        sp.update(
            {
                "checklist": ["new plan"],
                "questions_to_resolve": [],
                "key_findings": ["answer 1"],
                "irrelevant_paths": ["dead end"],
            }
        )
        self.assertTrue(sp.is_complete())
        self.assertEqual(sp.key_findings, ["answer 1"])
        self.assertEqual(sp.irrelevant_paths, ["dead end"])
        # A second update replaces everything again -- no merging.
        sp.update(
            {
                "checklist": [],
                "questions_to_resolve": ["Q2"],
                "key_findings": [],
                "irrelevant_paths": [],
            }
        )
        self.assertEqual(sp.questions_to_resolve, ["Q2"])
        self.assertEqual(sp.key_findings, [])

    def test_update_reports_resolved_questions(self) -> None:
        sp = Scratchpad.initial("d", ["Q1", "Q2"])
        delta = sp.update(
            {
                "checklist": [],
                "questions_to_resolve": ["Q2"],
                "key_findings": ["a1"],
                "irrelevant_paths": [],
            }
        )
        self.assertEqual(delta["questions_resolved"], ["Q1"])
        self.assertEqual(delta["open_questions"], 1)
        self.assertFalse(delta["complete"])

    def test_update_rejects_invalid_and_keeps_state(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        with self.assertRaises(ScratchpadError):
            sp.update({"checklist": []})
        self.assertEqual(sp.questions_to_resolve, ["Q1"])

    def test_from_dict_roundtrip(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        sp2 = Scratchpad.from_dict(sp.to_dict())
        self.assertEqual(sp2.to_dict(), sp.to_dict())
        with self.assertRaises(ScratchpadError):
            Scratchpad.from_dict({"nope": True})

    def test_extract_json_payload(self) -> None:
        self.assertEqual(extract_json_payload('{"a": 1}'), {"a": 1})
        wrapped = 'Here is my report:\n```json\n{"summary": "x"}\n```\nDone.'
        self.assertEqual(extract_json_payload(wrapped), {"summary": "x"})
        with self.assertRaises(ValueError):
            extract_json_payload("no json here at all")

    def test_validate_final_output(self) -> None:
        ok, parsed = validate_final_output(VALID_FINAL)
        self.assertTrue(ok)
        self.assertEqual(parsed["summary"], "done")
        ok, _ = validate_final_output("just prose")
        self.assertFalse(ok)
        ok, _ = validate_final_output(json.dumps({"findings": []}))
        self.assertFalse(ok)  # missing required "summary"
        schema = {"type": "object", "required": ["verdict"], "properties": {}}
        ok, parsed = validate_final_output('{"verdict": "pass"}', schema)
        self.assertTrue(ok)
        self.assertEqual(parsed["verdict"], "pass")

    def test_termination_check_complete(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        sp.update(
            {
                "checklist": [],
                "questions_to_resolve": [],
                "key_findings": [],
                "irrelevant_paths": [],
            }
        )
        d = termination_check(
            scratchpad=sp,
            response_text=VALID_FINAL,
            output_schema=None,
            turns_used=3,
            max_turns=50,
            time_exceeded=False,
        )
        self.assertTrue(d.finished)
        self.assertEqual(d.reason, "complete")

    def test_termination_check_open_questions_continues(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        d = termination_check(
            scratchpad=sp,
            response_text=VALID_FINAL,
            output_schema=None,
            turns_used=3,
            max_turns=50,
            time_exceeded=False,
        )
        self.assertFalse(d.finished)
        self.assertEqual(d.reason, "open_questions")
        self.assertIn("Q1", d.continue_prompt or "")

    def test_termination_check_invalid_output_continues(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        sp.update(
            {
                "checklist": [],
                "questions_to_resolve": [],
                "key_findings": [],
                "irrelevant_paths": [],
            }
        )
        d = termination_check(
            scratchpad=sp,
            response_text="not json",
            output_schema=None,
            turns_used=3,
            max_turns=50,
            time_exceeded=False,
        )
        self.assertFalse(d.finished)
        self.assertEqual(d.reason, "invalid_output")

    def test_termination_check_caps_finish(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        d = termination_check(
            scratchpad=sp, response_text="x", output_schema=None,
            turns_used=50, max_turns=50, time_exceeded=False,
        )
        self.assertTrue(d.finished)
        self.assertEqual(d.reason, "turn_cap")
        d = termination_check(
            scratchpad=sp, response_text="x", output_schema=None,
            turns_used=3, max_turns=50, time_exceeded=True,
        )
        self.assertTrue(d.finished)
        self.assertEqual(d.reason, "time_cap")

    def test_forced_wrapup_carries_scratchpad(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        sp.update(
            {
                "checklist": [],
                "questions_to_resolve": ["Q1"],
                "key_findings": ["partial"],
                "irrelevant_paths": ["dead"],
            }
        )
        wrap = forced_wrapup(sp, "last words", "turn_cap")
        self.assertTrue(wrap["forced_wrapup"])
        self.assertEqual(wrap["cap_reason"], "turn_cap")
        self.assertEqual(wrap["open_questions"], ["Q1"])
        self.assertEqual(wrap["findings"], ["partial"])
        self.assertIn("Q1", wrap["scratchpad"]["questions_to_resolve"])

    def test_record_verdict_folds_into_final_state(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        snapshot = record_verdict(sp, {"verdict": "pass", "confidence": 0.9})
        self.assertTrue(any("verdict" in f for f in sp.key_findings))
        self.assertIn("key_findings", snapshot)
        with self.assertRaises(ScratchpadError):
            record_verdict(sp, "not-a-dict")

    def test_preamble_mentions_protocol(self) -> None:
        text = investigator_system_preamble(None, ["seed q"])
        self.assertIn("Scratchpad", text)
        self.assertIn("questions_to_resolve", text)
        self.assertIn("seed q", text)
        self.assertIn("50", text)


# ---------------------------------------------------------------------------
# Scratchpad tool tests
# ---------------------------------------------------------------------------

class TestScratchpadTool(unittest.TestCase):
    def test_tool_updates_and_broadcasts(self) -> None:
        seen: list[dict] = []
        sp = Scratchpad.initial("d", ["Q1"])
        tool = ScratchpadTool(sp, on_update=seen.append)
        result = tool.run(
            {
                "checklist": [],
                "questions_to_resolve": [],
                "key_findings": ["a1"],
                "irrelevant_paths": [],
            },
            context=MagicMock(),
        )
        self.assertFalse(result.is_error)
        self.assertTrue(result.output["complete"])
        self.assertEqual(result.output["open_questions"], 0)
        self.assertEqual(tool.update_count, 1)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["key_findings"], ["a1"])

    def test_tool_rejects_invalid_payload(self) -> None:
        sp = Scratchpad.initial("d", ["Q1"])
        tool = ScratchpadTool(sp)
        result = tool.run({"checklist": []}, context=MagicMock())
        self.assertTrue(result.is_error)
        self.assertIn("invalid scratchpad payload", result.output["error"])
        self.assertEqual(sp.questions_to_resolve, ["Q1"])  # untouched
        self.assertEqual(tool.update_count, 0)

    def test_tool_self_allows_under_weak_handler(self) -> None:
        from src.model_tiers import weak_read_permission_handler

        sp = Scratchpad.initial("d")
        tool = ScratchpadTool(sp)
        ctx = MagicMock()
        ctx.permission_handler = weak_read_permission_handler
        # The tool's own check wins over the weak read-only handler, which
        # would otherwise deny a non-read tool name.
        res = tool.check_permissions({}, ctx)
        self.assertEqual(res.behavior.value, "allow")
        self.assertFalse(weak_read_permission_handler("Scratchpad", "m", None)[0])


# ---------------------------------------------------------------------------
# Investigator loop tests (end-to-end through the real agent loop)
# ---------------------------------------------------------------------------

class TestInvestigatorLoop(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.parent_registry = ToolRegistry()
        self.parent_registry.register(ReadStub())
        self.parent_registry.register(WriteStub())
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run(self, provider, **kwargs) -> dict:
        params = dict(
            task_id="t1",
            description="survey the repo",
            prompt="Survey the repo and report.",
            provider=provider,
            parent_registry=self.parent_registry,
            parent_context=self.context,
            max_turns=kwargs.pop("max_turns", 50),
            model=None,
            profile="investigator",
            seed_questions=kwargs.pop("seed_questions", ["Q1", "Q2"]),
            max_time_minutes=kwargs.pop("max_time_minutes", 10.0),
        )
        params.update(kwargs)
        return _run_child_loop(**params)

    def test_questions_shrink_across_turns(self) -> None:
        provider = RecordingProvider(
            "parent-model",
            [
                _response("plan", [_scratchpad_use(["Q1", "Q2"], tool_id="t1")]),
                _response("working", [_scratchpad_use(["Q2"], ["found Q1"], tool_id="t2")]),
                _response("almost", [_scratchpad_use([], ["found Q1", "found Q2"], tool_id="t3")]),
                _response(VALID_FINAL),
            ],
        )
        record = self._run(provider)
        self.assertEqual(record["status"], "done")
        self.assertTrue(record["scratchpad_complete"])
        self.assertEqual(record["scratchpad"]["questions_to_resolve"], [])
        self.assertEqual(record["scratchpad_updates"], 3)
        self.assertEqual(len(record["scratchpad_history"]), 3)
        self.assertFalse(record["forced_wrapup"])
        self.assertEqual(record["final_output"]["summary"], "done")
        self.assertEqual(record["num_turns"], 4)

    def test_finish_with_open_questions_is_continued(self) -> None:
        provider = RecordingProvider(
            "parent-model",
            [
                _response("plan", [_scratchpad_use(["Q1"], tool_id="t1")]),
                _response("I am done"),  # not JSON + questions open
                _response("fine", [_scratchpad_use([], ["a1"], tool_id="t3")]),
                _response(VALID_FINAL),
            ],
        )
        record = self._run(provider)
        # 4 provider calls proves the harness continued past the early stop.
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(record["status"], "done")
        self.assertTrue(record["scratchpad_complete"])
        self.assertFalse(record["forced_wrapup"])

    def test_finish_with_invalid_output_is_continued(self) -> None:
        provider = RecordingProvider(
            "parent-model",
            [
                _response("plan", [_scratchpad_use([], ["a1"], tool_id="t1")]),
                _response("prose report, no json"),
                _response(VALID_FINAL),
            ],
        )
        record = self._run(provider, seed_questions=[])
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(record["status"], "done")
        self.assertEqual(record["final_output"]["summary"], "done")

    def test_turn_cap_forces_wrapup(self) -> None:
        provider = LoopingScratchpadProvider()
        record = self._run(provider, max_turns=2)
        self.assertEqual(record["status"], "done")
        self.assertTrue(record["forced_wrapup"])
        self.assertEqual(record["num_turns"], 2)
        self.assertFalse(record["scratchpad_complete"])
        self.assertTrue(record["final_output"]["forced_wrapup"])
        self.assertEqual(record["final_output"]["cap_reason"], "turn_cap")
        parsed = json.loads(record["response_text"])
        self.assertTrue(parsed["forced_wrapup"])

    def test_time_cap_forces_wrapup(self) -> None:
        provider = LoopingScratchpadProvider()
        record = self._run(provider, max_time_minutes=0.001)
        self.assertEqual(record["status"], "done")
        self.assertTrue(record["forced_wrapup"])
        self.assertEqual(record["final_output"]["cap_reason"], "time_cap")

    def test_default_turn_cap_is_50(self) -> None:
        provider = LoopingScratchpadProvider()
        record = run_investigator_subagent(
            parent_registry=self.parent_registry,
            parent_context=self.context,
            provider=provider,
            description="never-ending",
            prompt="Never finish.",
            seed_questions=["Q1"],
            max_time_minutes=10.0,
        )
        self.assertEqual(record["num_turns"], INVESTIGATOR_MAX_TURNS)
        self.assertEqual(record["num_turns"], 50)
        self.assertTrue(record["forced_wrapup"])

    def test_temperature_pinned_to_01(self) -> None:
        provider = RecordingProvider("parent-model", [_response(VALID_FINAL)])
        self._run(provider, seed_questions=[])
        self.assertTrue(provider.calls)
        for _model, kwargs in provider.calls:
            self.assertEqual(kwargs.get("temperature"), INVESTIGATOR_TEMPERATURE)
            self.assertEqual(kwargs["temperature"], 0.1)
        # Parent provider object itself is untouched by the pin.
        self.assertNotIn("temperature", provider.__dict__)

    def test_investigator_uses_weak_profile_model(self) -> None:
        provider = RecordingProvider("parent-model", [_response(VALID_FINAL)])
        with patch(
            "src.tool_system.tools.subagent.weak_profile_model",
            return_value="small-model",
        ):
            record = self._run(provider, seed_questions=[])
        self.assertEqual(record["model"], "small-model")
        self.assertEqual(provider.calls[0][0], "small-model")

    def test_explicit_model_overrides_weak_profile_model(self) -> None:
        provider = RecordingProvider("parent-model", [_response(VALID_FINAL)])
        with patch(
            "src.tool_system.tools.subagent.weak_profile_model",
            return_value="small-model",
        ):
            record = self._run(provider, seed_questions=[], model="big-model")
        self.assertEqual(record["model"], "big-model")

    def test_mutation_tool_call_denied_end_to_end(self) -> None:
        provider = RecordingProvider(
            "parent-model",
            [
                _response("try write", [{"id": "t1", "name": "Write", "input": {}}]),
                _response("ok", [_scratchpad_use([], ["write denied"], tool_id="t2")]),
                _response(json.dumps({"summary": "write was denied"})),
            ],
        )
        record = self._run(provider, seed_questions=[])
        self.assertEqual(record["status"], "done")
        self.assertTrue(record["scratchpad_complete"])

    def test_weak_child_registry_denies_mutation(self) -> None:
        child = build_weak_child_registry(self.parent_registry)
        names = {t.spec().name.lower() for t in child.list_tools()}
        self.assertIn("read", names)
        self.assertNotIn("write", names)
        self.assertTrue(names <= weak_profile_allowed_tools())
        result = child.dispatch(
            ToolCall(name="Write", input={}), self.context
        )
        self.assertTrue(result.is_error)
        self.assertIn("unknown tool", str(result.output))

    def test_scratchpad_visible_to_child_registry(self) -> None:
        # The investigator's child registry = weak read set + Scratchpad.
        from src.tool_system.agent_loop import AgentLoopResult

        captured: dict[str, Any] = {}

        def fake_loop(*, conversation, provider, tool_registry, tool_context,
                      max_turns, **kw):
            captured["tools"] = {t.spec().name.lower() for t in tool_registry.list_tools()}
            return AgentLoopResult(response_text=VALID_FINAL, num_turns=1)

        provider = RecordingProvider("parent-model", [])
        with patch("src.tool_system.tools.subagent.run_agent_loop") as mock_loop:
            mock_loop.side_effect = fake_loop
            record = run_investigator_subagent(
                parent_registry=self.parent_registry,
                parent_context=self.context,
                provider=provider,
                description="d",
                prompt="p",
                seed_questions=["Q1"],
            )
        self.assertIn("scratchpad", captured["tools"])
        self.assertIn("read", captured["tools"])
        self.assertNotIn("write", captured["tools"])
        self.assertEqual(record["status"], "done")

    def test_on_scratchpad_update_callback(self) -> None:
        seen: list[dict] = []
        provider = RecordingProvider(
            "parent-model",
            [
                _response("plan", [_scratchpad_use(["Q1"], tool_id="t1")]),
                _response("done", [_scratchpad_use([], ["a1"], tool_id="t2")]),
                _response(VALID_FINAL),
            ],
        )
        record = run_investigator_subagent(
            parent_registry=self.parent_registry,
            parent_context=self.context,
            provider=provider,
            description="d",
            prompt="p",
            seed_questions=["Q1"],
            on_scratchpad_update=seen.append,
        )
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[-1]["questions_to_resolve"], [])
        self.assertEqual(record["status"], "done")


# ---------------------------------------------------------------------------
# Subagent tool wiring
# ---------------------------------------------------------------------------

class TestSubagentToolInvestigator(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.registry = ToolRegistry()
        self.registry.register(ReadStub())
        self.registry.register(WriteStub())
        self.registry.register(SubagentTool(self.registry))
        self.context = ToolContext(workspace_root=self.workspace)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _dispatch(self, tool_input: dict, provider: Any = None) -> ToolResult:
        if provider is not None:
            self.context.provider = provider
        return self.registry.dispatch(ToolCall(name="Subagent", input=tool_input), self.context)

    def test_investigator_profile_through_tool(self) -> None:
        provider = RecordingProvider(
            "parent-model",
            [
                _response("plan", [_scratchpad_use(["Q1"], tool_id="t1")]),
                _response("done", [_scratchpad_use([], ["a1"], tool_id="t2")]),
                _response(VALID_FINAL),
            ],
        )
        result = self._dispatch(
            {
                "description": "investigate",
                "prompt": "Investigate and report.",
                "profile": "investigator",
                "seed_questions": ["Q1"],
            },
            provider,
        )
        self.assertFalse(result.is_error)
        out = result.output
        self.assertEqual(out["profile"], "investigator")
        self.assertEqual(out["status"], "done")
        self.assertTrue(out["scratchpad_complete"])
        self.assertFalse(out["forced_wrapup"])
        self.assertIn("scratchpad", out)
        self.assertEqual(out["final_output"]["summary"], "done")

    def test_investigator_defaults_max_turns_50(self) -> None:
        provider = LoopingScratchpadProvider()
        result = self._dispatch(
            {
                "description": "never ends",
                "prompt": "Never finish.",
                "profile": "investigator",
            },
            provider,
        )
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["num_turns"], 50)
        self.assertTrue(result.output["forced_wrapup"])

    def test_investigator_custom_schema_enforced(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"verdict": {"type": "string"}},
            "required": ["verdict"],
        }
        provider = RecordingProvider(
            "parent-model",
            [
                _response("plan", [_scratchpad_use([], ["a1"], tool_id="t1")]),
                _response(json.dumps({"summary": "wrong shape"})),
                _response(json.dumps({"verdict": "pass"})),
            ],
        )
        result = self._dispatch(
            {
                "description": "review",
                "prompt": "Review and return a verdict.",
                "profile": "investigator",
                "output_schema": schema,
                "seed_questions": [],
            },
            provider,
        )
        self.assertFalse(result.is_error)
        # The wrong-shaped output was rejected by the gate; the run
        # continued until a schema-valid verdict arrived.
        self.assertEqual(len(provider.calls), 3)
        self.assertEqual(result.output["final_output"], {"verdict": "pass"})

    def test_invalid_profile_rejected(self) -> None:
        with self.assertRaises(ToolInputError):
            self._dispatch(
                {"description": "d", "prompt": "p", "profile": "nope"},
                RecordingProvider("m", []),
            )

    def test_invalid_investigator_inputs_rejected(self) -> None:
        # Validated at run() level (dispatch's input repair may legitimately
        # coerce e.g. a bare string into a one-element array first).
        tool = SubagentTool(self.registry)
        bad_inputs = [
            {"description": "d", "prompt": "p", "profile": "nope"},
            {"description": "d", "prompt": "p", "profile": "investigator",
             "seed_questions": [42]},
            {"description": "d", "prompt": "p", "profile": "investigator",
             "output_schema": 42},
            {"description": "d", "prompt": "p", "profile": "investigator",
             "max_time_minutes": "soon"},
        ]
        for bad in bad_inputs:
            with self.assertRaises(ToolInputError, msg=str(bad)):
                tool.run(bad, self.context)

    def test_standard_profile_unchanged(self) -> None:
        provider = RecordingProvider("parent-model", [_response("plain report")])
        result = self._dispatch(
            {"description": "plain", "prompt": "Say hi."}, provider
        )
        # Brief 020: a child that makes no tool calls and never calls a
        # terminal tool is a stall, not success.
        self.assertTrue(result.is_error)
        self.assertEqual(result.output["status"], "stalled")
        self.assertEqual(result.output["profile"], "standard")
        self.assertNotIn("scratchpad", result.output)


if __name__ == "__main__":
    unittest.main()
