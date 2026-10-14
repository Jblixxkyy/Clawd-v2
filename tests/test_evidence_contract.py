"""Evidence-before-submit contract on the completion tool (brief 014).

The terminal tool (StructuredOutput) carries the evidence contract in its
description: the model may only submit when it has verified the work by
execution (re-read the task, confirm artifacts from its own tool output, run
tests or run the solution end-to-end). The ``summary`` and ``verified`` fields
are schema-required so the shape is enforced, not suggested.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.tool_system.context import ToolContext
from src.tool_system.defaults import build_default_registry
from src.tool_system.errors import ToolInputError
from src.tool_system.protocol import ToolCall
from src.tool_system.tools.structured_output import StructuredOutputTool


class TestEvidenceContract(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.ctx = ToolContext(workspace_root=self.root)
        self.registry = build_default_registry(include_user_tools=False)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _desc(self) -> str:
        return StructuredOutputTool().spec().description

    # --- the contract ships in the tool description (prompt-level regression) ---
    def test_description_contains_evidence_requirements(self) -> None:
        desc = self._desc().lower()
        for marker in (
            "evidence-before-submit",
            "re-read the original task",
            "from your own tool output",
            "re-read each file",
            "format, types, and values",
            "run them and confirm they are green",
            "end-to-end",
            "read the output back",
            "never submit on assumption",
            "never claim work you did not execute",
        ):
            self.assertIn(marker, desc, f"contract marker missing from description: {marker!r}")

    def test_description_stays_prompt_diet_friendly(self) -> None:
        words = len(self._desc().split())
        self.assertLessEqual(words, 180, f"contract description must stay ~150 words, got {words}")

    # --- summary + verified are required by the schema ---
    def test_summary_and_verified_required_by_schema(self) -> None:
        schema = StructuredOutputTool().spec().input_schema
        required = list(schema.get("required", []))
        self.assertIn("summary", required)
        self.assertIn("verified", required)

    # --- a submission without `verified` is rejected at the registry layer ---
    def test_registry_rejects_submission_missing_verified(self) -> None:
        call = ToolCall(name="StructuredOutput", input={"summary": "did the thing"})
        with self.assertRaises(ToolInputError) as cm:
            self.registry.dispatch(call, self.ctx)
        self.assertIn("verified", str(cm.exception))

    def test_registry_rejects_submission_missing_summary(self) -> None:
        call = ToolCall(name="StructuredOutput", input={"verified": "re-read the file, output read back"})
        with self.assertRaises(ToolInputError):
            self.registry.dispatch(call, self.ctx)

    # --- empty fields are rejected even on the direct-call path ---
    def test_run_rejects_empty_fields(self) -> None:
        tool = StructuredOutputTool()
        with self.assertRaises(ToolInputError):
            tool.run({"summary": "x", "verified": "   "}, self.ctx)
        with self.assertRaises(ToolInputError):
            tool.run({"summary": "", "verified": "y"}, self.ctx)

    # --- happy path: a full evidence submission is accepted ---
    def test_valid_evidence_submission_accepted(self) -> None:
        call = ToolCall(
            name="StructuredOutput",
            input={
                "summary": "fixed the bug",
                "verified": "re-read the edited file; pytest run green (42 passed)",
            },
        )
        result = self.registry.dispatch(call, self.ctx)
        self.assertFalse(result.is_error)
        self.assertEqual(result.output["structured_output"]["summary"], "fixed the bug")
        self.assertEqual(
            result.output["structured_output"]["verified"],
            "re-read the edited file; pytest run green (42 passed)",
        )


if __name__ == "__main__":
    unittest.main()
