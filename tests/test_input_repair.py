"""Tests for brief 007: tool input repair (coerce before rejecting)."""

from __future__ import annotations

import unittest

from src.tool_system.context import ToolContext
from src.tool_system.errors import ToolInputError
from src.tool_system.input_repair import repair_input
from src.tool_system.permission_handler import PermissionResult
from src.tool_system.protocol import ToolCall, ToolResult
from src.tool_system.registry import ToolRegistry, ToolSpec
from src.tool_system.schema_validation import validate_json_schema


SCHEMA = {
    "type": "object",
    "properties": {
        "limit": {"type": "integer"},
        "enabled": {"type": "boolean"},
        "name": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "cfg": {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        },
    },
    "required": ["limit"],
}


class RepairUnitTests(unittest.TestCase):
    def test_stringified_root_object(self) -> None:
        out = repair_input('{"limit": "20", "enabled": "false"}', SCHEMA)
        self.assertEqual(out, {"limit": 20, "enabled": False})

    def test_stringified_nested_object(self) -> None:
        out = repair_input({"limit": 1, "cfg": '{"count": "3"}'}, SCHEMA)
        self.assertEqual(out, {"limit": 1, "cfg": {"count": 3}})

    def test_numeric_string(self) -> None:
        out = repair_input({"limit": "20"}, SCHEMA)
        self.assertEqual(out, {"limit": 20})

    def test_numeric_float_string(self) -> None:
        out = repair_input({"limit": "20.0"}, {"type": "object", "properties": {"limit": {"type": "number"}}})
        self.assertEqual(out["limit"], 20)

    def test_boolean_string(self) -> None:
        out = repair_input({"limit": 1, "enabled": "false"}, SCHEMA)
        self.assertEqual(out, {"limit": 1, "enabled": False})
        out = repair_input({"limit": 1, "enabled": "True"}, SCHEMA)
        self.assertEqual(out, {"limit": 1, "enabled": True})

    def test_null_placeholder_optional_field_dropped(self) -> None:
        out = repair_input({"limit": 1, "enabled": None}, SCHEMA)
        self.assertEqual(out, {"limit": 1})
        self.assertNotIn("enabled", out)

    def test_null_placeholder_required_field_kept(self) -> None:
        out = repair_input({"limit": None}, SCHEMA)
        self.assertIn("limit", out)
        self.assertIsNone(out["limit"])

    def test_empty_object_placeholder_dropped(self) -> None:
        out = repair_input({"limit": 1, "enabled": {}}, SCHEMA)
        self.assertEqual(out, {"limit": 1})

    def test_stringified_array(self) -> None:
        out = repair_input({"limit": 1, "tags": '["a", "b"]'}, SCHEMA)
        self.assertEqual(out, {"limit": 1, "tags": ["a", "b"]})

    def test_positional_tuple(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "pair": {
                    "type": "array",
                    "prefixItems": [{"type": "integer"}, {"type": "boolean"}],
                }
            },
        }
        out = repair_input({"pair": ["2", "false"]}, schema)
        self.assertEqual(out, {"pair": [2, False]})

    def test_typed_dict(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "counts": {"type": "object", "additionalProperties": {"type": "integer"}},
            },
        }
        out = repair_input({"counts": {"first": "2", "second": "3"}}, schema)
        self.assertEqual(out, {"counts": {"first": 2, "second": 3}})

    def test_nested_array_of_objects(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"count": {"type": "integer"}},
                    },
                }
            },
        }
        out = repair_input({"items": [{"count": "2"}, {"count": 5}]}, schema)
        self.assertEqual(out, {"items": [{"count": 2}, {"count": 5}]})

    def test_single_value_wrapped_for_array(self) -> None:
        out = repair_input({"limit": 1, "tags": "a"}, SCHEMA)
        self.assertEqual(out, {"limit": 1, "tags": ["a"]})

    def test_genuinely_wrong_types_still_fail(self) -> None:
        repaired = repair_input({"limit": "abc"}, SCHEMA)
        with self.assertRaises(ToolInputError):
            validate_json_schema(repaired, SCHEMA, root_name="Stub")
        repaired = repair_input({"limit": {"deep": "nope"}}, SCHEMA)
        with self.assertRaises(ToolInputError):
            validate_json_schema(repaired, SCHEMA, root_name="Stub")
        repaired = repair_input({"limit": 1, "enabled": "maybe"}, SCHEMA)
        with self.assertRaises(ToolInputError):
            validate_json_schema(repaired, SCHEMA, root_name="Stub")

    def test_ambiguous_values_left_alone(self) -> None:
        # A string that is not JSON stays a string for an object schema;
        # validation still fails as before.
        out = repair_input({"cfg": "not-json"}, SCHEMA)
        self.assertEqual(out["cfg"], "not-json")
        with self.assertRaises(ToolInputError):
            validate_json_schema(out, SCHEMA, root_name="Stub")

    def test_no_meaning_changing_coercion(self) -> None:
        # String values for string fields are untouched.
        out = repair_input({"limit": 1, "name": "20"}, SCHEMA)
        self.assertEqual(out["name"], "20")
        # JSON that parses but is not an object stays untouched for objects.
        out = repair_input({"cfg": "[1, 2]"}, SCHEMA)
        self.assertEqual(out["cfg"], "[1, 2]")

    def test_already_valid_input_unchanged(self) -> None:
        value = {"limit": 20, "enabled": True, "tags": ["a"]}
        self.assertIs(repair_input(value, SCHEMA), value)

    def test_depth_cap_does_not_recurse_forever(self) -> None:
        deep = {"limit": "1"}
        for _ in range(12):
            deep = {"nested": deep}
        schema = {"type": "object", "properties": {"nested": deep_schema(deep)}}
        # Should return without hanging or crashing; values beyond the cap
        # are simply left unrepaired.
        out = repair_input(deep, schema)
        self.assertIsInstance(out, dict)

    def test_ref_resolution(self) -> None:
        schema = {
            "type": "object",
            "properties": {"item": {"$ref": "#/$defs/item"}},
            "$defs": {"item": {"type": "object", "properties": {"n": {"type": "integer"}}}},
        }
        out = repair_input({"item": {"n": "7"}}, schema)
        self.assertEqual(out, {"item": {"n": 7}})


def deep_schema(deep: dict) -> dict:
    """Build a recursive object schema matching the nested dict shape."""
    if not isinstance(deep, dict) or len(deep) != 1 or "nested" not in deep:
        return {"type": "object", "properties": {"limit": {"type": "integer"}}}
    return {"type": "object", "properties": {"nested": deep_schema(deep["nested"])}}


class _StubTool:
    """Minimal tool: echoes its (repaired) input back."""

    def __init__(self) -> None:
        self.seen: list[dict] = []

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="RepairStub",
            description="stub",
            input_schema=SCHEMA,
            is_read_only=True,
        )

    def run(self, tool_input: dict, context: ToolContext) -> ToolResult:
        self.seen.append(tool_input)
        return ToolResult(name="RepairStub", output={"echo": tool_input})

    def check_permissions(self, tool_input: dict, context: ToolContext) -> PermissionResult:
        return PermissionResult.allow()


class RepairRegistryIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        from pathlib import Path

        self.tmp = tempfile.TemporaryDirectory()
        self.ctx = ToolContext(workspace_root=Path(self.tmp.name).resolve())
        self.stub = _StubTool()
        self.registry = ToolRegistry([self.stub])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_repair_runs_before_validation_in_execute_path(self) -> None:
        # Without repair this input fails validation ("limit" is a string).
        result = self.registry.dispatch(
            ToolCall(name="RepairStub", input={"limit": "20", "enabled": "false"}),
            self.ctx,
        )
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(self.stub.seen[0], {"limit": 20, "enabled": False})

    def test_stringified_root_input_repaired_in_execute_path(self) -> None:
        result = self.registry.dispatch(
            ToolCall(name="RepairStub", input='{"limit": "20"}'),
            self.ctx,
        )
        self.assertFalse(result.is_error, result.output)
        self.assertEqual(self.stub.seen[0], {"limit": 20})

    def test_genuinely_bad_input_still_errors_after_repair(self) -> None:
        with self.assertRaises(ToolInputError):
            self.registry.dispatch(
                ToolCall(name="RepairStub", input={"limit": "abc"}),
                self.ctx,
            )

    def test_missing_required_field_still_errors(self) -> None:
        with self.assertRaises(ToolInputError):
            self.registry.dispatch(
                ToolCall(name="RepairStub", input={"enabled": "true"}),
                self.ctx,
            )


if __name__ == "__main__":
    unittest.main()
