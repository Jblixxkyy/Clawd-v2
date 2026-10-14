# 007 — Tool input repair (coerce before rejecting)

## What
Repair malformed tool inputs when the schema unambiguously supports it, instead of rejecting them and costing a full model round-trip.

## Why
opencode has a dedicated plugin for this (`packages/core/src/plugin/tool-input-repair.ts` in ~/workspace/opencode-ref), running on the `execute.before` hook. It handles: stringified root/nested objects (`'{"limit":"20"}'` → `{limit: 20}`), numeric/boolean strings (`{limit: "20", enabled: "false"}` → `{limit: 20, enabled: false}`), null/empty-object placeholders for optional fields, stringified arrays, positional tuples, typed dicts. Small models constantly emit `{"limit": "20"}` — our `schema_validation.py` (`src/tool_system/schema_validation.py`) only validates and raises, so every slip costs a retry round-trip. Repair is strictly model-agnostic: it helps weak models most, harms none.

## Suggested approach
- Add `repair_input(value, schema)` in `src/tool_system/` (new module `input_repair.py`): walk the input against the tool's JSON schema, coerce only when unambiguous (string parses as the schema's number/boolean; JSON-string parses as the schema's object/array; null dropped for non-required fields). Depth-cap recursion at ~6 like opencode.
- Call it in the tool execute path (registry.py, where `validate_json_schema` is invoked) BEFORE validation. If repair is ambiguous, leave the value alone and let validation fail as today.
- Never coerce across types that change meaning (no string→object unless it parses as JSON).

## Expected tests
- Unit tests for each opencode repair case (stringified object, numeric string, boolean string, null placeholder, stringified array, nested).
- Test that genuinely wrong types still fail validation after repair.
- Test repair runs before validation in the execute path (integration via a stub tool).
- Full suite green.
