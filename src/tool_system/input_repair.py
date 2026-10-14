"""Tool input repair: coerce malformed tool inputs when the schema unambiguously supports it.

A Python port of opencode's ``tool-input-repair`` plugin
(``packages/core/src/plugin/tool-input-repair.ts``), which runs on the
``execute.before`` hook. Repair is strictly model-agnostic: it helps weak
models that emit ``{"limit": "20"}`` most, and harms none.

Repair applies only when the schema unambiguously supports it:
- Stringified root or nested object: ``'{"limit":"20"}'`` -> ``{limit: 20}``
- Numeric or boolean string: ``{limit: "20", enabled: "false"}`` -> ``{limit: 20, enabled: false}``
- Optional null or empty-object placeholder: ``{limit: null}`` -> ``{}``
- Stringified array or compatible item: ``{tags: '["a"]', count: "2"}`` -> ``{tags: ["a"], count: [2]}``
- Positional tuple: ``{pair: ["2", "false"]}`` -> ``{pair: [2, False]}``
- Typed dictionary: ``{counts: {first: "2"}}`` -> ``{counts: {first: 2}}``
- Nested fields: ``{items: [{count: "2"}]}`` -> ``{items: [{count: 2}]}``

Anything ambiguous is left untouched so validation fails exactly as it did
before repair existed.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

# Depth cap, mirroring opencode's maxDepth.
_MAX_DEPTH = 6

_TRUE_STRINGS = ("true",)
_FALSE_STRINGS = ("false",)


def repair_input(value: Any, schema: Mapping[str, Any]) -> Any:
    """Repair *value* against *schema*, returning the (possibly) repaired value.

    Non-destructive: returns the original object unchanged when no repair
    applies (only replaces values where a coercion unambiguously fits).
    """
    if not isinstance(schema, Mapping):
        return value
    return _repair(value, schema, schema, 0)


def _decode_json(text: str) -> Any:
    """Decode a JSON string, returning the _FAILED sentinel on failure."""
    try:
        return json.loads(text)
    except Exception:
        return _FAILED


class _Failed:
    pass


_FAILED = _Failed()


def _resolve_ref(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any] | None:
    ref = schema.get("$ref")
    if not isinstance(ref, str):
        return None
    if ref.startswith("#/$defs/") or ref.startswith("#/definitions/"):
        prefix, _, name = ref.rpartition("/")
        name = name.replace("~1", "/").replace("~0", "~")
        section = "$defs" if prefix == "#/$defs" else "definitions"
        target = root.get(section, {})
        if isinstance(target, Mapping):
            resolved = target.get(name)
            if isinstance(resolved, Mapping):
                return resolved
    return None


def _repair(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], depth: int) -> Any:
    if depth > _MAX_DEPTH:
        return value

    resolved = _resolve_ref(schema, root)
    if resolved is not None and resolved is not schema:
        return _repair(value, resolved, root, depth + 1)

    schema_type = schema.get("type")

    if isinstance(schema_type, list):
        if value is None and "null" in schema_type:
            return value
        if _json_type_of(value) in schema_type:
            return value
        concrete = [t for t in schema_type if t != "null"]
        if len(concrete) == 1:
            return _repair(value, {**schema, "type": concrete[0]}, root, depth + 1)
        return value

    if schema_type is None:
        branches = schema.get("anyOf") if isinstance(schema.get("anyOf"), list) else (
            schema.get("oneOf") if isinstance(schema.get("oneOf"), list) else None
        )
        if not branches or value is None:
            return value
        if any(_json_type_of(value) == (b.get("type") if isinstance(b, Mapping) else None) for b in branches if isinstance(b, Mapping)):
            return value
        candidates = [b for b in branches if isinstance(b, Mapping) and b.get("type") != "null"]
        if len(candidates) == 1:
            return _repair(value, candidates[0], root, depth + 1)
        return value

    if schema_type in ("number", "integer"):
        if not isinstance(value, str) or not value.strip():
            return value
        try:
            parsed = float(value) if schema_type == "number" else int(value, 10)
        except (ValueError, TypeError):
            return value
        if schema_type == "number":
            if not math.isfinite(parsed):
                return value
            # Prefer int when the string denotes a whole value, matching
            # JSON semantics (validation accepts int for "number" too).
            if float(parsed).is_integer():
                return int(parsed)
            return parsed
        return parsed

    if schema_type == "boolean":
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered == "true":
                return True
            if lowered == "false":
                return False
        return value

    if schema_type == "object":
        return _repair_object(value, schema, root, depth)
    if schema_type == "array":
        return _repair_array(value, schema, root, depth)
    return value


def _json_type_of(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"


def _is_plain_property(prop: Mapping[str, Any], composed: bool) -> bool:
    """A bare single-type property that provably rejects null and {}."""
    if composed or not isinstance(prop.get("type"), str) or prop.get("type") == "null":
        return False
    return not any(k in prop for k in ("anyOf", "oneOf", "allOf", "enum", "const", "$ref", "nullable"))


def _repair_object(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], depth: int) -> Any:
    if isinstance(value, str):
        decoded = _decode_json(value)
        if not isinstance(decoded, dict):
            # Not a JSON object (or not JSON at all): leave it untouched —
            # coercing a string into an object would change its meaning.
            return value
        value = decoded
    if not isinstance(value, dict):
        return value

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    required = set(required) if isinstance(required, list) else set()
    additional = schema.get("additionalProperties", True)
    composed = any(isinstance(schema.get(k), list) for k in ("allOf", "anyOf", "oneOf"))

    result: dict[str, Any] = {}
    changed = False
    for key, current in value.items():
        declared = key in properties
        prop: Mapping[str, Any] | None = None
        if declared:
            prop = properties[key] if isinstance(properties[key], Mapping) else None
        elif additional is not False and isinstance(additional, Mapping):
            # Typed dictionary: additionalProperties carries the value schema.
            prop = additional

        if not declared and additional is False:
            # Unexpected field: leave it (validation reports it); do not
            # silently drop fields outside repair's mandate.
            result[key] = current
            continue
        if prop is None:
            result[key] = current
            continue

        plain = _is_plain_property(prop, composed)
        placeholder = isinstance(current, dict) and not current and prop.get("type") != "object"
        if declared and key not in required and plain and (current is None or placeholder):
            # Null/empty-object placeholder for an optional field: drop it.
            changed = True
            continue

        repaired = _repair(current, prop, root, depth + 1)
        result[key] = repaired
        if repaired is not current:
            changed = True
    return result if changed else value


def _repair_array(value: Any, schema: Mapping[str, Any], root: Mapping[str, Any], depth: int) -> Any:
    candidate = value
    if isinstance(value, str):
        decoded = _decode_json(value)
        if decoded is not _FAILED:
            candidate = decoded

    prefix_items = schema.get("prefixItems")
    items = schema.get("items")

    if isinstance(candidate, list):
        if isinstance(prefix_items, list):
            # Positional tuple: repair each position against its own schema.
            tuple_schemas = prefix_items
            if isinstance(items, Mapping):
                tuple_schemas = prefix_items + [items] * max(0, len(candidate) - len(prefix_items))
            repaired = []
            changed = False
            for index, item in enumerate(candidate):
                member = tuple_schemas[index] if index < len(tuple_schemas) else None
                if isinstance(member, Mapping):
                    new_item = _repair(item, member, root, depth + 1)
                else:
                    new_item = item
                repaired.append(new_item)
                if new_item is not item:
                    changed = True
            return repaired if changed else candidate
        if isinstance(items, Mapping):
            repaired = [_repair(item, items, root, depth + 1) for item in candidate]
            return repaired if any(n is not o for n, o in zip(repaired, candidate)) else candidate
        return candidate

    # Non-list value where an array is expected: wrap only if the value
    # cleanly repairs into the item type (opencode's "compatible item" case).
    if prefix_items or not isinstance(items, Mapping):
        return value
    repaired_item = _repair(candidate, items, root, depth + 1)
    item_type = items.get("type")
    if item_type == "object":
        compatible = isinstance(repaired_item, dict)
    elif item_type == "array":
        compatible = isinstance(repaired_item, list)
    elif item_type == "integer":
        compatible = isinstance(repaired_item, int) and not isinstance(repaired_item, bool)
    elif item_type == "number":
        compatible = isinstance(repaired_item, (int, float)) and not isinstance(repaired_item, bool) and math.isfinite(float(repaired_item))
    elif item_type in ("string", "boolean"):
        compatible = isinstance(repaired_item, str) if item_type == "string" else isinstance(repaired_item, bool)
    else:
        compatible = False
    return [repaired_item] if compatible else value
