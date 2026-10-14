"""Persistent 'always allow' permission rules for the REPL.

Rules are stored as JSON in ~/.clawd/permissions.json, e.g.:
    {"always_allow": ["Read", "Glob", "Grep"]}

A tool listed here skips the interactive permission prompt entirely.
This is the terminal counterpart to Claude Code's "always allow" rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_store_path() -> Path:
    return Path.home() / ".clawd" / "permissions.json"


def load_rules() -> dict[str, Any]:
    """Load the rules file, returning {} when missing or corrupt."""
    path = get_store_path()
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}


def save_rules(rules: dict[str, Any]) -> None:
    path = get_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, indent=2), encoding="utf-8")


def _normalized(rules: dict[str, Any]) -> list[str]:
    raw = rules.get("always_allow") or []
    if not isinstance(raw, list):
        return []
    return sorted({str(t) for t in raw if str(t).strip()})


def list_always_allowed() -> list[str]:
    """Tool names that skip the permission prompt."""
    return _normalized(load_rules())


def is_always_allowed(tool_name: str) -> bool:
    return tool_name in _normalized(load_rules())


def allow_always(tool_name: str) -> None:
    """Remember 'always allow' for a tool. Idempotent."""
    rules = load_rules()
    allowed = _normalized(rules)
    if tool_name not in allowed:
        allowed.append(tool_name)
        rules["always_allow"] = sorted(allowed)
        save_rules(rules)


def forget_tool(tool_name: str) -> bool:
    """Remove a tool from always-allow. Returns True if it was present."""
    rules = load_rules()
    allowed = _normalized(rules)
    if tool_name not in allowed:
        return False
    allowed.remove(tool_name)
    rules["always_allow"] = allowed
    save_rules(rules)
    return True


def clear_rules() -> None:
    save_rules({"always_allow": []})
