"""Planner subagent profile (brief 016).

Data/config for the 005-style read-only planner agent. This module defines
the profile only — the subagent loop itself (``tools/subagent.py``) is owned
by another builder this wave, which consumes ``PLANNER_PROFILE`` and
``build_planner_registry()``.

Defense in depth (crush pattern): the prompt says "don't mutate", and the
registry handed to the planner physically excludes mutation tools, so a
planner cannot call Write/Edit/Bash even if the model tries — dispatch
fails with "unknown tool".

Marker-delimited plan output (``<!-- PLAN_START -->`` ... ``<!-- PLAN_END -->``)
lets the UI detect a finished plan and open an approve/edit dialog.
"""

from __future__ import annotations

from typing import Any

from .plan_policy import (
    PLAN_MARKER_END,
    PLAN_MARKER_START,
    REQUIRED_SECTIONS,
)

#: Marker delimiters the planner must wrap its final plan in.
PLANNER_PLAN_MARKERS = (PLAN_MARKER_START, PLAN_MARKER_END)

#: Tools the planner may never receive, even if the parent registry has them.
#: (Physical exclusion is enforced by build_planner_registry; this set
#: documents the intent and backs the "no mutation tools" test.)
PLANNER_FORBIDDEN_TOOLS = frozenset({
    "write", "edit", "bash", "powershell",
    "mcp", "skill", "croncreate", "crondelete",
    "subagent", "agent", "toolsearch",
})

#: Plan-mode way-out tools the planner keeps so it can finish.
PLANNER_KEPT_STATE_TOOLS = frozenset({"exitplanmode", "exit_plan_mode"})


def detect_plan_markers(text: str) -> tuple[str, str] | None:
    """Extract the marker-delimited plan from *text* for the UI dialog.

    Returns (start_marker, plan_body) when both markers are present with
    content between them, else None.
    """
    if not isinstance(text, str):
        return None
    start = text.find(PLAN_MARKER_START)
    end = text.find(PLAN_MARKER_END)
    if start == -1 or end == -1 or end <= start:
        return None
    body = text[start + len(PLAN_MARKER_START):end].strip()
    if not body:
        return None
    return PLAN_MARKER_START, body


PLANNER_SYSTEM_PROMPT = """\
You are a read-only PLANNER. You explore, investigate, and design — you never
modify anything. Your toolset physically excludes all mutation tools
(Write/Edit/Bash/shell/skills/MCP); attempting to call one fails at dispatch.
Work only with read-only tools: Read, Glob, Grep, WebFetch, WebSearch, LSP.

Self-review workflow (do this before drafting the plan):
1. Synthesize what you learned from investigation into a short brief.
2. List gaps, contradictions, and unverified assumptions in that brief.
3. Re-read the relevant sources to close as many gaps as you can.
4. List the remaining unknowns explicitly — do not silently guess.

Then draft the implementation plan using EXACTLY these sections:
{sections}

IMPLEMENTATION STEPS must be numbered steps, each a goal+method+reference
tuple:
  1. Goal: <what this step achieves>
     Method: <how to do it>
     Reference: <file/symbol/doc that grounds it>

TESTING AND VALIDATION must name the concrete commands or checks that prove
each step worked.

Output rules:
- Wrap the final plan in markers so the UI can open an approve/edit dialog:
  {marker_start}
  ...plan...
  {marker_end}
- Output NOTHING after the end marker.
- Do not implement anything. Do not ask for approval yourself — the parent
  session handles approval via exit_plan_mode.
""".format(
    sections="\n".join(f"- {s}" for s in REQUIRED_SECTIONS),
    marker_start=PLAN_MARKER_START,
    marker_end=PLAN_MARKER_END,
)

#: The planner profile, consumed by the subagent-loop builder.
PLANNER_PROFILE: dict[str, Any] = {
    "name": "planner",
    "description": (
        "Read-only planning agent. Explores with read-only tools and returns "
        "a marker-delimited implementation plan. Cannot mutate anything: "
        "mutation tools are physically excluded from its registry."
    ),
    "system_prompt": PLANNER_SYSTEM_PROMPT,
    "tool_policy": "read-only + plan-state tools only",
    "forbidden_tools": sorted(PLANNER_FORBIDDEN_TOOLS),
    "kept_state_tools": sorted(PLANNER_KEPT_STATE_TOOLS),
    "plan_markers": list(PLANNER_PLAN_MARKERS),
    "plan_sections": list(REQUIRED_SECTIONS),
    # Marker detection for the UI approve/edit dialog:
    "plan_detector": "planner_profile.detect_plan_markers",
}


def build_planner_registry(parent_registry: Any) -> Any:
    """Build the planner's physically-restricted tool registry.

    Keeps every tool whose spec is read-only, plus the plan-mode exit
    (the way out), and drops everything else. Unknown/mutation tools then
    fail dispatch with "unknown tool" — the planner cannot call them even
    if the model tries.

    Mirrors ``tools/subagent.py::build_weak_child_registry`` without
    importing it (avoids an import cycle; subagent.py is owned by the
    024 builder this wave).
    """
    from .registry import ToolRegistry

    child = ToolRegistry()
    for tool in parent_registry.list_tools():
        name = tool.spec().name.lower()
        if name in PLANNER_FORBIDDEN_TOOLS:
            continue
        if name in PLANNER_KEPT_STATE_TOOLS:
            child.register(tool)
            continue
        try:
            read_only = bool(tool.spec().is_read_only)
        except Exception:
            read_only = False
        if read_only:
            child.register(tool)
    return child
