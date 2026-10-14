"""Delta instruction system (opencode parity, brief 002).

Instead of resending the full system prompt every turn, the agent loop sends
the full baseline once (turn 0) and only the *changed* instruction sections
on later turns. Adapted from opencode's instruction-state observe/commit
flow (packages/core/src/session/instruction-state.ts):

- Instruction sources are named dynamic sections:
  - ``context``: ``build_context_prompt`` -- git status / CLAUDE.md can
    change mid-session.
  - ``skills``: ``build_skills_section`` -- skills install mid-session and
    are discovered instantly.
  - ``mcp``: ``build_mcp_section`` -- MCP servers connect/disconnect
    mid-session; empty when none are connected (prompt diet).
- Static sections (date line, merged core, conditional tool guidance, env
  block, Windows-only shell) form the cached baseline, sent every turn.
- Turn 0 sends baseline + all dynamic sections (the full prompt).
- Turns > 0 send baseline + a delta block with only added/updated/removed
  dynamic sections. When nothing changed, the turn carries just the baseline.

Token savings compound every turn, and smaller per-turn prompts behave
better on small models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .context import ToolContext


# Dynamic instruction sources, in stable render order.
_DYNAMIC_SOURCES = ("context", "skills", "mcp")


@dataclass
class InstructionDelta:
    """One changed dynamic section since the last turn."""

    name: str
    kind: str  # "added" | "updated" | "removed"
    text: str = ""  # new section text ("" for removed)


@dataclass
class DeltaReport:
    """Per-turn evidence for the delta system."""

    turn: int
    changed: bool
    deltas: list[InstructionDelta] = field(default_factory=list)
    baseline_bytes: int = 0
    full_bytes: int = 0  # baseline + all dynamic sections (turn-0 size)
    system_bytes: int = 0  # what this turn actually sends


def _render_delta_block(deltas: list[InstructionDelta]) -> str:
    """Render changed sections as a compact update block."""
    lines = [
        "[Instruction update: only the changed sections are shown below. "
        "Unchanged sections from the start of the session still apply.]",
    ]
    for d in deltas:
        if d.kind == "removed":
            lines.append(f"## {d.name} (removed -- no longer applies)")
        else:
            lines.append(f"## {d.name} ({d.kind})")
            if d.text.strip():
                lines.append(d.text)
    return "\n".join(lines)


class DeltaInstructionTracker:
    """Tracks instruction state across turns and emits per-turn system text.

    Usage from the agent loop::

        tracker = DeltaInstructionTracker(style_prompt, tool_context, names)
        turn0_system = tracker.full_prompt()      # baseline + everything
        system_text, report = tracker.next_system(turn=1)  # baseline + delta
    """

    def __init__(
        self,
        style_prompt: str,
        tool_context: "ToolContext",
        registered_tools: set[str] | None = None,
        script_mode: bool = False,
    ) -> None:
        self._style_prompt = style_prompt
        self._tool_context = tool_context
        self._registered_tools = registered_tools
        self._script_mode = script_mode
        self._baseline: str | None = None
        self._last_dynamic: dict[str, str] | None = None

    # -- section builders -------------------------------------------------

    def _build_baseline(self) -> str:
        """Static sections: date line, core, tool guidance, env, shell."""
        # Imported lazily to keep module import order cycle-free.
        from .system_prompt import (
            build_agent_guidelines,
            build_env_block,
            build_tool_guidance,
            todays_date_line,
        )

        core = f"{todays_date_line()}\n\n{self._style_prompt}"
        guidance = build_tool_guidance(self._registered_tools, script_mode=self._script_mode)
        if guidance.strip():
            core = f"{core}\n\n{guidance}"
        sections = [core]
        try:
            env_block = build_env_block(self._tool_context)
        except Exception:
            env_block = ""
        if env_block.strip():
            sections.append(env_block)
        try:
            shell = build_agent_guidelines()
        except Exception:
            shell = ""
        if shell.strip():
            sections.append(shell)
        return "\n\n".join(s for s in sections if s.strip())

    def _build_dynamic(self) -> dict[str, str]:
        """Dynamic sections, recomputed every turn."""
        from .system_prompt import build_mcp_section, build_skills_section
        from ..context_system import build_context_prompt

        try:
            context = build_context_prompt(
                self._tool_context.workspace_root,
                cwd=self._tool_context.cwd,
            )
        except Exception:
            context = ""
        try:
            skills = build_skills_section(self._tool_context)
        except Exception:
            skills = ""
        try:
            mcp = build_mcp_section(self._tool_context)
        except Exception:
            mcp = ""
        return {"context": context, "skills": skills, "mcp": mcp}

    # -- public API ---------------------------------------------------------

    def baseline(self) -> str:
        if self._baseline is None:
            self._baseline = self._build_baseline()
        return self._baseline

    def full_prompt(self) -> str:
        """Turn-0 system text: baseline + every dynamic section."""
        dynamic = self._build_dynamic()
        self._last_dynamic = dynamic
        parts = [self.baseline()]
        for name in _DYNAMIC_SOURCES:
            text = dynamic.get(name, "")
            if text.strip():
                parts.append(text)
        return "\n\n".join(parts)

    def next_system(self, turn: int) -> tuple[str, DeltaReport]:
        """System text for turn > 0: baseline + only the changed sections."""
        if self._last_dynamic is None:
            # full_prompt() was never called; treat this turn as turn 0.
            full = self.full_prompt()
            return full, DeltaReport(
                turn=turn,
                changed=True,
                baseline_bytes=len(self.baseline().encode()),
                full_bytes=len(full.encode()),
                system_bytes=len(full.encode()),
            )
        dynamic = self._build_dynamic()
        deltas: list[InstructionDelta] = []
        for name in _DYNAMIC_SOURCES:
            old = (self._last_dynamic.get(name) or "").strip()
            new = (dynamic.get(name) or "").strip()
            if old == new:
                continue
            if not old and new:
                kind = "added"
            elif old and not new:
                kind = "removed"
            else:
                kind = "updated"
            deltas.append(InstructionDelta(name=name, kind=kind, text=dynamic.get(name, "")))
        self._last_dynamic = dynamic
        baseline = self.baseline()
        if deltas:
            system_text = f"{baseline}\n\n{_render_delta_block(deltas)}"
        else:
            system_text = baseline
        report = DeltaReport(
            turn=turn,
            changed=bool(deltas),
            deltas=deltas,
            baseline_bytes=len(baseline.encode()),
            full_bytes=self._last_full_bytes(),
            system_bytes=len(system_text.encode()),
        )
        return system_text, report

    # -- bookkeeping helpers --------------------------------------------------

    def _last_full_bytes(self) -> int:
        """Byte size of the turn-0 full prompt (baseline + dynamic at send)."""
        parts = [self.baseline()]
        for name in _DYNAMIC_SOURCES:
            text = (self._last_dynamic or {}).get(name, "")
            if text.strip():
                parts.append(text)
        return len("\n\n".join(parts).encode())


def summarize_delta_report(report: DeltaReport) -> dict[str, Any]:
    """JSON-friendly token evidence for one turn's delta report."""
    return {
        "turn": report.turn,
        "changed": report.changed,
        "deltas": [
            {"name": d.name, "kind": d.kind, "bytes": len(d.text.encode())}
            for d in report.deltas
        ],
        "baseline_bytes": report.baseline_bytes,
        "full_bytes": report.full_bytes,
        "system_bytes": report.system_bytes,
        "saved_bytes": report.full_bytes - report.system_bytes,
    }
