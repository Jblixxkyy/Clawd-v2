"""Forced scratchpad protocol for investigator subagents (brief 024).

Ported from gemini-cli's ``codebase-investigator`` agent: a mandatory
scratchpad artifact that externalizes a subagent's working memory so the
*harness* -- not the model -- decides when the investigation is done.

The artifact has four lists::

    {
        "checklist": [...],            # steps the investigation will take
        "questions_to_resolve": [...], # open questions; empty == done
        "key_findings": [...],         # confirmed facts worth reporting
        "irrelevant_paths": [...],     # explored and ruled out, not retried
    }

Semantics (mirroring 020's TodoWrite):

* The scratchpad is seeded by the harness before turn 1, so it always
  exists even if the model never calls the tool.
* Updates are whole-overwrite: the ``Scratchpad`` tool takes the entire
  artifact each call and the harness replaces its state wholesale.
* Termination is defined by the harness: the subagent may only finish
  when ``questions_to_resolve`` is empty AND its final message validates
  against the task's JSON schema. Otherwise the harness continues it
  (within the turn/time caps). See :func:`termination_check`.
* At the caps the harness performs a forced wrap-up: it synthesizes the
  final artifact from the scratchpad state instead of truncating
  silently (opencode max-steps pattern). See :func:`forced_wrapup`.

The investigator profile itself (read-only toolset from 028's
``weak_profile_allowed_tools``, model from 028's ``weak_profile_model``,
50 turns, 10 minutes, temperature 0.1) is wired in
``src/tool_system/tools/subagent.py`` (:func:`run_investigator_subagent`),
which reuses this module. The 018 reviewer/judge builder can call that
same entry point directly -- a reviewer's rubric verdict becomes the
scratchpad's final state via :func:`record_verdict`.

Live UI rendering (collapsible per-subagent panel) is PENDING for the
web UI: every scratchpad update is recorded in ``scratchpad_history``
(snapshots) and broadcast through the ``on_scratchpad_update`` callback
passed to :func:`run_investigator_subagent`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..tool_system.protocol import ToolResult
from ..tool_system.registry import ToolSpec
from ..tool_system.permission_handler import PermissionResult

# --- Investigator profile defaults (brief 024) ------------------------------
# Read-only toolset + model come from 028's weak profile
# (model_tiers.weak_profile_allowed_tools / weak_profile_model).

SCRATCHPAD_TOOL_NAME = "Scratchpad"

INVESTIGATOR_MAX_TURNS = 50
INVESTIGATOR_MAX_TIME_MINUTES = 10.0
INVESTIGATOR_TEMPERATURE = 0.1

#: The four artifact keys, in canonical order.
SCRATCHPAD_KEYS = (
    "checklist",
    "questions_to_resolve",
    "key_findings",
    "irrelevant_paths",
)

#: Fallback schema when the caller does not supply one. The investigator
#: protocol always ends with machine-readable output; at minimum the
#: child must summarize what it found.
DEFAULT_INVESTIGATOR_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary"],
}


class ScratchpadError(ValueError):
    """Raised when a scratchpad update payload fails validation."""


# --- Artifact ---------------------------------------------------------------

@dataclass
class Scratchpad:
    """The investigator's working-memory artifact.

    All four lists hold plain strings. Updates replace the whole
    artifact (whole-overwrite, like 020's todos); validation rejects
    malformed payloads instead of merging them.
    """

    checklist: list[str] = field(default_factory=list)
    questions_to_resolve: list[str] = field(default_factory=list)
    key_findings: list[str] = field(default_factory=list)
    irrelevant_paths: list[str] = field(default_factory=list)

    @classmethod
    def initial(
        cls,
        description: str,
        seed_questions: Optional[list[str]] = None,
    ) -> "Scratchpad":
        """Seed the scratchpad before turn 1.

        The harness always creates the artifact up front so the
        termination check has something to read even if the model never
        calls the ``Scratchpad`` tool. The model's first job is to
        refine this seed into a real plan.
        """
        questions = [q for q in (seed_questions or []) if isinstance(q, str) and q.strip()]
        if not questions:
            questions = [f"Investigate and report: {description.strip()}"]
        return cls(
            checklist=[
                "Create/refresh the scratchpad via the Scratchpad tool",
                "Work through questions_to_resolve one at a time",
                "Return the final report as JSON matching the required schema",
            ],
            questions_to_resolve=questions,
            key_findings=[],
            irrelevant_paths=[],
        )

    @staticmethod
    def validate_payload(payload: Any) -> list[str]:
        """Validate a whole-overwrite payload; returns a list of errors."""
        errors: list[str] = []
        if not isinstance(payload, dict):
            return ["scratchpad payload must be an object with the keys: "
                    + ", ".join(SCRATCHPAD_KEYS)]
        for key in SCRATCHPAD_KEYS:
            if key not in payload:
                errors.append(f"missing required key: '{key}'")
                continue
            value = payload[key]
            if not isinstance(value, list):
                errors.append(f"'{key}' must be a list of strings")
                continue
            for i, item in enumerate(value):
                if not isinstance(item, str):
                    errors.append(f"'{key}[{i}]' must be a string")
        return errors

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Whole-overwrite update. Returns a delta summary.

        Raises :class:`ScratchpadError` on invalid payloads; the stored
        state is untouched when validation fails.
        """
        errors = self.validate_payload(payload)
        if errors:
            raise ScratchpadError("; ".join(errors))
        resolved = [q for q in self.questions_to_resolve
                    if q not in payload["questions_to_resolve"]]
        for key in SCRATCHPAD_KEYS:
            setattr(self, key, list(payload[key]))
        return {
            "questions_resolved": resolved,
            "open_questions": len(self.questions_to_resolve),
            "complete": self.is_complete(),
        }

    def is_complete(self) -> bool:
        """Termination clause 1: no open questions."""
        return len(self.questions_to_resolve) == 0

    def open_questions_count(self) -> int:
        return len(self.questions_to_resolve)

    def to_dict(self) -> dict[str, Any]:
        return {key: list(getattr(self, key)) for key in SCRATCHPAD_KEYS}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scratchpad":
        errors = cls.validate_payload(data)
        if errors:
            raise ScratchpadError("; ".join(errors))
        return cls(**{key: list(data[key]) for key in SCRATCHPAD_KEYS})

    def render_markdown(self) -> str:
        """Human-readable rendering (used in prompts and the future UI)."""
        def _section(title: str, items: list[str]) -> str:
            body = "\n".join(f"- {i}" for i in items) if items else "- (none)"
            return f"### {title}\n{body}"

        return "\n".join([
            _section("Checklist", self.checklist),
            _section("Questions to resolve", self.questions_to_resolve),
            _section("Key findings", self.key_findings),
            _section("Irrelevant paths", self.irrelevant_paths),
        ])


# --- The Scratchpad tool (lives in the investigator's child registry) ------

class ScratchpadTool:
    """Whole-overwrite scratchpad updates for the investigator subagent.

    The artifact is harness-internal state (like 020's todos), not a
    workspace mutation, so the tool self-allows its own permission check:
    the 028 weak read-only permission handler would otherwise deny it.
    """

    def __init__(
        self,
        scratchpad: Scratchpad,
        on_update: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> None:
        self._scratchpad = scratchpad
        self._on_update = on_update
        self.update_count = 0

    @property
    def scratchpad(self) -> Scratchpad:
        return self._scratchpad

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=SCRATCHPAD_TOOL_NAME,
            description=(
                "Update your investigation scratchpad. You MUST call this "
                "on your first turn to create your working plan, and re-emit "
                "the FULL scratchpad on every subsequent turn (whole "
                "overwrite -- always send all four lists, not deltas). "
                "Move answered items from questions_to_resolve into "
                "key_findings; log explored-but-useless paths in "
                "irrelevant_paths so you do not retry them. The harness "
                "only lets you finish when questions_to_resolve is EMPTY "
                "and your final message is valid JSON matching the required "
                "schema -- stopping early just gets you continued."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    key: {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": _SCRATCHPAD_KEY_HELP[key],
                    }
                    for key in SCRATCHPAD_KEYS
                },
                "required": list(SCRATCHPAD_KEYS),
            },
            is_read_only=True,
        )

    def check_permissions(
        self, tool_input: dict[str, Any], context: Any
    ) -> PermissionResult:
        # Artifact-internal state, not a workspace mutation: always allowed,
        # even under the weak profile's read-only permission handler.
        return PermissionResult.allow()

    def run(self, tool_input: dict[str, Any], context: Any) -> ToolResult:
        try:
            delta = self._scratchpad.update(tool_input or {})
        except ScratchpadError as exc:
            return ToolResult(
                name=SCRATCHPAD_TOOL_NAME,
                output={
                    "error": f"invalid scratchpad payload: {exc}",
                    "hint": "Send all four keys (checklist, "
                            "questions_to_resolve, key_findings, "
                            "irrelevant_paths), each a list of strings.",
                },
                is_error=True,
            )
        self.update_count += 1
        snapshot = self._scratchpad.to_dict()
        if self._on_update is not None:
            try:
                self._on_update(snapshot)
            except Exception:
                pass  # UI callbacks must never break the investigation
        return ToolResult(
            name=SCRATCHPAD_TOOL_NAME,
            output={
                "scratchpad": snapshot,
                "open_questions": self._scratchpad.open_questions_count(),
                "complete": self._scratchpad.is_complete(),
                "questions_resolved": delta["questions_resolved"],
            },
            is_error=False,
        )


_SCRATCHPAD_KEY_HELP = {
    "checklist": "Steps you plan to take (update as the plan changes).",
    "questions_to_resolve": (
        "Open questions. EMPTY this list to finish -- the harness keeps "
        "you going until it is empty."
    ),
    "key_findings": "Confirmed facts worth reporting in the final output.",
    "irrelevant_paths": "Explored and ruled out -- do not revisit these.",
}


# --- Final-output handling --------------------------------------------------

def extract_json_payload(text: str) -> Any:
    """Tolerantly extract JSON from a model's final message.

    Accepts raw JSON or prose-wrapped JSON (first ``{``/``[`` to the
    matching last ``}``/``]``, codex assessment.rs style). Raises
    ``ValueError`` when nothing parses.
    """
    if not isinstance(text, str):
        raise ValueError("final output is not text")
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = stripped.find(open_c)
        end = stripped.rfind(close_c)
        if 0 <= start < end:
            try:
                return json.loads(stripped[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                continue
    raise ValueError("final output is not valid JSON (no JSON object/array found)")


def validate_final_output(
    text: str,
    schema: Optional[dict[str, Any]] = None,
) -> tuple[bool, Any]:
    """Termination clause 2: the final message must validate as JSON.

    Returns ``(True, parsed)`` on success, ``(False, error_message)`` on
    failure. Never raises.
    """
    schema = schema or DEFAULT_INVESTIGATOR_OUTPUT_SCHEMA
    try:
        parsed = extract_json_payload(text)
    except ValueError as exc:
        return False, str(exc)
    try:
        import jsonschema

        jsonschema.validate(instance=parsed, schema=schema)
    except ImportError:
        # Minimal fallback: object shape + required keys only.
        if not isinstance(parsed, dict):
            return False, "final output must be a JSON object"
        missing = [k for k in schema.get("required", []) if k not in parsed]
        if missing:
            return False, f"final output missing required keys: {missing}"
        return True, parsed
    except Exception as exc:  # jsonschema.ValidationError etc.
        return False, f"final output does not match schema: {exc}"
    return True, parsed


# --- Termination gate -------------------------------------------------------

@dataclass
class TerminationDecision:
    """Outcome of the harness termination check after a loop run."""

    finished: bool
    reason: str  # complete | open_questions | invalid_output |
                 # turn_cap | time_cap | cancelled
    continue_prompt: Optional[str] = None


def termination_check(
    *,
    scratchpad: Scratchpad,
    response_text: str,
    output_schema: Optional[dict[str, Any]] = None,
    turns_used: int,
    max_turns: int,
    time_exceeded: bool,
    cancelled: bool = False,
) -> TerminationDecision:
    """Decide whether the investigator may finish or must continue.

    The subagent may only finish when ``questions_to_resolve`` is empty
    AND the final output validates against the schema; otherwise the
    harness continues it (within the turn/time caps).
    """
    if cancelled:
        return TerminationDecision(
            finished=True,
            reason="time_cap" if time_exceeded else "cancelled",
        )
    if turns_used >= max_turns:
        return TerminationDecision(finished=True, reason="turn_cap")
    if time_exceeded:
        return TerminationDecision(finished=True, reason="time_cap")

    if not scratchpad.is_complete():
        open_qs = "\n".join(f"- {q}" for q in scratchpad.questions_to_resolve)
        return TerminationDecision(
            finished=False,
            reason="open_questions",
            continue_prompt=(
                "You stopped with open questions on your scratchpad, so the "
                "investigation is NOT done. Continue now:\n"
                f"{open_qs}\n"
                "Update your scratchpad via the Scratchpad tool as you work, "
                "and keep going until questions_to_resolve is empty AND your "
                "final message is valid JSON matching the required schema."
            ),
        )

    valid, detail = validate_final_output(response_text, output_schema)
    if not valid:
        return TerminationDecision(
            finished=False,
            reason="invalid_output",
            continue_prompt=(
                f"Your final message did not satisfy the required output "
                f"schema: {detail}. Respond again with ONLY the JSON object "
                f"matching the schema -- no prose wrapper."
            ),
        )

    return TerminationDecision(finished=True, reason="complete")


def forced_wrapup(
    scratchpad: Scratchpad,
    response_text: str,
    reason: str,
) -> dict[str, Any]:
    """Synthesize the final artifact when a cap fires (opencode max-steps
    pattern): the run always ends with a usable artifact built from the
    scratchpad, never a silent truncation."""
    return {
        "summary": (
            f"Investigation hit the harness cap ({reason}) before the "
            f"scratchpad was complete. Partial results below; "
            f"{scratchpad.open_questions_count()} question(s) remain open."
        ),
        "findings": list(scratchpad.key_findings),
        "forced_wrapup": True,
        "cap_reason": reason,
        "open_questions": list(scratchpad.questions_to_resolve),
        "irrelevant_paths": list(scratchpad.irrelevant_paths),
        "last_response": (response_text or "")[:2000],
        "scratchpad": scratchpad.to_dict(),
    }


# --- 018 reviewer reuse -----------------------------------------------------

def record_verdict(
    scratchpad: Scratchpad, verdict: dict[str, Any]
) -> dict[str, Any]:
    """Fold a reviewer's rubric verdict into the scratchpad's final state.

    For 018's reviewer subagents: their rubric verdict becomes the
    scratchpad's final state -- the verdict is appended to key_findings
    (so the wrap-up artifact carries it) and the returned dict is the
    final scratchpad snapshot. Returns the snapshot dict.
    """
    if not isinstance(verdict, dict):
        raise ScratchpadError("verdict must be a dict")
    entry = "verdict: " + json.dumps(verdict, sort_keys=True)
    if entry not in scratchpad.key_findings:
        scratchpad.key_findings.append(entry)
    return scratchpad.to_dict()


def investigator_system_preamble(
    output_schema: Optional[dict[str, Any]] = None,
    seed_questions: Optional[list[str]] = None,
) -> str:
    """Protocol text injected ahead of the investigator's task prompt."""
    schema = output_schema or DEFAULT_INVESTIGATOR_OUTPUT_SCHEMA
    schema_text = json.dumps(schema, indent=2)
    seed_text = ""
    if seed_questions:
        seed_text = ("\nSeeded open questions (refine these on your first "
                     "Scratchpad call):\n" + "\n".join(f"- {q}" for q in seed_questions))
    return f"""## Investigator protocol (mandatory -- the harness enforces this)

You are a tightly-scaffolded investigation subagent. You work READ-ONLY:
any mutation tool call is denied, so plan around reads (Read, Glob, Grep,
WebFetch, WebSearch) only.

1. FIRST TURN: call the Scratchpad tool to create your working scratchpad
   (checklist / questions_to_resolve / key_findings / irrelevant_paths).{seed_text}
2. EVERY TURN: re-emit the FULL updated scratchpad via the Scratchpad tool
   (whole overwrite -- all four lists, every time). Move answered items
   out of questions_to_resolve into key_findings; log dead ends in
   irrelevant_paths.
3. YOU MAY ONLY FINISH when BOTH hold:
   (a) questions_to_resolve is EMPTY, and
   (b) your final message is a single JSON object matching this schema:
{schema_text}
   If you stop early, the harness WILL continue you -- stopping is not an
   escape hatch.
4. Caps: {INVESTIGATOR_MAX_TURNS} turns / {INVESTIGATOR_MAX_TIME_MINUTES:g} minutes.
   At the cap the harness force-wraps from your scratchpad state, so keep
   the scratchpad current -- it is your deliverable of last resort.
5. Be precise and evidence-backed. No invented results; log what is missing.

## Task
"""
