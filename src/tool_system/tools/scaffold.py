"""Scaffold tool tier (brief 017).

Four first-class scaffold tools that structure the agent's reasoning instead of
leaving it to free text, ported from continue's CLI tool tier
(``reportFailure`` / ``exit`` / ``AskQuestion``) with opencode's ``question.ts``
refinements (recommended option rendered first with an explicit
"(Recommended)" suffix; dismissal tunneled as a typed defect the model
continues from) and cline's contract rule forbidding jailbreak options
(e.g. "toggle to Act mode").

Tools
-----
- ``Checklist`` — ordered ``{label, status}`` plan-as-structured-state.
  Read-only with respect to the world (no side effects); updated via
  whole-list overwrite. Pairs with the todo invariants of brief 020
  (statuses include ``blocked``; at most one ``in_progress``).
- ``AskQuestion`` — forces ambiguity into a typed 2-5 option channel.
  Options that would change agent mode/permissions are rejected by the
  contract. Dismissal raises :class:`ToolFailure` (continuable), and
  dismissal-with-feedback carries the feedback as the failure reason.
- ``ReportFailure`` — terminal: marks the run failed with a structured
  reason and the list of attempted approaches. Emits a log-ready record
  for the run log / the 015 done-gate's records.
- ``Exit`` (alias ``Status``) — loop hygiene: sets ``isComplete`` with an
  outcome enum (``done`` / ``failed``) so the harness can distinguish a
  clean finish from a silent stall. This is NOT a second completion tool:
  020's ``complete_task`` remains the terminal success path; ``Exit`` is
  the low-level loop-termination signal.

Integration notes (for the integrator; nothing here is wired yet)
-----------------------------------------------------------------
- Registry: add to ``src/tool_system/defaults.py`` (owned by brief 014's
  builder this wave).
- Loop: ``src/tool_system/agent_loop.py`` (CRLF) should catch
  :class:`ToolFailure` from ``tool.run()`` and feed it back to the model
  as a continuable error result (not loop termination), and treat
  :func:`is_terminal_result` / :func:`evaluate_run_end` as the two-clause
  completion check.
- Plan mode: ``registry.py``'s plan-mode gate special-cases ``ExitPlanMode``;
  consider allowing ``Exit``/``ReportFailure`` alongside it (a run can
  fail or exit while planning).
- Web UI: each tool's output carries a ``"widget"`` key
  (``checklist`` / ``ask_question`` / ``report_failure`` / ``exit``) so
  the UI can render compact structured widgets instead of raw JSON dumps
  (``src/web_app.py`` is CRLF for this work).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..context import ToolContext
from ..errors import ToolInputError, ToolPermissionError
from ..protocol import ToolResult
from ..registry import ToolSpec

__all__ = [
    "ToolFailure",
    "QuestionDismissed",
    "ChecklistTool",
    "AskQuestionTool",
    "ReportFailureTool",
    "ExitTool",
    "SCAFFOLD_TOOLS",
    "CHECKLIST_STATUSES",
    "EXIT_OUTCOMES",
    "FORBIDDEN_OPTION_PATTERNS",
    "RECOMMENDED_SUFFIX",
    "is_terminal_result",
    "evaluate_run_end",
    "render_question",
]


# ---------------------------------------------------------------------------
# Typed control-flow markers
# ---------------------------------------------------------------------------


class ToolFailure(Exception):
    """A typed failure the model can continue from.

    Raised (not returned) by scaffold tools when the interaction defects
    without killing the run — e.g. the user dismisses an ``AskQuestion``
    prompt. ``continuable`` is always True: the agent loop must feed the
    reason back to the model as an error result and let it continue. It is
    never a dead end and never loop termination.

    Integration (PENDING, agent_loop.py): catch ``ToolFailure`` around
    ``tool.run()`` and convert to a continuable tool-error result whose
    content is ``failure.reason``.
    """

    def __init__(self, reason: str, *, kind: str = "failure") -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("ToolFailure requires a non-empty reason")
        super().__init__(reason)
        self.reason: str = reason
        self.kind: str = kind
        self.continuable: bool = True


class QuestionDismissed(Exception):
    """Raised by an ask handler / UI layer when the user dismisses a question.

    Carries optional free-text ``feedback``; ``AskQuestionTool`` converts this
    into a :class:`ToolFailure` whose reason is the feedback (or a default
    dismissal message), so the model continues from the defect instead of
    hitting a dead end.
    """

    def __init__(self, feedback: Optional[str] = None) -> None:
        super().__init__(feedback or "question dismissed")
        self.feedback: Optional[str] = feedback


RECOMMENDED_SUFFIX = " (Recommended)"

CHECKLIST_STATUSES = ("pending", "in_progress", "completed", "blocked")

EXIT_OUTCOMES = ("done", "failed")

# cline's AskQuestion contract: options must not change agent mode or
# permissions (e.g. "toggle to Act mode"). Case-insensitive regexes matched
# against "label + description".
FORBIDDEN_OPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bact\s*mode\b", re.IGNORECASE),
    re.compile(r"\bplan\s*mode\b", re.IGNORECASE),
    re.compile(r"\btoggle\b.{0,20}\bmode\b", re.IGNORECASE),
    re.compile(r"\b(change|switch|enter|exit)\b.{0,20}\bmode\b", re.IGNORECASE),
    re.compile(r"\bpermission\s*mode\b", re.IGNORECASE),
    re.compile(r"\b(bypass|disable|ignore|drop)\b.{0,30}\bpermission", re.IGNORECASE),
    re.compile(r"\b(permission|privilege|access)\b.{0,30}\b(elevat|escalat|grant)", re.IGNORECASE),
    re.compile(r"\belevat", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bignore\b.{0,30}\b(guardrail|safeguard|policy|policies)\b", re.IGNORECASE),
    re.compile(r"\bdisable\b.{0,30}\b(guardrail|safeguard)\b", re.IGNORECASE),
)


def _check_option_contract(label: str, description: str) -> Optional[str]:
    """Return the offending pattern source if an option breaks the contract."""
    text = f"{label} {description}"
    for pattern in FORBIDDEN_OPTION_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def render_question(question: str, options: list[dict[str, Any]]) -> dict[str, Any]:
    """Render a question for presentation: recommended option first.

    The recommended option's label gets the explicit ``(Recommended)`` suffix
    (opencode's ``question.ts`` refinement). Returns a new dict; the input is
    not mutated.
    """
    ordered = sorted(options, key=lambda o: not bool(o.get("recommended")))
    rendered_options = []
    for opt in ordered:
        label = opt["label"]
        if opt.get("recommended"):
            label = f"{label}{RECOMMENDED_SUFFIX}"
        rendered_options.append({"label": label, "description": opt["description"]})
    return {"question": question, "options": rendered_options}


def is_terminal_result(result: ToolResult) -> bool:
    """True when a tool result signals the run should stop.

    Integration helper (PENDING, agent_loop.py): a result is terminal when it
    came from ``Exit``/``Status`` (``isComplete`` set) or ``ReportFailure``
    (``outcome == "failed"``).
    """
    output = result.output
    if not isinstance(output, dict):
        return False
    if output.get("isComplete") is True:
        return True
    return output.get("outcome") == "failed" and result.name in {"ReportFailure", "Exit", "Status"}


def evaluate_run_end(
    *,
    exit_called: bool,
    exit_outcome: Optional[str] = None,
    final_message_had_tool_calls: bool,
) -> dict[str, Any]:
    """Two-clause loop-hygiene check (continue's termination protocol).

    - A terminal tool (``Exit``/``Status``) set ``isComplete`` -> the run is
      complete with the recorded outcome.
    - Otherwise the run did NOT complete cleanly: an assistant message with
      no tool calls and no ``Exit`` is a stall, not a finish (020 treats the
      absence of ``complete_task`` + no tool calls as stall, not success),
      and a loop that ended while tool calls were still in flight is a
      stall too.

    Returns ``{"isComplete": bool, "outcome": "done"|"failed"|"stall",
    "reason": str}`` for the run log.
    """
    if exit_called:
        outcome = exit_outcome or "done"
        return {
            "isComplete": True,
            "outcome": outcome,
            "reason": "terminal tool set isComplete",
        }
    if final_message_had_tool_calls:
        return {
            "isComplete": False,
            "outcome": "stall",
            "reason": "loop ended while tool calls were still in flight; no Exit recorded",
        }
    return {
        "isComplete": False,
        "outcome": "stall",
        "reason": "assistant message with no tool calls but no Exit: silent stall, not done",
    }


# ---------------------------------------------------------------------------
# Checklist
# ---------------------------------------------------------------------------


class ChecklistTool:
    """Ordered {label, status} plan-as-structured-state. Read-only vs the
    world (no side effects); updated via whole-list overwrite."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Checklist",
            description=(
                "Maintain an ordered checklist of {label, status} items as structured "
                "tool state (not prose). Read-only with respect to the world: it never "
                "touches files, shell, or network. Update via whole-list overwrite: "
                "every call submits the ENTIRE current list. Statuses: pending, "
                "in_progress, completed, blocked. At most one item may be in_progress "
                "at a time. Use it to keep the plan visible and honest."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string", "minLength": 1},
                                "status": {
                                    "type": "string",
                                    "enum": list(CHECKLIST_STATUSES),
                                },
                            },
                            "required": ["label", "status"],
                        },
                    }
                },
                "required": ["items"],
            },
            is_read_only=True,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        items = tool_input.get("items")
        if not isinstance(items, list) or not items:
            raise ToolInputError("items must be a non-empty array")
        seen_labels: set[str] = set()
        in_progress = 0
        normalized: list[dict[str, str]] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ToolInputError(f"items[{i}] must be an object")
            label = item.get("label")
            status = item.get("status")
            if not isinstance(label, str) or not label.strip():
                raise ToolInputError(f"items[{i}].label must be a non-empty string")
            if status not in CHECKLIST_STATUSES:
                raise ToolInputError(
                    f"items[{i}].status must be one of {', '.join(CHECKLIST_STATUSES)}"
                )
            if label in seen_labels:
                raise ToolInputError(f"items[{i}].label duplicates an earlier item: {label!r}")
            seen_labels.add(label)
            if status == "in_progress":
                in_progress += 1
            normalized.append({"label": label, "status": status})
        if in_progress > 1:
            raise ToolInputError(
                "at most one checklist item may be in_progress at a time "
                f"(found {in_progress}); submit the whole list with the others "
                "moved to pending, blocked, or completed"
            )
        return ToolResult(
            name="Checklist",
            output={"items": normalized, "widget": "checklist"},
        )


# ---------------------------------------------------------------------------
# AskQuestion
# ---------------------------------------------------------------------------


class AskQuestionTool:
    """Force ambiguity into a typed 2-5 option channel. Dismissal raises a
    continuable ToolFailure; mode/permission-changing options are rejected."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="AskQuestion",
            description=(
                "Ask a single multiple-choice question when you are genuinely blocked "
                "on a decision the user must make. Provide 2-5 options, each with a "
                "label and description; mark at most one option recommended:true and it "
                "is presented first with a (Recommended) suffix. CONTRACT: no option "
                "may change the agent's mode or permissions (e.g. 'toggle to Act "
                "mode', 'bypass permissions', 'elevate privileges') — such options "
                "are rejected. If the user dismisses the question, you receive a "
                "typed failure you can continue from — never stall, never re-ask "
                "the same question unchanged."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "question": {"type": "string", "minLength": 1},
                    "options": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 5,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "label": {"type": "string", "minLength": 1},
                                "description": {"type": "string", "minLength": 1},
                                "recommended": {"type": "boolean"},
                            },
                            "required": ["label", "description"],
                        },
                    },
                    "answer": {
                        "type": "string",
                        "description": (
                            "Pre-supplied answer (option label) for non-interactive "
                            "runs and harness replays. Omit for interactive use."
                        ),
                    },
                },
                "required": ["question", "options"],
            },
            is_read_only=True,
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        question = tool_input.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ToolInputError("question must be a non-empty string")
        options = self._validate_options(tool_input.get("options"))
        rendered = render_question(question, options)

        answer = tool_input.get("answer")
        if answer is not None:
            self._validate_answer(answer, options)
            return ToolResult(
                name="AskQuestion",
                output={
                    "question": question,
                    "options": rendered["options"],
                    "answer": self._base_label(answer),
                    "answered_via": "pre-supplied",
                    "widget": "ask_question",
                },
            )

        if context.ask_user is None:
            raise ToolPermissionError(
                "AskQuestion requires user interaction but no ask_user handler "
                "is configured"
            )
        try:
            raw = context.ask_user([rendered])
        except QuestionDismissed as dismissed:
            raise self._dismissal_failure(dismissed.feedback) from dismissed
        if not isinstance(raw, dict) or not raw:
            raise self._dismissal_failure(None)
        picked = raw.get(question)
        if picked is None:
            # Handler keyed answers by the rendered payload; fall back to the
            # single value when only one answer was returned.
            values = [v for v in raw.values() if isinstance(v, str) and v]
            picked = values[0] if len(values) == 1 else None
        if picked is None:
            raise self._dismissal_failure(None)
        self._validate_answer(picked, options)
        return ToolResult(
            name="AskQuestion",
            output={
                "question": question,
                "options": rendered["options"],
                "answer": self._base_label(picked),
                "widget": "ask_question",
            },
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _base_label(label: str) -> str:
        if label.endswith(RECOMMENDED_SUFFIX):
            return label[: -len(RECOMMENDED_SUFFIX)]
        return label

    def _validate_options(self, options: Any) -> list[dict[str, Any]]:
        if not isinstance(options, list) or not (2 <= len(options) <= 5):
            raise ToolInputError(
                "options must be an array of 2-5 items "
                f"(got {len(options) if isinstance(options, list) else type(options).__name__})"
            )
        seen: set[str] = set()
        recommended = 0
        normalized: list[dict[str, Any]] = []
        for i, opt in enumerate(options):
            if not isinstance(opt, dict):
                raise ToolInputError(f"options[{i}] must be an object")
            label = opt.get("label")
            description = opt.get("description")
            if not isinstance(label, str) or not label.strip():
                raise ToolInputError(f"options[{i}].label must be a non-empty string")
            if not isinstance(description, str) or not description.strip():
                raise ToolInputError(f"options[{i}].description must be a non-empty string")
            if label in seen:
                raise ToolInputError(f"options[{i}].label duplicates an earlier option: {label!r}")
            seen.add(label)
            offending = _check_option_contract(label, description)
            if offending is not None:
                raise ToolInputError(
                    f"options[{i}] {label!r} violates the AskQuestion contract: options "
                    "must not change agent mode or permissions "
                    f"(matched {offending!r})"
                )
            is_recommended = bool(opt.get("recommended"))
            recommended += 1 if is_recommended else 0
            normalized.append(
                {"label": label, "description": description, "recommended": is_recommended}
            )
        if recommended > 1:
            raise ToolInputError("at most one option may be marked recommended")
        return normalized

    def _validate_answer(self, answer: Any, options: list[dict[str, Any]]) -> None:
        if not isinstance(answer, str) or not answer:
            raise ToolInputError("answer must be a non-empty option label")
        labels = {opt["label"] for opt in options}
        if self._base_label(answer) not in labels:
            raise ToolInputError(
                f"answer {answer!r} does not match any option label "
                f"({', '.join(sorted(labels))})"
            )

    @staticmethod
    def _dismissal_failure(feedback: Optional[str]) -> ToolFailure:
        if isinstance(feedback, str) and feedback.strip():
            return ToolFailure(
                f"question dismissed with feedback: {feedback.strip()}",
                kind="dismissed_with_feedback",
            )
        return ToolFailure(
            "question dismissed by user without an answer; continue with your best "
            "judgment or re-ask with refined options",
            kind="dismissed",
        )


# ---------------------------------------------------------------------------
# ReportFailure
# ---------------------------------------------------------------------------


class ReportFailureTool:
    """Terminal: mark the run failed with a structured reason and the list of
    attempted approaches. Emits a log-ready record."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="ReportFailure",
            description=(
                "TERMINAL. Call when the task is unrecoverable: you have exhausted "
                "reasonable approaches and cannot complete it. Marks the run failed "
                "with a structured reason and the list of approaches you attempted, "
                "so the run log records exactly what went wrong. Never call this "
                "to bail out early — it is the failure exit, not a shortcut."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Structured reason the task cannot be completed.",
                    },
                    "attempted": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                        "description": "Approaches attempted before giving up.",
                    },
                },
                "required": ["reason", "attempted"],
            },
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        reason = tool_input.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ToolInputError("reason must be a non-empty string")
        attempted = tool_input.get("attempted")
        if not isinstance(attempted, list) or not attempted:
            raise ToolInputError("attempted must be a non-empty array of strings")
        for i, step in enumerate(attempted):
            if not isinstance(step, str) or not step.strip():
                raise ToolInputError(f"attempted[{i}] must be a non-empty string")
        log_record = {
            "tool": "ReportFailure",
            "outcome": "failed",
            "reason": reason.strip(),
            "attempted": [s.strip() for s in attempted],
        }
        return ToolResult(
            name="ReportFailure",
            output={
                "isComplete": True,
                "outcome": "failed",
                "reason": reason.strip(),
                "attempted": [s.strip() for s in attempted],
                "widget": "report_failure",
                "log_record": log_record,
            },
        )


# ---------------------------------------------------------------------------
# Exit / Status
# ---------------------------------------------------------------------------


class ExitTool:
    """Loop hygiene: set isComplete with an outcome enum (done/failed).

    This is the low-level loop-termination signal, NOT a second completion
    tool — 020's complete_task remains the terminal success path."""

    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="Exit",
            description=(
                "Loop-hygiene exit: sets isComplete with an outcome (done/failed) "
                "so the harness can distinguish a clean finish from a silent stall. "
                "Use done when the loop's work is genuinely finished; use failed "
                "with a summary when stopping without success. This is the "
                "termination signal — it does not replace complete_task as the "
                "structured success path."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "outcome": {
                        "type": "string",
                        "enum": list(EXIT_OUTCOMES),
                        "description": "done: loop finished cleanly. failed: stopping without success.",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short summary of the final state for the run log.",
                    },
                },
                "required": ["outcome"],
            },
            aliases=("Status",),
            strict=True,
        )

    def run(self, tool_input: dict[str, Any], context: ToolContext) -> ToolResult:
        outcome = tool_input.get("outcome")
        if outcome not in EXIT_OUTCOMES:
            raise ToolInputError(f"outcome must be one of {', '.join(EXIT_OUTCOMES)}")
        summary = tool_input.get("summary")
        if summary is not None and not isinstance(summary, str):
            raise ToolInputError("summary must be a string when provided")
        return ToolResult(
            name="Exit",
            output={
                "isComplete": True,
                "outcome": outcome,
                "summary": (summary or "").strip(),
                "widget": "exit",
                "log_record": {
                    "tool": "Exit",
                    "outcome": outcome,
                    "summary": (summary or "").strip(),
                },
            },
        )


SCAFFOLD_TOOLS: tuple[type, ...] = (
    ChecklistTool,
    AskQuestionTool,
    ReportFailureTool,
    ExitTool,
)
