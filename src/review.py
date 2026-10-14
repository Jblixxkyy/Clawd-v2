"""Reviewer loop (brief 018).

On-demand ``/review`` machinery, modeled on codex's reviewer and continue's
``cn check``:

- ``snapshot_diff`` captures the worktree's uncommitted changes.
- ``build_review_prompt`` renders a codex-style rubric prompt (P0-P3
  findings, exact code ranges, strict-JSON verdict schema).
- ``ReviewerSession`` spawns a fresh-context reviewer through an injected
  reviewer runner, parses the verdict tolerantly (prose-wrapped JSON is
  accepted; missing fields are inferred), injects the verdict back as a
  user-role message so the worker iterates, and logs a structured
  ``review_verdict`` run-log event (OpenHands SecurityAnalysisEvent pattern).
- Reviewers may return a unified-diff **patch capsule** alongside the
  verdict (``ReviewVerdict.patch``); it is exposed as data for a UI
  apply/discard widget.

Spawning uses an injected ``reviewer_runner(prompt, model) -> str``. The
production runner is a thin wrapper over the existing ``Subagent`` tool API
(``src/tool_system/tools/subagent.py`` — owned by the 024 builder; never
edited here). Wiring that runner into the live command path is a PENDING
integration item; until then ``/review`` falls back to emitting the
reviewer prompt as text the worker can hand to the Subagent tool itself.

Model default: the session's cheap/secondary model via
``src.model_tiers.utility_model`` (028) — reviewers do not burn main-model
budget.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from src.model_tiers import tier_for_model, utility_model

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITIES = ("P0", "P1", "P2", "P3")
FAIL_SEVERITIES = ("P0", "P1")  # findings that flip an absent verdict to fail

DEFAULT_MAX_REVIEW_ITERATIONS = 2
DEFAULT_REVIEWER_MODEL: Optional[str] = None

REVIEW_CONFIG_KEY = "review"  # config["review"] = {"model": ..., "max_iterations": ...}

_REVIEW_EVENT_KIND = "review_verdict"
_JUDGE_EVENT_KIND = "judge_verdict"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class ReviewFinding:
    severity: str = "P2"
    file: str = ""
    start_line: int = 0
    end_line: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "file": self.file,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "description": self.description,
        }


@dataclass
class ReviewVerdict:
    status: str = "pass"  # "pass" | "fail"
    confidence: float = 0.5
    findings: list[ReviewFinding] = field(default_factory=list)
    patch: Optional[str] = None  # unified-diff patch capsule (continue pattern)
    raw: str = ""
    model: Optional[str] = None

    @property
    def findings_count(self) -> int:
        return len(self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "findings": [f.to_dict() for f in self.findings],
            "patch": self.patch,
            "model": self.model,
        }


# ---------------------------------------------------------------------------
# Diff snapshot
# ---------------------------------------------------------------------------


def snapshot_diff(root: Optional[Path | str] = None) -> str:
    """Return the worktree's uncommitted changes as a unified diff string.

    Uses ``git diff HEAD`` so both staged and unstaged changes are covered.
    Returns "" when there is nothing to review or git is unavailable. Never
    raises.
    """
    try:
        cwd = str(Path(root).expanduser()) if root is not None else None
        proc = subprocess.run(
            ["git", "diff", "HEAD", "--no-color", "--unified=3"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        diff = proc.stdout or ""
    except Exception:
        return ""
    diff = diff.strip()
    if not diff:
        return ""
    # Truncate very large diffs so the reviewer prompt stays sane.
    limit = 60_000
    if len(diff) > limit:
        diff = diff[:limit] + "\n... [diff truncated at 60k chars]"
    return diff


# ---------------------------------------------------------------------------
# Rubric prompt (codex-style)
# ---------------------------------------------------------------------------

REVIEW_RUBRIC = """You are a code reviewer. Review the unified diff below carefully.

## Severity rubric
- **P0**: definite bug, security vulnerability, data loss, crash, or logic error that will fire in production. Must fix before merge.
- **P1**: probable bug, race condition, broken edge case, or significant maintainability hazard. Should fix before merge.
- **P2**: minor issue — style drift, unclear naming, missing test coverage for changed code, nit worth addressing.
- **P3**: informational — observation or suggestion, no action required.

## Rules
- Report ONLY issues you are confident about. Do not invent problems.
- Every finding MUST reference an exact location: the file path from the diff and the 1-based line range (start_line, end_line) in the NEW file version.
- If the diff is clean and correct, say so — verdict "pass" with an empty findings list.

## Output
Respond with a JSON object (you may wrap it in prose or a code fence):

{
  "verdict": "pass" | "fail",
  "confidence": 0.0,
  "findings": [
    {
      "severity": "P0" | "P1" | "P2" | "P3",
      "file": "path/to/file.py",
      "start_line": 12,
      "end_line": 15,
      "description": "what is wrong and why"
    }
  ],
  "patch": "<optional unified diff fixing the findings, or null>"
}

"verdict" is "fail" if there is any P0 or P1 finding, "pass" otherwise.
"confidence" is your confidence in the verdict from 0.0 to 1.0.
"findings" is a (possibly empty) array, ordered by severity then file.
"patch" is an optional unified diff you propose as the fix (apply/discard widget data), or null.
"""


def build_review_prompt(diff_text: str, context_note: str = "") -> str:
    """Render the full reviewer prompt: rubric + diff (+ optional note)."""
    parts = [REVIEW_RUBRIC.strip(), "", "## Diff under review", "", "```diff", diff_text.strip(), "```"]
    if context_note:
        parts += ["", "## Context", "", context_note.strip()]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Tolerant verdict parsing (codex assessment.rs style)
# ---------------------------------------------------------------------------


def extract_json_blob(text: str) -> Optional[str]:
    """Extract the first balanced JSON object from possibly prose-wrapped text.

    Handles ```json / ``` fences and trailing prose. Returns None when no
    balanced object is found.
    """
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # Fallback: first "{" to last "}" (unbalanced scan — best effort).
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1]
    return None


def _coerce_severity(value: Any) -> str:
    sev = str(value or "P2").strip().upper()
    return sev if sev in SEVERITIES else "P2"


def _coerce_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _extract_diff_fence(text: str) -> Optional[str]:
    """Pull a ```diff fenced block out of raw text (patch capsule)."""
    match = __import__("re").search(r"```diff\s*\n(.*?)```", text, __import__("re").S)
    if match:
        body = match.group(1).strip()
        return body or None
    return None


def parse_review_verdict(text: str, model: Optional[str] = None) -> ReviewVerdict:
    """Parse a reviewer's raw output into a ReviewVerdict, tolerantly.

    Accepts prose-wrapped JSON and infers missing fields:
    - verdict absent → "fail" if any P0/P1 finding, else "pass"
    - confidence absent → 0.5
    - findings absent → []
    - patch absent → null (also checks ```diff fences in the raw text)

    Unparseable output becomes a fail verdict with the raw text attached, so
    a confused reviewer can never silently pass.
    """
    raw = text or ""
    blob = extract_json_blob(raw)
    data: dict[str, Any] = {}
    parsed_ok = False
    if blob:
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                data = parsed
                parsed_ok = True
        except (ValueError, TypeError):
            data = {}

    findings: list[ReviewFinding] = []
    for item in data.get("findings") or []:
        if isinstance(item, str):
            findings.append(ReviewFinding(description=item))
        elif isinstance(item, dict):
            findings.append(
                ReviewFinding(
                    severity=_coerce_severity(item.get("severity")),
                    file=str(item.get("file") or ""),
                    start_line=_coerce_int(item.get("start_line") or item.get("line")),
                    end_line=_coerce_int(item.get("end_line") or item.get("line")),
                    description=str(item.get("description") or ""),
                )
            )

    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in ("pass", "fail"):
        verdict = "fail" if any(f.severity in FAIL_SEVERITIES for f in findings) else "pass"
    if not parsed_ok and raw.strip():
        # The reviewer produced no parseable verdict at all — a confused
        # reviewer must never silently pass.
        verdict = "fail"

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    patch = data.get("patch")
    patch = str(patch).strip() if isinstance(patch, str) and patch.strip() else None
    if patch is None:
        # continue pattern: also accept a ```diff fence outside the JSON
        patch = _extract_diff_fence(raw)

    return ReviewVerdict(
        status=verdict,
        confidence=confidence,
        findings=findings,
        patch=patch,
        raw=raw,
        model=model,
    )


# ---------------------------------------------------------------------------
# Run-log verdict events (OpenHands SecurityAnalysisEvent pattern)
# ---------------------------------------------------------------------------


def make_verdict_event(
    kind: str,
    verdict: Any,
    model: Optional[str],
    confidence: float,
    findings_count: int = 0,
    iteration: int = 0,
    extra: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a structured verdict event dict (SecurityAnalysisEvent shape)."""
    event: dict[str, Any] = {
        "kind": kind,
        "verdict": verdict,
        "model": model,
        "tier": tier_for_model(model, config),
        "confidence": confidence,
        "findings_count": findings_count,
        "iteration": iteration,
    }
    if extra:
        event.update(extra)
    return event


def log_verdict_event(
    history: Any,
    event: dict[str, Any],
) -> None:
    """Append a structured verdict event to the run log.

    ``history`` is a ``HistoryLog``-like object with ``add(title, detail)``.
    Silently no-ops when the history object is missing or broken.
    """
    try:
        if history is None or not hasattr(history, "add"):
            return
        history.add(event.get("kind", "verdict"), json.dumps(event, ensure_ascii=False))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Worker-facing rendering (injected as a user-role message)
# ---------------------------------------------------------------------------


def format_verdict_for_worker(verdict: ReviewVerdict) -> str:
    """Render a verdict as the user-role message the worker iterates on."""
    lines = [
        f"<review verdict=\"{verdict.status}\" confidence={verdict.confidence:.2f}>",
        "",
    ]
    if verdict.status == "pass":
        lines.append("The reviewer found no issues. Continue with your work.")
    else:
        lines.append(f"The reviewer found {verdict.findings_count} issue(s). Fix them, then run /review again.")
        lines.append("")
        for f in verdict.findings:
            loc = f.file or "?"
            if f.start_line:
                loc += f":{f.start_line}" + (f"-{f.end_line}" if f.end_line and f.end_line != f.start_line else "")
            lines.append(f"- [{f.severity}] {loc}: {f.description}")
        if verdict.patch:
            lines.append("")
            lines.append("The reviewer also suggested this patch:")
            lines.append("```diff")
            lines.append(verdict.patch)
            lines.append("```")
    lines += ["", "</review>"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reviewer session orchestration
# ---------------------------------------------------------------------------


def make_subagent_reviewer_runner(
    subagent_tool: Any,
    tool_context: Any,
    description: str = "review code diff",
    max_turns: int = 10,
    profile: str = "weak",
) -> Callable[[str, Optional[str]], str]:
    """Build a reviewer_runner over the existing Subagent tool API.

    Never edited: ``SubagentTool`` itself is owned by the 024 builder. This
    only *calls* ``tool.run(tool_input, context)`` with a description, the
    reviewer prompt, an explicit cheap model, and the read-only "weak"
    profile (reviewers only need to read the diff).

    Returns a callable ``runner(prompt, model) -> raw_text``.
    """
    def runner(prompt: str, model: Optional[str]) -> str:
        tool_input: dict[str, Any] = {
            "description": description,
            "prompt": prompt,
            "max_turns": max_turns,
            "profile": profile,
        }
        if model:
            tool_input["model"] = model
        result = subagent_tool.run(tool_input, tool_context)
        output = getattr(result, "output", None)
        if isinstance(output, dict):
            if output.get("error") and not output.get("response_text"):
                raise RuntimeError(f"reviewer subagent error: {output['error']}")
            text = output.get("response_text") or output.get("output") or ""
            return str(text)
        return str(output or "")

    return runner


class ReviewerSession:
    """One /review pass: spawn reviewer, parse verdict, inject, log, iterate.

    ``reviewer_runner`` is ``(prompt, model) -> raw_text`` (a stub in tests,
    ``make_subagent_reviewer_runner`` in production once wired). ``get_diff``
    is called before every pass so the worker can fix findings and be
    re-reviewed. The verdict is injected into ``conversation`` via
    ``add_user_message`` (fallback: append to ``conversation.messages``).
    """

    def __init__(
        self,
        reviewer_runner: Callable[[str, Optional[str]], str],
        *,
        model: Optional[str] = None,
        max_iterations: int = DEFAULT_MAX_REVIEW_ITERATIONS,
        history: Any = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.runner = reviewer_runner
        self.model = model
        self.max_iterations = max(1, max_iterations)
        self.history = history
        self.config = config or {}

    def _inject(self, conversation: Any, verdict: ReviewVerdict) -> None:
        message = format_verdict_for_worker(verdict)
        try:
            if hasattr(conversation, "add_user_message"):
                conversation.add_user_message(message)
            elif hasattr(conversation, "messages"):
                conversation.messages.append({"role": "user", "content": message})
        except Exception:
            pass

    def run(
        self,
        get_diff: Callable[[], str],
        conversation: Any,
        context_note: str = "",
    ) -> ReviewVerdict:
        last: Optional[ReviewVerdict] = None
        for iteration in range(self.max_iterations):
            diff = get_diff() or ""
            if not diff.strip():
                last = ReviewVerdict(status="pass", confidence=1.0, model=self.model)
                self._inject(conversation, last)
                return last
            prompt = build_review_prompt(diff, context_note)
            raw = self.runner(prompt, self.model)
            verdict = parse_review_verdict(raw, model=self.model)
            last = verdict
            log_verdict_event(
                self.history,
                make_verdict_event(
                    _REVIEW_EVENT_KIND,
                    verdict.status,
                    self.model,
                    verdict.confidence,
                    findings_count=verdict.findings_count,
                    iteration=iteration,
                    extra={"patch_present": verdict.patch is not None},
                    config=self.config,
                ),
            )
            self._inject(conversation, verdict)
            if verdict.status == "pass":
                return verdict
        return last or ReviewVerdict(status="pass", confidence=1.0, model=self.model)


# ---------------------------------------------------------------------------
# /review command helpers (registered in command_system/builtins.py)
# ---------------------------------------------------------------------------


def parse_review_args(args: str) -> dict[str, Any]:
    """Parse /review arguments: [--model MODEL] [--max-iterations N]."""
    argv = shlex.split(args or "")
    model: Optional[str] = None
    max_iterations = DEFAULT_MAX_REVIEW_ITERATIONS
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--model" and i + 1 < len(argv):
            model = argv[i + 1].strip() or None
            i += 2
        elif a.startswith("--model="):
            model = a.split("=", 1)[1].strip() or None
            i += 1
        elif a == "--max-iterations" and i + 1 < len(argv):
            try:
                max_iterations = max(1, int(argv[i + 1]))
            except ValueError:
                pass
            i += 2
        elif a.startswith("--max-iterations="):
            try:
                max_iterations = max(1, int(a.split("=", 1)[1]))
            except ValueError:
                pass
            i += 1
        else:
            i += 1
    return {"model": model, "max_iterations": max_iterations}


def resolve_review_model(args_model: Optional[str], config: Optional[dict[str, Any]]) -> Optional[str]:
    """Reviewer model: explicit flag > config review.model > small tier."""
    if args_model:
        return args_model
    cfg = (config or {}).get(REVIEW_CONFIG_KEY) or {}
    if cfg.get("model"):
        return str(cfg["model"])
    return utility_model(config=config) or DEFAULT_REVIEWER_MODEL


def resolve_review_max_iterations(args_value: int, config: Optional[dict[str, Any]]) -> int:
    cfg = (config or {}).get(REVIEW_CONFIG_KEY) or {}
    try:
        return max(1, int(cfg.get("max_iterations", args_value)))
    except (TypeError, ValueError):
        return args_value
