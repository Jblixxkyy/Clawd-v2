"""Judge-driven goal loop (brief 018), modeled on OpenHands.

After a run ends, a judge renders the transcript as plain ``role: text``
(the system prompt is deliberately EXCLUDED for cost) and returns a
``GoalVerdict{complete, missing[]}``. ``JudgeController`` decides
continue-vs-stop with an iteration cap: if the verdict reports missing items
and budget remains, the "what's missing" follow-up is fed back as the next
prompt and the run resumes; otherwise it stops. Per-run critic config with
``success_threshold``/``max_iterations`` mirrors OpenHands
``IterativeRefinementConfig``.

The judge model defaults to the session's cheap/secondary model via
``src.model_tiers.utility_model`` (028). Every verdict is logged as a
structured run-log event (OpenHands SecurityAnalysisEvent pattern).

Think events (023): the judge reads think contents *defensively* — content
blocks or message attributes that look like thinking are included when
present, but nothing hard-requires them to exist.

Post-run hook-in point for ``src/tool_system/agent_loop.py`` is PENDING
(the integrator owns that file); this module exposes everything the hook
needs: ``JudgeController``, ``should_judge_run``, and ``render_transcript``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.model_tiers import tier_for_model, utility_model
from src.review import extract_json_blob, log_verdict_event, make_verdict_event

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MAX_JUDGE_ITERATIONS = 3
DEFAULT_SUCCESS_THRESHOLD = 0.7
DEFAULT_JUDGE_MODEL: Optional[str] = None

JUDGE_CONFIG_KEY = "judge"  # config["judge"] = {"enabled": bool, "model": ..., "max_iterations": ..., "success_threshold": ...}

_JUDGE_EVENT_KIND = "judge_verdict"

DEFAULT_FOLLOWUP_TEMPLATE = (
    "The run is not complete yet. The following items are still missing:\n"
    "{missing}\n\n"
    "Continue the task and address every missing item above. "
    "When you believe everything is done, finish normally."
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class GoalVerdict:
    """complete/missing verdict from the judge."""

    complete: bool = False
    missing: list[str] = field(default_factory=list)
    confidence: float = 0.5
    score: Optional[float] = None  # 0..1 critic score; used with success_threshold
    raw: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "missing": self.missing,
            "confidence": self.confidence,
            "score": self.score,
        }


@dataclass
class CriticConfig:
    """Per-run critic configuration (OpenHands IterativeRefinementConfig)."""

    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD
    max_iterations: int = DEFAULT_MAX_JUDGE_ITERATIONS
    followup_template: str = DEFAULT_FOLLOWUP_TEMPLATE


# ---------------------------------------------------------------------------
# Transcript rendering (system prompt EXCLUDED for cost)
# ---------------------------------------------------------------------------


def _block_text(block: Any) -> tuple[str, str]:
    """Return (kind, text) for a content block, defensively.

    kind is "text" for normal text, "think" for reasoning/think content
    (023 ThinkEvent shape or {"type": "think"} blocks). Think content is
    read via getattr/.get chains and never hard-required.
    """
    if isinstance(block, str):
        return "text", block
    if isinstance(block, dict):
        btype = str(block.get("type") or "").lower()
        if btype == "think":
            return "think", str(block.get("text") or block.get("thought") or "")
        if btype == "text":
            return "text", str(block.get("text") or "")
        # tool_use / tool_result: keep the name + a short summary for the judge
        if btype == "tool_use":
            return "text", f"[tool call: {block.get('name', '?')}]"
        if btype == "tool_result":
            content = block.get("content") or ""
            if isinstance(content, list):
                content = " ".join(str(getattr(c, "text", c)) for c in content if c)
            text = str(content)
            if len(text) > 500:
                text = text[:500] + "…"
            return "text", f"[tool result]{' (error)' if block.get('is_error') else ''}: {text}"
        return "text", str(block.get("text") or block.get("content") or "")
    # Object-style blocks (dataclasses): getattr, never hard-require.
    text = getattr(block, "text", None)
    if isinstance(text, str):
        cls = type(block).__name__.lower()
        kind = "think" if "think" in cls else "text"
        return kind, text
    thought = getattr(block, "thought", None)  # 023 ThinkEvent field (defensive)
    if isinstance(thought, str):
        return "think", thought
    return "text", ""


def _message_text(content: Any) -> str:
    """Flatten message content (str or list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        pieces: list[str] = []
        for block in content:
            kind, text = _block_text(block)
            if not text:
                continue
            if kind == "think":
                pieces.append(f"(thinking) {text}")
            else:
                pieces.append(text)
        return "\n".join(pieces)
    return str(content)


def render_transcript(messages: Any) -> str:
    """Render a transcript as plain ``role: text`` lines for the judge.

    Accepts a Conversation (has ``get_messages``), a list of dicts, or a
    list of Message objects. The ``system`` role is deliberately EXCLUDED
    (cost). Think content is included defensively via ``_block_text``.
    """
    if messages is None:
        return ""
    if hasattr(messages, "get_messages"):
        try:
            messages = messages.get_messages()
        except Exception:
            return ""
    lines: list[str] = []
    for msg in messages or []:
        if isinstance(msg, dict):
            role = str(msg.get("role") or "unknown").lower()
            text = _message_text(msg.get("content"))
        else:
            role = str(getattr(msg, "role", "unknown") or "unknown").lower()
            content = getattr(msg, "content", None)
            # Defensive think-event read (023, in progress): a message that
            # carries thought content without a text body.
            think_text = getattr(msg, "thought", None)
            text = _message_text(content)
            if isinstance(think_text, str) and think_text.strip():
                text = (text + "\n" if text else "") + f"(thinking) {think_text.strip()}"
        if role == "system":
            continue
        lines.append(f"{role}: {text}".rstrip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge prompt + tolerant parsing
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a judge evaluating whether an agent run completed its goal.

## Transcript (role: text, system prompt excluded)
```transcript
{transcript}
```

## Task
Decide whether the run achieved its goal. Be strict: "complete" only if the transcript shows the goal's concrete outcomes were reached, not merely attempted.

## Output
Respond with a JSON object (you may wrap it in prose or a code fence):

{
  "complete": true | false,
  "missing": ["item still missing 1", "item still missing 2"],
  "confidence": 0.0,
  "score": 0.0
}

- "complete": true only if nothing is missing.
- "missing": concrete items still outstanding (empty when complete).
- "confidence": 0.0-1.0 confidence in the verdict.
- "score": optional 0.0-1.0 overall completion score.
"""


def build_judge_prompt(transcript_text: str) -> str:
    return JUDGE_PROMPT.replace("{transcript}", transcript_text.strip())


def parse_goal_verdict(text: str) -> GoalVerdict:
    """Parse a judge's raw output into a GoalVerdict, tolerantly.

    Missing fields are inferred:
    - complete absent → True when ``missing`` is empty and score (if any) >=
      success threshold; else False
    - missing absent → []
    - confidence absent → 0.5
    """
    raw = text or ""
    blob = extract_json_blob(raw)
    data: dict[str, Any] = {}
    if blob:
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                data = parsed
        except (ValueError, TypeError):
            data = {}

    missing_raw = data.get("missing") or []
    missing: list[str] = []
    if isinstance(missing_raw, str):
        missing = [missing_raw] if missing_raw.strip() else []
    elif isinstance(missing_raw, (list, tuple)):
        missing = [str(m) for m in missing_raw if str(m).strip()]

    score: Optional[float] = None
    if data.get("score") is not None:
        try:
            score = max(0.0, min(1.0, float(data["score"])))
        except (TypeError, ValueError):
            score = None

    complete_raw = data.get("complete")
    if isinstance(complete_raw, bool):
        complete = complete_raw
    elif isinstance(complete_raw, str):
        complete = complete_raw.strip().lower() in ("true", "yes", "1", "pass")
    else:
        complete = not missing and (score is None or score >= DEFAULT_SUCCESS_THRESHOLD)

    try:
        confidence = float(data.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return GoalVerdict(
        complete=complete,
        missing=missing,
        confidence=confidence,
        score=score,
        raw=raw,
    )


def build_followup_prompt(verdict: GoalVerdict, template: str = DEFAULT_FOLLOWUP_TEMPLATE) -> str:
    missing = "\n".join(f"- {item}" for item in verdict.missing) or "- (unspecified)"
    return template.replace("{missing}", missing)


# ---------------------------------------------------------------------------
# Judge controller (continue-vs-stop)
# ---------------------------------------------------------------------------


def should_judge_run(config: Optional[dict[str, Any]]) -> bool:
    """Whether the judge-driven goal loop is enabled for this run (opt-in)."""
    cfg = (config or {}).get(JUDGE_CONFIG_KEY) or {}
    return bool(cfg.get("enabled", False))


def resolve_judge_config(config: Optional[dict[str, Any]]) -> CriticConfig:
    cfg = (config or {}).get(JUDGE_CONFIG_KEY) or {}
    try:
        threshold = float(cfg.get("success_threshold", DEFAULT_SUCCESS_THRESHOLD))
    except (TypeError, ValueError):
        threshold = DEFAULT_SUCCESS_THRESHOLD
    try:
        max_iterations = max(0, int(cfg.get("max_iterations", DEFAULT_MAX_JUDGE_ITERATIONS)))
    except (TypeError, ValueError):
        max_iterations = DEFAULT_MAX_JUDGE_ITERATIONS
    template = str(cfg.get("followup_template") or DEFAULT_FOLLOWUP_TEMPLATE)
    return CriticConfig(
        success_threshold=max(0.0, min(1.0, threshold)),
        max_iterations=max_iterations,
        followup_template=template,
    )


def resolve_judge_model(config: Optional[dict[str, Any]]) -> Optional[str]:
    cfg = (config or {}).get(JUDGE_CONFIG_KEY) or {}
    if cfg.get("model"):
        return str(cfg["model"])
    return utility_model(config=config) or DEFAULT_JUDGE_MODEL


class JudgeController:
    """Post-run judge: renders the transcript, judges, and drives resume/stop.

    ``judge_runner`` is ``(transcript_text, model) -> raw_text`` (a stub in
    tests; a cheap-model subagent call in production once wired).
    ``worker`` is ``(followup_prompt) -> None``: it resumes the run with the
    "what's missing" prompt; ``get_messages`` re-reads the transcript after
    each resume.
    """

    def __init__(
        self,
        judge_runner: Callable[[str, Optional[str]], str],
        *,
        config: Optional[dict[str, Any]] = None,
        critic: Optional[CriticConfig] = None,
        history: Any = None,
    ) -> None:
        self.runner = judge_runner
        self.config = config or {}
        self.critic = critic or resolve_judge_config(self.config)
        self.history = history
        self.model = resolve_judge_model(self.config)

    def _judge_once(self, messages: Any, iteration: int) -> GoalVerdict:
        transcript = render_transcript(messages)
        raw = self.runner(transcript, self.model)
        verdict = parse_goal_verdict(raw)
        # score-based completion honoring the critic's success threshold
        if verdict.score is not None and not verdict.missing:
            verdict.complete = verdict.score >= self.critic.success_threshold
        log_verdict_event(
            self.history,
            make_verdict_event(
                _JUDGE_EVENT_KIND,
                verdict.complete,
                self.model,
                verdict.confidence,
                findings_count=len(verdict.missing),
                iteration=iteration,
                extra={"score": verdict.score, "missing": verdict.missing},
                config=self.config,
            ),
        )
        return verdict

    def run(
        self,
        get_messages: Callable[[], Any],
        worker: Callable[[str], None],
    ) -> GoalVerdict:
        """Judge the run; resume on missing items until complete or cap."""
        last: Optional[GoalVerdict] = None
        for iteration in range(self.critic.max_iterations + 1):
            verdict = self._judge_once(get_messages(), iteration)
            last = verdict
            if verdict.complete:
                return verdict
            if iteration >= self.critic.max_iterations:
                return verdict
            worker(build_followup_prompt(verdict, self.critic.followup_template))
        return last or GoalVerdict(complete=False, missing=["no verdict produced"])
