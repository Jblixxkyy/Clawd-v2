"""030 — Relevance-shaped skill loading (small-model lifter).

A weak model with 50 skills dumped into context drowns; the same model with
2 skills auto-activated by triggers/markers behaves far smarter.

This module implements the *automatic, relevance-shaped* loading layer:

- Skill frontmatter extensions: ``triggers:`` (keyword list), ``markers:``
  (repo markers like ``uv.lock``, ``package.json``, ``.ps1``), optional
  ``context_cmd:`` (shell command whose stdout is injected on activation).
- Activation: on session start and relevant events (file mentions, tool
  results), skills are scored by trigger/marker match and auto-loaded
  progressively — SKILL.md body only for activated skills, one-line index
  for the rest.
- Progressive disclosure tiers: name+description index for everything,
  full body on activation, full body on first use (``promote``).
- Per-model-family prompt variants: small overrides file
  ``prompt-variants/<family>.md`` in any skills dir, keyed by model family.
- Catalog pruning: ``prune_catalog`` drops tools the active model family
  demonstrably doesn't need. Config-driven, conservative default: NO
  pruning.

This is distinct from brief 004 (skills Q&A eval) and brief 009 (explicit
@-mentions): it never touches those paths.

The clean entry point for external wiring (session start, repl, web app)::

    from src.skills.activation import ActivationContext, activate_skills

    ctx = ActivationContext(project_root=cwd, text=user_message,
                            model_family=detect_model_family(model_id))
    activation = activate_skills(ctx)
    prompt_block = activation.render()

External wiring into the system prompt / agent loop is intentionally NOT
done here (other builders own those files); it is reported as pending.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .loader import _candidate_user_skills_dirs, find_all_skills
from .model import PromptSkill

# Event types that can trigger a (re-)activation pass.
EVENT_SESSION_START = "session_start"
EVENT_USER_MESSAGE = "user_message"
EVENT_FILE_MENTION = "file_mention"
EVENT_TOOL_RESULT = "tool_result"

EVENTS = (
    EVENT_SESSION_START,
    EVENT_USER_MESSAGE,
    EVENT_FILE_MENTION,
    EVENT_TOOL_RESULT,
)

# ---------------------------------------------------------------------------
# Model families
# ---------------------------------------------------------------------------

# family -> substring hints matched (case-insensitive) against the model id.
_FAMILY_HINTS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("claude", ("claude", "anthropic")),
    ("gpt", ("gpt-", "gpt4", "gpt-4", "gpt-5", "o1", "o3", "openai", "codex")),
    ("gemini", ("gemini",)),
    ("grok", ("grok", "xai")),
    ("llama", ("llama",)),
    ("qwen", ("qwen",)),
    ("mistral", ("mistral", "mixtral")),
    ("deepseek", ("deepseek",)),
    ("glm", ("glm", "zhipu")),
    ("nemotron", ("nemotron",)),
    ("minimax", ("minimax",)),
)


def detect_model_family(model_id: Optional[str]) -> str:
    """Map a model id to a coarse family key for prompt-variant lookup.

    Unknown / missing ids map to ``"default"``. Pure string matching —
    same semantics for every family, only wording may vary per family.
    """
    if not model_id:
        return "default"
    low = str(model_id).lower()
    for family, hints in _FAMILY_HINTS:
        if any(h in low for h in hints):
            return family
    return "default"


# ---------------------------------------------------------------------------
# Activation context
# ---------------------------------------------------------------------------


@dataclass
class ActivationContext:
    """Everything a scoring pass needs. Built by the caller per event."""

    project_root: Optional[Path] = None
    user_skills_dir: Optional[Path] = None
    text: str = ""  # user message / file mention / tool result text
    event: str = EVENT_SESSION_START
    model_family: Optional[str] = None
    # Budget for the whole rendered skill block (index + activated bodies).
    token_budget: int = 8000
    # Scoring weights.
    marker_score: int = 10
    trigger_score: int = 2
    # Master switch; when False nothing auto-activates (manual/promote only).
    auto_activate: bool = True
    # Seconds allowed for a skill's context_cmd.
    context_cmd_timeout: float = 5.0


def default_context(
    *,
    project_root: Optional[Path] = None,
    user_skills_dir: Optional[Path] = None,
    text: str = "",
    event: str = EVENT_SESSION_START,
    model_id: Optional[str] = None,
) -> ActivationContext:
    """Build an ActivationContext with values from the config file.

    Reads ``src.config.skill_loading_config()`` lazily so this module stays
    import-light; never raises — falls back to hardcoded defaults.
    """
    try:
        from ..config import skill_loading_config

        cfg = skill_loading_config()
    except Exception:
        cfg = {}
    return ActivationContext(
        project_root=project_root,
        user_skills_dir=user_skills_dir,
        text=text,
        event=event,
        model_family=detect_model_family(model_id),
        token_budget=int(cfg.get("token_budget", 8000) or 8000),
        auto_activate=bool(cfg.get("auto_activate", True)),
        context_cmd_timeout=float(cfg.get("context_cmd_timeout", 5.0) or 5.0),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_GLOB_MAGIC = re.compile(r"[*?\[\]]")


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_glob(root: Path, pattern: str) -> List[Path]:
    try:
        return list(root.glob(pattern))
    except OSError:
        return []


def marker_present(marker: str, project_root: Path) -> bool:
    """True when a frontmatter marker matches the project tree.

    A marker may be:
    - an exact relative path: ``uv.lock``, ``src/main.py``
    - a glob: ``*.toml``, ``packages/*/package.json``
    - a bare extension: ``.ps1`` (matches any file with that extension)
    - a bare filename searched recursively: ``uv.lock`` also matches
      ``sub/dir/uv.lock`` via the recursive fallback.
    """
    marker = (marker or "").strip()
    if not marker:
        return False
    root = Path(project_root)
    if _safe_exists(root / marker):
        return True
    if _GLOB_MAGIC.search(marker):
        return bool(_safe_glob(root, marker))
    if marker.startswith(".") and "/" not in marker and "\\" not in marker:
        # Bare extension: any file ending with it, at any depth.
        return bool(_safe_glob(root, f"**/*{marker}"))
    # Bare filename: recursive search.
    return bool(_safe_glob(root, f"**/{marker}"))


def _keyword_in_text(keyword: str, text: str) -> bool:
    """Case-insensitive keyword hit.

    Word-boundary match for plain alphanumeric keywords (so a ``go``
    trigger doesn't fire on "going"); plain substring for keywords
    containing non-word characters.
    """
    kw = (keyword or "").strip()
    if not kw:
        return False
    if re.fullmatch(r"[\w-]+", kw):
        return re.search(r"\b" + re.escape(kw) + r"\b", text, re.IGNORECASE) is not None
    return kw.lower() in text.lower()


def score_skill(
    skill: PromptSkill, context: ActivationContext
) -> Tuple[int, List[str]]:
    """Score one skill against the activation context.

    Returns ``(score, reasons)`` where reasons are human-readable match
    notes like ``"marker:uv.lock"`` / ``"trigger:pdf"``. Score > 0 means
    the skill should auto-activate (when auto_activate is on).
    """
    score = 0
    reasons: List[str] = []
    if context.project_root is not None:
        for marker in skill.markers or ():
            if marker_present(marker, context.project_root):
                score += context.marker_score
                reasons.append(f"marker:{marker}")
    if context.text:
        for keyword in skill.triggers or ():
            if _keyword_in_text(keyword, context.text):
                score += context.trigger_score
                reasons.append(f"trigger:{keyword}")
    return score, reasons


# ---------------------------------------------------------------------------
# context_cmd
# ---------------------------------------------------------------------------


def run_context_cmd(
    skill: PromptSkill,
    context: ActivationContext,
    *,
    max_chars: int = 2000,
) -> Optional[str]:
    """Run a skill's ``context_cmd`` and return its stdout.

    Runs once per activation (result cached on the Activation). Never
    raises: timeouts and failures yield ``None`` (no injected noise).
    Same trust level as the skill body itself — it comes from an installed
    skill the user already trusts.
    """
    cmd = (skill.context_cmd or "").strip()
    if not cmd:
        return None
    cwd = context.project_root
    if cwd is None and skill.skill_root:
        cwd = Path(skill.skill_root)
    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=context.context_cmd_timeout,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    out = (proc.stdout or "").strip()
    if not out:
        return None
    return out[:max_chars]


# ---------------------------------------------------------------------------
# Per-model-family prompt variants
# ---------------------------------------------------------------------------


def _skills_dirs_for(context: ActivationContext) -> List[Path]:
    """All skill directories a variant file could live in."""
    dirs: List[Path] = []
    if context.user_skills_dir is not None:
        dirs.append(Path(context.user_skills_dir).expanduser().resolve())
    else:
        dirs.extend(_candidate_user_skills_dirs())
    if context.project_root is not None:
        pr = Path(context.project_root).expanduser().resolve()
        for sub in ("skills",):
            for base in (".clawd", ".claude"):
                d = pr / base / sub
                if d not in dirs:
                    dirs.append(d)
    return dirs


def load_prompt_variant(
    model_family: Optional[str],
    skills_dirs: Sequence[Path],
) -> Optional[str]:
    """Load a per-model-family wording override for the skills block.

    Looks for ``prompt-variants/<family>.md`` in each skills dir (first hit
    wins). The variant only rewords the surrounding prompt copy — same
    semantics, tuned per family. Returns ``None`` when absent (plain
    rendering).
    """
    if not model_family:
        return None
    filename = f"{model_family}.md"
    for d in skills_dirs:
        candidate = Path(d) / "prompt-variants" / filename
        try:
            if candidate.is_file():
                text = candidate.read_text(encoding="utf-8").strip()
                return text or None
        except OSError:
            continue
    return None


# ---------------------------------------------------------------------------
# Catalog pruning (conservative: OFF unless configured)
# ---------------------------------------------------------------------------


def prune_catalog(
    tool_names: Sequence[str],
    model_family: Optional[str],
    prune_map: Optional[Dict[str, Sequence[str]]] = None,
) -> List[str]:
    """Drop tools the active model family demonstrably doesn't need.

    ``prune_map`` is ``{family: [tool, ...]}`` from config key
    ``skills.prune_tools``. Default (no map / family not listed): NO
    pruning — the advertised catalog passes through untouched. Pruning is
    opt-in per family, never silent.
    """
    tools = list(tool_names)
    if not prune_map or not model_family:
        return tools
    drop = set(prune_map.get(model_family, []) or [])
    if not drop:
        return tools
    return [t for t in tools if t not in drop]


# ---------------------------------------------------------------------------
# Activation result + progressive disclosure rendering
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    try:
        from ..token_estimation import count_tokens

        return count_tokens(text)
    except Exception:
        return max(1, len(text) // 4)


class Activation:
    """Result of one :func:`activate_skills` pass.

    Progressive disclosure tiers:
    - Tier 0: model-family variant preamble (optional).
    - Tier 1: one-line index (``/name — description``) for EVERY skill.
    - Tier 2: full SKILL.md body (+ context_cmd output) for activated
      skills, and for skills promoted via :meth:`promote` on first use.

    Bodies are dropped back to index-only (lowest score first) when the
    rendered block would exceed the token budget.
    """

    def __init__(
        self,
        context: ActivationContext,
        *,
        skills: Sequence[PromptSkill],
        scores: Dict[str, Tuple[int, List[str]]],
        variant_preamble: Optional[str] = None,
    ) -> None:
        self.context = context
        self.scores = scores
        self.variant_preamble = variant_preamble
        by_score = sorted(
            skills, key=lambda s: (-scores.get(s.name, (0, []))[0], s.name)
        )
        self.active: List[PromptSkill] = [
            s for s in by_score if scores.get(s.name, (0, []))[0] > 0
        ]
        active_names = {s.name for s in self.active}
        self.inactive: List[PromptSkill] = [
            s for s in by_score if s.name not in active_names
        ]
        # context_cmd outputs, computed once per activation.
        self.context_outputs: Dict[str, str] = {}
        for skill in self.active:
            out = run_context_cmd(skill, context)
            if out:
                self.context_outputs[skill.name] = out
        # Skills promoted to the body tier by first use (progressive disclosure).
        self._promoted: Dict[str, PromptSkill] = {}

    # -- tier 2 on first use -------------------------------------------------
    def promote(self, name: str) -> bool:
        """Load a skill's full body on first use (tier 1 -> tier 2).

        Returns True when the skill is now in the body tier.
        """
        for skill in self.inactive:
            if skill.name == name:
                self._promoted[name] = skill
                return True
        return name in self._promoted or any(s.name == name for s in self.active)

    @property
    def body_tier(self) -> List[PromptSkill]:
        """Skills whose full body is included in the rendered block."""
        seen = {s.name for s in self.active}
        ordered = list(self.active)
        for name, skill in self._promoted.items():
            if name not in seen:
                ordered.append(skill)
                seen.add(name)
        return ordered

    # -- rendering -----------------------------------------------------------
    def _index_line(self, skill: PromptSkill) -> str:
        score, reasons = self.scores.get(skill.name, (0, []))
        tag = f" [auto-activated: {', '.join(reasons)}]" if score > 0 else ""
        desc = (skill.description or "").strip().splitlines()[0][:160]
        return f"- /{skill.name} — {desc}{tag}"

    def _body_block(self, skill: PromptSkill) -> str:
        parts = [f"### {skill.name}", skill.markdown_content.strip()]
        out = self.context_outputs.get(skill.name)
        if out:
            parts.append(f"<context-cmd name=\"{skill.name}\">\n{out}\n</context-cmd>")
        return "\n".join(p for p in parts if p)

    def render(self) -> str:
        """Render the full skill block for the system prompt."""
        lines: List[str] = []
        if self.variant_preamble:
            lines.append(self.variant_preamble.strip())
            lines.append("")
        total = len(self.active) + len(self.inactive)
        lines.append(
            f"Available skills: {total} installed, "
            f"{len(self.active)} auto-activated for this context."
        )
        bodies = self.body_tier
        # Budget enforcement: drop lowest-scoring bodies first.
        index_only: List[PromptSkill] = []
        while True:
            candidate = [s for s in bodies if s not in index_only]
            block = self._render_with(candidate)
            if _estimate_tokens(block) <= self.context.token_budget:
                return block
            if not candidate:
                return block
            # Demote the lowest-scoring body.
            lowest = min(
                candidate, key=lambda s: self.scores.get(s.name, (0, []))[0]
            )
            index_only.append(lowest)

    def _render_with(self, bodies: Sequence[PromptSkill]) -> str:
        lines: List[str] = []
        if self.variant_preamble:
            lines.append(self.variant_preamble.strip())
            lines.append("")
        total = len(self.active) + len(self.inactive)
        lines.append(
            f"Available skills: {total} installed, "
            f"{len(self.active)} auto-activated for this context."
        )
        body_names = {s.name for s in bodies}
        if bodies:
            lines.append("")
            lines.append("## Activated skill details")
            for skill in bodies:
                lines.append("")
                lines.append(self._body_block(skill))
        lines.append("")
        lines.append("## Skill index (mention /name or use the skill to load its full guide)")
        for skill in list(self.active) + list(self.inactive):
            if skill.name in body_names:
                continue
            lines.append(self._index_line(skill))
        return "\n".join(lines).strip() + "\n"

    def token_estimate(self) -> int:
        """Estimated tokens of :meth:`render`."""
        return _estimate_tokens(self.render())

    # -- re-activation on later events ---------------------------------------
    def refresh(self, event: str, text: str) -> List[str]:
        """Re-score all skills against a new event; activate new matches.

        Returns the names of newly activated skills (context_cmd outputs
        are computed for them). Existing activations are kept.
        """
        ctx = ActivationContext(
            project_root=self.context.project_root,
            user_skills_dir=self.context.user_skills_dir,
            text=text,
            event=event,
            model_family=self.context.model_family,
            token_budget=self.context.token_budget,
            marker_score=self.context.marker_score,
            trigger_score=self.context.trigger_score,
            auto_activate=self.context.auto_activate,
            context_cmd_timeout=self.context.context_cmd_timeout,
        )
        newly: List[str] = []
        known = {s.name for s in self.active}
        for skill in self.inactive:
            if skill.name in known:
                continue
            score, reasons = score_skill(skill, ctx)
            if score > 0 and ctx.auto_activate:
                self.scores[skill.name] = (score, reasons)
                self.active.append(skill)
                out = run_context_cmd(skill, ctx)
                if out:
                    self.context_outputs[skill.name] = out
                newly.append(skill.name)
        self.active.sort(key=lambda s: (-self.scores.get(s.name, (0, []))[0], s.name))
        self.inactive = [s for s in self.inactive if s.name not in set(newly)]
        # Promoted skills that just activated are no longer "promoted extras".
        for name in newly:
            self._promoted.pop(name, None)
        return newly


def activate_skills(context: ActivationContext) -> Activation:
    """Score every discoverable skill and build its progressive-disclosure block.

    This is the clean entry point for external wiring (session start in the
    repl / agent loop / web app). It loads skills via the standard loader,
    scores them against ``context`` (markers + trigger keywords), runs
    ``context_cmd`` for activated skills, and applies any per-model-family
    prompt variant.
    """
    skills = find_all_skills(
        project_root=context.project_root,
        user_skills_dir=context.user_skills_dir,
    )
    scores: Dict[str, Tuple[int, List[str]]] = {}
    for skill in skills:
        scores[skill.name] = score_skill(skill, context)
    if not context.auto_activate:
        scores = {name: (0, reasons) for name, (_, reasons) in scores.items()}
    variant = load_prompt_variant(
        context.model_family, _skills_dirs_for(context)
    )
    return Activation(
        context, skills=skills, scores=scores, variant_preamble=variant
    )
