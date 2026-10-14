"""Skills Q&A regression grading (brief 004).

The Oct 11 scenario: ask "what skills do you have?" and check whether the
model answers from the injected ``## Available Skills`` inventory instead of
inventing generic skills from base-model priors.

Grading rule (heuristic, documented as such):
- PASS: every required fixture skill name appears in the answer
  (case-insensitive, with or without the leading "/").
- FAIL: any required name is missing, or any blocklisted generic invention
  appears. The blocklist holds the actual inventions observed on Oct 11
  (nemotron-3.5-lightning-30b, 3x: "technical writing", "thermodynamics").

Known failure mode (documented, not a bug in this grader): small models may
ignore the inventory even when it is provably injected. On 2026-10-13 the
dieted prompt (brief 001) made the same 30B model answer correctly, so this
task doubles as a regression test for prompt changes: re-fatten the prompt
and the 30B may start inventing again.
"""

from __future__ import annotations

# Generic "skills" the 30B model invented on Oct 11 instead of reading the
# injected inventory. Kept as the documented failure signature.
BLOCKLISTED_GENERICS: tuple[str, ...] = ("thermodynamics", "technical writing")


def grade_skills_answer(
    response_text: str | None,
    required_skills: list[str] | tuple[str, ...],
    blocklist: list[str] | tuple[str, ...] = BLOCKLISTED_GENERICS,
) -> tuple[bool, str]:
    """Grade a "what skills do you have?" answer.

    Returns (passed, reason). Never raises: ungradeable input fails closed.
    """
    try:
        text = (response_text or "").lower()
        required = [str(s) for s in (required_skills or [])]
        missing = [s for s in required if s.lower() not in text]
        if missing:
            return False, (
                "answer does not come from the injected inventory; "
                f"missing required skill(s): {', '.join(missing)}"
            )
        invented = [b for b in (blocklist or []) if str(b).lower() in text]
        if invented:
            return False, (
                "answer invents generic skills from priors instead of the "
                f"inventory: {', '.join(invented)}"
            )
        return True, (
            "answer names the injected inventory skills: "
            f"{', '.join(required) if required else '(none required)'}"
        )
    except Exception as exc:  # fail closed, never break the harness
        return False, f"grader error: {exc}"
