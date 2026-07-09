"""
Post-processing heuristics for solver numeric answers.
"""

from __future__ import annotations

_PERCENTAGE_KEYWORDS = (
    "percent",
    "percentage",
    "ratio",
    "relation",
    "proportion",
)

_RATIO_CONTEXT_KEYWORDS = (
    "in relation to",
    "relative to",
    "as a percent",
    "as a percentage",
    "percentage change",
    "percent change",
)


def is_percentage_question(question: str) -> bool:
    """Return True when the question likely expects a decimal ratio/percentage."""
    q = question.lower()
    if any(kw in q for kw in _PERCENTAGE_KEYWORDS):
        return True
    if "change" in q and any(ctx in q for ctx in ("relation", "relative", "percent")):
        return True
    return any(ctx in q for ctx in _RATIO_CONTEXT_KEYWORDS)


def maybe_fix_percentage_scale(
    answer: float, question: str, is_percentage: bool
) -> float:
    """
    Correct common 100x percentage scale errors.

    When the model returns -21.05 instead of -0.2105, divide by 100.
    """
    if not (is_percentage or is_percentage_question(question)):
        return answer

    if abs(answer) <= 1.0:
        return answer

    # Only scale when the answer looks like a whole-number percentage.
    if abs(answer) > 1.0:
        return answer / 100.0
    return answer


def finalize_answer(
    answer: float, question: str, is_percentage: bool
) -> float:
    """Apply all post-processing heuristics to a numeric answer."""
    return maybe_fix_percentage_scale(answer, question, is_percentage)
