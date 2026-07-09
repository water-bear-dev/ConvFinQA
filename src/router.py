"""
Routing logic for single-call vs two-step retrieve-then-compute solver.
"""

from __future__ import annotations

import os
import re
from typing import Literal

Route = Literal["two_step", "single"]

_MATH_KEYWORDS = (
    "difference",
    "change",
    "increase",
    "decrease",
    "subtract",
    "add",
    "sum",
    "total",
    "percent",
    "percentage",
    "ratio",
    "relation",
    "proportion",
    "times",
    "multiply",
    "divided",
)

_TOPIC_SWITCH_PATTERNS = (
    r"credit facilit",
    r"senior note",
    r"term loan",
    r"operating lease",
    r"capital lease",
    r"what about",
    r"associated with",
    r"related to",
    r"for the",
)

_REASK_PATTERNS = (
    r"what was it\b",
    r"and in \d{4}",
    r"in \d{4}\?",
    r"that value",
    r"previous year",
    r"the same",
    r"what was the .+ in \d{4}",
)


def is_reask_question(question: str) -> bool:
    """Return True when the question re-requests a prior lookup."""
    q = question.lower()
    return any(re.search(pat, q) for pat in _REASK_PATTERNS)


def _looks_like_topic_switch(question: str) -> bool:
    q = question.lower()
    return any(re.search(pat, q) for pat in _TOPIC_SWITCH_PATTERNS)


def _is_pure_lookup(question: str) -> bool:
    q = question.lower()
    return not any(kw in q for kw in _MATH_KEYWORDS)


def route_turn(
    turn_idx: int,
    question: str,
    has_type2: bool,
) -> Route:
    """
    Choose solver route: two-step retrieve+compute or single-call.

    Env override: SOLVER_ROUTER=auto|single|two_step (default auto).
    """
    override = os.getenv("SOLVER_ROUTER", "auto").strip().lower()
    if override == "single":
        return "single"
    if override == "two_step":
        return "two_step"

    if turn_idx == 0:
        return "two_step"

    if is_reask_question(question):
        return "two_step"

    if has_type2 and _looks_like_topic_switch(question):
        return "two_step"

    if _is_pure_lookup(question):
        return "two_step"

    return "single"
