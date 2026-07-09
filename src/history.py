"""
Conversation history types for multi-turn solving.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnState:
    """One prior dialogue turn with optional active metric for Type II tracking."""

    question: str
    answer: float
    metric: str | None = None


def normalize_history(
    history: list[TurnState] | list[tuple[str, float]],
) -> list[TurnState]:
    """Convert tuple history to TurnState list."""
    if not history:
        return []
    if isinstance(history[0], TurnState):
        return history  # type: ignore[return-value]
    return [TurnState(question=q, answer=a) for q, a in history]  # type: ignore[misc]
