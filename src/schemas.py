"""
Structured output schema for the LLM solver.

Gemini is configured to return JSON that conforms to this schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SolverResponse(BaseModel):
    # Local models occasionally omit fields; we keep this optional to avoid hard failures
    # and rely on downstream code to treat missing reasoning as low-confidence output.
    reasoning: str | None = Field(
        default=None, description="Brief explanation grounded in the document."
    )
    expression: str | None = Field(
        default=None,
        description='Optional arithmetic expression, e.g. "206588 - 181001".',
    )
    # Accept either a number or a string; local models often emit formatted answers
    # (e.g. \"$4 million\"). The solver normalizes these to floats when possible.
    answer: float | str | None = Field(
        default=None,
        description="Fallback numeric answer if no expression is provided.",
    )
    is_percentage: bool = Field(
        default=False,
        description="True if the answer is a percentage (returned as a decimal).",
    )

