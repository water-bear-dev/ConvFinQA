"""
Structured output schema for the LLM solver.

Gemini is configured to return JSON that conforms to this schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class SolverResponse(BaseModel):
    reasoning: str = Field(description="Brief explanation grounded in the document.")
    expression: str | None = Field(
        default=None,
        description='Optional arithmetic expression, e.g. "206588 - 181001".',
    )
    answer: float | None = Field(
        default=None,
        description="Fallback numeric answer if no expression is provided.",
    )
    is_percentage: bool = Field(
        default=False,
        description="True if the answer is a percentage (returned as a decimal).",
    )

