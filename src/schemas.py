"""
Structured output schema for the LLM solver.

Gemini is configured to return JSON that conforms to this schema.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CellRetrieval(BaseModel):
    row_label: str = Field(description="Exact table row label used.")
    column_header: str = Field(description="Exact table column header used.")
    value: float = Field(description="Numeric cell value.")
    metric: str = Field(description="Human-readable description of the metric.")


class RetrievalResponse(BaseModel):
    cells: list[CellRetrieval] = Field(default_factory=list)
    active_metric: str | None = Field(
        default=None, description="Current metric thread for Type II dialogues."
    )
    reasoning: str | None = Field(default=None)


class SolverResponse(BaseModel):
    reasoning: str | None = Field(
        default=None, description="Brief explanation grounded in the document."
    )
    expression: str | None = Field(
        default=None,
        description='Optional arithmetic expression, e.g. "206588 - 181001".',
    )
    answer: float | str | None = Field(
        default=None,
        description="Fallback numeric answer if no expression is provided.",
    )
    is_percentage: bool = Field(
        default=False,
        description="True if the answer is a percentage (returned as a decimal).",
    )
    source: str | None = Field(
        default=None,
        description='Optional cell citation, e.g. "row: 2007, column: revenue".',
    )
