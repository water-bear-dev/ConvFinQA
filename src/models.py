"""
Pydantic models matching the cleaned ConvFinQA dataset schema (see dataset.md).

Gold labels for evaluation live in dialogue.executed_answers (floats/strings).
Human-readable answers (may include % formatting) are in dialogue.conv_answers.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Financial document: narrative text surrounding a data table."""

    pre_text: str = Field(description="Text before the table")
    post_text: str = Field(description="Text after the table")
    table: dict[str, dict[str, float | str | int]] = Field(
        description="Table as row_label -> column_label -> value"
    )


class Dialogue(BaseModel):
    """Multi-turn conversation with gold programs and executed answers."""

    conv_questions: list[str] = Field(description="User questions per turn")
    conv_answers: list[str] = Field(
        description="Human-readable gold answers (may include % formatting)"
    )
    turn_program: list[str] = Field(
        description="FinQA DSL program per turn (useful for analysis, not runtime)"
    )
    executed_answers: list[float | str] = Field(
        description="Numeric gold values used for execution-accuracy evaluation"
    )
    qa_split: list[bool] = Field(
        description="0 = Type I turn, 1 = Type II (second question thread)"
    )


class Features(BaseModel):
    """Metadata added during cleaning to describe record characteristics."""

    num_dialogue_turns: int = Field(description="Number of turns in the dialogue")
    has_type2_question: bool = Field(description="True if hybrid (Type II) conversation")
    has_duplicate_columns: bool = Field(
        description="Table has duplicate column headers (may be suffixed)"
    )
    has_non_numeric_values: bool = Field(
        description="Table contains non-numeric cell values"
    )


class ConvFinQARecord(BaseModel):
    """One financial document with a scripted multi-turn dialogue."""

    id: str = Field(description="Unique record identifier")
    doc: Document = Field(description="The financial document")
    dialogue: Dialogue = Field(description="The conversational dialogue")
    features: Features = Field(description="Record metadata")

