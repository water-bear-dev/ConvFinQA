"""
Prompts for the ConvFinQA solver.

Keep prompts concise and rule-based; rely on structured output via schema.
"""

from __future__ import annotations

from .history import TurnState

SYSTEM_PROMPT = """You answer multi-turn questions about a financial document containing text and a markdown table.

Rules:
1) Use ONLY the provided document and prior turn answers. Do not invent facts.
2) Resolve coreference using the PRIOR TURNS section (e.g. "that value", "in 2008", "the difference").
3) If arithmetic is needed, put the computation in `expression` using numbers from the document or prior answers. Do not compute in prose.
4) Percentages MUST be returned as decimals (e.g. 14.136% -> 0.14136). If the question asks for a percent, set is_percentage=true.
   Example: change from 19 to 15 -> expression: (15-19)/19, answer: -0.21053 (NOT -21.05).
5) Output MUST be a single JSON object with keys: reasoning, expression, answer, is_percentage, source.
6) `answer` MUST be a raw number (no currency symbols, commas, units like "million", or extra words). If you cannot produce a number, set answer=null.
7) Always include `reasoning` (1-2 short sentences). For lookups, cite the exact row label and column header in reasoning and `source`.
8) For follow-up year/metric questions, re-read the table for the value. Use prior turns only to resolve WHAT is being asked, not to copy numbers.
9) For Type II topic switches (e.g. senior notes -> credit facilities), identify the new line item before computing.
"""

RETRIEVE_SYSTEM_PROMPT = """You extract relevant table cells from a financial document. Do NOT perform arithmetic.

Rules:
1) Use ONLY the provided document and prior turns to resolve what metric is being asked.
2) Return JSON with keys: cells (list), active_metric, reasoning.
3) Each cell must include row_label, column_header, value (number), and metric (description).
4) Quote exact row labels and column headers from the table. Do not guess similar columns.
5) For Type II switches, set active_metric to the new line item being discussed.
6) If multiple cells are needed, include all of them. No expressions or computed answers.
"""

COMPUTE_SYSTEM_PROMPT = """You compute the numeric answer for a financial QA turn using retrieved cells and prior turns.

Rules:
1) Use retrieved cell values and prior turn answers as operands. Do not invent numbers.
2) Put arithmetic in `expression`. Percentages as decimals (14.1% -> 0.141). Set is_percentage=true when appropriate.
   Example: percentage change (15-19)/19 -> answer: -0.21053, NOT -21.05.
3) Output JSON: reasoning, expression, answer, is_percentage, source.
4) `answer` must be a raw number only.
5) If prior answers may be wrong and retrieved cells disagree, prefer retrieved cell values.
"""


def _format_history(history: list[TurnState]) -> str:
    if not history:
        return "(none)"
    lines: list[str] = []
    for turn in history:
        metric_note = f" [metric: {turn.metric}]" if turn.metric else ""
        lines.append(f"Q: {turn.question}\nA: {turn.answer}{metric_note}")
    return "\n\n".join(lines)


def build_user_prompt(
    doc_text: str, history: list[TurnState], question: str
) -> str:
    """Build a prompt with document, prior turns, and current question."""
    return (
        "=== DOCUMENT ===\n"
        f"{doc_text}\n\n"
        "=== PRIOR TURNS ===\n"
        f"{_format_history(history)}\n\n"
        "=== CURRENT QUESTION ===\n"
        f"{question}\n"
    )


def build_retrieve_prompt(
    doc_text: str, history: list[TurnState], question: str
) -> str:
    """Build a retrieval-only prompt."""
    return (
        "=== DOCUMENT ===\n"
        f"{doc_text}\n\n"
        "=== PRIOR TURNS ===\n"
        f"{_format_history(history)}\n\n"
        "=== CURRENT QUESTION ===\n"
        f"{question}\n\n"
        "Extract the relevant table cell(s) needed to answer this question. "
        "Do not compute the final answer."
    )


def build_compute_prompt(
    retrieved_cells: list[dict],
    history: list[TurnState],
    question: str,
    active_metric: str | None = None,
    correction_note: str | None = None,
) -> str:
    """Build a compute prompt from retrieved cells."""
    cell_lines: list[str] = []
    for cell in retrieved_cells:
        cell_lines.append(
            f"- row={cell['row_label']}, column={cell['column_header']}, "
            f"value={cell['value']}, metric={cell['metric']}"
        )
    cells_text = "\n".join(cell_lines) if cell_lines else "(none)"

    parts = [
        "=== RETRIEVED CELLS ===",
        cells_text,
        "",
        "=== PRIOR TURNS ===",
        _format_history(history),
        "",
    ]
    if active_metric:
        parts.extend([f"=== ACTIVE METRIC ===\n{active_metric}", ""])
    if correction_note:
        parts.extend([f"=== CORRECTION ===\n{correction_note}", ""])
    parts.extend(["=== CURRENT QUESTION ===", question])
    return "\n".join(parts)
