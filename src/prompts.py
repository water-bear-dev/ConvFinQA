"""
Prompts for the ConvFinQA solver.

Keep prompts concise and rule-based; rely on structured output via schema.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You answer multi-turn questions about a financial document containing text and a markdown table.

Rules:
1) Use ONLY the provided document and prior turn answers. Do not invent facts.
2) Resolve coreference using the PRIOR TURNS section (e.g. "that value", "in 2008", "the difference").
3) If arithmetic is needed, put the computation in `expression` using numbers from the document or prior answers. Do not compute in prose.
4) Percentages MUST be returned as decimals (e.g. 14.136% -> 0.14136). If the question asks for a percent, set is_percentage=true.
5) Return a JSON object matching the schema. If you cannot answer, set answer=null and explain why in reasoning.
"""


def build_user_prompt(
    doc_text: str, history: list[tuple[str, float]], question: str
) -> str:
    """Build a prompt with document, prior turns, and current question."""
    history_lines: list[str] = []
    for q, a in history:
        history_lines.append(f"Q: {q}\nA: {a}")
    history_text = "\n\n".join(history_lines) if history_lines else "(none)"

    return (
        "=== DOCUMENT ===\n"
        f"{doc_text}\n\n"
        "=== PRIOR TURNS ===\n"
        f"{history_text}\n\n"
        "=== CURRENT QUESTION ===\n"
        f"{question}\n"
    )

