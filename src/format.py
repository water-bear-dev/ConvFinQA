"""
Format financial documents for LLM prompts or CLI display.

Converts the nested table dict into a markdown table so row/column
alignment is preserved for both humans and models.
"""

from __future__ import annotations

from .models import Document


def format_document(doc: Document) -> str:
    """
    Format pre-text, markdown table, and post-text into one string.

    Column headers are sorted alphabetically for deterministic output.
    """
    formatted_str = f"Document Pre-text:\n{doc.pre_text}\n\n"

    if doc.table:
        all_headers: set[str] = set()
        for row_data in doc.table.values():
            all_headers.update(row_data.keys())
        sorted_headers = sorted(all_headers)

        header_row = "| Row Label | " + " | ".join(sorted_headers) + " |"
        separator_row = "|-----------|" + "---|" * len(sorted_headers)

        formatted_str += "Table:\n"
        formatted_str += header_row + "\n"
        formatted_str += separator_row + "\n"

        for row_label, row_data in doc.table.items():
            values = [str(row_data.get(header, "")) for header in sorted_headers]
            formatted_str += f"| {row_label} | " + " | ".join(values) + " |\n"
    else:
        formatted_str += "Table: No table data available.\n"

    formatted_str += f"\nDocument Post-text:\n{doc.post_text}\n"
    return formatted_str

