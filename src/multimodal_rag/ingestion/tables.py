"""Shared table handling used by every format parser that has a native
table structure (PDF via unstructured, DOCX, PPTX).
"""

from ..providers.factory import get_llm


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render a rectangular grid of cell strings as a markdown table."""
    if not rows:
        return ""
    header, *body = rows
    lines = [
        _row_line(header),
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend(_row_line(row) for row in body)
    return "\n".join(lines)


def _row_line(row: list[str]) -> str:
    return "| " + " | ".join(cell.replace("\n", " ").strip() for cell in row) + " |"


def summarize_table(markdown_table: str) -> str | None:
    """Optional one-sentence LLM summary of a table. Returns None if no LLM
    provider is configured or the call fails — a summary is a nice-to-have,
    never a reason to fail ingestion.
    """
    try:
        llm = get_llm()
    except Exception:
        return None
    try:
        return llm.generate(
            [
                {
                    "role": "user",
                    "content": (
                        "Summarize this table in one short sentence, for someone "
                        "deciding whether to read it in full:\n\n" + markdown_table
                    ),
                }
            ]
        )
    except Exception:
        return None
