"""Markdown parser: markdown-it-py token stream -> common Element schema.

Markdown is a plain-text structured format, so this is the simplest of the
four parsers: headings map directly to titles, GFM tables are already
markdown syntax so conversion is nearly a pass-through, and images are
`![alt](path)` links resolved relative to the source file.
"""

from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from .schema import Element, ElementMetadata, ElementType
from .tables import rows_to_markdown, summarize_table
from .vision import ImageDescriber

# GFM tables aren't in strict CommonMark, so enable the (already-built-in)
# table rule explicitly rather than pulling in a separate plugin package.
_MD = MarkdownIt("commonmark").enable("table")


def parse_markdown(path: Path, summarize_tables: bool = False) -> list[Element]:
    source = path.read_text(encoding="utf-8")
    tokens = _MD.parse(source)
    describer = ImageDescriber()

    elements: list[Element] = []
    position = 0
    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token.type == "heading_open":
            elements.append(
                Element(
                    type=ElementType.TITLE,
                    text=tokens[i + 1].content,
                    metadata=ElementMetadata(source_file=str(path), position=position),
                )
            )
            position += 1
            i += 3  # heading_open, inline, heading_close
            continue

        if token.type == "paragraph_open":
            inline = tokens[i + 1]
            elements.append(
                _paragraph_or_image_element(inline, path, position, describer)
            )
            position += 1
            i += 3  # paragraph_open, inline, paragraph_close
            continue

        if token.type == "table_open":
            rows, end = _extract_table_rows(tokens, i)
            markdown_table = rows_to_markdown(rows)
            summary = summarize_table(markdown_table) if summarize_tables else None
            elements.append(
                Element(
                    type=ElementType.TABLE,
                    text=markdown_table,
                    table_summary=summary,
                    metadata=ElementMetadata(source_file=str(path), position=position),
                )
            )
            position += 1
            i = end + 1  # skip past table_close
            continue

        i += 1

    return elements


def _paragraph_or_image_element(
    inline: Token, path: Path, position: int, describer: ImageDescriber
) -> Element:
    children = inline.children or []
    # A paragraph containing *only* an image (`![alt](src)`) is an image
    # element, not text — this is markdown's usual "figure" convention.
    if len(children) == 1 and children[0].type == "image":
        src = str(children[0].attrs.get("src", ""))
        image_path = (path.parent / src).resolve()
        image_bytes = image_path.read_bytes() if image_path.exists() else b""
        description, status = describer.describe(image_bytes)
        return Element(
            type=ElementType.IMAGE,
            image_bytes=image_bytes,
            description=description,
            description_status=status,
            metadata=ElementMetadata(source_file=str(path), position=position),
        )

    return Element(
        type=ElementType.PARAGRAPH,
        text=inline.content,
        metadata=ElementMetadata(source_file=str(path), position=position),
    )


def _extract_table_rows(tokens: list[Token], table_open_index: int) -> tuple[list[list[str]], int]:
    """Walk from a table_open token to its table_close, returning the cell
    grid and the index of table_close."""
    rows: list[list[str]] = []
    j = table_open_index + 1
    while tokens[j].type != "table_close":
        if tokens[j].type == "tr_open":
            row: list[str] = []
            k = j + 1
            while tokens[k].type != "tr_close":
                if tokens[k].type in ("th_open", "td_open"):
                    row.append(tokens[k + 1].content)
                k += 1
            rows.append(row)
            j = k
        j += 1
    return rows, j
