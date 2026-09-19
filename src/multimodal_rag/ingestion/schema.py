"""Common internal representation every format-specific parser targets.

Each source document becomes a flat list of Elements. Downstream code
(chunking, embeddings, retrieval) depends only on this schema — never on
the parsing library that produced it. Same ports/adapters idea as the
providers layer, applied to ingestion.
"""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class ElementType(StrEnum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    IMAGE = "image"
    CHART = "chart"


class ElementMetadata(BaseModel):
    source_file: str
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    """Excel sheet name (or None for a single-sheet source like CSV) --
    same role as page/slide, mirrored for the same reason: a citation
    from a spreadsheet should say which sheet it came from, not just
    which document."""
    position: int


class Element(BaseModel):
    type: ElementType
    text: str | None = None
    """Title/paragraph text, or a table rendered as markdown."""

    image_bytes: bytes | None = None
    """Raw bytes for image/chart elements."""

    description: str | None = None
    """Vision-generated caption for image/chart elements (what gets embedded)."""

    description_status: Literal["generated", "placeholder"] = "generated"
    """Whether `description` came from a real VisionProvider or a fallback
    placeholder because none was available at ingestion time."""

    table_summary: str | None = None
    """Optional short LLM summary, tables only."""

    metadata: ElementMetadata
