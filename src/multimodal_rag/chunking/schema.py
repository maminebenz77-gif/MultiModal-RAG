"""Common chunk representation every chunking strategy targets."""

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    source_file: str
    element_positions: list[int] = []
    """Which Element.metadata.position values fed this chunk. Best-effort:
    strategies that flatten elements to raw text before splitting
    (fixed-size, recursive, semantic) can't reliably recover this, so it
    stays empty for them. Strategies that walk the Element list directly
    (structure-aware, parent-child) populate it exactly."""

    element_types: list[str] = []
    """The ElementType values (as strings) present in this chunk. A
    structure-aware section chunk can legitimately span several types
    (TITLE, PARAGRAPH, TABLE, ...) — a single "the type" isn't always
    well-defined, so this is a list, not one value. Same best-effort
    rule as element_positions."""

    pages: list[int] = []
    slides: list[int] = []
    """Page/slide numbers of the elements that fed this chunk — needed
    for citations (source file + page/slide), which is why this exists.
    A chunk can span multiple elements on different pages, so these are
    deduplicated, order-preserving lists, not single values. Same
    best-effort rule as element_positions/element_types: populated
    exactly by strategies that walk the Element list directly, empty for
    the flatten-first strategies. Empty for Markdown/DOCX sources, which
    have no native page/slide concept at all."""


class Chunk(BaseModel):
    id: str
    text: str
    parent_id: str | None = None
    """Set only on "child" chunks produced by the parent-child strategy."""

    is_parent: bool = False
    """True only for the "parent" (big) chunk in a parent-child pair.
    Distinct from parent_id being None, which is ALSO true for ordinary
    standalone chunks that have no parent-child structure at all — this
    field exists so stores can specifically exclude parent chunks from
    being returned as direct search hits (see VectorStore.search()/
    KeywordStore.search()) without also hiding unrelated standalone
    chunks. A parent is meant to be reached only by resolving up from
    one of its children, never matched directly."""

    metadata: ChunkMetadata
