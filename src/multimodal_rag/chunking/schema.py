"""Common chunk representation every chunking strategy targets."""

from pydantic import BaseModel


class ChunkElement(BaseModel):
    """A snapshot of one Element (ingestion/schema.py) that fed a chunk
    -- kept around so a chunk can later be displayed as an assembly of
    its real parts (an actual rendered table, an actual image) rather
    than only the flattened `Chunk.text` blob. Same best-effort rule as
    the rest of ChunkMetadata: populated by strategies that walk the
    Element list directly, empty for flatten-first ones."""

    type: str
    """An ElementType value ("title" | "paragraph" | "table" | "image" |
    "chart"), kept as a plain str (not the enum) so this schema has no
    import dependency on the ingestion layer -- same reasoning as
    ChunkMetadata.element_types already being list[str]."""

    text: str | None = None
    """Title/paragraph text, or a table already rendered as markdown --
    same content ChunkMetadata.element_types would already tell you WAS
    a table, just now with the actual content to render as one."""

    image_base64: str | None = None
    """A downscaled thumbnail, image/chart elements only -- see
    image_utils.downscale_image(). None for every other type, and for
    an image/chart whose bytes weren't available at ingest time."""

    description: str | None = None
    """Vision-generated caption, image/chart elements only -- shown
    alongside image_base64, not instead of it."""

    page: int | None = None
    slide: int | None = None


class ChunkMetadata(BaseModel):
    source_file: str

    doc_id: str = ""
    """Stable document identity (sha256 of the filename -- see
    api/routers/ingest.py), distinct from source_file (the human-readable
    filename shown in citations). Set post-hoc by the ingest router, in
    the same loop that restores source_file to the filename -- not by any
    chunker, since it's intrinsic to the DOCUMENT, not to how a chunk was
    cut, and it's already embedded as the prefix of Chunk.id
    (chunking/ids.py's chunk_id()), so recording it here just makes an
    existing fact addressable. Defaults to "" so old chunks upserted
    before this field existed are visibly incomplete rather than
    silently wrong -- see stores.qdrant_store._to_point, which
    deliberately does NOT fall back to source_file here."""

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

    elements: list[ChunkElement] = []
    """The actual elements behind element_types, in order -- lets a
    caller render this chunk as its real constituent parts instead of
    just the flattened text. Same best-effort rule as element_positions/
    element_types. A parent-child child chunk carries its PARENT's
    elements verbatim (see ParentChildChunker) -- a child is an
    arbitrary character-range slice of the parent's flattened text, so
    precise per-child element attribution isn't attempted here, same
    pre-existing approximation element_types/pages/slides already make
    for children."""

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
