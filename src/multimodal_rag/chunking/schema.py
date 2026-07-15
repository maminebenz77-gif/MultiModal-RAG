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


class Chunk(BaseModel):
    id: str
    text: str
    parent_id: str | None = None
    """Set only on "child" chunks produced by the parent-child strategy."""

    metadata: ChunkMetadata
