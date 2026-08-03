"""Grounded RAG answer types."""

from pydantic import BaseModel

from ..stores.schema import SearchResult


class Citation(BaseModel):
    marker: int
    """The ⟦N⟧ number used inline in the answer text."""

    chunk_id: str
    source: str
    pages: list[int] = []
    slides: list[int] = []


class RagAnswer(BaseModel):
    answer: str
    """Raw model output, including inline ⟦N⟧ markers as generated."""

    citations: list[Citation]
    """Structured lookup for each marker actually used — built by US from
    the numbered context we constructed, not trusted from the model's own
    (unreliable) recall of filenames/page numbers."""

    refused: bool
    """True if the model indicated the answer isn't in the context."""

    retrieved_chunks: list[SearchResult] = []
    """Every chunk that made it into the generation context, not just the
    ones actually cited -- lets a caller (e.g. the frontend's "retrieved
    chunks" panel) see what a retrieval method returned even when the
    model didn't end up citing all of it."""
