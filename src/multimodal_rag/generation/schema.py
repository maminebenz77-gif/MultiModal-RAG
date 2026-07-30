"""Grounded RAG answer types."""

from pydantic import BaseModel


class Citation(BaseModel):
    marker: int
    """The [N] number used inline in the answer text."""

    chunk_id: str
    source: str
    pages: list[int] = []
    slides: list[int] = []


class RagAnswer(BaseModel):
    answer: str
    """Raw model output, including inline [N] markers as generated."""

    citations: list[Citation]
    """Structured lookup for each marker actually used — built by US from
    the numbered context we constructed, not trusted from the model's own
    (unreliable) recall of filenames/page numbers."""

    refused: bool
    """True if the model indicated the answer isn't in the context."""
