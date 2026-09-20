"""Grounded RAG answer types."""

from pydantic import BaseModel

from ..chunking.schema import ChunkElement
from ..stores.schema import SearchResult


class Citation(BaseModel):
    marker: int
    """The ⟦N⟧ number used inline in the answer text."""

    chunk_id: str
    source: str
    doc_id: str = ""
    """The document's stable id (see ChunkMetadata.doc_id), distinct from
    `source` (the display filename) -- lets a caller reliably match a
    citation back to a specific document even if two documents share a
    filename display quirk. "" for citations recorded before this field
    existed."""

    pages: list[int] = []
    slides: list[int] = []

    text: str = ""
    elements: list[ChunkElement] = []
    """A snapshot of the cited chunk's own text/elements at the moment it
    was cited -- persisted (see api/db.py's citations table) so a
    reloaded conversation can still show real chunk detail, not just
    the bare marker/source/location, even if the corpus has since
    changed or that chunk no longer exists there."""


class RagAnswer(BaseModel):
    answer: str
    """Raw model output, including inline ⟦N⟧ markers as generated."""

    citations: list[Citation]
    """Structured lookup for each marker actually used — built by US from
    the numbered context we constructed, not trusted from the model's own
    (unreliable) recall of filenames/page numbers."""

    refused: bool
    """True if the model indicated the answer isn't in the context."""

    needs_clarification: bool = False
    """True if the model asked the user a clarifying question instead of
    searching/answering -- `answer` holds that question, `citations` and
    `retrieved_chunks` are empty, and `refused` is False (this isn't "no
    answer in the corpus", it's "not enough information in the request
    yet to know what to search for"). See generation/agent.py."""

    retrieved_chunks: list[SearchResult] = []
    """Every chunk that made it into the generation context, not just the
    ones actually cited -- lets a caller (e.g. the frontend's "retrieved
    chunks" panel) see what a retrieval method returned even when the
    model didn't end up citing all of it."""
