"""Pydantic request/response models for the API layer. These are
deliberately separate from the internal domain schemas (Chunk,
SearchResult, RagAnswer, ...) even where they overlap heavily --
the API's shape is a promise to external callers and needs to be able
to evolve independently of internal refactors.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..retrieval.schema import RetrievalMethod


class IngestResponse(BaseModel):
    doc_id: str
    """sha256 of the FILENAME, not the file's bytes -- doc_id identifies
    "this document" as a stable slot that survives edits, so re-ingesting
    the same filename with slightly different content is recognized as
    an update to the SAME document (and only the chunks that actually
    changed get re-embedded) rather than looking like an unrelated new
    document. A rename is therefore a new doc_id, even if the content is
    byte-identical to something already ingested -- the trade-off that
    makes chunk-level diffing on edits possible at all with a single
    identity concept. See DocumentSummary.content_hash for the "is this
    exact content already what's stored" check."""

    filename: str
    status: Literal["ingested", "already_ingested", "duplicate_content"]
    """"already_ingested": this exact content (same doc_id AND
    content_hash) was already in the corpus -- parse/chunk/embed/index
    were skipped entirely. A changed content_hash under an existing
    doc_id still returns "ingested", not "already_ingested" -- some real
    work happened, even if fewer chunks than a fresh document.
    "duplicate_content": this exact content already exists in the
    corpus under a DIFFERENT filename (checked globally, not scoped to
    this doc_id) -- nothing was ingested; see duplicate_of."""

    duplicate_of: str | None = None
    """Set only when status == "duplicate_content": the filename this
    content is already stored under."""

    num_parent_chunks: int
    num_child_chunks: int
    ingested_at: datetime
    ingest_warnings: list[str] = Field(default_factory=list)
    """Non-fatal ingest warnings (e.g., parser fallback from PDF hi_res
    to fast mode)."""


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    content_hash: str
    """sha256 of the file's actual bytes -- compared against a new
    upload's hash to detect "nothing changed" (skip everything) vs
    "content changed under this filename" (diff chunks) vs "never seen
    this filename before" (full ingest)."""

    num_parent_chunks: int
    num_child_chunks: int
    ingested_at: datetime


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]


class WipeResponse(BaseModel):
    status: Literal["wiped"]
    documents_deleted: int
    chunks_deleted: int
    """Query/feedback history is NOT touched by a wipe -- it's a log of
    past activity, not corpus state, and stays meaningful even after
    the corpus itself is reset."""


class ProviderOverride(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None


class RuntimeOverrides(BaseModel):
    llm: ProviderOverride | None = None
    embedder: ProviderOverride | None = None


class ConversationTurn(BaseModel):
    question: str = Field(min_length=1, max_length=4_000)
    answer: str = Field(min_length=1, max_length=12_000)


class QueryRequest(BaseModel):
    question: str
    history: list[ConversationTurn] = Field(default_factory=list, max_length=10)
    """Earlier user/assistant turns, oldest first. This is used only to
    rewrite a follow-up into a standalone retrieval question; it is not
    included as evidence in the grounded answer prompt."""

    retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID_RRF
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = False
    """Retrieve a broader candidate pool, then re-rank it with a
    cross-encoder before the top_k cut -- higher precision, higher
    latency/cost. Requires a Reranker to be configured for this
    deployment (see api/main.py); if not, the request fails."""

    doc_ids: list[str] | None = None
    """Restricts results to these document IDs -- a post-retrieval
    filter (see Retriever usage in routers/query.py), not a native store
    query. Fine at this corpus size; would need real store-level
    filtering to scale."""

    runtime_overrides: RuntimeOverrides | None = None
    """Optional per-request provider overrides from the UI. When unset,
    the backend uses the active .env profile defaults."""


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    source: str
    pages: list[int]
    slides: list[int]


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    score: float
    text: str
    source: str
    pages: list[int]
    slides: list[int]


class QueryResponse(BaseModel):
    query_id: str
    """UUID identifying this query -- POST /feedback references it."""

    question: str
    answer: str
    citations: list[CitationOut]
    refused: bool
    retrieval_method: RetrievalMethod
    retrieved_chunks: list[RetrievedChunkOut]
    """Every chunk that made it into the generation context -- lets a
    caller see what a retrieval method actually returned, not just what
    the model ended up citing."""


class FeedbackRequest(BaseModel):
    query_id: str
    rating: Literal["up", "down"]
    comment: str | None = None


class FeedbackResponse(BaseModel):
    feedback_id: str
    status: Literal["recorded"]


class MetricsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    total_queries: int
    refusal_rate: float
    feedback_up: int
    feedback_down: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant: Literal["up", "down"]
    elasticsearch: Literal["up", "down"]
