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
    """sha256 of the uploaded file's bytes -- re-uploading identical
    content always maps to the same doc_id, so ingestion is naturally
    idempotent at the identity level even before real change-detection
    is built."""

    filename: str
    status: Literal["ingested"]
    num_parent_chunks: int
    num_child_chunks: int
    ingested_at: datetime


class DocumentSummary(BaseModel):
    doc_id: str
    filename: str
    num_parent_chunks: int
    num_child_chunks: int
    ingested_at: datetime


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]


class QueryRequest(BaseModel):
    question: str
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
