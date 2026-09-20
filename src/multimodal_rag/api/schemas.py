"""Pydantic request/response models for the API layer. These are
deliberately separate from the internal domain schemas (Chunk,
SearchResult, RagAnswer, ...) even where they overlap heavily --
the API's shape is a promise to external callers and needs to be able
to evolve independently of internal refactors.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from ..metadata import Classification
from ..retrieval.schema import RetrievalMethod


class DocumentMetadataOut(BaseModel):
    """The tags a document carries -- see metadata.DocumentMetadata for
    the internal domain model this mirrors. Kept separate from it (this
    file's own stated rule) so the API's shape can evolve independently."""

    classification: Classification
    private: bool = False
    owner: str | None = None
    author: str | None = None
    doc_date: str | None = None
    data_type: str | None = None
    tags: list[str] = []


class DocumentMetadataUpdate(BaseModel):
    """PATCH /documents/{doc_id}'s request body. Every field optional --
    PATCH semantics: only fields the caller actually sends are changed
    (see Database.update_document_metadata, which reads this via
    model_dump(exclude_unset=True), not via which fields are None --
    private=False and private=<omitted> are different requests)."""

    classification: Classification | None = None
    private: bool | None = None
    owner: str | None = None
    author: str | None = None
    doc_date: str | None = None
    data_type: str | None = None
    tags: list[str] | None = None


class IngestResponse(BaseModel):
    doc_id: str
    """A hash of (uploader, FILENAME), not the file's bytes -- doc_id identifies
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

    metadata: DocumentMetadataOut
    """What this doc_id is now recorded as -- for "duplicate_content",
    this is the ALREADY-STORED document's metadata (nothing new was
    ingested under this doc_id, so there's nothing else to show); for
    "already_ingested", the existing document's metadata, UNCHANGED by
    whatever this request's own metadata said (see PATCH
    /documents/{doc_id} for how metadata is actually updated on an
    already-ingested document)."""


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
    metadata: DocumentMetadataOut


class DocumentsResponse(BaseModel):
    documents: list[DocumentSummary]


class WipeResponse(BaseModel):
    status: Literal["wiped"]
    documents_deleted: int
    chunks_deleted: int
    """Query/feedback history is NOT touched by a wipe -- it's a log of
    past activity, not corpus state, and stays meaningful even after
    the corpus itself is reset."""


class DocumentDeleteResponse(BaseModel):
    status: Literal["deleted"]
    doc_id: str
    chunks_deleted: int
    """Query/feedback history referencing this document (if any) is NOT
    touched -- same reasoning as WipeResponse."""


class ProviderOverride(BaseModel):
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None


class RuntimeOverrides(BaseModel):
    llm: ProviderOverride | None = None
    embedder: ProviderOverride | None = None


class QueryRequest(BaseModel):
    question: str
    conversation_id: str | None = None
    """Omit to start a new conversation (the server generates one and
    returns it in QueryResponse.conversation_id); pass a previously
    returned id to continue it -- the server loads that conversation's
    history and feeds it to the agent as real conversation messages (see
    generation/agent.py), which decides for itself, from the actual
    history, whether/how to search again for a follow-up. An unknown id
    is a 404, not a silently-started new conversation."""

    retrieval_method: RetrievalMethod = RetrievalMethod.HYBRID_RRF
    top_k: int = Field(default=5, ge=1, le=50)
    rerank: bool = False
    """Retrieve a broader candidate pool, then re-rank it with a
    cross-encoder before the top_k cut -- higher precision, higher
    latency/cost. Requires a Reranker to be configured for this
    deployment (see api/main.py); if not, the request fails."""

    doc_ids: list[str] | None = None
    """Restricts results to these document IDs -- the stable ids GET
    /documents hands out (sha256 of the filename), NOT filenames. A
    post-retrieval filter (see Retriever usage in routers/query.py), not
    a native store query. Fine at this corpus size; would need real
    store-level filtering to scale."""

    runtime_overrides: RuntimeOverrides | None = None
    """Optional per-request provider overrides from the UI. When unset,
    the backend uses the active .env profile defaults."""


class ChunkElementOut(BaseModel):
    type: str
    text: str | None = None
    image_base64: str | None = None
    description: str | None = None
    page: int | None = None
    slide: int | None = None


class CitationOut(BaseModel):
    marker: int
    chunk_id: str
    source: str
    doc_id: str = ""
    """The document's stable id (see generation.schema.Citation.doc_id).
    "" for citations recorded before this field existed."""

    pages: list[int]
    slides: list[int]
    text: str = ""
    elements: list[ChunkElementOut] = []
    """A snapshot of the cited chunk's own text/elements, persisted
    alongside the citation (see api/db.py) so a reloaded conversation
    can still show real chunk detail -- an actual table, an actual
    image -- not just the bare marker/source/location. Empty for
    citations recorded before this field existed."""


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    score: float
    text: str
    source: str
    doc_id: str = ""
    """The document's stable id (see stores.schema.SearchResult.doc_id)."""

    pages: list[int]
    slides: list[int]
    elements: list[ChunkElementOut] = []
    """The chunk's real constituent parts (title/paragraph/table/image),
    for rendering it as more than its flattened `text`. Empty for
    chunks produced before this field existed (not yet re-ingested) or
    by flatten-first chunking strategies. This is the full retrieved-
    candidate set, live-turn-only (not persisted) -- see CitationOut
    for the cited subset, which IS persisted."""


class QueryResponse(BaseModel):
    query_id: str
    """UUID identifying this query -- POST /feedback references it."""

    conversation_id: str
    """Always populated -- either the id the caller passed in, or the
    one the server just created because none was given. The caller's
    next /query call for this conversation should pass this back."""

    question: str
    answer: str
    citations: list[CitationOut]
    refused: bool
    needs_clarification: bool = False
    """True if `answer` is a clarifying question the agent asked back
    instead of searching -- the request was too ambiguous to know what
    to search for. Not a refusal: no search happened, so `citations` and
    `retrieved_chunks` are empty. The caller's next /query call (same
    conversation_id) should just carry the user's reply as `question`."""

    retrieval_method: RetrievalMethod
    retrieved_chunks: list[RetrievedChunkOut]
    """Every chunk that made it into the generation context -- lets a
    caller see what a retrieval method actually returned, not just what
    the model ended up citing."""


class ConversationMessageOut(BaseModel):
    query_id: str
    question: str
    answer: str
    citations: list[CitationOut]
    refused: bool
    needs_clarification: bool
    retrieval_method: str
    created_at: datetime


class ConversationResponse(BaseModel):
    conversation_id: str
    messages: list[ConversationMessageOut]
    """Oldest first -- the full turn history for this conversation, each
    with its own citations, for reloading/resuming (see GET
    /conversations/{conversation_id})."""


class ConversationSummaryOut(BaseModel):
    conversation_id: str
    preview: str
    """An LLM-generated title once one exists (see generation/title.py),
    otherwise the conversation's raw first question -- enough to
    recognize it in a picker without fetching the full turn history."""

    message_count: int
    updated_at: datetime
    """When its most recent turn was recorded."""


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummaryOut]
    """Most recently active first (see GET /conversations)."""


class ConversationDeleteResponse(BaseModel):
    status: Literal["deleted"]
    conversation_id: str


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
    avg_latency_ms: float
    """Average of queries.latency_ms across every recorded query that has
    one -- 0.0 if none do yet (a fresh database, or every row predates
    this column). Deliberately just the average, not p50/p95 -- this
    endpoint stays the cheap, always-on local summary; percentile
    breakdowns and cost-per-query are a separate, heavier concern."""

    feedback_rate: float
    """(feedback_up + feedback_down) / total_queries -- what fraction of
    queries got ANY explicit feedback at all, not the up/down split
    (that's feedback_up/feedback_down already). 0.0 when there are no
    queries yet."""


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    qdrant: Literal["up", "down"]
    elasticsearch: Literal["up", "down"]
