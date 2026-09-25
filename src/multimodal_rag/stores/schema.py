"""Common search result representation, shared by vector (Qdrant) and
keyword (Elasticsearch) stores — one shape for both is what makes it
possible to compare or later combine BM25 and vector results directly.
"""

from pydantic import BaseModel

from ..chunking.schema import ChunkElement


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    source: str
    doc_id: str
    element_types: list[str]
    elements: list[ChunkElement] = []
    """The actual elements behind element_types -- see
    ChunkMetadata.elements. Empty for chunks produced before this field
    existed (not yet re-ingested) or by flatten-first chunking
    strategies."""

    pages: list[int] = []
    slides: list[int] = []
    """Page/slide numbers the chunk came from — needed for citations.
    Empty when the source chunker couldn't determine them (see
    ChunkMetadata.pages/slides)."""

    parent_id: str | None = None
    """Set only when this chunk was produced by the parent-child chunking
    strategy and IS a child — lets a caller resolve back to the fuller
    parent chunk (see Retriever's resolve_parent_context) while still
    citing this precise child chunk. None for every other chunk."""

    model_id: str | None = None
    """None for keyword (BM25) results — no embedding model is involved."""

    version: int = 1
    doc_family_id: str | None = None
    effective_from: str | None = None
    status: str = "current"
    """Version lineage (see metadata.DocumentMetadata), read straight off the
    chunk's stored payload -- not re-verified against sqlite the way access
    control is (ScopedRetriever's post-check). A slightly stale ranking
    nudge is a quality issue; a slightly stale ACL is a leak, which is why
    only the second one gets the expensive double-check. Defaults match a
    document with no lineage tags at all: a first version, current."""

    tags: list[str] = []
    author: str | None = None
    doc_date: str | None = None
    classification: str = "public"
    private: bool = False
    """The rest of DocumentMetadata (see metadata.py), read straight off the
    same stored payload as the lineage fields above -- for observability
    only (this is what makes a chunk's tags/author/date/classification
    visible on the retrieval span in Langfuse; see
    retriever._summarize_results). NEVER the access-control authority:
    ScopedRetriever's mandatory security filter and sqlite post-check
    already decide visibility before a result ever reaches here, so
    `classification`/`private` on a SearchResult are what THIS chunk
    happens to carry, not a second place to re-derive who may see it.
    Defaults match an untagged, public, shared document."""

    vector: list[float] | None = None
    """Only populated when a caller explicitly asks for it (e.g. MMR's
    diversity computation needs candidate vectors, not just scores) —
    fetching vectors has a real bandwidth cost, so it's opt-in, not
    returned by default. Always None for keyword (BM25) results."""


class ConsistencyReport(BaseModel):
    """Result of comparing chunk_ids across a VectorStore and a
    KeywordStore that are supposed to represent the same corpus — see
    stores.indexer.HybridIndexer.check_consistency()."""

    only_in_vector_store: list[str]
    only_in_keyword_store: list[str]

    @property
    def is_consistent(self) -> bool:
        return not self.only_in_vector_store and not self.only_in_keyword_store
