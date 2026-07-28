"""Common search result representation, shared by vector (Qdrant) and
keyword (Elasticsearch) stores — one shape for both is what makes it
possible to compare or later combine BM25 and vector results directly.
"""

from pydantic import BaseModel


class SearchResult(BaseModel):
    chunk_id: str
    score: float
    text: str
    source: str
    doc_id: str
    element_types: list[str]
    model_id: str | None = None
    """None for keyword (BM25) results — no embedding model is involved."""

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
