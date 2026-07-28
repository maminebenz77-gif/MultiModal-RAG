"""Coordinates writes to both stores, so the two demo scripts' pattern of
calling vector_store.upsert() and keyword_store.index_chunks() as two
separate, uncoordinated calls stops being the only option.

Not true cross-database atomicity — Qdrant and Elasticsearch have no
shared transaction mechanism, so that's not achievable without a much
bigger architecture (a single source-of-truth store both stores sync
from independently, the way real search infrastructure eventually
solves this — not proportionate to build here). What this DOES give:
the failure becomes visible and specific (which chunk_ids are now
inconsistent) instead of silent, plus a way to detect drift from any
other cause via check_consistency().
"""

from ..chunking.schema import Chunk
from ..providers.schema import EmbeddingVector
from .base import KeywordStore, VectorStore
from .schema import ConsistencyReport


class IndexConsistencyError(RuntimeError):
    """Raised when the vector store upsert succeeded but keyword
    indexing failed even after retries — the two stores are now
    inconsistent for the given chunk_ids. Carries those chunk_ids so the
    caller knows exactly what to reconcile, rather than discovering it
    later via a confusing search result.
    """

    def __init__(self, message: str, chunk_ids: list[str]) -> None:
        super().__init__(message)
        self.chunk_ids = chunk_ids


class HybridIndexer:
    def __init__(self, vector_store: VectorStore, keyword_store: KeywordStore) -> None:
        self._vector_store = vector_store
        self._keyword_store = keyword_store

    def index(self, chunks: list[Chunk], vectors: list[EmbeddingVector]) -> None:
        """Upsert into the vector store, then index into the keyword
        store. Each store already retries transient failures internally
        (see UpsertBatchError / ElasticsearchStore's retry wrapping) —
        this only handles the case where the second step fails
        *persistently* after the first step already succeeded."""
        self._vector_store.upsert(chunks, vectors)

        try:
            self._keyword_store.index_chunks(chunks)
        except Exception as exc:
            chunk_ids = [chunk.id for chunk in chunks]
            raise IndexConsistencyError(
                f"Vector store upsert succeeded but keyword indexing failed even "
                f"after retries; {len(chunk_ids)} chunk_ids are now present in the "
                "vector store but missing or stale in the keyword store.",
                chunk_ids=chunk_ids,
            ) from exc

    def check_consistency(self) -> ConsistencyReport:
        """Compare the chunk_ids actually present in each store. Catches
        drift regardless of how it happened — a failed index() call, a
        deletion that only touched one store, manual intervention,
        anything that didn't go through this coordinator at all."""
        vector_ids = set(self._vector_store.list_chunk_ids())
        keyword_ids = set(self._keyword_store.list_chunk_ids())
        return ConsistencyReport(
            only_in_vector_store=sorted(vector_ids - keyword_ids),
            only_in_keyword_store=sorted(keyword_ids - vector_ids),
        )
