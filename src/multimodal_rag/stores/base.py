"""Abstract VectorStore interface ("port") every backend implements.

Same ports/adapters shape as providers/chunking: downstream code depends
only on this interface and multimodal_rag.stores.factory.get_vector_store()
— never on a concrete store class — so the backend (Qdrant today, maybe
Pinecone/Weaviate/pgvector later) can change without touching anything
that calls it.
"""

from abc import ABC, abstractmethod

from ..chunking.schema import Chunk
from ..providers.schema import EmbeddingVector
from .schema import SearchResult


class VectorStore(ABC):
    @abstractmethod
    def create_collection(
        self,
        dimension: int,
        distance: str = "cosine",
        m: int = 16,
        ef_construct: int = 100,
        indexing_threshold: int = 20000,
    ) -> None:
        """Create (or recreate) the collection, sized for `dimension`-length
        vectors. m/ef_construct/indexing_threshold are exposed so HNSW
        behavior can be experimented with directly, not buried."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[EmbeddingVector]) -> None:
        """Insert or update `chunks` with their corresponding `vectors`
        (matched by list position). Refuses to mix vectors from different
        embedding models in one call. Implementations should retry
        transient failures and, if a batch fails persistently, keep
        going with the rest rather than losing already-succeeded work —
        see qdrant_store.UpsertBatchError for the concrete contract."""

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int = 5, ef_search: int | None = None
    ) -> list[SearchResult]:
        """Find the top_k chunks closest to query_vector. ef_search is the
        query-time recall/latency knob — exposed per-call, no rebuild
        needed to change it."""
