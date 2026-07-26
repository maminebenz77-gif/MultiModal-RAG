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
        """Prepare a new collection version, sized for `dimension`-length
        vectors. m/ef_construct/indexing_threshold are exposed so HNSW
        behavior can be experimented with directly, not buried.

        Must NOT destroy or affect whatever is currently live — callers
        keep searching the previous version, unaffected, until publish()
        atomically cuts over. A production pipeline that tore down the
        live collection before the replacement was ready and verified
        would have no rollback and no way to diff what changed if the
        new version turned out to be broken."""

    @abstractmethod
    def upsert(self, chunks: list[Chunk], vectors: list[EmbeddingVector]) -> None:
        """Insert or update `chunks` with their corresponding `vectors`
        (matched by list position) into whichever version is currently
        being built (or the live one, if no new version is pending).
        Refuses to mix vectors from different embedding models in one
        call. Implementations should retry transient failures and, if a
        batch fails persistently, keep going with the rest rather than
        losing already-succeeded work — see qdrant_store.UpsertBatchError
        for the concrete contract."""

    @abstractmethod
    def publish(self) -> None:
        """Atomically make the most recently created+populated collection
        version live, replacing whatever was live before. search() never
        sees a partial/empty state during ingestion — this is a single
        atomic cutover, not a delete-then-rebuild."""

    @abstractmethod
    def search(
        self, query_vector: EmbeddingVector, top_k: int = 5, ef_search: int | None = None
    ) -> list[SearchResult]:
        """Find the top_k chunks closest to query_vector, against
        whichever version is currently live. ef_search is the query-time
        recall/latency knob — exposed per-call, no rebuild needed to
        change it.

        Takes an EmbeddingVector, not a raw list[float], specifically so
        the caller can't search without declaring which model produced
        the query vector. Implementations should verify that model_id
        against what's actually stored — a dimension match alone doesn't
        guarantee compatibility: two different models can share a
        dimension count while encoding meaning in incompatible spaces,
        which would silently return confident-looking, meaningless
        results rather than an error."""
