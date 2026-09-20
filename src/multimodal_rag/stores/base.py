"""Abstract store interfaces ("ports") every backend implements.

Same ports/adapters shape as providers/chunking: downstream code depends
only on these interfaces and multimodal_rag.stores.factory's
get_vector_store()/get_keyword_store() — never on a concrete store class
— so the backend (Qdrant/Elasticsearch today, maybe others later) can
change without touching anything that calls it.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..chunking.schema import Chunk
from ..metadata import DocumentMetadata
from ..providers.schema import EmbeddingVector
from .filters import SearchFilter
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
    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[EmbeddingVector],
        doc_metadata: DocumentMetadata | None = None,
    ) -> None:
        """Insert or update `chunks` with their corresponding `vectors`
        (matched by list position) into whichever version is currently
        being built (or the live one, if no new version is pending).
        Refuses to mix vectors from different embedding models in one
        call. Implementations should retry transient failures and, if a
        batch fails persistently, keep going with the rest rather than
        losing already-succeeded work — see qdrant_store.UpsertBatchError
        for the concrete contract.

        doc_metadata, if given, is merged into every chunk's payload
        (doc_metadata.to_payload()) -- the ONLY document-level fields
        (classification, author, tags, ...) a fresh write carries.
        Omitted (None) for callers that don't have it yet (most tests,
        the demo scripts); every chunk from ingestion proper always
        passes it. See set_document_metadata() for changing it LATER,
        without a re-embed."""

    @abstractmethod
    def publish(self) -> None:
        """Atomically make the most recently created+populated collection
        version live, replacing whatever was live before. search() never
        sees a partial/empty state during ingestion — this is a single
        atomic cutover, not a delete-then-rebuild."""

    @abstractmethod
    def search(
        self,
        query_vector: EmbeddingVector,
        top_k: int = 5,
        ef_search: int | None = None,
        with_vectors: bool = False,
        search_filter: SearchFilter | None = None,
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
        results rather than an error.

        with_vectors=True populates SearchResult.vector — needed by MMR's
        diversity computation, off by default since fetching vectors has
        a real bandwidth cost most callers don't need to pay.

        search_filter, if given, is applied natively by the backend
        (pre-filtering, before/during the similarity search itself), not
        as a post-hoc Python filter over the returned top_k — the two are
        NOT equivalent: post-filtering a fixed-size top_k can silently
        return fewer than top_k results (or zero) once the filter is
        selective, since results the filter would have kept never got a
        chance to be in that top_k to begin with. Implementations should
        have a payload index on every field ever passed here — an
        unindexed filter degrades from a correctness precondition to a
        slow, potentially lossy scan (see docs/technical-decisions.md
        §7)."""

    @abstractmethod
    def list_chunk_ids(self) -> list[str]:
        """All chunk_ids currently in the live collection. Used for
        cross-store consistency checks (see stores.indexer.HybridIndexer)
        — not something most retrieval code needs directly."""

    @abstractmethod
    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        """Fetch one specific chunk by its chunk_id, directly — not a
        similarity search, no score involved. None if no chunk with that
        ID exists in the live collection. Used to resolve a child chunk's
        parent for parent-child retrieval (see Retriever's
        resolve_parent_context)."""

    @abstractmethod
    def ensure_ready(self, dimension: int) -> None:
        """Idempotently make sure a live collection exists, sized for
        `dimension`-length vectors: creates and publishes an empty one if
        none exists yet, otherwise a no-op. Safe to call on every service
        startup — unlike create_collection(), it never touches an
        already-live collection, so it can't discard previously ingested
        data the way blindly recreating on every boot would."""

    @abstractmethod
    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove specific chunks by ID. Used when re-ingesting an edited
        document: chunk_id is content-addressed (chunking/ids.py), so a
        chunk whose text changed gets a new ID and its old one becomes an
        orphan unless explicitly deleted — this is that explicit
        deletion. A no-op for any chunk_id that isn't present."""

    @abstractmethod
    def set_document_metadata(self, doc_id: str, fields: dict[str, Any]) -> None:
        """Merge `fields` into the stored payload of every chunk
        belonging to `doc_id` -- a payload PATCH (only the given keys
        change; every other stored field, metadata or not, is left
        alone), not a re-embed and not a full replace. `fields` is
        typically DocumentMetadata.to_payload() (the whole document, at
        ingest time) or a partial dict of just what a caller actually
        changed (PATCH /documents/{doc_id}) -- this method makes no
        distinction between the two, since a partial merge onto an
        already-complete payload IS the whole update either way.
        A no-op if `fields` is empty or no chunk matches `doc_id`."""

    @abstractmethod
    def ping(self) -> bool:
        """Cheap reachability check — True if the backend responds at
        all, regardless of whether a live collection exists yet. Used by
        GET /health; deliberately doesn't touch application data."""


class KeywordStore(ABC):
    """Lexical (BM25) search — the counterpart to VectorStore. No
    model_id concern exists here at all: there's no embedding model
    involved, so nothing to mix or verify. Still deliberately simpler
    than VectorStore's blue-green versioning — that responded to a
    specific, demonstrated failure mode for Qdrant that has no
    counterpart here.
    """

    @abstractmethod
    def create_index(self) -> None:
        """Create (or recreate) the index with a sensible default
        analyzer for chunk text."""

    @abstractmethod
    def index_chunks(
        self, chunks: list[Chunk], doc_metadata: DocumentMetadata | None = None
    ) -> None:
        """Index `chunks` for BM25 search. Re-indexing the same chunk_id
        updates it in place.

        doc_metadata -- see VectorStore.upsert()'s identical parameter."""

    @abstractmethod
    def search(
        self, query: str, top_k: int = 5, search_filter: SearchFilter | None = None
    ) -> list[SearchResult]:
        """BM25 keyword search: rank chunks by lexical relevance to
        `query` (term frequency, inverse document frequency, length
        normalization) — not semantic similarity.

        search_filter -- see VectorStore.search()'s identical parameter.
        Implementations should put it in a query context that doesn't
        contribute to the relevance score (Elasticsearch's bool.filter,
        not bool.must) — a document either satisfies the filter or it
        doesn't; it shouldn't be able to outrank another document by
        matching the filter "better" than another."""

    @abstractmethod
    def list_chunk_ids(self) -> list[str]:
        """All chunk_ids currently in the index. Used for cross-store
        consistency checks (see stores.indexer.HybridIndexer)."""

    @abstractmethod
    def ensure_ready(self) -> None:
        """Idempotently make sure the index exists: creates it if not,
        no-op if it does. Safe to call on every service startup —
        create_index() itself is destructive (drops and recreates), so
        calling THAT unconditionally on every boot would silently wipe
        every previously indexed chunk."""

    @abstractmethod
    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove specific chunks by ID -- the keyword-store counterpart
        to VectorStore.delete_chunks(), for the same reason (cleaning up
        orphaned chunk_ids left behind when a re-ingested document's
        content changes). A no-op for any chunk_id that isn't present."""

    @abstractmethod
    def set_document_metadata(self, doc_id: str, fields: dict[str, Any]) -> None:
        """The keyword-store counterpart to VectorStore's identical
        method -- see its docstring."""

    @abstractmethod
    def ping(self) -> bool:
        """Cheap reachability check — True if the backend responds at
        all, regardless of whether the index exists yet. Used by
        GET /health; deliberately doesn't touch application data."""
