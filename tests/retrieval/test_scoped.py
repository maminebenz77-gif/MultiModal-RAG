"""ScopedRetriever: the mandatory security filter + sqlite post-check
that sits between a caller (or the agent) and the real Retriever.

Integration-style, like test_retriever.py: real Qdrant/Elasticsearch,
plus a real sqlite Database (tmp_path) since the post-check reads from
it directly -- that's the whole point being tested, not something a
mock of Database could stand in for.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from multimodal_rag.api.db import Database
from multimodal_rag.chunking.schema import Chunk, ChunkMetadata
from multimodal_rag.identity import Principal
from multimodal_rag.metadata import DocumentMetadata
from multimodal_rag.providers.base import EmbeddingProvider
from multimodal_rag.providers.schema import EmbeddingVector
from multimodal_rag.retrieval.retriever import Retriever
from multimodal_rag.retrieval.schema import RetrievalMethod
from multimodal_rag.retrieval.scoped import ScopedRetriever
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore
from multimodal_rag.stores.indexer import HybridIndexer
from multimodal_rag.stores.qdrant_store import QdrantStore

_COLLECTION = "test_scoped_retriever"


class FakeEmbedder(EmbeddingProvider):
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        return [
            EmbeddingVector(vector=self._vectors_by_text[t], model_id="fake-model", dimension=2)
            for t in texts
        ]


@pytest.fixture
def vector_store() -> Iterator[QdrantStore]:
    s = QdrantStore(url="http://localhost:6333", collection_name=_COLLECTION)
    s.create_collection(dimension=2, indexing_threshold=0)
    s.publish()
    yield s
    physical = s._current_alias_target()
    if physical is not None:
        s._client.delete_collection(physical)


@pytest.fixture
def keyword_store() -> Iterator[ElasticsearchStore]:
    s = ElasticsearchStore(url="http://localhost:9200", index_name=_COLLECTION)
    s.create_index()
    yield s
    s._client.indices.delete(index=_COLLECTION, ignore_unavailable=True)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    return Database(tmp_path / "state.db")


def _ingest(
    indexer: HybridIndexer,
    db: Database,
    embedder: FakeEmbedder,
    *,
    doc_id: str,
    text: str,
    classification: str,
    private: bool = False,
    owner: str | None = None,
) -> None:
    """Mirrors what routers/ingest.py actually does: a store write AND a
    sqlite row, both carrying the same metadata -- the post-check reads
    the sqlite side, the store-level filter reads the payload side, and
    a real ingest keeps both in sync."""
    metadata = DocumentMetadata(
        classification=classification, private=private, owner=owner  # type: ignore[arg-type]
    )
    chunk = Chunk(
        id=f"{doc_id}::a::0",
        text=text,
        metadata=ChunkMetadata(
            source_file=doc_id, doc_id=doc_id, element_positions=[0], element_types=["title"]
        ),
    )
    indexer.index([chunk], embedder.embed([text]), doc_metadata=metadata)
    db.upsert_document(
        Principal.unrestricted(), doc_id, doc_id, f"hash-{doc_id}", 1, 0, metadata=metadata
    )


def test_a_public_document_is_visible_to_a_low_clearance_principal(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore, db: Database
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "public content": [1.0, 0.0]})
    indexer = HybridIndexer(vector_store, keyword_store)
    _ingest(indexer, db, embedder, doc_id="doc-a", text="public content", classification="public")

    retriever = ScopedRetriever(
        Retriever(vector_store, keyword_store, embedder),
        Principal(principal_id="user:bob", clearance="public"),
        db,
    )
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=5)

    assert [r.chunk_id for r in results] == ["doc-a::a::0"]


def test_a_document_above_the_callers_clearance_is_invisible(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore, db: Database
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "secret content": [1.0, 0.0]})
    indexer = HybridIndexer(vector_store, keyword_store)
    _ingest(indexer, db, embedder, doc_id="doc-a", text="secret content", classification="c3")

    retriever = ScopedRetriever(
        Retriever(vector_store, keyword_store, embedder),
        Principal(principal_id="user:bob", clearance="public"),
        db,
    )
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=5)

    assert results == []


def test_a_document_at_the_callers_exact_clearance_is_visible(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore, db: Database
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "c2 content": [1.0, 0.0]})
    indexer = HybridIndexer(vector_store, keyword_store)
    _ingest(indexer, db, embedder, doc_id="doc-a", text="c2 content", classification="c2")

    retriever = ScopedRetriever(
        Retriever(vector_store, keyword_store, embedder),
        Principal(principal_id="user:bob", clearance="c2"),
        db,
    )
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=5)

    assert [r.chunk_id for r in results] == ["doc-a::a::0"]


def test_a_private_document_is_visible_only_to_its_owner(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore, db: Database
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "alices notes": [1.0, 0.0]})
    indexer = HybridIndexer(vector_store, keyword_store)
    _ingest(
        indexer,
        db,
        embedder,
        doc_id="doc-a",
        text="alices notes",
        classification="public",
        private=True,
        owner="user:alice",
    )

    owner_retriever = ScopedRetriever(
        Retriever(vector_store, keyword_store, embedder),
        Principal(principal_id="user:alice", clearance="public"),
        db,
    )
    stranger_retriever = ScopedRetriever(
        Retriever(vector_store, keyword_store, embedder),
        Principal(principal_id="user:mallory", clearance="c3"),
        db,
    )

    assert [r.chunk_id for r in owner_retriever.retrieve("query", RetrievalMethod.COSINE, 5)] == [
        "doc-a::a::0"
    ]
    # Higher clearance does NOT substitute for ownership -- privacy and
    # clearance are independent checks, both must pass.
    assert stranger_retriever.retrieve("query", RetrievalMethod.COSINE, 5) == []


def test_an_unowned_private_document_is_visible_to_nobody_including_high_clearance(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore, db: Database
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "orphaned": [1.0, 0.0]})
    indexer = HybridIndexer(vector_store, keyword_store)
    _ingest(
        indexer, db, embedder, doc_id="doc-a", text="orphaned", classification="public",
        private=True, owner=None,
    )

    retriever = ScopedRetriever(
        Retriever(vector_store, keyword_store, embedder),
        Principal(principal_id="user:anyone", clearance="c3"),
        db,
    )
    assert retriever.retrieve("query", RetrievalMethod.COSINE, 5) == []


def test_an_admin_principal_sees_everything_including_others_private_documents(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore, db: Database
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "alices private note": [1.0, 0.0]})
    indexer = HybridIndexer(vector_store, keyword_store)
    _ingest(
        indexer,
        db,
        embedder,
        doc_id="doc-a",
        text="alices private note",
        classification="c3",
        private=True,
        owner="user:alice",
    )

    retriever = ScopedRetriever(
        Retriever(vector_store, keyword_store, embedder), Principal.unrestricted(), db
    )
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=5)

    assert [r.chunk_id for r in results] == ["doc-a::a::0"]


def test_post_check_drops_a_result_the_catalogue_no_longer_recognizes(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore, db: Database
) -> None:
    """Simulates drift: the chunk is still in the search index, but its
    sqlite row is gone (e.g. a delete that touched the store but not the
    catalogue). Fail closed -- absent from the catalogue means invisible,
    not "trust what's on the payload.\""""
    embedder = FakeEmbedder({"query": [1.0, 0.0], "orphaned in the index": [1.0, 0.0]})
    indexer = HybridIndexer(vector_store, keyword_store)
    metadata = DocumentMetadata(classification="public")
    chunk = Chunk(
        id="doc-a::a::0",
        text="orphaned in the index",
        metadata=ChunkMetadata(
            source_file="doc-a", doc_id="doc-a", element_positions=[0], element_types=["title"]
        ),
    )
    indexer.index([chunk], embedder.embed(["orphaned in the index"]), doc_metadata=metadata)
    # Deliberately NOT calling db.upsert_document -- the sqlite row never existed.

    retriever = ScopedRetriever(
        Retriever(vector_store, keyword_store, embedder),
        Principal(principal_id="user:bob", clearance="c3"),
        db,
    )
    assert retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=5) == []
