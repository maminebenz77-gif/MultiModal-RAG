"""Integration tests against real local Qdrant + Elasticsearch."""

from collections.abc import Iterator

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkMetadata
from multimodal_rag.providers.schema import EmbeddingVector
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore
from multimodal_rag.stores.indexer import HybridIndexer, IndexConsistencyError
from multimodal_rag.stores.qdrant_store import QdrantStore

_NAME = "test_hybrid_indexer"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.retry.time.sleep", lambda seconds: None)


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            source_file="doc.md", element_positions=[0], element_types=["title"]
        ),
    )


def _vector(values: list[float], model_id: str = "test-model") -> EmbeddingVector:
    return EmbeddingVector(vector=values, model_id=model_id, dimension=len(values))


@pytest.fixture
def vector_store() -> Iterator[QdrantStore]:
    s = QdrantStore(url="http://localhost:6333", collection_name=_NAME)
    s.create_collection(dimension=2, indexing_threshold=0)
    s.publish()
    yield s
    physical = s._current_alias_target()
    if physical is not None:
        s._client.delete_collection(physical)


@pytest.fixture
def keyword_store() -> Iterator[ElasticsearchStore]:
    s = ElasticsearchStore(url="http://localhost:9200", index_name=_NAME)
    s.create_index()
    yield s
    s._client.indices.delete(index=_NAME, ignore_unavailable=True)


def test_index_writes_to_both_stores(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    chunks = [_chunk("a", "hello world"), _chunk("b", "goodbye world")]
    vectors = [_vector([1.0, 0.0]), _vector([0.0, 1.0])]

    indexer.index(chunks, vectors)

    assert sorted(vector_store.list_chunk_ids()) == ["a", "b"]
    assert sorted(keyword_store.list_chunk_ids()) == ["a", "b"]


def test_index_raises_index_consistency_error_when_keyword_indexing_fails(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_fails(chunks: list[Chunk]) -> None:
        raise RuntimeError("persistent ES failure")

    monkeypatch.setattr(keyword_store, "index_chunks", always_fails)

    indexer = HybridIndexer(vector_store, keyword_store)
    chunks = [_chunk("a", "hello world")]
    vectors = [_vector([1.0, 0.0])]

    with pytest.raises(IndexConsistencyError) as exc_info:
        indexer.index(chunks, vectors)

    assert exc_info.value.chunk_ids == ["a"]
    # The vector store write already succeeded and isn't rolled back --
    # the two stores are now genuinely, visibly inconsistent.
    assert vector_store.list_chunk_ids() == ["a"]
    assert keyword_store.list_chunk_ids() == []


def test_delete_removes_from_both_stores(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index(
        [_chunk("a", "hello world"), _chunk("b", "goodbye world")],
        [_vector([1.0, 0.0]), _vector([0.0, 1.0])],
    )

    indexer.delete(["b"])

    assert vector_store.list_chunk_ids() == ["a"]
    assert keyword_store.list_chunk_ids() == ["a"]


def test_delete_with_empty_list_is_a_no_op(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index([_chunk("a", "hello")], [_vector([1.0, 0.0])])

    indexer.delete([])

    assert vector_store.list_chunk_ids() == ["a"]


def test_delete_raises_index_consistency_error_when_keyword_deletion_fails(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index([_chunk("a", "hello")], [_vector([1.0, 0.0])])

    def always_fails(chunk_ids: list[str]) -> None:
        raise RuntimeError("persistent ES failure")

    monkeypatch.setattr(keyword_store, "delete_chunks", always_fails)

    with pytest.raises(IndexConsistencyError) as exc_info:
        indexer.delete(["a"])

    assert exc_info.value.chunk_ids == ["a"]
    # The vector store deletion already succeeded -- now inconsistent
    # with the keyword store, which still has the stale chunk.
    assert vector_store.list_chunk_ids() == []
    assert keyword_store.list_chunk_ids() == ["a"]


def test_check_consistency_reports_no_drift_when_in_sync(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index([_chunk("a", "hello")], [_vector([1.0, 0.0])])

    report = indexer.check_consistency()

    assert report.is_consistent
    assert report.only_in_vector_store == []
    assert report.only_in_keyword_store == []


def test_check_consistency_reports_chunks_only_in_vector_store(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    vector_store.upsert([_chunk("a", "hello")], [_vector([1.0, 0.0])])

    indexer = HybridIndexer(vector_store, keyword_store)
    report = indexer.check_consistency()

    assert not report.is_consistent
    assert report.only_in_vector_store == ["a"]
    assert report.only_in_keyword_store == []


def test_check_consistency_reports_chunks_only_in_keyword_store(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    keyword_store.index_chunks([_chunk("a", "hello")])

    indexer = HybridIndexer(vector_store, keyword_store)
    report = indexer.check_consistency()

    assert not report.is_consistent
    assert report.only_in_vector_store == []
    assert report.only_in_keyword_store == ["a"]
