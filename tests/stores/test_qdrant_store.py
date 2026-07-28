"""Integration tests against the real local Qdrant instance (see
docker-compose.yml) rather than a mocked client — a vector store is
mostly a thin wrapper around real network calls, so mocking the client
would mostly test the mock. Uses a dedicated test collection (alias),
torn down after each test.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkMetadata
from multimodal_rag.providers.schema import EmbeddingVector
from multimodal_rag.stores.qdrant_store import ModelMismatchError, QdrantStore, UpsertBatchError

_COLLECTION = "test_collection"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.retry.time.sleep", lambda seconds: None)


def _chunk(chunk_id: str, text: str, source: str = "doc.md") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(source_file=source, element_positions=[0], element_types=["title"]),
    )


def _vector(values: list[float], model_id: str = "test-model") -> EmbeddingVector:
    return EmbeddingVector(vector=values, model_id=model_id, dimension=len(values))


@pytest.fixture
def store() -> Iterator[QdrantStore]:
    s = QdrantStore(url="http://localhost:6333", collection_name=_COLLECTION)
    s.create_collection(dimension=4, indexing_threshold=0)
    s.publish()
    yield s
    physical = s._current_alias_target()
    if physical is not None:
        s._client.delete_collection(physical)


def test_create_collection_rejects_unknown_distance(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="Unknown distance"):
        store.create_collection(dimension=4, distance="manhattan")


def test_publish_without_pending_collection_raises(store: QdrantStore) -> None:
    with pytest.raises(RuntimeError, match="No pending collection"):
        store.publish()


def test_create_collection_does_not_affect_a_live_published_version(
    store: QdrantStore,
) -> None:
    # store fixture already published an (empty) v1. Populate it.
    store.upsert([_chunk("doc.md::a::0", "v1 data")], [_vector([1.0, 0.0, 0.0, 0.0])])

    # Start building v2 but do NOT publish it yet.
    store.create_collection(dimension=4, indexing_threshold=0)
    store.upsert([_chunk("doc.md::b::0", "v2 data")], [_vector([0.0, 1.0, 0.0, 0.0])])

    # search() still goes through the alias, which still points at v1.
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::a::0"]


def test_publish_swaps_atomically_and_removes_the_previous_version(
    store: QdrantStore,
) -> None:
    store.upsert([_chunk("doc.md::a::0", "v1 data")], [_vector([1.0, 0.0, 0.0, 0.0])])
    old_physical = store._current_alias_target()
    assert old_physical is not None

    store.create_collection(dimension=4, indexing_threshold=0)
    store.upsert([_chunk("doc.md::b::0", "v2 data")], [_vector([0.0, 1.0, 0.0, 0.0])])
    store.publish()

    results = store.search(_vector([0.0, 1.0, 0.0, 0.0]), top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::b::0"]
    assert not store._client.collection_exists(old_physical)


def test_upsert_and_search_roundtrip(store: QdrantStore) -> None:
    chunks = [_chunk("doc.md::a::0", "about GPUs"), _chunk("doc.md::a::1", "about soup recipes")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])]
    store.upsert(chunks, vectors)

    results = store.search(_vector([0.9, 0.1, 0.0, 0.0]), top_k=2)

    assert results[0].chunk_id == "doc.md::a::0"
    assert results[0].text == "about GPUs"
    assert results[0].source == "doc.md"
    assert results[0].doc_id == "doc.md"
    assert results[0].element_types == ["title"]
    assert results[0].model_id == "test-model"
    assert results[0].score > results[1].score


def test_search_rejects_a_query_vector_from_a_different_model(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0], "model-a")])

    with pytest.raises(ModelMismatchError, match="model-a.*model-b|model-b.*model-a"):
        store.search(_vector([1.0, 0.0, 0.0, 0.0], "model-b"), top_k=1)


def test_search_allows_a_matching_model(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0], "model-a")])

    results = store.search(_vector([1.0, 0.0, 0.0, 0.0], "model-a"), top_k=1)
    assert len(results) == 1


def test_search_on_empty_collection_does_not_raise_model_mismatch(store: QdrantStore) -> None:
    # store fixture publishes an empty collection — nothing to compare
    # the query vector's model against yet, so nothing should be rejected.
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0], "any-model"), top_k=1)
    assert results == []


def test_search_omits_vector_by_default(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0])])
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].vector is None


def test_search_with_vectors_populates_vector(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0])])
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1, with_vectors=True)
    assert results[0].vector == [1.0, 0.0, 0.0, 0.0]


def test_upsert_rejects_mixed_models(store: QdrantStore) -> None:
    chunks = [_chunk("doc.md::a::0", "a"), _chunk("doc.md::a::1", "b")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0], "model-a"), _vector([0.0, 1.0, 0.0, 0.0], "model-b")]
    with pytest.raises(ValueError, match="mix vectors"):
        store.upsert(chunks, vectors)


def test_upsert_rejects_mismatched_lengths(store: QdrantStore) -> None:
    chunks = [_chunk("doc.md::a::0", "a")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])]
    with pytest.raises(ValueError, match="same length"):
        store.upsert(chunks, vectors)


def test_reupserting_same_chunk_id_updates_rather_than_duplicates(store: QdrantStore) -> None:
    chunk = _chunk("doc.md::a::0", "original text")
    store.upsert([chunk], [_vector([1.0, 0.0, 0.0, 0.0])])

    updated = _chunk("doc.md::a::0", "updated text")
    store.upsert([updated], [_vector([1.0, 0.0, 0.0, 0.0])])

    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert len(results) == 1
    assert results[0].text == "updated text"


def test_search_accepts_explicit_ef_search(store: QdrantStore) -> None:
    chunks = [_chunk("doc.md::a::0", "a"), _chunk("doc.md::a::1", "b")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])]
    store.upsert(chunks, vectors)

    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1, ef_search=64)
    assert len(results) == 1


def test_upsert_splits_into_multiple_batches(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store._batch_size = 1
    call_sizes: list[int] = []
    real_upsert = store._client.upsert

    def spy_upsert(**kwargs: Any) -> Any:
        points = kwargs["points"]
        assert isinstance(points, list)
        call_sizes.append(len(points))
        return real_upsert(**kwargs)

    monkeypatch.setattr(store._client, "upsert", spy_upsert)

    chunks = [_chunk(f"doc.md::a::{i}", f"text {i}") for i in range(3)]
    vectors = [_vector([float(i), 0.0, 0.0, 0.0]) for i in range(3)]
    store.upsert(chunks, vectors)

    assert call_sizes == [1, 1, 1]


def test_upsert_retries_transient_failure_then_succeeds(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"count": 0}
    real_upsert = store._client.upsert

    def flaky_upsert(**kwargs: Any) -> Any:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient network blip")
        return real_upsert(**kwargs)

    monkeypatch.setattr(store._client, "upsert", flaky_upsert)

    chunk = _chunk("doc.md::a::0", "hello")
    store.upsert([chunk], [_vector([1.0, 0.0, 0.0, 0.0])])

    assert attempts["count"] == 3
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].chunk_id == "doc.md::a::0"


def test_upsert_persistent_batch_failure_raises_but_preserves_others(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store._batch_size = 1
    store._max_retries = 2
    real_upsert = store._client.upsert

    def sometimes_failing_upsert(**kwargs: Any) -> Any:
        points = kwargs["points"]
        assert isinstance(points, list)
        if points[0].payload["chunk_id"] == "doc.md::bad::0":
            raise RuntimeError("persistent failure")
        return real_upsert(**kwargs)

    monkeypatch.setattr(store._client, "upsert", sometimes_failing_upsert)

    good = _chunk("doc.md::good::0", "fine")
    bad = _chunk("doc.md::bad::0", "will fail")

    with pytest.raises(UpsertBatchError) as exc_info:
        store.upsert([good, bad], [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])])

    error = exc_info.value
    assert error.succeeded_chunk_ids == ["doc.md::good::0"]
    assert error.failed_chunk_ids == ["doc.md::bad::0"]

    # The good chunk is durably stored despite the other batch's failure.
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert any(r.chunk_id == "doc.md::good::0" for r in results)


def test_list_chunk_ids_empty_when_nothing_upserted(store: QdrantStore) -> None:
    assert store.list_chunk_ids() == []


def test_list_chunk_ids_returns_all_upserted_ids(store: QdrantStore) -> None:
    chunks = [_chunk(f"doc.md::a::{i}", f"text {i}") for i in range(5)]
    vectors = [_vector([float(i), 0.0, 0.0, 0.0]) for i in range(5)]
    store.upsert(chunks, vectors)

    assert sorted(store.list_chunk_ids()) == sorted(c.id for c in chunks)
