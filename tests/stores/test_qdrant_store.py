"""Integration tests against the real local Qdrant instance (see
docker-compose.yml) rather than a mocked client — a vector store is
mostly a thin wrapper around real network calls, so mocking the client
would mostly test the mock. Uses a dedicated test collection, torn down
after each test.
"""

from collections.abc import Iterator

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkMetadata
from multimodal_rag.providers.schema import EmbeddingVector
from multimodal_rag.stores.qdrant_store import QdrantStore

_COLLECTION = "test_collection"


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
    yield s
    s._client.delete_collection(_COLLECTION)


def test_create_collection_is_idempotent(store: QdrantStore) -> None:
    store.create_collection(dimension=4, indexing_threshold=0)
    store.create_collection(dimension=4, indexing_threshold=0)


def test_create_collection_rejects_unknown_distance(store: QdrantStore) -> None:
    with pytest.raises(ValueError, match="Unknown distance"):
        store.create_collection(dimension=4, distance="manhattan")


def test_upsert_and_search_roundtrip(store: QdrantStore) -> None:
    chunks = [_chunk("doc.md::a::0", "about GPUs"), _chunk("doc.md::a::1", "about soup recipes")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])]
    store.upsert(chunks, vectors)

    results = store.search(query_vector=[0.9, 0.1, 0.0, 0.0], top_k=2)

    assert results[0].chunk_id == "doc.md::a::0"
    assert results[0].text == "about GPUs"
    assert results[0].source == "doc.md"
    assert results[0].doc_id == "doc.md"
    assert results[0].element_types == ["title"]
    assert results[0].model_id == "test-model"
    assert results[0].score > results[1].score


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

    results = store.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=10)
    assert len(results) == 1
    assert results[0].text == "updated text"


def test_search_accepts_explicit_ef_search(store: QdrantStore) -> None:
    chunks = [_chunk("doc.md::a::0", "a"), _chunk("doc.md::a::1", "b")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])]
    store.upsert(chunks, vectors)

    results = store.search(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=1, ef_search=64)
    assert len(results) == 1
