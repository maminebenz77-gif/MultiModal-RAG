"""Integration tests against the real local Elasticsearch instance (see
docker-compose.yml) rather than a mocked client, matching the same
rationale as the Qdrant tests — a store is mostly a thin wrapper around
real network calls.
"""

from collections.abc import Iterator

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkMetadata
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore

_INDEX = "test_index"


def _chunk(chunk_id: str, text: str, source: str = "doc.md") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(source_file=source, element_positions=[0], element_types=["title"]),
    )


@pytest.fixture
def store() -> Iterator[ElasticsearchStore]:
    s = ElasticsearchStore(url="http://localhost:9200", index_name=_INDEX)
    s.create_index()
    yield s
    s._client.indices.delete(index=_INDEX, ignore_unavailable=True)


def test_create_index_is_idempotent(store: ElasticsearchStore) -> None:
    store.create_index()
    store.create_index()


def test_index_and_search_roundtrip(store: ElasticsearchStore) -> None:
    chunks = [
        _chunk("doc.md::a::0", "the GPU ran out of memory during batch inference"),
        _chunk("doc.md::a::1", "the soup needed more salt and pepper"),
    ]
    store.index_chunks(chunks)

    results = store.search("GPU memory", top_k=2)

    assert results[0].chunk_id == "doc.md::a::0"
    assert results[0].text == "the GPU ran out of memory during batch inference"
    assert results[0].source == "doc.md"
    assert results[0].doc_id == "doc.md"
    assert results[0].element_types == ["title"]
    assert results[0].model_id is None


def test_search_ranks_more_relevant_document_higher(store: ElasticsearchStore) -> None:
    chunks = [
        _chunk("doc.md::a::0", "latency latency latency: the internal gateway was slow"),
        _chunk("doc.md::a::1", "a brief mention of latency in passing"),
    ]
    store.index_chunks(chunks)

    results = store.search("latency", top_k=2)

    assert results[0].chunk_id == "doc.md::a::0"
    assert results[0].score > results[1].score


def test_reindexing_same_chunk_id_updates_rather_than_duplicates(
    store: ElasticsearchStore,
) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "original text about latency")])
    store.index_chunks([_chunk("doc.md::a::0", "updated text about throughput")])

    results = store.search("throughput", top_k=10)
    assert len(results) == 1
    assert results[0].text == "updated text about throughput"


def test_search_with_no_matches_returns_empty_list(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "something about GPUs")])
    results = store.search("nonexistent_term_xyz", top_k=5)
    assert results == []
