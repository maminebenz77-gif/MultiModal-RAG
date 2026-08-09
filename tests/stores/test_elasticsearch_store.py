"""Integration tests against the real local Elasticsearch instance (see
docker-compose.yml) rather than a mocked client, matching the same
rationale as the Qdrant tests — a store is mostly a thin wrapper around
real network calls.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkMetadata
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore

_INDEX = "test_index"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.retry.time.sleep", lambda seconds: None)


def _chunk(
    chunk_id: str,
    text: str,
    source: str = "doc.md",
    pages: list[int] | None = None,
    parent_id: str | None = None,
    is_parent: bool = False,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        parent_id=parent_id,
        is_parent=is_parent,
        metadata=ChunkMetadata(
            source_file=source,
            element_positions=[0],
            element_types=["title"],
            pages=pages or [],
        ),
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


def test_ping_returns_true_when_reachable(store: ElasticsearchStore) -> None:
    assert store.ping() is True


def test_ping_returns_false_when_unreachable() -> None:
    unreachable = ElasticsearchStore(url="http://localhost:1", index_name="whatever")
    assert unreachable.ping() is False


def test_list_chunk_ids_empty_when_nothing_indexed(store: ElasticsearchStore) -> None:
    assert store.list_chunk_ids() == []


def test_list_chunk_ids_returns_all_indexed_ids(store: ElasticsearchStore) -> None:
    chunks = [_chunk(f"doc.md::a::{i}", f"text {i}") for i in range(5)]
    store.index_chunks(chunks)

    assert sorted(store.list_chunk_ids()) == sorted(c.id for c in chunks)


def test_delete_chunks_removes_them_from_the_index(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "keep me"), _chunk("doc.md::b::0", "delete me")])
    store.delete_chunks(["doc.md::b::0"])

    assert store.list_chunk_ids() == ["doc.md::a::0"]


def test_delete_chunks_is_a_no_op_for_unknown_ids(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "text")])
    store.delete_chunks(["nonexistent"])  # should not raise
    assert store.list_chunk_ids() == ["doc.md::a::0"]


def test_delete_chunks_with_empty_list_is_a_no_op(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "text")])
    store.delete_chunks([])
    assert store.list_chunk_ids() == ["doc.md::a::0"]


def test_index_chunks_retries_transient_failure_then_succeeds(
    store: ElasticsearchStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"count": 0}
    from elasticsearch.helpers import bulk as real_bulk

    def flaky_bulk(client: Any, actions: Any) -> Any:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient network blip")
        return real_bulk(client, actions)

    monkeypatch.setattr("multimodal_rag.stores.elasticsearch_store.bulk", flaky_bulk)

    store.index_chunks([_chunk("doc.md::a::0", "hello")])

    assert attempts["count"] == 3
    assert store.list_chunk_ids() == ["doc.md::a::0"]


def test_pages_round_trip_through_search(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "stored content", pages=[2, 3])])
    results = store.search("stored content", top_k=1)
    assert results[0].pages == [2, 3]


def test_parent_id_round_trips_through_search(store: ElasticsearchStore) -> None:
    store.index_chunks(
        [_chunk("doc.md::child::0", "stored content", parent_id="doc.md::parent::0")]
    )
    results = store.search("stored content", top_k=1)
    assert results[0].parent_id == "doc.md::parent::0"


def test_parent_id_is_none_when_chunk_has_no_parent(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "stored content")])
    results = store.search("stored content", top_k=1)
    assert results[0].parent_id is None


def test_is_parent_chunks_are_excluded_from_search(store: ElasticsearchStore) -> None:
    store.index_chunks(
        [
            _chunk("doc.md::parent::0", "shared latency wording", is_parent=True),
            _chunk(
                "doc.md::child::0", "shared latency wording", parent_id="doc.md::parent::0"
            ),
        ]
    )
    results = store.search("shared latency wording", top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::child::0"]


def test_ensure_ready_creates_the_index_when_missing() -> None:
    fresh = ElasticsearchStore(url="http://localhost:9200", index_name="test_ensure_ready")
    try:
        assert not fresh._client.indices.exists(index="test_ensure_ready")
        fresh.ensure_ready()
        assert fresh._client.indices.exists(index="test_ensure_ready")
    finally:
        fresh._client.indices.delete(index="test_ensure_ready", ignore_unavailable=True)


def test_ensure_ready_does_not_wipe_an_existing_index(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "stored content")])

    store.ensure_ready()

    results = store.search("stored content", top_k=1)
    assert len(results) == 1
