"""Integration tests against the real local Elasticsearch instance (see
docker-compose.yml) rather than a mocked client, matching the same
rationale as the Qdrant tests — a store is mostly a thin wrapper around
real network calls.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkElement, ChunkMetadata
from multimodal_rag.metadata import DocumentMetadata
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore
from multimodal_rag.stores.filters import SearchFilter

_INDEX = "test_index"


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.retry.time.sleep", lambda seconds: None)


def _chunk(
    chunk_id: str,
    text: str,
    source: str = "doc.md",
    doc_id: str | None = None,
    pages: list[int] | None = None,
    parent_id: str | None = None,
    is_parent: bool = False,
    elements: list[ChunkElement] | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        parent_id=parent_id,
        is_parent=is_parent,
        metadata=ChunkMetadata(
            source_file=source,
            # See test_qdrant_store.py's identical _chunk() helper for why
            # this defaults to `source`.
            doc_id=doc_id if doc_id is not None else source,
            element_positions=[0],
            element_types=["title"],
            elements=elements or [],
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


def test_index_chunks_with_doc_metadata_merges_it_into_the_source(
    store: ElasticsearchStore,
) -> None:
    metadata = DocumentMetadata(classification="c2", author="Alice", tags=["policy"])
    store.index_chunks([_chunk("doc.md::a::0", "chunk one")], metadata)

    doc = store._client.get(index=store._index_name, id="doc.md::a::0")
    assert doc["_source"]["classification"] == "c2"
    assert doc["_source"]["author"] == "Alice"
    assert doc["_source"]["tags"] == ["policy"]


def test_index_chunks_without_doc_metadata_writes_no_metadata_fields(
    store: ElasticsearchStore,
) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "chunk one")])

    doc = store._client.get(index=store._index_name, id="doc.md::a::0")
    assert "classification" not in doc["_source"]


def test_set_document_metadata_patches_existing_chunks_without_touching_others(
    store: ElasticsearchStore,
) -> None:
    store.index_chunks(
        [
            _chunk("a.md::x::0", "doc a chunk", doc_id="a"),
            _chunk("b.md::x::0", "doc b chunk", doc_id="b"),
        ]
    )

    store.set_document_metadata("a", {"classification": "c3", "tags": ["urgent"]})

    doc_a = store._client.get(index=store._index_name, id="a.md::x::0")
    assert doc_a["_source"]["classification"] == "c3"
    assert doc_a["_source"]["tags"] == ["urgent"]
    doc_b = store._client.get(index=store._index_name, id="b.md::x::0")
    assert "classification" not in doc_b["_source"]


def test_set_document_metadata_is_a_no_op_for_empty_fields(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "chunk one")])
    store.set_document_metadata("doc.md", {})  # must not raise

    doc = store._client.get(index=store._index_name, id="doc.md::a::0")
    assert "classification" not in doc["_source"]


def test_doc_id_is_independent_of_the_display_filename(store: ElasticsearchStore) -> None:
    # Same identity bug as test_qdrant_store.py's equivalent test -- doc_id
    # used to be silently overwritten with the filename ("source") on
    # write. Prove they're tracked as two distinct fields end to end.
    chunk = _chunk("real-doc.md::a::0", "content", source="real-doc.md", doc_id="sha256-abc123")
    store.index_chunks([chunk])

    results = store.search("content", top_k=1)
    assert results[0].source == "real-doc.md"
    assert results[0].doc_id == "sha256-abc123"


def test_search_populates_lineage_fields_from_the_source(store: ElasticsearchStore) -> None:
    metadata = DocumentMetadata(
        classification="public", doc_family_id="fam-1", version=2, effective_from="2026-01-01"
    )
    store.index_chunks([_chunk("doc.md::a::0", "chunk one")], metadata)

    result = store.search("chunk", top_k=1)[0]

    assert (result.version, result.doc_family_id, result.effective_from, result.status) == (
        2,
        "fam-1",
        "2026-01-01",
        "current",
    )


def test_search_defaults_lineage_fields_for_a_chunk_with_none(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "chunk one")])

    result = store.search("chunk", top_k=1)[0]

    assert (result.version, result.doc_family_id, result.effective_from, result.status) == (
        1,
        None,
        None,
        "current",
    )


def test_search_filters_by_date_range(store: ElasticsearchStore) -> None:
    old = DocumentMetadata(classification="public", doc_date="2023-01-01")
    new = DocumentMetadata(classification="public", doc_date="2025-01-01")
    store.index_chunks(
        [_chunk("old.md::a::0", "shared wording", source="old.md", doc_id="old")], old
    )
    store.index_chunks(
        [_chunk("new.md::a::0", "shared wording", source="new.md", doc_id="new")], new
    )

    results = store.search(
        "shared wording",
        top_k=10,
        search_filter=SearchFilter(date_range={"doc_date": ("2024-01-01", "2025-12-31")}),
    )

    assert [r.chunk_id for r in results] == ["new.md::a::0"]


def test_search_filters_by_date_range_inclusive_of_the_boundary_date(
    store: ElasticsearchStore,
) -> None:
    metadata = DocumentMetadata(classification="public", doc_date="2024-12-31")
    store.index_chunks([_chunk("doc.md::a::0", "content")], metadata)

    results = store.search(
        "content",
        top_k=10,
        search_filter=SearchFilter(date_range={"doc_date": ("2024-01-01", "2024-12-31")}),
    )

    assert [r.chunk_id for r in results] == ["doc.md::a::0"]


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


def test_elements_round_trip_through_search(store: ElasticsearchStore) -> None:
    elements = [
        ChunkElement(type="title", text="Section A"),
        ChunkElement(type="table", text="| A | B |\n| --- | --- |\n| 1 | 2 |"),
        ChunkElement(type="image", image_base64="aGVsbG8=", description="A photo."),
    ]
    store.index_chunks([_chunk("doc.md::a::0", "stored content", elements=elements)])

    results = store.search("stored content", top_k=1)

    assert results[0].elements == elements


def test_elements_defaults_to_empty_list_when_absent(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "stored content")])
    results = store.search("stored content", top_k=1)
    assert results[0].elements == []


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
