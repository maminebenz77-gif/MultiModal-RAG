"""Integration tests against the real local Elasticsearch instance (see
docker-compose.yml) rather than a mocked client — a store is mostly a
thin wrapper around real network calls, so mocking the client would
mostly test the mock.

ElasticsearchStore (keyword role) and ElasticsearchVectorStore (vector
role) share one physical index/alias — see elasticsearch_store.py's
module docstring for why they're two classes. The `store` fixture
below sets up both roles against the same index, mirroring how every
real call site (api/main.py's lifespan, generation/demo.py, ...)
always calls the vector role's ensure_ready(dimension)/create_collection()
before the keyword role's ensure_ready()/create_index().
"""

from collections.abc import Iterator
from typing import Any

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkElement, ChunkMetadata
from multimodal_rag.metadata import DocumentMetadata
from multimodal_rag.providers.schema import EmbeddingVector
from multimodal_rag.stores.elasticsearch_store import (
    ElasticsearchStore,
    ElasticsearchVectorStore,
    ModelMismatchError,
    UpsertBatchError,
)
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
            # Defaults to `source` so a test that doesn't care about the
            # doc_id/filename distinction doesn't have to spell it out.
            doc_id=doc_id if doc_id is not None else source,
            element_positions=[0],
            element_types=["title"],
            elements=elements or [],
            pages=pages or [],
        ),
    )


def _vector(values: list[float], model_id: str = "test-model") -> EmbeddingVector:
    return EmbeddingVector(vector=values, model_id=model_id, dimension=len(values))


@pytest.fixture
def store() -> Iterator[ElasticsearchStore]:
    """The keyword role, paired with a vector role sharing the same
    index -- see this module's docstring for why ensure_ready() has to
    run through the vector role first."""
    kw = ElasticsearchStore(url="http://localhost:9200", index_name=_INDEX)
    ElasticsearchVectorStore(kw).ensure_ready(dimension=4)
    yield kw
    physical = kw._current_alias_target()
    if physical is not None:
        kw._client.indices.delete(index=physical, ignore_unavailable=True)


@pytest.fixture
def vector_store(store: ElasticsearchStore) -> ElasticsearchVectorStore:
    return ElasticsearchVectorStore(store)


# ------------------------------------------------------------- shared setup


def test_create_index_is_a_no_op_once_the_shared_index_exists(store: ElasticsearchStore) -> None:
    store.create_index()
    store.create_index()


def test_create_index_raises_without_a_paired_vector_role_first() -> None:
    kw = ElasticsearchStore(url="http://localhost:9200", index_name="test_no_vector_role")
    with pytest.raises(RuntimeError, match="create_collection"):
        kw.create_index()


def test_ensure_ready_raises_without_a_paired_vector_role_first() -> None:
    kw = ElasticsearchStore(url="http://localhost:9200", index_name="test_no_vector_role")
    with pytest.raises(RuntimeError, match="paired VectorStore"):
        kw.ensure_ready()


def test_ensure_ready_does_not_wipe_an_existing_index(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "stored content")])

    store.ensure_ready()

    results = store.search("stored content", top_k=1)
    assert len(results) == 1


# ----------------------------------------------- merge safety (both roles)


def test_vector_and_keyword_writes_merge_rather_than_clobber_each_other(
    store: ElasticsearchStore, vector_store: ElasticsearchVectorStore
) -> None:
    """The whole reason both roles write via `update`+`doc_as_upsert`
    instead of a plain `index` replace: a document written by one role
    must not lose the other role's fields. Confirmed live during
    development; pinned down here as a real regression test."""
    chunk = _chunk("doc.md::a::0", "vacation policy")
    vector_store.upsert([chunk], [_vector([1.0, 0.0, 0.0, 0.0])])
    store.index_chunks([chunk])

    keyword_hit = store.search("vacation policy", top_k=1)[0]
    vector_hit = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1, with_vectors=True)[0]
    assert keyword_hit.chunk_id == vector_hit.chunk_id == "doc.md::a::0"
    assert vector_hit.vector == [1.0, 0.0, 0.0, 0.0]
    assert keyword_hit.text == "vacation policy"


def test_writing_keyword_role_first_then_vector_role_also_merges(
    store: ElasticsearchStore, vector_store: ElasticsearchVectorStore
) -> None:
    # Order shouldn't matter -- HybridIndexer.index() always calls
    # upsert() then index_chunks(), but nothing about the merge
    # mechanism itself depends on that order.
    chunk = _chunk("doc.md::a::0", "warranty terms")
    store.index_chunks([chunk])
    vector_store.upsert([chunk], [_vector([0.0, 1.0, 0.0, 0.0])])

    result = vector_store.search(_vector([0.0, 1.0, 0.0, 0.0]), top_k=1, with_vectors=True)[0]
    assert result.text == "warranty terms"
    assert result.vector == [0.0, 1.0, 0.0, 0.0]


# ------------------------------------------------------ keyword role (BM25)


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

    doc = store._client.get(index=store._alias, id="doc.md::a::0")
    assert doc["_source"]["classification"] == "c2"
    assert doc["_source"]["author"] == "Alice"
    assert doc["_source"]["tags"] == ["policy"]


def test_index_chunks_without_doc_metadata_writes_no_metadata_fields(
    store: ElasticsearchStore,
) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "chunk one")])

    doc = store._client.get(index=store._alias, id="doc.md::a::0")
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

    doc_a = store._client.get(index=store._alias, id="a.md::x::0")
    assert doc_a["_source"]["classification"] == "c3"
    assert doc_a["_source"]["tags"] == ["urgent"]
    doc_b = store._client.get(index=store._alias, id="b.md::x::0")
    assert "classification" not in doc_b["_source"]


def test_set_document_metadata_is_a_no_op_for_empty_fields(store: ElasticsearchStore) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "chunk one")])
    store.set_document_metadata("doc.md", {})  # must not raise

    doc = store._client.get(index=store._alias, id="doc.md::a::0")
    assert "classification" not in doc["_source"]


def test_doc_id_is_independent_of_the_display_filename(store: ElasticsearchStore) -> None:
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


def test_search_populates_tags_author_date_and_privacy_from_the_source(
    store: ElasticsearchStore,
) -> None:
    metadata = DocumentMetadata(
        classification="c2",
        private=True,
        author="Alice",
        doc_date="2026-01-01",
        tags=["runbook", "q1"],
    )
    store.index_chunks([_chunk("doc.md::a::0", "chunk one")], metadata)

    result = store.search("chunk", top_k=1)[0]

    assert result.tags == ["runbook", "q1"]
    assert result.author == "Alice"
    assert result.doc_date == "2026-01-01"
    assert result.classification == "c2"
    assert result.private is True


def test_search_defaults_tags_author_date_and_privacy_for_a_chunk_with_none(
    store: ElasticsearchStore,
) -> None:
    store.index_chunks([_chunk("doc.md::a::0", "chunk one")])

    result = store.search("chunk", top_k=1)[0]

    assert result.tags == []
    assert result.author is None
    assert result.doc_date is None
    assert result.classification == "public"
    assert result.private is False


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
    """A whole-round failure (e.g. the connection dropping mid-request,
    not one rejected document) is retried the same way
    retry_with_backoff would retry any other operation -- see
    _bulk_with_partial_failure's docstring."""
    attempts = {"count": 0}
    from elasticsearch.helpers import streaming_bulk as real_streaming_bulk

    def flaky_streaming_bulk(client: Any, actions: Any, **kwargs: Any) -> Any:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient network blip")
        yield from real_streaming_bulk(client, actions, **kwargs)

    monkeypatch.setattr(
        "multimodal_rag.stores.elasticsearch_store.streaming_bulk", flaky_streaming_bulk
    )

    store.index_chunks([_chunk("doc.md::a::0", "hello")])

    assert attempts["count"] == 3
    assert store.list_chunk_ids() == ["doc.md::a::0"]


def test_index_chunks_persistent_item_failure_raises_but_preserves_others(
    store: ElasticsearchStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One chunk that never succeeds, no matter how many rounds are
    tried, must not take the other, perfectly good chunk down with it --
    the whole reason for tracking success/failure per chunk_id instead
    of per call."""
    from elasticsearch.helpers import streaming_bulk as real_streaming_bulk

    def one_chunk_always_rejected(client: Any, actions: Any, **kwargs: Any) -> Any:
        # The "bad" chunk is never actually sent to Elasticsearch -- a
        # real persistent failure means the write never lands, not that
        # it lands and we just choose to report it as failed.
        good = [a for a in actions if a["_id"] != "doc.md::bad::0"]
        if good:
            yield from real_streaming_bulk(client, good, **kwargs)
        if any(a["_id"] == "doc.md::bad::0" for a in actions):
            yield False, {
                "update": {"_id": "doc.md::bad::0", "status": 400, "error": "simulated"}
            }

    monkeypatch.setattr(
        "multimodal_rag.stores.elasticsearch_store.streaming_bulk", one_chunk_always_rejected
    )

    chunks = [_chunk("doc.md::good::0", "keep me"), _chunk("doc.md::bad::0", "always fails")]

    with pytest.raises(UpsertBatchError) as exc_info:
        store.index_chunks(chunks)

    assert exc_info.value.succeeded_chunk_ids == ["doc.md::good::0"]
    assert exc_info.value.failed_chunk_ids == ["doc.md::bad::0"]
    # The good chunk is still searchable, refreshed, despite the other
    # chunk failing every round -- a persistent per-item failure must
    # never unwind already-succeeded work.
    assert store.list_chunk_ids() == ["doc.md::good::0"]


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


# ------------------------------------------------ vector role (kNN, blue-green)


def test_create_collection_rejects_unknown_distance(vector_store: ElasticsearchVectorStore) -> None:
    with pytest.raises(ValueError, match="Unknown distance"):
        vector_store.create_collection(dimension=4, distance="manhattan")


def test_publish_without_pending_collection_raises(vector_store: ElasticsearchVectorStore) -> None:
    with pytest.raises(RuntimeError, match="No pending index"):
        vector_store.publish()


def test_create_collection_does_not_affect_a_live_published_version(
    store: ElasticsearchStore, vector_store: ElasticsearchVectorStore
) -> None:
    # store/vector_store fixtures already published an (empty) v1. Populate it.
    vector_store.upsert([_chunk("doc.md::a::0", "v1 data")], [_vector([1.0, 0.0, 0.0, 0.0])])

    # Start building v2 but do NOT publish it yet.
    vector_store.create_collection(dimension=4)
    vector_store.upsert([_chunk("doc.md::b::0", "v2 data")], [_vector([0.0, 1.0, 0.0, 0.0])])

    # search() still goes through the alias, which still points at v1.
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::a::0"]


def test_publish_swaps_atomically_and_removes_the_previous_version(
    store: ElasticsearchStore, vector_store: ElasticsearchVectorStore
) -> None:
    vector_store.upsert([_chunk("doc.md::a::0", "v1 data")], [_vector([1.0, 0.0, 0.0, 0.0])])
    old_physical = store._current_alias_target()
    assert old_physical is not None

    vector_store.create_collection(dimension=4)
    vector_store.upsert([_chunk("doc.md::b::0", "v2 data")], [_vector([0.0, 1.0, 0.0, 0.0])])
    vector_store.publish()

    results = vector_store.search(_vector([0.0, 1.0, 0.0, 0.0]), top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::b::0"]
    assert not store._client.indices.exists(index=old_physical)


def test_upsert_and_search_roundtrip(vector_store: ElasticsearchVectorStore) -> None:
    chunks = [_chunk("doc.md::a::0", "about GPUs"), _chunk("doc.md::a::1", "about soup recipes")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])]
    vector_store.upsert(chunks, vectors)

    results = vector_store.search(_vector([0.9, 0.1, 0.0, 0.0]), top_k=2)

    assert results[0].chunk_id == "doc.md::a::0"
    assert results[0].text == "about GPUs"
    assert results[0].source == "doc.md"
    assert results[0].doc_id == "doc.md"
    assert results[0].element_types == ["title"]
    assert results[0].model_id == "test-model"
    assert results[0].score > results[1].score


def test_upsert_with_doc_metadata_merges_it_into_every_chunks_document(
    store: ElasticsearchStore, vector_store: ElasticsearchVectorStore
) -> None:
    metadata = DocumentMetadata(classification="c2", author="Alice", tags=["policy"])
    chunks = [_chunk("doc.md::a::0", "chunk one"), _chunk("doc.md::a::1", "chunk two")]
    vector_store.upsert(
        chunks, [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])], metadata
    )

    for chunk in chunks:
        doc = store._client.get(index=store._alias, id=chunk.id)
        assert doc["_source"]["classification"] == "c2"
        assert doc["_source"]["author"] == "Alice"
        assert doc["_source"]["tags"] == ["policy"]


def test_upsert_without_doc_metadata_writes_no_metadata_fields(
    store: ElasticsearchStore, vector_store: ElasticsearchVectorStore
) -> None:
    vector_store.upsert([_chunk("doc.md::a::0", "chunk one")], [_vector([1.0, 0.0, 0.0, 0.0])])

    doc = store._client.get(index=store._alias, id="doc.md::a::0")
    assert "classification" not in doc["_source"]


def test_search_rejects_a_query_vector_from_a_different_model(
    vector_store: ElasticsearchVectorStore,
) -> None:
    vector_store.upsert(
        [_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0], "model-a")]
    )

    with pytest.raises(ModelMismatchError, match="model-a.*model-b|model-b.*model-a"):
        vector_store.search(_vector([1.0, 0.0, 0.0, 0.0], "model-b"), top_k=1)


def test_search_allows_a_matching_model(vector_store: ElasticsearchVectorStore) -> None:
    vector_store.upsert(
        [_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0], "model-a")]
    )

    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0], "model-a"), top_k=1)
    assert len(results) == 1


def test_search_on_empty_collection_does_not_raise_model_mismatch(
    vector_store: ElasticsearchVectorStore,
) -> None:
    # store/vector_store fixtures publish an empty index -- nothing to
    # compare the query vector's model against yet, so nothing should be
    # rejected.
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0], "any-model"), top_k=1)
    assert results == []


def test_search_omits_vector_by_default(vector_store: ElasticsearchVectorStore) -> None:
    vector_store.upsert([_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0])])
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].vector is None


def test_search_with_vectors_populates_vector(vector_store: ElasticsearchVectorStore) -> None:
    vector_store.upsert([_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0])])
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1, with_vectors=True)
    assert results[0].vector == [1.0, 0.0, 0.0, 0.0]


def test_upsert_rejects_mixed_models(vector_store: ElasticsearchVectorStore) -> None:
    chunks = [_chunk("doc.md::a::0", "a"), _chunk("doc.md::a::1", "b")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0], "model-a"), _vector([0.0, 1.0, 0.0, 0.0], "model-b")]
    with pytest.raises(ValueError, match="mix vectors"):
        vector_store.upsert(chunks, vectors)


def test_upsert_rejects_mismatched_lengths(vector_store: ElasticsearchVectorStore) -> None:
    chunks = [_chunk("doc.md::a::0", "a")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])]
    with pytest.raises(ValueError, match="same length"):
        vector_store.upsert(chunks, vectors)


def test_reupserting_same_chunk_id_updates_rather_than_duplicates(
    vector_store: ElasticsearchVectorStore,
) -> None:
    chunk = _chunk("doc.md::a::0", "original text")
    vector_store.upsert([chunk], [_vector([1.0, 0.0, 0.0, 0.0])])

    updated = _chunk("doc.md::a::0", "updated text")
    vector_store.upsert([updated], [_vector([1.0, 0.0, 0.0, 0.0])])

    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert len(results) == 1
    assert results[0].text == "updated text"


def test_search_accepts_explicit_ef_search(vector_store: ElasticsearchVectorStore) -> None:
    chunks = [_chunk("doc.md::a::0", "a"), _chunk("doc.md::a::1", "b")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])]
    vector_store.upsert(chunks, vectors)

    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1, ef_search=64)
    assert len(results) == 1


def test_upsert_retries_transient_failure_then_succeeds(
    vector_store: ElasticsearchVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = {"count": 0}
    from elasticsearch.helpers import streaming_bulk as real_streaming_bulk

    def flaky_streaming_bulk(client: Any, actions: Any, **kwargs: Any) -> Any:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise RuntimeError("transient network blip")
        yield from real_streaming_bulk(client, actions, **kwargs)

    monkeypatch.setattr(
        "multimodal_rag.stores.elasticsearch_store.streaming_bulk", flaky_streaming_bulk
    )

    vector_store.upsert([_chunk("doc.md::a::0", "hello")], [_vector([1.0, 0.0, 0.0, 0.0])])

    assert attempts["count"] == 3
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].chunk_id == "doc.md::a::0"


def test_upsert_persistent_item_failure_raises_but_preserves_others(
    vector_store: ElasticsearchVectorStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from elasticsearch.helpers import streaming_bulk as real_streaming_bulk

    def one_chunk_always_rejected(client: Any, actions: Any, **kwargs: Any) -> Any:
        good = [a for a in actions if a["_id"] != "doc.md::bad::0"]
        if good:
            yield from real_streaming_bulk(client, good, **kwargs)
        if any(a["_id"] == "doc.md::bad::0" for a in actions):
            yield False, {
                "update": {"_id": "doc.md::bad::0", "status": 400, "error": "simulated"}
            }

    monkeypatch.setattr(
        "multimodal_rag.stores.elasticsearch_store.streaming_bulk", one_chunk_always_rejected
    )

    chunks = [_chunk("doc.md::good::0", "keep me"), _chunk("doc.md::bad::0", "always fails")]
    vectors = [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])]

    with pytest.raises(UpsertBatchError) as exc_info:
        vector_store.upsert(chunks, vectors)

    assert exc_info.value.succeeded_chunk_ids == ["doc.md::good::0"]
    assert exc_info.value.failed_chunk_ids == ["doc.md::bad::0"]
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::good::0"]


def test_list_chunk_ids_empty_when_nothing_upserted(vector_store: ElasticsearchVectorStore) -> None:
    assert vector_store.list_chunk_ids() == []


def test_list_chunk_ids_returns_all_upserted_ids(vector_store: ElasticsearchVectorStore) -> None:
    chunks = [_chunk(f"doc.md::a::{i}", f"text {i}") for i in range(5)]
    # float(i) starting at i=0 would give an all-zero vector for the
    # first chunk -- ES's cosine similarity rejects zero-magnitude
    # vectors outright ("does not support vectors with zero magnitude").
    # Never an issue with real embeddings (never all-zero by
    # construction); +1 sidesteps it here.
    vectors = [_vector([float(i) + 1, 0.0, 0.0, 0.0]) for i in range(5)]
    vector_store.upsert(chunks, vectors)

    assert sorted(vector_store.list_chunk_ids()) == sorted(c.id for c in chunks)


def test_pages_round_trip_through_vector_search(vector_store: ElasticsearchVectorStore) -> None:
    vector_store.upsert(
        [_chunk("doc.md::a::0", "stored", pages=[2, 3])], [_vector([1.0, 0.0, 0.0, 0.0])]
    )
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].pages == [2, 3]


def test_parent_id_round_trips_through_vector_search(
    vector_store: ElasticsearchVectorStore,
) -> None:
    vector_store.upsert(
        [_chunk("doc.md::child::0", "stored", parent_id="doc.md::parent::0")],
        [_vector([1.0, 0.0, 0.0, 0.0])],
    )
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].parent_id == "doc.md::parent::0"


def test_elements_round_trip_through_vector_search(
    vector_store: ElasticsearchVectorStore,
) -> None:
    elements = [
        ChunkElement(type="title", text="Section A"),
        ChunkElement(type="table", text="| A | B |\n| --- | --- |\n| 1 | 2 |"),
        ChunkElement(type="image", image_base64="aGVsbG8=", description="A photo."),
    ]
    vector_store.upsert(
        [_chunk("doc.md::a::0", "stored", elements=elements)], [_vector([1.0, 0.0, 0.0, 0.0])]
    )

    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)

    assert results[0].elements == elements


def test_is_parent_chunks_are_excluded_from_vector_search(
    vector_store: ElasticsearchVectorStore,
) -> None:
    vector_store.upsert(
        [
            _chunk("doc.md::parent::0", "parent text", is_parent=True),
            _chunk("doc.md::child::0", "child text", parent_id="doc.md::parent::0"),
        ],
        [_vector([1.0, 0.0, 0.0, 0.0]), _vector([1.0, 0.0, 0.0, 0.0])],
    )
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::child::0"]


def test_get_by_chunk_id_returns_the_matching_chunk(
    vector_store: ElasticsearchVectorStore,
) -> None:
    vector_store.upsert(
        [_chunk("doc.md::parent::0", "parent text", pages=[1, 2])],
        [_vector([1.0, 0.0, 0.0, 0.0])],
    )
    fetched = vector_store.get_by_chunk_id("doc.md::parent::0")
    assert fetched is not None
    assert fetched.id == "doc.md::parent::0"
    assert fetched.text == "parent text"
    assert fetched.metadata.pages == [1, 2]


def test_get_by_chunk_id_includes_elements(vector_store: ElasticsearchVectorStore) -> None:
    elements = [ChunkElement(type="title", text="Section A")]
    vector_store.upsert(
        [_chunk("doc.md::parent::0", "parent text", elements=elements)],
        [_vector([1.0, 0.0, 0.0, 0.0])],
    )
    fetched = vector_store.get_by_chunk_id("doc.md::parent::0")
    assert fetched is not None
    assert fetched.metadata.elements == elements


def test_get_by_chunk_id_returns_none_for_unknown_id(
    vector_store: ElasticsearchVectorStore,
) -> None:
    assert vector_store.get_by_chunk_id("nonexistent") is None


def test_get_by_chunk_id_is_still_fetchable_for_a_parent_chunk(
    vector_store: ElasticsearchVectorStore,
) -> None:
    vector_store.upsert(
        [_chunk("doc.md::parent::0", "parent text", is_parent=True)],
        [_vector([1.0, 0.0, 0.0, 0.0])],
    )
    fetched = vector_store.get_by_chunk_id("doc.md::parent::0")
    assert fetched is not None
    assert fetched.is_parent is True


def test_delete_chunks_removes_them_from_vector_search(
    vector_store: ElasticsearchVectorStore,
) -> None:
    vector_store.upsert(
        [_chunk("doc.md::a::0", "keep me"), _chunk("doc.md::b::0", "delete me")],
        [_vector([1.0, 0.0, 0.0, 0.0]), _vector([1.0, 0.0, 0.0, 0.0])],
    )
    vector_store.delete_chunks(["doc.md::b::0"])

    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::a::0"]


def test_ping_returns_true_when_reachable_on_vector_role(
    vector_store: ElasticsearchVectorStore,
) -> None:
    assert vector_store.ping() is True


def test_ensure_ready_creates_a_live_index_when_none_exists() -> None:
    kw = ElasticsearchStore(url="http://localhost:9200", index_name="test_es_ensure_ready")
    vec = ElasticsearchVectorStore(kw)
    try:
        assert kw._current_alias_target() is None
        vec.ensure_ready(dimension=4)
        assert kw._current_alias_target() is not None
        vec.upsert([_chunk("doc.md::a::0", "text")], [_vector([1.0, 0.0, 0.0, 0.0])])
        results = vec.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert results[0].chunk_id == "doc.md::a::0"
    finally:
        physical = kw._current_alias_target()
        if physical is not None:
            kw._client.indices.delete(index=physical, ignore_unavailable=True)


def test_ensure_ready_is_a_no_op_when_an_index_already_exists(
    store: ElasticsearchStore, vector_store: ElasticsearchVectorStore
) -> None:
    vector_store.upsert([_chunk("doc.md::a::0", "text")], [_vector([1.0, 0.0, 0.0, 0.0])])
    live_before = store._current_alias_target()

    vector_store.ensure_ready(dimension=4)

    assert store._current_alias_target() == live_before
    results = vector_store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].chunk_id == "doc.md::a::0"


def test_reembed_cutover_leaves_old_data_live_until_publish(
    store: ElasticsearchStore, vector_store: ElasticsearchVectorStore
) -> None:
    """Simulates an embedding-model change: build a whole new version
    alongside the live one, confirm both the old model's queries and the
    new model's queries behave correctly before/after the atomic swap."""
    vector_store.upsert(
        [_chunk("doc.md::a::0", "v1 data")], [_vector([1.0, 0.0, 0.0, 0.0], "model-v1")]
    )

    vector_store.create_collection(dimension=4)
    vector_store.upsert(
        [_chunk("doc.md::a::0", "v1 data")], [_vector([0.0, 0.0, 1.0, 0.0], "model-v2")]
    )

    # Still serving the old version until publish.
    assert len(vector_store.search(_vector([1.0, 0.0, 0.0, 0.0], "model-v1"), top_k=5)) == 1

    vector_store.publish()

    with pytest.raises(ModelMismatchError):
        vector_store.search(_vector([1.0, 0.0, 0.0, 0.0], "model-v1"), top_k=5)
    assert len(vector_store.search(_vector([0.0, 0.0, 1.0, 0.0], "model-v2"), top_k=5)) == 1
