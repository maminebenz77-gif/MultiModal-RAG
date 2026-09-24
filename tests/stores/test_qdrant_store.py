"""Integration tests against the real local Qdrant instance (see
docker-compose.yml) rather than a mocked client — a vector store is
mostly a thin wrapper around real network calls, so mocking the client
would mostly test the mock. Uses a dedicated test collection (alias),
torn down after each test.
"""

from collections.abc import Iterator
from typing import Any

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkElement, ChunkMetadata
from multimodal_rag.metadata import DocumentMetadata
from multimodal_rag.providers.schema import EmbeddingVector
from multimodal_rag.stores.filters import SearchFilter
from multimodal_rag.stores.qdrant_store import (
    _PAYLOAD_INDEXES,
    ModelMismatchError,
    QdrantStore,
    UpsertBatchError,
)

_COLLECTION = "test_collection"


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
            # Defaults to `source` so every pre-existing call site (which
            # only ever set `source`) keeps asserting doc_id == "doc.md"
            # unchanged; tests that care about doc_id vs. source being
            # DIFFERENT (the identity bug this field exists to fix) pass
            # it explicitly.
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


def _indexed_fields(store: QdrantStore, collection_name: str) -> set[str]:
    return set(store._client.get_collection(collection_name).payload_schema or {})


def test_create_collection_indexes_payload_fields_before_publish(
    store: QdrantStore,
) -> None:
    # The indexes must exist on the PENDING collection, before the alias ever
    # points at it -- otherwise a freshly published version is briefly
    # searchable but unindexed.
    store.create_collection(dimension=4, indexing_threshold=0)
    pending = store._pending_collection
    assert pending is not None

    assert _PAYLOAD_INDEXES.keys() <= _indexed_fields(store, pending)


def test_ensure_ready_indexes_an_already_live_collection(store: QdrantStore) -> None:
    # Simulate a collection created before payload indexing existed: drop the
    # indexes from the live version, then let startup heal it. Without this,
    # an existing deployment would need a full rebuild to gain an index.
    physical = store._current_alias_target()
    assert physical is not None
    for field_name in _PAYLOAD_INDEXES:
        store._client.delete_payload_index(collection_name=physical, field_name=field_name)
    assert _indexed_fields(store, physical) == set()

    store.ensure_ready(dimension=4)

    assert _PAYLOAD_INDEXES.keys() <= _indexed_fields(store, physical)
    # Healing must not have rebuilt anything -- same physical collection.
    assert store._current_alias_target() == physical


def test_ensure_ready_survives_a_payload_index_failure(
    store: QdrantStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ensure_ready() runs at service startup. An index that conflicts with an
    # old collection's schema must degrade filter performance, not take the
    # whole API down at boot.
    def boom(**kwargs: Any) -> None:
        raise RuntimeError("index schema conflict")

    monkeypatch.setattr(store._client, "create_payload_index", boom)

    store.ensure_ready(dimension=4)  # must not raise


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


def test_upsert_with_doc_metadata_merges_it_into_every_chunks_payload(
    store: QdrantStore,
) -> None:
    metadata = DocumentMetadata(classification="c2", author="Alice", tags=["policy"])
    chunks = [_chunk("doc.md::a::0", "chunk one"), _chunk("doc.md::a::1", "chunk two")]
    store.upsert(chunks, [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])], metadata)

    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=2)
    for result in results:
        fetched = store.get_by_chunk_id(result.chunk_id)
        assert fetched is not None
    # Fetch raw payloads via search's own top-level fields isn't enough
    # for classification/tags (SearchResult doesn't surface them yet) --
    # scroll the client directly to check what actually got stored.
    points, _ = store._client.scroll(collection_name=store._alias, limit=10, with_payload=True)
    for point in points:
        assert point.payload is not None
        assert point.payload["classification"] == "c2"
        assert point.payload["author"] == "Alice"
        assert point.payload["tags"] == ["policy"]


def test_upsert_without_doc_metadata_writes_no_metadata_fields(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "chunk one")], [_vector([1.0, 0.0, 0.0, 0.0])])

    points, _ = store._client.scroll(collection_name=store._alias, limit=10, with_payload=True)
    assert points[0].payload is not None
    assert "classification" not in points[0].payload


def test_set_document_metadata_patches_existing_chunks_without_touching_others(
    store: QdrantStore,
) -> None:
    store.upsert(
        [
            _chunk("a.md::x::0", "doc a chunk", doc_id="a"),
            _chunk("b.md::x::0", "doc b chunk", doc_id="b"),
        ],
        [_vector([1.0, 0.0, 0.0, 0.0]), _vector([0.0, 1.0, 0.0, 0.0])],
    )

    store.set_document_metadata("a", {"classification": "c3", "tags": ["urgent"]})

    fetched_a = store.get_by_chunk_id("a.md::x::0")
    assert fetched_a is not None
    points, _ = store._client.scroll(collection_name=store._alias, limit=10, with_payload=True)
    by_chunk_id = {p.payload["chunk_id"]: p.payload for p in points if p.payload is not None}
    assert by_chunk_id["a.md::x::0"]["classification"] == "c3"
    assert by_chunk_id["a.md::x::0"]["tags"] == ["urgent"]
    # Doc b's chunk was never touched by the filter -- no metadata keys
    # leaked onto it.
    assert "classification" not in by_chunk_id["b.md::x::0"]


def test_set_document_metadata_is_a_no_op_for_empty_fields(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "chunk one")], [_vector([1.0, 0.0, 0.0, 0.0])])
    store.set_document_metadata("doc.md", {})  # must not raise

    points, _ = store._client.scroll(collection_name=store._alias, limit=10, with_payload=True)
    assert "classification" not in points[0].payload


def test_doc_id_is_independent_of_the_display_filename(store: QdrantStore) -> None:
    # This is the identity bug (Phase 1): doc_id used to be silently
    # overwritten with the filename ("source") on write. Prove they're
    # tracked as two distinct payload fields end to end.
    chunk = _chunk("real-doc.md::a::0", "content", source="real-doc.md", doc_id="sha256-abc123")
    store.upsert([chunk], [_vector([1.0, 0.0, 0.0, 0.0])])

    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].source == "real-doc.md"
    assert results[0].doc_id == "sha256-abc123"

    fetched = store.get_by_chunk_id("real-doc.md::a::0")
    assert fetched is not None
    assert fetched.metadata.source_file == "real-doc.md"
    assert fetched.metadata.doc_id == "sha256-abc123"


def test_search_populates_lineage_fields_from_the_payload(store: QdrantStore) -> None:
    metadata = DocumentMetadata(
        classification="public", doc_family_id="fam-1", version=2, effective_from="2026-01-01"
    )
    store.upsert(
        [_chunk("doc.md::a::0", "chunk")], [_vector([1.0, 0.0, 0.0, 0.0])], doc_metadata=metadata
    )

    result = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)[0]

    assert (result.version, result.doc_family_id, result.effective_from, result.status) == (
        2,
        "fam-1",
        "2026-01-01",
        "current",
    )


def test_search_defaults_lineage_fields_for_a_chunk_with_none(store: QdrantStore) -> None:
    """A chunk written with no doc_metadata (most tests, the demo scripts) --
    must round-trip as "a current first version", not raise or come back
    None where a caller expects an int/str."""
    store.upsert([_chunk("doc.md::a::0", "chunk")], [_vector([1.0, 0.0, 0.0, 0.0])])

    result = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)[0]

    assert (result.version, result.doc_family_id, result.effective_from, result.status) == (
        1,
        None,
        None,
        "current",
    )


def test_search_filters_by_date_range(store: QdrantStore) -> None:
    old = DocumentMetadata(classification="public", doc_date="2023-01-01")
    new = DocumentMetadata(classification="public", doc_date="2025-01-01")
    store.upsert(
        [_chunk("old.md::a::0", "shared wording", source="old.md", doc_id="old")],
        [_vector([1.0, 0.0, 0.0, 0.0])],
        doc_metadata=old,
    )
    store.upsert(
        [_chunk("new.md::a::0", "shared wording", source="new.md", doc_id="new")],
        [_vector([1.0, 0.0, 0.0, 0.0])],
        doc_metadata=new,
    )

    results = store.search(
        _vector([1.0, 0.0, 0.0, 0.0]),
        top_k=10,
        search_filter=SearchFilter(date_range={"doc_date": ("2024-01-01", "2025-12-31")}),
    )

    assert [r.chunk_id for r in results] == ["new.md::a::0"]


def test_search_filters_by_date_range_inclusive_of_the_boundary_date(
    store: QdrantStore,
) -> None:
    metadata = DocumentMetadata(classification="public", doc_date="2024-12-31")
    store.upsert(
        [_chunk("doc.md::a::0", "content")], [_vector([1.0, 0.0, 0.0, 0.0])], doc_metadata=metadata
    )

    results = store.search(
        _vector([1.0, 0.0, 0.0, 0.0]),
        top_k=10,
        search_filter=SearchFilter(date_range={"doc_date": ("2024-01-01", "2024-12-31")}),
    )

    assert [r.chunk_id for r in results] == ["doc.md::a::0"]


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


def test_pages_round_trip_through_search(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "stored", pages=[2, 3])], [_vector([1.0, 0.0, 0.0, 0.0])])
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].pages == [2, 3]


def test_parent_id_round_trips_through_search(store: QdrantStore) -> None:
    store.upsert(
        [_chunk("doc.md::child::0", "stored", parent_id="doc.md::parent::0")],
        [_vector([1.0, 0.0, 0.0, 0.0])],
    )
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].parent_id == "doc.md::parent::0"


def test_parent_id_is_none_when_chunk_has_no_parent(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0])])
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].parent_id is None


def test_elements_round_trip_through_search(store: QdrantStore) -> None:
    elements = [
        ChunkElement(type="title", text="Section A"),
        ChunkElement(type="table", text="| A | B |\n| --- | --- |\n| 1 | 2 |"),
        ChunkElement(type="image", image_base64="aGVsbG8=", description="A photo."),
    ]
    store.upsert(
        [_chunk("doc.md::a::0", "stored", elements=elements)], [_vector([1.0, 0.0, 0.0, 0.0])]
    )

    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)

    assert results[0].elements == elements


def test_elements_defaults_to_empty_list_when_absent(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "stored")], [_vector([1.0, 0.0, 0.0, 0.0])])
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].elements == []


def test_is_parent_chunks_are_excluded_from_search(store: QdrantStore) -> None:
    store.upsert(
        [
            _chunk("doc.md::parent::0", "parent text", is_parent=True),
            _chunk("doc.md::child::0", "child text", parent_id="doc.md::parent::0"),
        ],
        [_vector([1.0, 0.0, 0.0, 0.0]), _vector([1.0, 0.0, 0.0, 0.0])],
    )
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::child::0"]


def test_is_parent_chunks_are_still_fetchable_by_id(store: QdrantStore) -> None:
    store.upsert(
        [_chunk("doc.md::parent::0", "parent text", is_parent=True)],
        [_vector([1.0, 0.0, 0.0, 0.0])],
    )
    fetched = store.get_by_chunk_id("doc.md::parent::0")
    assert fetched is not None
    assert fetched.is_parent is True
    assert fetched.text == "parent text"


def test_get_by_chunk_id_returns_the_matching_chunk(store: QdrantStore) -> None:
    store.upsert(
        [_chunk("doc.md::parent::0", "parent text", pages=[1, 2])],
        [_vector([1.0, 0.0, 0.0, 0.0])],
    )
    fetched = store.get_by_chunk_id("doc.md::parent::0")
    assert fetched is not None
    assert fetched.id == "doc.md::parent::0"
    assert fetched.text == "parent text"
    assert fetched.metadata.pages == [1, 2]


def test_get_by_chunk_id_includes_elements(store: QdrantStore) -> None:
    elements = [ChunkElement(type="title", text="Section A")]
    store.upsert(
        [_chunk("doc.md::parent::0", "parent text", elements=elements)],
        [_vector([1.0, 0.0, 0.0, 0.0])],
    )
    fetched = store.get_by_chunk_id("doc.md::parent::0")
    assert fetched is not None
    assert fetched.metadata.elements == elements


def test_get_by_chunk_id_returns_none_for_unknown_id(store: QdrantStore) -> None:
    assert store.get_by_chunk_id("nonexistent") is None


def test_delete_chunks_removes_them_from_search(store: QdrantStore) -> None:
    store.upsert(
        [_chunk("doc.md::a::0", "keep me"), _chunk("doc.md::b::0", "delete me")],
        [_vector([1.0, 0.0, 0.0, 0.0]), _vector([1.0, 0.0, 0.0, 0.0])],
    )
    store.delete_chunks(["doc.md::b::0"])

    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=10)
    assert [r.chunk_id for r in results] == ["doc.md::a::0"]


def test_delete_chunks_is_a_no_op_for_unknown_ids(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "text")], [_vector([1.0, 0.0, 0.0, 0.0])])
    store.delete_chunks(["nonexistent"])  # should not raise
    assert len(store.list_chunk_ids()) == 1


def test_delete_chunks_with_empty_list_is_a_no_op(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "text")], [_vector([1.0, 0.0, 0.0, 0.0])])
    store.delete_chunks([])
    assert len(store.list_chunk_ids()) == 1


def test_ping_returns_true_when_reachable(store: QdrantStore) -> None:
    assert store.ping() is True


def test_ping_returns_false_when_unreachable() -> None:
    unreachable = QdrantStore(url="http://localhost:1", collection_name="whatever")
    assert unreachable.ping() is False


def test_ensure_ready_creates_a_live_collection_when_none_exists() -> None:
    fresh = QdrantStore(url="http://localhost:6333", collection_name="test_ensure_ready")
    try:
        assert fresh._current_alias_target() is None
        fresh.ensure_ready(dimension=4)
        assert fresh._current_alias_target() is not None
        fresh.upsert([_chunk("doc.md::a::0", "text")], [_vector([1.0, 0.0, 0.0, 0.0])])
        results = fresh.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
        assert results[0].chunk_id == "doc.md::a::0"
    finally:
        physical = fresh._current_alias_target()
        if physical is not None:
            fresh._client.delete_collection(physical)


def test_ensure_ready_is_a_no_op_when_a_collection_already_exists(store: QdrantStore) -> None:
    store.upsert([_chunk("doc.md::a::0", "text")], [_vector([1.0, 0.0, 0.0, 0.0])])
    live_before = store._current_alias_target()

    store.ensure_ready(dimension=4)

    assert store._current_alias_target() == live_before
    results = store.search(_vector([1.0, 0.0, 0.0, 0.0]), top_k=1)
    assert results[0].chunk_id == "doc.md::a::0"
