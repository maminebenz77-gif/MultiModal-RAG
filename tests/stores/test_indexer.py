"""Integration tests against the real local Elasticsearch.

Both `vector_store` and `keyword_store` below are two role-views onto
the SAME shared index (see elasticsearch_store.py's module docstring) --
this changes what "consistency" between them can actually mean. Before
(Qdrant + Elasticsearch, two real databases), a chunk could genuinely
exist in one and not the other. Now, writing via EITHER role touches
the same document, so `list_chunk_ids()` returns the identical set for
both roles by construction -- there is no longer a way for the two
"stores" to drift apart the old way. That's not a gap in this test
file; it's the actual, intended payoff of merging them (see the
migration plan's Context section: one fewer place metadata/chunks can
disagree). The tests below that used to prove "only in vector store" /
"only in keyword store" scenarios are replaced with one test that
documents this new reality directly, so a future regression (the two
roles somehow NOT sharing state) would be caught immediately.
"""

from collections.abc import Iterator

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkMetadata
from multimodal_rag.providers.schema import EmbeddingVector
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore, ElasticsearchVectorStore
from multimodal_rag.stores.indexer import HybridIndexer, IndexConsistencyError

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
def vector_store() -> Iterator[ElasticsearchVectorStore]:
    kw = ElasticsearchStore(url="http://localhost:9200", index_name=_NAME)
    vec = ElasticsearchVectorStore(kw)
    vec.ensure_ready(dimension=2)
    yield vec
    physical = kw._current_alias_target()
    if physical is not None:
        kw._client.indices.delete(index=physical, ignore_unavailable=True)


@pytest.fixture
def keyword_store(vector_store: ElasticsearchVectorStore) -> ElasticsearchStore:
    # The SAME backend the vector_store fixture wraps -- see this
    # module's docstring for why they must share state, not each get
    # their own.
    return vector_store._store


def test_index_writes_to_both_stores(
    vector_store: ElasticsearchVectorStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    chunks = [_chunk("a", "hello world"), _chunk("b", "goodbye world")]
    vectors = [_vector([1.0, 0.0]), _vector([0.0, 1.0])]

    indexer.index(chunks, vectors)

    assert sorted(vector_store.list_chunk_ids()) == ["a", "b"]
    assert sorted(keyword_store.list_chunk_ids()) == ["a", "b"]


def test_index_raises_index_consistency_error_when_keyword_indexing_fails(
    vector_store: ElasticsearchVectorStore,
    keyword_store: ElasticsearchStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def always_fails(chunks: list[Chunk], doc_metadata: object = None) -> None:
        raise RuntimeError("persistent ES failure")

    monkeypatch.setattr(keyword_store, "index_chunks", always_fails)

    indexer = HybridIndexer(vector_store, keyword_store)
    chunks = [_chunk("a", "hello world")]
    vectors = [_vector([1.0, 0.0])]

    with pytest.raises(IndexConsistencyError) as exc_info:
        indexer.index(chunks, vectors)

    assert exc_info.value.chunk_ids == ["a"]
    # The vector store's write (upsert()) already succeeded before the
    # mocked keyword_store.index_chunks() call raised -- the chunk is
    # durably there, same as before this migration.
    assert vector_store.list_chunk_ids() == ["a"]


def test_delete_removes_from_both_stores(
    vector_store: ElasticsearchVectorStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index(
        [_chunk("a", "hello world"), _chunk("b", "goodbye world")],
        [_vector([1.0, 0.0]), _vector([0.0, 1.0])],
    )

    indexer.delete(["b"])

    assert vector_store.list_chunk_ids() == ["a"]
    assert keyword_store.list_chunk_ids() == ["a"]


def test_delete_all_wipes_both_stores_and_returns_the_count(
    vector_store: ElasticsearchVectorStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index(
        [_chunk("a", "hello world"), _chunk("b", "goodbye world")],
        [_vector([1.0, 0.0]), _vector([0.0, 1.0])],
    )

    deleted = indexer.delete_all()

    assert deleted == 2
    assert vector_store.list_chunk_ids() == []
    assert keyword_store.list_chunk_ids() == []


def test_delete_document_removes_only_that_documents_chunks(
    vector_store: ElasticsearchVectorStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index(
        [
            _chunk("doc-a::structure::0::hash1", "hello world"),
            _chunk("doc-a::structure::1::hash2", "more of doc a"),
            _chunk("doc-b::structure::0::hash3", "a different document"),
        ],
        [_vector([1.0, 0.0]), _vector([0.0, 1.0]), _vector([1.0, 1.0])],
    )

    deleted = indexer.delete_document("doc-a")

    assert deleted == 2
    assert vector_store.list_chunk_ids() == ["doc-b::structure::0::hash3"]
    assert keyword_store.list_chunk_ids() == ["doc-b::structure::0::hash3"]


def test_delete_document_with_no_matching_chunks_is_a_harmless_no_op(
    vector_store: ElasticsearchVectorStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index([_chunk("doc-a::structure::0::hash1", "hello")], [_vector([1.0, 0.0])])

    deleted = indexer.delete_document("doc-nonexistent")

    assert deleted == 0
    assert vector_store.list_chunk_ids() == ["doc-a::structure::0::hash1"]


def test_delete_with_empty_list_is_a_no_op(
    vector_store: ElasticsearchVectorStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index([_chunk("a", "hello")], [_vector([1.0, 0.0])])

    indexer.delete([])

    assert vector_store.list_chunk_ids() == ["a"]


def test_delete_raises_index_consistency_error_when_keyword_deletion_fails(
    vector_store: ElasticsearchVectorStore,
    keyword_store: ElasticsearchStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # ElasticsearchVectorStore.delete_chunks() is a pure delegate to this
    # SAME keyword_store.delete_chunks() method (see
    # elasticsearch_store.py) -- patching the method outright would fail
    # HybridIndexer.delete()'s FIRST call (vector_store's) too, not just
    # the second (keyword_store's) it's meant to simulate. A call-count
    # fake fails only the second invocation, which is what this test
    # actually wants to exercise: HybridIndexer's own try/except around
    # that second call, regardless of whether the two "stores" happen to
    # share an implementation underneath.
    real_delete_chunks = keyword_store.delete_chunks
    calls = {"count": 0}

    def fail_on_second_call(chunk_ids: list[str]) -> None:
        calls["count"] += 1
        if calls["count"] >= 2:
            raise RuntimeError("persistent ES failure")
        real_delete_chunks(chunk_ids)

    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index([_chunk("a", "hello")], [_vector([1.0, 0.0])])

    monkeypatch.setattr(keyword_store, "delete_chunks", fail_on_second_call)

    with pytest.raises(IndexConsistencyError) as exc_info:
        indexer.delete(["a"])

    assert exc_info.value.chunk_ids == ["a"]
    assert calls["count"] == 2
    # The vector store's delete already ran for real (the first,
    # unfaked call) before the second, faked call raised -- the chunk is
    # genuinely gone.
    assert vector_store.list_chunk_ids() == []


def test_check_consistency_reports_no_drift_when_in_sync(
    vector_store: ElasticsearchVectorStore, keyword_store: ElasticsearchStore
) -> None:
    indexer = HybridIndexer(vector_store, keyword_store)
    indexer.index([_chunk("a", "hello")], [_vector([1.0, 0.0])])

    report = indexer.check_consistency()

    assert report.is_consistent
    assert report.only_in_vector_store == []
    assert report.only_in_keyword_store == []


def test_check_consistency_is_trivially_always_consistent_now(
    vector_store: ElasticsearchVectorStore, keyword_store: ElasticsearchStore
) -> None:
    """Documents the real, structural consequence of merging the two
    stores: `check_consistency()` compares `vector_store.list_chunk_ids()`
    against `keyword_store.list_chunk_ids()`, but both roles now read
    the identical underlying index -- even a chunk written through only
    ONE role (here, the keyword role alone, no vector ever upserted)
    shows up in both lists, because there's only one list. The old
    "only in vector store" / "only in keyword store" drift this method
    was built to catch (see stores/indexer.py's module docstring) simply
    cannot happen anymore -- there's no second database left to drift
    from. If this test ever starts failing, it means the two roles have
    stopped sharing state, which would be a real regression."""
    keyword_store.index_chunks([_chunk("a", "keyword-only write, no vector")])

    report = HybridIndexer(vector_store, keyword_store).check_consistency()

    assert report.is_consistent
    assert report.only_in_vector_store == []
    assert report.only_in_keyword_store == []
