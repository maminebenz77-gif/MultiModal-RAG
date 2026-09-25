"""Integration tests against real local Qdrant + Elasticsearch, but with
a fake embedder/reranker for exact, controlled vectors — testing MMR's
diversity trade-off and RRF's fusion precisely requires exact control
over similarity values that real embeddings don't offer.
"""

from collections.abc import Iterator
from datetime import date
from unittest.mock import patch

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkMetadata
from multimodal_rag.providers.base import EmbeddingProvider, Reranker
from multimodal_rag.providers.schema import EmbeddingVector
from multimodal_rag.retrieval import retriever as retriever_module
from multimodal_rag.retrieval.retriever import Retriever
from multimodal_rag.retrieval.schema import RetrievalMethod
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore
from multimodal_rag.stores.filters import SearchFilter
from multimodal_rag.stores.qdrant_store import QdrantStore
from multimodal_rag.stores.schema import SearchResult

_COLLECTION = "test_retrieval"
_MODEL_ID = "fake-model"


class FakeEmbedder(EmbeddingProvider):
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        return [
            EmbeddingVector(vector=self._vectors_by_text[t], model_id=_MODEL_ID, dimension=2)
            for t in texts
        ]


class FakeReranker(Reranker):
    def __init__(self, order: list[int]) -> None:
        self._order = order

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        return self._order


def _chunk(
    chunk_id: str,
    text: str,
    parent_id: str | None = None,
    source: str = "doc.md",
    doc_id: str | None = None,
    is_parent: bool = False,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        parent_id=parent_id,
        is_parent=is_parent,
        metadata=ChunkMetadata(
            # Defaults to `source` -- see test_qdrant_store.py's identical
            # _chunk() helper for why. The doc_ids filter tests below pass
            # doc_id explicitly, DIFFERENT from source, to prove the
            # filter matches the real stable id, not the display filename
            # (the identity bug this field exists to fix).
            source_file=source,
            doc_id=doc_id if doc_id is not None else source,
            element_positions=[0],
            element_types=["title"],
        ),
    )


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


def test_cosine_returns_top_k_by_similarity(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "close match": [0.99, 0.01],
            "far match": [0.1, 0.99],
        }
    )
    chunks = [_chunk("a", "close match"), _chunk("b", "far match")]
    vectors = embedder.embed([c.text for c in chunks])
    vector_store.upsert(chunks, vectors)

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=2)

    assert [r.chunk_id for r in results] == ["a", "b"]


def test_bm25_delegates_to_keyword_store(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"a-chunk": [1.0, 0.0], "b-chunk": [0.0, 1.0]})
    chunks = [
        _chunk("a", "the GPU ran out of memory"),
        _chunk("b", "the soup needed more salt"),
    ]
    keyword_store.index_chunks(chunks)

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve("GPU memory", method=RetrievalMethod.BM25, top_k=2)

    assert results[0].chunk_id == "a"
    assert results[0].model_id is None


def test_mmr_with_lambda_one_matches_pure_relevance_ranking(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "a": [1.0, 0.0],
            "b": [1.0, 0.01],
            "c": [0.6, 0.8],
        }
    )
    chunks = [_chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")]
    vectors = embedder.embed([c.text for c in chunks])
    vector_store.upsert(chunks, vectors)

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query", method=RetrievalMethod.MMR, top_k=3, mmr_lambda=1.0, candidate_pool=3
    )

    # lambda=1 -> no diversity penalty at all -> identical to plain relevance ranking.
    assert [r.chunk_id for r in results] == ["a", "b", "c"]


def test_mmr_with_low_lambda_prefers_diversity_over_redundant_high_relevance(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "a": [1.0, 0.0],
            "b": [1.0, 0.01],  # near-duplicate of "a" -- high relevance, high redundancy
            "c": [0.6, 0.8],  # lower relevance, but genuinely different from "a"
        }
    )
    chunks = [_chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")]
    vectors = embedder.embed([c.text for c in chunks])
    vector_store.upsert(chunks, vectors)

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query", method=RetrievalMethod.MMR, top_k=3, mmr_lambda=0.3, candidate_pool=3
    )

    # "a" is always picked first (highest relevance, nothing selected yet
    # to be redundant with). At low lambda, "c" should be preferred over
    # "b" for the second slot despite "b" scoring higher on raw relevance,
    # because "b" is nearly redundant with "a" and "c" isn't.
    assert results[0].chunk_id == "a"
    assert results[1].chunk_id == "c"


def test_hybrid_rrf_favors_a_chunk_ranked_in_both_lists(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {
            "GPU memory": [1.0, 0.0],  # the query text itself
            "GPU memory error occurred": [1.0, 0.0],  # strong on both signals
            "insufficient VRAM during the run": [0.95, 0.05],  # vector-close, no shared terms
            "a GPU memory issue was logged separately": [0.0, 1.0],  # vector-far, shares terms
        }
    )
    chunks = [
        _chunk("both", "GPU memory error occurred"),
        _chunk("vector-only", "insufficient VRAM during the run"),
        _chunk("bm25-only", "a GPU memory issue was logged separately"),
    ]
    vectors = embedder.embed([c.text for c in chunks])
    vector_store.upsert(chunks, vectors)
    keyword_store.index_chunks(chunks)

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "GPU memory", method=RetrievalMethod.HYBRID_RRF, top_k=3, candidate_pool=3
    )

    # "both" should out-rank items that only appear in one list, since its
    # RRF score sums contributions from both rankings.
    assert results[0].chunk_id == "both"


def test_rerank_reorders_results_via_the_reranker(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.9, 0.1], "c": [0.8, 0.2]}
    )
    chunks = [_chunk("a", "a"), _chunk("b", "b"), _chunk("c", "c")]
    vectors = embedder.embed([c.text for c in chunks])
    vector_store.upsert(chunks, vectors)

    # Reranker reverses cosine's natural order: c, b, a.
    reranker = FakeReranker(order=[2, 1, 0])
    retriever = Retriever(vector_store, keyword_store, embedder, reranker=reranker)

    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=2, rerank=True, candidate_pool=3
    )

    assert [r.chunk_id for r in results] == ["c", "b"]


def test_rerank_without_a_reranker_raises(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "a": [1.0, 0.0]})
    chunks = [_chunk("a", "a")]
    vector_store.upsert(chunks, embedder.embed(["a"]))

    retriever = Retriever(vector_store, keyword_store, embedder, reranker=None)

    with pytest.raises(ValueError, match="requires a Reranker"):
        retriever.retrieve("query", method=RetrievalMethod.COSINE, rerank=True)


def test_resolve_parent_context_substitutes_parent_text_but_keeps_child_chunk_id(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "child text": [1.0, 0.0], "parent text, much longer": [0.5, 0.5]}
    )
    parent = _chunk("parent", "parent text, much longer", is_parent=True)
    child = _chunk("child", "child text", parent_id="parent")
    vector_store.upsert([parent, child], embedder.embed([parent.text, child.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=1, resolve_parent_context=True
    )

    assert results[0].chunk_id == "child"
    assert results[0].text == "parent text, much longer"


def test_resolve_parent_context_dedupes_multiple_children_of_the_same_parent(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {
            "query": [1.0, 0.0],
            "parent text": [0.5, 0.5],
            "child one": [1.0, 0.0],
            "child two": [0.99, 0.01],
        }
    )
    parent = _chunk("parent", "parent text", is_parent=True)
    child_one = _chunk("child-1", "child one", parent_id="parent")
    child_two = _chunk("child-2", "child two", parent_id="parent")
    vector_store.upsert(
        [parent, child_one, child_two],
        embedder.embed([parent.text, child_one.text, child_two.text]),
    )

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=2, resolve_parent_context=True
    )

    # Both children matched (both close to "query") and share a parent --
    # the LLM should see that parent's text once, not twice, so only the
    # higher-ranked child should survive resolution.
    assert len(results) == 1
    assert results[0].chunk_id == "child-1"
    assert results[0].text == "parent text"


def test_resolve_parent_context_leaves_parentless_chunks_unchanged(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "standalone text": [1.0, 0.0]})
    chunk = _chunk("a", "standalone text")
    vector_store.upsert([chunk], embedder.embed([chunk.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=1, resolve_parent_context=True
    )

    assert results[0].chunk_id == "a"
    assert results[0].text == "standalone text"


def test_resolve_parent_context_off_by_default_leaves_child_text_as_is(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "child text": [1.0, 0.0], "parent text": [0.5, 0.5]}
    )
    parent = _chunk("parent", "parent text", is_parent=True)
    child = _chunk("child", "child text", parent_id="parent")
    vector_store.upsert([parent, child], embedder.embed([parent.text, child.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=1)

    assert results[0].chunk_id == "child"
    assert results[0].text == "child text"


def test_doc_ids_filter_excludes_chunks_from_other_documents(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "from doc a": [1.0, 0.0], "from doc b": [0.9, 0.1]}
    )
    # source (the display filename) deliberately does NOT match doc_id
    # (the real stable id) here -- this is what test_query_doc_ids_filter
    # in test_query.py's real-world equivalent would look like once
    # ChunkMetadata.doc_id is populated correctly: the filter must match
    # against doc_id, not the filename that used to leak into that field.
    chunk_a = _chunk("a", "from doc a", source="doc-a.md", doc_id="sha256-doc-a")
    chunk_b = _chunk("b", "from doc b", source="doc-b.md", doc_id="sha256-doc-b")
    vector_store.upsert([chunk_a, chunk_b], embedder.embed([chunk_a.text, chunk_b.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=2, doc_ids=["sha256-doc-b"]
    )

    assert [r.chunk_id for r in results] == ["b"]


def test_doc_ids_filter_does_not_match_the_display_filename(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    # Guards directly against the identity bug regressing: passing the
    # FILENAME (what doc_ids used to silently accept) must match nothing,
    # now that doc_id and source are tracked separately.
    embedder = FakeEmbedder({"query": [1.0, 0.0], "from doc a": [1.0, 0.0]})
    chunk_a = _chunk("a", "from doc a", source="doc-a.md", doc_id="sha256-doc-a")
    vector_store.upsert([chunk_a], embedder.embed([chunk_a.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=2, doc_ids=["doc-a.md"]
    )

    assert results == []


def test_doc_ids_filter_finds_matches_the_old_post_retrieval_design_would_have_missed(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    # The concrete argument for store-level (not post-retrieval) filtering:
    # build a corpus where the chunks that match the filter are the LEAST
    # similar to the query among the whole corpus -- ranked outside any
    # top_k a plain similarity search would return. The retired design
    # fetched candidates by similarity FIRST, then filtered by doc_id
    # afterward -- so it would never even have SEEN these two chunks, and
    # would have returned nothing. A native store-level filter restricts
    # the search itself, so it finds them regardless of how they'd
    # otherwise rank.
    noise_vectors = {f"noise {i}": [1.0 - i * 0.01, i * 0.01] for i in range(10)}
    target_vectors = {"target a": [0.0, 1.0], "target b": [0.01, 0.99]}
    embedder = FakeEmbedder({"query": [1.0, 0.0], **noise_vectors, **target_vectors})

    noise_chunks = [
        _chunk(f"n{i}", text, source="other.md", doc_id="other")
        for i, text in enumerate(noise_vectors)
    ]
    target_chunks = [
        _chunk("t-a", "target a", source="target.md", doc_id="target"),
        _chunk("t-b", "target b", source="target.md", doc_id="target"),
    ]
    all_chunks = noise_chunks + target_chunks
    vector_store.upsert(all_chunks, embedder.embed([c.text for c in all_chunks]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=2, doc_ids=["target"]
    )

    assert {r.chunk_id for r in results} == {"t-a", "t-b"}


def test_doc_ids_none_means_no_filtering(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "from doc a": [1.0, 0.0], "from doc b": [0.9, 0.1]}
    )
    chunk_a = _chunk("a", "from doc a", source="doc-a.md", doc_id="sha256-doc-a")
    chunk_b = _chunk("b", "from doc b", source="doc-b.md", doc_id="sha256-doc-b")
    vector_store.upsert([chunk_a, chunk_b], embedder.embed([chunk_a.text, chunk_b.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=2)

    assert {r.chunk_id for r in results} == {"a", "b"}


def test_search_filter_and_doc_ids_intersect_rather_than_stack_independently(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    """The mechanism ScopedRetriever depends on: retrieve()'s own
    search_filter parameter must compose with doc_ids via merge()
    (narrowing), not just get silently ignored or override it."""
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "from doc a": [1.0, 0.0], "from doc b": [0.9, 0.1]}
    )
    chunk_a = _chunk("a", "from doc a", source="doc-a.md", doc_id="sha256-doc-a")
    chunk_b = _chunk("b", "from doc b", source="doc-b.md", doc_id="sha256-doc-b")
    vector_store.upsert([chunk_a, chunk_b], embedder.embed([chunk_a.text, chunk_b.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)

    # doc_ids allows both; search_filter (standing in for a security
    # clause) allows only "a" -- the intersection must be just "a".
    results = retriever.retrieve(
        "query",
        method=RetrievalMethod.COSINE,
        top_k=2,
        doc_ids=["sha256-doc-a", "sha256-doc-b"],
        search_filter=SearchFilter(any_of={"doc_id": ["sha256-doc-a"]}),
    )

    assert [r.chunk_id for r in results] == ["a"]


def test_search_filter_alone_narrows_results_without_doc_ids(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "from doc a": [1.0, 0.0], "from doc b": [0.9, 0.1]}
    )
    chunk_a = _chunk("a", "from doc a", source="doc-a.md", doc_id="sha256-doc-a")
    chunk_b = _chunk("b", "from doc b", source="doc-b.md", doc_id="sha256-doc-b")
    vector_store.upsert([chunk_a, chunk_b], embedder.embed([chunk_a.text, chunk_b.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query",
        method=RetrievalMethod.COSINE,
        top_k=2,
        search_filter=SearchFilter(any_of={"doc_id": ["sha256-doc-b"]}),
    )

    assert [r.chunk_id for r in results] == ["b"]


def _patch_traced_span():
    # A MagicMock stands in cleanly here: MagicMock auto-implements
    # __enter__/__exit__, so `with traced_span(...) as span:` works with
    # no further setup, and the yielded `span` is itself a MagicMock --
    # safe to pass straight into update_span_output without configuring
    # anything else.
    return patch.object(retriever_module, "traced_span")


def test_retrieve_opens_a_retriever_span_for_the_whole_call(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "a": [1.0, 0.0]})
    vector_store.upsert([_chunk("a", "a")], embedder.embed(["a"]))
    retriever = Retriever(vector_store, keyword_store, embedder)

    with _patch_traced_span() as mock_traced_span:
        retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=1)

    first_call = mock_traced_span.call_args_list[0]
    # Method is in the span NAME, not just metadata -- metadata is real
    # (see retriever.py's comment), but Langfuse's default trace-tree view
    # doesn't surface it without expanding a panel, so the name carries it
    # too, for a method that's visible at a glance.
    assert first_call.args[0] == "retrieve[cosine]"
    assert first_call.kwargs["as_type"] == "retriever"
    assert first_call.kwargs["input"] == "query"
    assert first_call.kwargs["metadata"] == {
        "method": "cosine",
        "top_k": 1,
        "rerank": False,
        "candidate_pool": 20,
        "rrf_k": None,
        "mmr_lambda": None,
    }


def test_cosine_traces_embed_query_and_qdrant_search_as_child_spans(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "a": [1.0, 0.0]})
    vector_store.upsert([_chunk("a", "a")], embedder.embed(["a"]))
    retriever = Retriever(vector_store, keyword_store, embedder)

    with _patch_traced_span() as mock_traced_span:
        retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=1)

    calls = mock_traced_span.call_args_list
    names = [c.args[0] for c in calls]
    assert names == ["retrieve[cosine]", "embed_query", "qdrant_search"]
    assert calls[1].kwargs["as_type"] == "embedding"
    assert calls[2].kwargs["as_type"] == "span"


def test_bm25_traces_elasticsearch_search_as_a_child_span(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"a-chunk": [1.0, 0.0]})
    keyword_store.index_chunks([_chunk("a", "the GPU ran out of memory")])
    retriever = Retriever(vector_store, keyword_store, embedder)

    with _patch_traced_span() as mock_traced_span:
        retriever.retrieve("GPU memory", method=RetrievalMethod.BM25, top_k=1)

    names = [c.args[0] for c in mock_traced_span.call_args_list]
    assert names == ["retrieve[bm25]", "elasticsearch_search"]


def test_hybrid_rrf_traces_embed_qdrant_and_elasticsearch(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "a": [1.0, 0.0]})
    chunk = _chunk("a", "a")
    vector_store.upsert([chunk], embedder.embed(["a"]))
    keyword_store.index_chunks([chunk])
    retriever = Retriever(vector_store, keyword_store, embedder)

    with _patch_traced_span() as mock_traced_span:
        retriever.retrieve("query", method=RetrievalMethod.HYBRID_RRF, top_k=1)

    names = [c.args[0] for c in mock_traced_span.call_args_list]
    assert names == [
        "retrieve[hybrid_rrf]",
        "embed_query",
        "qdrant_search",
        "elasticsearch_search",
        "rrf_fuse",
    ]


def test_rerank_traces_the_reranker_call_as_a_child_span(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.9, 0.1]})
    chunks = [_chunk("a", "a"), _chunk("b", "b")]
    vector_store.upsert(chunks, embedder.embed(["a", "b"]))
    reranker = FakeReranker(order=[1, 0])
    retriever = Retriever(vector_store, keyword_store, embedder, reranker=reranker)

    with _patch_traced_span() as mock_traced_span:
        retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=2, rerank=True)

    names = [c.args[0] for c in mock_traced_span.call_args_list]
    assert names == ["retrieve[cosine]", "embed_query", "qdrant_search", "rerank"]


def test_mmr_traces_the_selection_loop_as_a_child_span(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "a": [1.0, 0.0], "b": [0.9, 0.1]})
    chunks = [_chunk("a", "a"), _chunk("b", "b")]
    vector_store.upsert(chunks, embedder.embed(["a", "b"]))
    retriever = Retriever(vector_store, keyword_store, embedder)

    with _patch_traced_span() as mock_traced_span:
        retriever.retrieve("query", method=RetrievalMethod.MMR, top_k=2)

    calls = mock_traced_span.call_args_list
    names = [c.args[0] for c in calls]
    assert names == ["retrieve[mmr]", "embed_query", "qdrant_search", "mmr_select"]
    select_call = calls[3]
    assert select_call.kwargs["metadata"]["mmr_lambda"] == 0.5
    assert select_call.kwargs["metadata"]["candidates"] == 2


def test_retrieve_attaches_a_result_summary_as_the_span_output(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "close match": [0.99, 0.01]})
    vector_store.upsert([_chunk("a", "close match")], embedder.embed(["close match"]))
    retriever = Retriever(vector_store, keyword_store, embedder)

    with patch.object(retriever_module, "update_span_output") as mock_update:
        retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=1)

    # The LAST call is for the top-level "retrieve" span -- child spans
    # (qdrant_search has its own output, embed_query has none) update
    # first, in call order.
    output = mock_update.call_args_list[-1].args[1]
    assert len(output) == 1
    summary = output[0]
    assert summary["chunk_id"] == "a"
    assert summary["source"] == "doc.md"
    assert summary["pages"] == []
    assert summary["slides"] == []
    assert summary["text_preview"] == "close match"
    assert isinstance(summary["score"], float)


def test_summarize_results_truncates_long_text_for_trace_legibility() -> None:
    long_text = "x" * 500
    result = SearchResult(
        chunk_id="a",
        score=0.5,
        text=long_text,
        source="doc.md",
        doc_id="doc",
        element_types=["paragraph"],
    )

    summary = retriever_module._summarize_results([result])

    assert len(summary[0]["text_preview"]) == retriever_module._TEXT_PREVIEW_LENGTH
    assert summary[0]["text_preview"] == long_text[: retriever_module._TEXT_PREVIEW_LENGTH]


def test_summarize_results_includes_tags_author_date_and_classification() -> None:
    # What makes a retrieved chunk's own metadata visible on its
    # Langfuse retrieval span -- not just score/text, but what it IS
    # (tags, author, date, confidentiality level).
    result = SearchResult(
        chunk_id="a",
        score=0.5,
        text="text",
        source="doc.md",
        doc_id="doc",
        element_types=["paragraph"],
        tags=["runbook"],
        author="Alice",
        doc_date="2026-01-01",
        classification="c2",
        private=True,
    )

    summary = retriever_module._summarize_results([result])[0]

    assert summary["tags"] == ["runbook"]
    assert summary["author"] == "Alice"
    assert summary["doc_date"] == "2026-01-01"
    assert summary["classification"] == "c2"
    assert summary["private"] is True


# --------------------------------------------------- recency tilt / collapse


def _dated_chunk(chunk_id: str, text: str, doc_id: str) -> Chunk:
    """A plain chunk -- lineage (effective_from/version/doc_family_id)
    isn't a Chunk/ChunkMetadata field (see _upsert_with_lineage): it's
    document metadata, injected at the store-write boundary."""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata=ChunkMetadata(
            source_file=doc_id, doc_id=doc_id, element_positions=[0], element_types=["title"]
        ),
    )


def _upsert_with_lineage(vector_store, chunks_and_vectors, **metadata_kwargs):
    from multimodal_rag.metadata import DocumentMetadata

    for chunk, vector in chunks_and_vectors:
        metadata = DocumentMetadata(classification="public", **metadata_kwargs)
        vector_store.upsert([chunk], [vector], doc_metadata=metadata)


def test_recency_tilt_breaks_a_near_tie_in_favor_of_the_newer_result(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "old text": [1.0, 0.0], "new text": [0.999, 0.001]}
    )
    old = _dated_chunk("old", "old text", "old-doc")
    new = _dated_chunk("new", "new text", "new-doc")
    _upsert_with_lineage(
        vector_store, [(old, embedder.embed(["old text"])[0])], effective_from="2020-01-01"
    )
    _upsert_with_lineage(
        vector_store,
        [(new, embedder.embed(["new text"])[0])],
        effective_from=date.today().isoformat(),
    )

    retriever = Retriever(vector_store, keyword_store, embedder, recency_tilt_weight=0.2)
    # Without a tilt, "old" (near-perfect similarity) outranks "new"
    # (slightly lower). With one, the small residual gap is exactly what
    # the tilt is sized to close.
    no_tilt = Retriever(vector_store, keyword_store, embedder, recency_tilt_weight=0.0)
    assert [r.chunk_id for r in no_tilt.retrieve("query", RetrievalMethod.COSINE, top_k=2)] == [
        "old",
        "new",
    ]

    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=2)
    assert [r.chunk_id for r in results] == ["new", "old"]


def test_recency_tilt_cannot_override_a_real_relevance_gap(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "clearly relevant": [1.0, 0.0], "barely relevant": [0.1, 0.99]}
    )
    relevant = _dated_chunk("relevant", "clearly relevant", "old-doc")
    barely = _dated_chunk("barely", "barely relevant", "new-doc")
    _upsert_with_lineage(
        vector_store,
        [(relevant, embedder.embed(["clearly relevant"])[0])],
        effective_from="2015-01-01",
    )
    _upsert_with_lineage(
        vector_store,
        [(barely, embedder.embed(["barely relevant"])[0])],
        effective_from=date.today().isoformat(),
    )

    # An intentionally extreme weight -- even this must not flip a real gap.
    retriever = Retriever(vector_store, keyword_store, embedder, recency_tilt_weight=0.2)
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=2)

    assert results[0].chunk_id == "relevant"


def test_a_result_with_no_effective_from_gets_no_boost_and_no_penalty(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    embedder = FakeEmbedder({"query": [1.0, 0.0], "undated": [1.0, 0.0]})
    chunk = _dated_chunk("undated", "undated", "doc")
    vector_store.upsert([chunk], embedder.embed(["undated"]))  # no doc_metadata at all

    retriever = Retriever(vector_store, keyword_store, embedder, recency_tilt_weight=0.5)
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=1)

    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_two_current_versions_of_one_family_collapse_to_the_higher_version(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    """Not just a superseded/current problem -- two uploads that were never
    linked by supersedes_doc_id can both be "current" in the same family."""
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "v1 text": [1.0, 0.0], "v2 text": [0.99, 0.01]}
    )
    v1 = _dated_chunk("v1", "v1 text", "fam")
    v2 = _dated_chunk("v2", "v2 text", "fam")
    _upsert_with_lineage(
        vector_store, [(v1, embedder.embed(["v1 text"])[0])],
        doc_family_id="fam", version=1, effective_from="2024-01-01",
    )
    _upsert_with_lineage(
        vector_store, [(v2, embedder.embed(["v2 text"])[0])],
        doc_family_id="fam", version=2, effective_from="2025-01-01",
    )

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=5)

    assert [r.chunk_id for r in results] == ["v2"]


def test_untagged_documents_are_never_collapsed_into_each_other(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    """No doc_family_id must mean "its own family of one", not "the same
    family as every other untagged document"."""
    embedder = FakeEmbedder({"query": [1.0, 0.0], "a text": [1.0, 0.0], "b text": [0.99, 0.0]})
    chunk_a = _dated_chunk("a", "a text", "doc-a")
    chunk_b = _dated_chunk("b", "b text", "doc-b")
    vector_store.upsert([chunk_a, chunk_b], embedder.embed(["a text", "b text"]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=5)

    assert {r.chunk_id for r in results} == {"a", "b"}


def test_recency_tilt_is_not_applied_after_reranking(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    """A cross-encoder reranker judges (query, text) pairs directly -- it
    never sees .score, and its own returned order isn't driven by .score
    either (see _rerank's docstring). Re-sorting its output by a tilted
    score afterward wouldn't nudge its judgment, it would silently
    overwrite it -- so the tilt must not run at all once reranked,
    however recent one result is."""
    embedder = FakeEmbedder({"query": [1.0, 0.0], "old": [1.0, 0.0], "new": [0.8, 0.6]})
    old = _dated_chunk("old", "old", "old-doc")
    new = _dated_chunk("new", "new", "new-doc")
    _upsert_with_lineage(
        vector_store, [(old, embedder.embed(["old"])[0])], effective_from="2015-01-01"
    )
    _upsert_with_lineage(
        vector_store, [(new, embedder.embed(["new"])[0])], effective_from=date.today().isoformat()
    )

    # The reranker deliberately puts the OLDER result first -- its own
    # judgment, which an extreme tilt weight must not override.
    reranker = FakeReranker(order=[0, 1])  # candidates arrive [old, new] from cosine
    retriever = Retriever(
        vector_store, keyword_store, embedder, reranker=reranker, recency_tilt_weight=1.0
    )

    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=2, rerank=True, candidate_pool=2
    )

    assert [r.chunk_id for r in results] == ["old", "new"]


def test_family_collapse_still_runs_on_the_full_reranked_pool_not_just_survivors(
    vector_store: QdrantStore, keyword_store: ElasticsearchStore
) -> None:
    """The reason _rerank stops truncating: family collapse (unlike the
    tilt) is safe and still runs after reranking -- if it cut to top_k
    first, a stale duplicate could occupy a slot while the current
    version of the same family sat just outside it, uncollapsed."""
    embedder = FakeEmbedder(
        {"query": [1.0, 0.0], "stale dup": [1.0, 0.0], "current": [0.9, 0.0]}
    )
    stale = _dated_chunk("stale", "stale dup", "fam-old")
    current = _dated_chunk("current", "current", "fam-new")
    _upsert_with_lineage(
        vector_store, [(stale, embedder.embed(["stale dup"])[0])],
        doc_family_id="fam", version=1,
    )
    _upsert_with_lineage(
        vector_store, [(current, embedder.embed(["current"])[0])],
        doc_family_id="fam", version=2,
    )

    # The reranker ranks the stale duplicate first -- both still enter a
    # top_k=1 cut unless collapse removes the lower version first.
    reranker = FakeReranker(order=[0, 1])  # unchanged: stale, current
    retriever = Retriever(vector_store, keyword_store, embedder, reranker=reranker)

    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=1, rerank=True, candidate_pool=2
    )

    assert [r.chunk_id for r in results] == ["current"]
