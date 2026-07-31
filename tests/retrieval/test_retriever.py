"""Integration tests against real local Qdrant + Elasticsearch, but with
a fake embedder/reranker for exact, controlled vectors — testing MMR's
diversity trade-off and RRF's fusion precisely requires exact control
over similarity values that real embeddings don't offer.
"""

from collections.abc import Iterator

import pytest

from multimodal_rag.chunking.schema import Chunk, ChunkMetadata
from multimodal_rag.providers.base import EmbeddingProvider, Reranker
from multimodal_rag.providers.schema import EmbeddingVector
from multimodal_rag.retrieval.retriever import Retriever
from multimodal_rag.retrieval.schema import RetrievalMethod
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore
from multimodal_rag.stores.qdrant_store import QdrantStore

_COLLECTION = "test_retrieval"
_MODEL_ID = "fake-model"


class FakeEmbedder(EmbeddingProvider):
    def __init__(self, vectors_by_text: dict[str, list[float]]) -> None:
        self._vectors_by_text = vectors_by_text

    def embed(self, texts: list[str]) -> list[EmbeddingVector]:
        return [
            EmbeddingVector(
                vector=self._vectors_by_text[t], model_id=_MODEL_ID, dimension=2
            )
            for t in texts
        ]


class FakeReranker(Reranker):
    def __init__(self, order: list[int]) -> None:
        self._order = order

    def rerank(self, query: str, documents: list[str]) -> list[int]:
        return self._order


def _chunk(chunk_id: str, text: str, parent_id: str | None = None) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        parent_id=parent_id,
        metadata=ChunkMetadata(
            source_file="doc.md", element_positions=[0], element_types=["title"]
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
    parent = _chunk("parent", "parent text, much longer")
    child = _chunk("child", "child text", parent_id="parent")
    vector_store.upsert([parent, child], embedder.embed([parent.text, child.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve(
        "query", method=RetrievalMethod.COSINE, top_k=1, resolve_parent_context=True
    )

    assert results[0].chunk_id == "child"
    assert results[0].text == "parent text, much longer"


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
    parent = _chunk("parent", "parent text")
    child = _chunk("child", "child text", parent_id="parent")
    vector_store.upsert([parent, child], embedder.embed([parent.text, child.text]))

    retriever = Retriever(vector_store, keyword_store, embedder)
    results = retriever.retrieve("query", method=RetrievalMethod.COSINE, top_k=1)

    assert results[0].chunk_id == "child"
    assert results[0].text == "child text"
