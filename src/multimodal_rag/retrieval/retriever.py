"""Retriever: constructed once with its dependencies (stores, embedder,
optional reranker), called many times with different queries/methods —
method selection happens per call via the `method` parameter, not by
swapping which Retriever you constructed.
"""

from ..providers.base import EmbeddingProvider, Reranker
from ..similarity import cosine_similarity
from ..stores.base import KeywordStore, VectorStore
from ..stores.schema import SearchResult
from .schema import RetrievalMethod


class Retriever:
    def __init__(
        self,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embedder: EmbeddingProvider,
        reranker: Reranker | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._keyword_store = keyword_store
        self._embedder = embedder
        self._reranker = reranker

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod = RetrievalMethod.HYBRID_RRF,
        top_k: int = 5,
        rerank: bool = False,
        mmr_lambda: float = 0.5,
        rrf_k: int = 60,
        candidate_pool: int = 20,
        resolve_parent_context: bool = False,
    ) -> list[SearchResult]:
        # Reranking works on a broader candidate set than the final
        # top_k -- the whole point is retrieve-broad-then-rerank-down.
        pool_size = candidate_pool if rerank else top_k

        if method == RetrievalMethod.COSINE:
            results = self._cosine(query, pool_size)
        elif method == RetrievalMethod.MMR:
            results = self._mmr(query, pool_size, mmr_lambda, candidate_pool)
        elif method == RetrievalMethod.BM25:
            results = self._bm25(query, pool_size)
        elif method == RetrievalMethod.HYBRID_RRF:
            results = self._hybrid_rrf(query, pool_size, rrf_k, candidate_pool)
        else:
            raise ValueError(f"Unknown retrieval method: {method!r}")

        if rerank:
            # Reranking runs on the precise child text (matching the
            # child against the query is what the cross-encoder is good
            # at) -- parent context is substituted in afterward, only for
            # the results that actually made the final cut.
            results = self._rerank(query, results, top_k)
        else:
            results = results[:top_k]

        if resolve_parent_context:
            results = self._resolve_parent_context(results)
        return results

    def _resolve_parent_context(self, results: list[SearchResult]) -> list[SearchResult]:
        """For each result that's a child chunk (has parent_id set),
        replace its `text` with the PARENT chunk's fuller text -- more
        context for the LLM to generate from -- while every other field
        (chunk_id, source, pages, ...) keeps pointing at the CHILD, so
        citations still resolve to the small, precise chunk that was
        actually matched, not the parent's broader text."""
        resolved = []
        for result in results:
            if result.parent_id is None:
                resolved.append(result)
                continue
            parent = self._vector_store.get_by_chunk_id(result.parent_id)
            if parent is None:
                resolved.append(result)
                continue
            resolved.append(result.model_copy(update={"text": parent.text}))
        return resolved

    def _cosine(self, query: str, top_k: int) -> list[SearchResult]:
        query_vector = self._embedder.embed([query])[0]
        return self._vector_store.search(query_vector, top_k=top_k)

    def _bm25(self, query: str, top_k: int) -> list[SearchResult]:
        return self._keyword_store.search(query, top_k=top_k)

    def _mmr(
        self, query: str, top_k: int, mmr_lambda: float, candidate_pool: int
    ) -> list[SearchResult]:
        query_vector = self._embedder.embed([query])[0]
        candidates = self._vector_store.search(
            query_vector, top_k=candidate_pool, with_vectors=True
        )
        if not candidates:
            return []

        selected: list[SearchResult] = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            best = max(remaining, key=lambda c: self._mmr_score(c, selected, mmr_lambda))
            selected.append(best)
            remaining.remove(best)
        return selected

    @staticmethod
    def _mmr_score(
        candidate: SearchResult, selected: list[SearchResult], mmr_lambda: float
    ) -> float:
        assert candidate.vector is not None, "MMR requires candidate vectors (with_vectors=True)"
        relevance = candidate.score
        if not selected:
            return mmr_lambda * relevance
        redundancy = max(
            cosine_similarity(candidate.vector, s.vector) for s in selected if s.vector is not None
        )
        return mmr_lambda * relevance - (1 - mmr_lambda) * redundancy

    def _hybrid_rrf(
        self, query: str, top_k: int, rrf_k: int, candidate_pool: int
    ) -> list[SearchResult]:
        query_vector = self._embedder.embed([query])[0]
        vector_results = self._vector_store.search(query_vector, top_k=candidate_pool)
        keyword_results = self._keyword_store.search(query, top_k=candidate_pool)

        # Fused by RANK, not raw score -- a BM25 score and a cosine score
        # aren't measuring the same thing and can't be meaningfully
        # rescaled onto each other, but "ranked #1" means the same thing
        # regardless of which method produced that ranking.
        scores: dict[str, float] = {}
        by_id: dict[str, SearchResult] = {}
        for rank, result in enumerate(vector_results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1 / (rrf_k + rank)
            by_id[result.chunk_id] = result
        for rank, result in enumerate(keyword_results, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1 / (rrf_k + rank)
            by_id.setdefault(result.chunk_id, result)

        ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
        return [
            by_id[chunk_id].model_copy(update={"score": scores[chunk_id]})
            for chunk_id in ranked_ids
        ]

    def _rerank(
        self, query: str, candidates: list[SearchResult], top_k: int
    ) -> list[SearchResult]:
        if self._reranker is None:
            raise ValueError("rerank=True requires a Reranker to be provided to the Retriever.")
        if not candidates:
            return []
        order = self._reranker.rerank(query, [c.text for c in candidates])
        return [candidates[i] for i in order[:top_k]]
