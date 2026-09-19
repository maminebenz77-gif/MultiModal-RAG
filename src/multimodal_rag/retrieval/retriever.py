"""Retriever: constructed once with its dependencies (stores, embedder,
optional reranker), called many times with different queries/methods —
method selection happens per call via the `method` parameter, not by
swapping which Retriever you constructed.

Every retrieve() call is traced (see tracing.py) -- one "retriever" span
covering the whole call, with the actual embedder/Qdrant/Elasticsearch/
reranker calls each as their own child span, so a trace shows real
per-step timing rather than one undifferentiated duration. This is a
no-op when Langfuse isn't configured (see tracing.py's own contract), so
it costs nothing here -- one cheap None-check per call -- when tracing
is off, which is true for the vast majority of callers (run_eval.py,
demo.py, most tests).
"""

from typing import Any

from ..providers.base import EmbeddingProvider, Reranker
from ..providers.schema import EmbeddingVector
from ..similarity import cosine_similarity
from ..stores.base import KeywordStore, VectorStore
from ..stores.filters import SearchFilter
from ..stores.schema import SearchResult
from ..tracing import traced_span, update_span_output
from .schema import RetrievalMethod

_TEXT_PREVIEW_LENGTH = 300
"""A trace should be legible in the Langfuse UI, not a dump of the full
resolved parent text (up to 8000 characters -- see
_resolve_parent_context) or the chunk's full base64-image-bearing
elements. A preview is what you'd actually want to eyeball; the real
content is still one click away in the app itself."""


def _summarize_results(results: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": r.chunk_id,
            "source": r.source,
            "score": r.score,
            "pages": r.pages,
            "slides": r.slides,
            "text_preview": r.text[:_TEXT_PREVIEW_LENGTH],
        }
        for r in results
    ]


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
        doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        with traced_span(
            # Method in the span NAME, not just metadata -- metadata is
            # real (confirmed live via the API), but Langfuse's default
            # trace-tree view doesn't surface it without expanding a
            # panel, so "which method ran" wasn't actually visible at a
            # glance. The name always shows.
            f"retrieve[{method.value}]",
            as_type="retriever",
            input=query,
            metadata={
                "method": method.value,
                "top_k": top_k,
                "rerank": rerank,
                # Without these, a child qdrant_search/elasticsearch_search
                # showing top_k=20 has no visible connection back to the
                # top_k=5 this call was actually asked for -- candidate_pool
                # is the reason the widened fetch happens at all, and
                # rrf_k/mmr_lambda are the fusion/selection parameters that
                # turn that wider pool into the final result, whichever of
                # them this method actually uses.
                "candidate_pool": candidate_pool,
                "rrf_k": rrf_k if method == RetrievalMethod.HYBRID_RRF else None,
                "mmr_lambda": mmr_lambda if method == RetrievalMethod.MMR else None,
            },
        ) as span:
            # doc_ids is sugar over a real, store-level filter (see
            # stores.filters.SearchFilter) -- built once here and handed
            # to whichever store call(s) `method` makes below, so every
            # retrieval method gets document scoping identically instead
            # of each reimplementing it. `is not None` (not truthy) so
            # doc_ids=[] keeps its old meaning -- "match no document" --
            # rather than silently becoming "no filter"; see
            # SearchFilter.any_of's docstring.
            search_filter = (
                SearchFilter(any_of={"doc_id": doc_ids}) if doc_ids is not None else None
            )

            # Reranking needs a broader candidate set than the final top_k
            # to still have top_k left over after re-scoring. Document
            # filtering no longer does: the store now returns results
            # that already satisfy search_filter, not a superset a Python
            # loop used to narrow down afterward -- so there's nothing
            # left to compensate for by over-fetching.
            pool_size = candidate_pool if rerank else top_k

            if method == RetrievalMethod.COSINE:
                results = self._cosine(query, pool_size, search_filter)
            elif method == RetrievalMethod.MMR:
                results = self._mmr(query, pool_size, mmr_lambda, candidate_pool, search_filter)
            elif method == RetrievalMethod.BM25:
                results = self._bm25(query, pool_size, search_filter)
            elif method == RetrievalMethod.HYBRID_RRF:
                results = self._hybrid_rrf(query, pool_size, rrf_k, candidate_pool, search_filter)
            else:
                raise ValueError(f"Unknown retrieval method: {method!r}")

            if rerank:
                # Always runs on precise CHILD text, never parent text --
                # parent chunks can never appear here at all, since
                # VectorStore/KeywordStore.search() exclude is_parent=True
                # chunks natively. Parent context is substituted in
                # afterward, only for the results that actually made the cut.
                results = self._rerank(query, results, top_k)
            else:
                results = results[:top_k]

            if resolve_parent_context:
                results = self._resolve_parent_context(results)

            update_span_output(span, _summarize_results(results))
            return results

    def _resolve_parent_context(self, results: list[SearchResult]) -> list[SearchResult]:
        """For each result that's a child chunk (has parent_id set),
        replace its `text` with the PARENT chunk's fuller text -- more
        context for the LLM to generate from -- while every other field
        (chunk_id, source, pages, ...) keeps pointing at the CHILD, so
        citations still resolve to the small, precise chunk that was
        actually matched, not the parent's broader text.

        If two DIFFERENT children of the SAME parent both made it into
        `results` (a real possibility -- both are separately embedded,
        separately ranked), only the higher-ranked one is kept. Without
        this, the LLM would see the same parent section's text twice,
        burning context budget for zero new information."""
        resolved = []
        seen_parent_ids: set[str] = set()
        for result in results:
            if result.parent_id is None:
                resolved.append(result)
                continue
            if result.parent_id in seen_parent_ids:
                continue
            seen_parent_ids.add(result.parent_id)
            parent = self._vector_store.get_by_chunk_id(result.parent_id)
            if parent is None:
                resolved.append(result)
                continue
            resolved.append(result.model_copy(update={"text": parent.text}))
        return resolved

    def _embed(self, query: str) -> EmbeddingVector:
        with traced_span("embed_query", as_type="embedding", input=query):
            return self._embedder.embed([query])[0]

    def _vector_search(
        self, query: str, query_vector: EmbeddingVector, **kwargs: Any
    ) -> list[SearchResult]:
        # search_filter is a pydantic model, not JSON-serializable as-is --
        # dumped to a plain dict for the trace, separately from `kwargs`
        # itself, which is passed to the real store call unchanged.
        trace_metadata = dict(kwargs)
        search_filter = trace_metadata.get("search_filter")
        if search_filter is not None:
            trace_metadata["search_filter"] = search_filter.model_dump()
        with traced_span(
            "qdrant_search", as_type="span", input=query, metadata=trace_metadata
        ) as span:
            results = self._vector_store.search(query_vector, **kwargs)
            update_span_output(span, _summarize_results(results))
            return results

    def _keyword_search(
        self, query: str, top_k: int, search_filter: SearchFilter | None = None
    ) -> list[SearchResult]:
        trace_metadata: dict[str, Any] = {"top_k": top_k}
        if search_filter is not None:
            trace_metadata["search_filter"] = search_filter.model_dump()
        with traced_span(
            "elasticsearch_search", as_type="span", input=query, metadata=trace_metadata
        ) as span:
            results = self._keyword_store.search(query, top_k=top_k, search_filter=search_filter)
            update_span_output(span, _summarize_results(results))
            return results

    def _cosine(
        self, query: str, top_k: int, search_filter: SearchFilter | None = None
    ) -> list[SearchResult]:
        query_vector = self._embed(query)
        return self._vector_search(query, query_vector, top_k=top_k, search_filter=search_filter)

    def _bm25(
        self, query: str, top_k: int, search_filter: SearchFilter | None = None
    ) -> list[SearchResult]:
        return self._keyword_search(query, top_k, search_filter)

    def _mmr(
        self,
        query: str,
        top_k: int,
        mmr_lambda: float,
        candidate_pool: int,
        search_filter: SearchFilter | None = None,
    ) -> list[SearchResult]:
        query_vector = self._embed(query)
        candidates = self._vector_search(
            query,
            query_vector,
            top_k=candidate_pool,
            with_vectors=True,
            search_filter=search_filter,
        )
        if not candidates:
            return []

        # Same gap as RRF fusion: the diversity-vs-relevance trade-off
        # that's the entire point of MMR happened in an untraced Python
        # loop -- a trace showed candidate_pool vector results in, then
        # nothing explaining which ones got picked or why over the
        # others.
        with traced_span(
            "mmr_select",
            as_type="span",
            input=query,
            metadata={"mmr_lambda": mmr_lambda, "candidates": len(candidates)},
        ) as span:
            selected: list[SearchResult] = []
            mmr_scores: list[float] = []
            remaining = list(candidates)
            while remaining and len(selected) < top_k:
                best = max(remaining, key=lambda c: self._mmr_score(c, selected, mmr_lambda))
                mmr_scores.append(self._mmr_score(best, selected, mmr_lambda))
                selected.append(best)
                remaining.remove(best)
            # Same _summarize_results() shape every other span in this
            # file uses -- see _hybrid_rrf's identical fix. `score` here
            # is already the candidate's raw relevance (MMR never
            # rewrites it); `mmr_score` is the only genuinely new field,
            # the actual relevance-vs-redundancy number that decided the
            # pick, in selection order.
            summaries = _summarize_results(selected)
            for summary, mmr_score in zip(summaries, mmr_scores, strict=True):
                summary["mmr_score"] = mmr_score
            update_span_output(span, summaries)
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
        self,
        query: str,
        top_k: int,
        rrf_k: int,
        candidate_pool: int,
        search_filter: SearchFilter | None = None,
    ) -> list[SearchResult]:
        query_vector = self._embed(query)
        vector_results = self._vector_search(
            query, query_vector, top_k=candidate_pool, search_filter=search_filter
        )
        keyword_results = self._keyword_search(query, candidate_pool, search_filter)

        # This computation -- turning candidate_pool-sized vector AND
        # keyword result lists into the final top_k -- was previously
        # invisible: pure Python between two traced searches and the
        # traced rerank/output, with no span of its own. A trace showed
        # 20 vector candidates in, 20 keyword candidates in, then nothing
        # until either a rerank span or the outer retrieve span's own
        # output -- the actual fusion decision (why these five, not those)
        # never appeared anywhere.
        with traced_span(
            "rrf_fuse",
            as_type="span",
            input=query,
            metadata={
                "rrf_k": rrf_k,
                "vector_candidates": len(vector_results),
                "keyword_candidates": len(keyword_results),
            },
        ) as span:
            # Fused by RANK, not raw score -- a BM25 score and a cosine
            # score aren't measuring the same thing and can't be
            # meaningfully rescaled onto each other, but "ranked #1" means
            # the same thing regardless of which method produced that
            # ranking.
            scores: dict[str, float] = {}
            by_id: dict[str, SearchResult] = {}
            sources: dict[str, list[str]] = {}
            for rank, result in enumerate(vector_results, start=1):
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1 / (rrf_k + rank)
                by_id[result.chunk_id] = result
                sources.setdefault(result.chunk_id, []).append("vector")
            for rank, result in enumerate(keyword_results, start=1):
                scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1 / (rrf_k + rank)
                by_id.setdefault(result.chunk_id, result)
                sources.setdefault(result.chunk_id, []).append("keyword")

            ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:top_k]
            fused = [
                by_id[chunk_id].model_copy(update={"score": scores[chunk_id]})
                for chunk_id in ranked_ids
            ]
            # Same _summarize_results() shape every other span in this
            # file uses (chunk_id, source, score, pages, slides,
            # text_preview) -- a first version of this span built its own
            # thinner shape from scratch and dropped text_preview, so it
            # was the one span in the whole trace where you couldn't
            # actually tell WHICH chunk got fused in, just its id/score.
            # `score` is already the fused score here, via the
            # model_copy() above -- only `found_by` is genuinely new
            # information _summarize_results() has no field for.
            summaries = _summarize_results(fused)
            for summary, chunk_id in zip(summaries, ranked_ids, strict=True):
                # "vector", "keyword", or both -- both is the interesting
                # case: a chunk both methods agreed on.
                summary["found_by"] = sources[chunk_id]
            update_span_output(span, summaries)
            return fused

    def _rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        if self._reranker is None:
            raise ValueError("rerank=True requires a Reranker to be provided to the Retriever.")
        if not candidates:
            return []
        with traced_span(
            "rerank", as_type="span", input=query, metadata={"candidates": len(candidates)}
        ) as span:
            order = self._reranker.rerank(query, [c.text for c in candidates])
            reranked = [candidates[i] for i in order[:top_k]]
            update_span_output(span, _summarize_results(reranked))
            return reranked
