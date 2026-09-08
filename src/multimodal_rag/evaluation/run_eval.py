"""Retrieval-method comparison against the golden set (data/golden_set.json).

Ingests the golden set's corpus into a scratch collection, then for every
RetrievalMethod (plus one hybrid_rrf+rerank row, if a Reranker is
configured for this profile) runs every non-refusal golden question
through:

  - Retriever.retrieve() directly, for Layer-1 (recall@k / MRR / nDCG@k)
    -- no LLM calls, deterministic.
  - RagChain (single-shot retrieve-then-generate, same baseline demo.py
    uses -- deliberately not AgentChain, so every method gets exactly one
    generation per question, not tool-calling variance) for Layer-2
    (faithfulness / relevance / hallucination_rate), via a hand-rolled
    LLM-as-judge (evaluation/judge.py).

So "hybrid+rerank beat plain cosine on faithfulness" is a number from this
table, not a guess.

Run: `uv run python -m multimodal_rag.evaluation.run_eval`
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..chunking.parent_child import ParentChildChunker
from ..chunking.schema import Chunk
from ..generation.chain import RagChain
from ..generation.context import format_context_block
from ..ingestion import parse_document
from ..providers.factory import get_embedder, get_reranker
from ..retrieval.retriever import Retriever
from ..retrieval.schema import RetrievalMethod
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from .judge import JudgeParseError, score_faithfulness, score_relevance
from .retrieval_metrics import mrr, ndcg_at_k, recall_at_k

_GOLDEN_SET_PATH = Path(__file__).resolve().parents[3] / "data" / "golden_set.json"
_SAMPLE_DOCS_DIR = Path(__file__).resolve().parents[3] / "data" / "samples"
_CORPUS_FILES = [
    "chunking_demo.md",
    "sample.md",
    # The remaining three exist purely as topically-distinct distractors --
    # without them, the corpus was small enough (2 overlapping documents)
    # that recall@5 saturated at 1.0 for every method regardless of ranking
    # quality, since both documents almost always fit inside 5 slots
    # anyway. These add real competition: same technical vocabulary
    # ("latency", "embedding", "index") but genuinely different subjects,
    # which is what actually stresses a retriever's ability to tell
    # "relevant" from "just uses similar words."
    "vector_index_strategies.md",
    "chunking_strategy_notes.md",
    "embedding_service_postmortem.md",
]
_COLLECTION = "retrieval_eval"
_TOP_K = 3
"""Deliberately smaller than the 5-document corpus. recall_at_k can only
ever fail to find a relevant document when there are more documents in
the corpus than fit in top-k -- at _TOP_K >= corpus size, every method
trivially returns every document and recall@k saturates at ~1.0 for all
of them regardless of ranking quality (this is exactly what happened at
_TOP_K=5 against this same 5-document corpus; caught by actually running
it, not by reasoning about it up front)."""

_HALLUCINATION_THRESHOLD = 0.7
"""An answer scoring below this on faithfulness counts toward
hallucination_rate. Arbitrary but not unusual as a cutoff -- the point is
having ONE fixed, stated threshold that every method is held to the same
way, not the exact number."""


@dataclass
class _MethodScores:
    label: str
    recall: float
    mrr: float
    ndcg: float
    faithfulness: float
    relevance: float
    hallucination_rate: float
    false_refusals: int
    """Golden items NOT marked expect_refusal that the chain refused to
    answer anyway -- a real failure mode (the method found nothing usable
    even though an answer exists), tracked separately from
    hallucination_rate since it's the opposite problem: saying too little,
    not saying something unsupported."""
    judge_failures: int
    """Items skipped because the judge's own output couldn't be parsed
    (see judge.JudgeParseError) -- excluded from the faithfulness/
    relevance averages rather than silently corrupting them."""


def _load_golden_set() -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.loads(_GOLDEN_SET_PATH.read_text()))


def _ingest_one(filename: str) -> list[Chunk]:
    """Same doc_id-then-filename-swap dance as routers/ingest.py, and for
    the same reason: chunk_id() (chunking/ids.py) needs a stable per-file
    doc_id to hash against, but the golden set's ground truth is keyed by
    the human-readable filename (see data/golden_set.json), not doc_id --
    so source_file gets swapped back to `filename` right after chunking.
    """
    path = _SAMPLE_DOCS_DIR / filename
    doc_id = hashlib.sha256(filename.encode()).hexdigest()
    elements = parse_document(path)
    for element in elements:
        element.metadata.source_file = doc_id
    chunks = ParentChildChunker().chunk(elements)
    for chunk in chunks:
        chunk.metadata.source_file = filename
    return chunks


def _score_method(
    retriever: Retriever,
    items: list[dict[str, Any]],
    method: RetrievalMethod,
    *,
    rerank: bool,
) -> _MethodScores:
    chain = RagChain(
        retriever, method=method, top_k=_TOP_K, rerank=rerank, resolve_parent_context=True
    )

    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    faithfulness_scores: list[float] = []
    relevance_scores: list[float] = []
    false_refusals = 0
    judge_failures = 0

    for item in items:
        results = retriever.retrieve(item["question"], method=method, top_k=_TOP_K, rerank=rerank)
        recalls.append(recall_at_k(results, item["expected_sources"]))
        mrrs.append(mrr(results, item["expected_sources"]))
        ndcgs.append(ndcg_at_k(results, item["expected_sources"]))

        rag_answer = chain.answer(item["question"])
        if rag_answer.refused:
            # A refusal makes no factual claims and isn't trying to
            # address the question -- scoring it against judges built for
            # substantive answers would be meaningless, not honest. It's
            # still a real failure for an item that WAS answerable, just
            # a different one -- tracked separately below.
            false_refusals += 1
            continue

        context_text = "\n\n".join(
            format_context_block(i, r) for i, r in enumerate(rag_answer.retrieved_chunks, start=1)
        )
        try:
            faithfulness_scores.append(score_faithfulness(rag_answer.answer, context_text))
            relevance_scores.append(score_relevance(item["question"], rag_answer.answer))
        except JudgeParseError:
            judge_failures += 1

    n = len(items)
    hallucination_rate = (
        sum(1 for s in faithfulness_scores if s < _HALLUCINATION_THRESHOLD)
        / len(faithfulness_scores)
        if faithfulness_scores
        else 0.0
    )
    return _MethodScores(
        label=f"{method.value}{' +rerank' if rerank else ''}",
        recall=sum(recalls) / n,
        mrr=sum(mrrs) / n,
        ndcg=sum(ndcgs) / n,
        faithfulness=sum(faithfulness_scores) / len(faithfulness_scores)
        if faithfulness_scores
        else 0.0,
        relevance=sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0.0,
        hallucination_rate=hallucination_rate,
        false_refusals=false_refusals,
        judge_failures=judge_failures,
    )


def _print_table(rows: list[_MethodScores]) -> None:
    columns = [
        ("Method", 20, "<"),
        (f"Recall@{_TOP_K}", 11, ".3f"),
        ("MRR", 8, ".3f"),
        (f"nDCG@{_TOP_K}", 10, ".3f"),
        ("Faithful", 10, ".3f"),
        ("Relevant", 10, ".3f"),
        ("Halluc.", 9, ".3f"),
    ]
    header = "".join(f"{name:<{width}}" for name, width, _ in columns)
    print(f"\n{header}")
    print("-" * len(header))
    for row in rows:
        values = [
            row.label,
            row.recall,
            row.mrr,
            row.ndcg,
            row.faithfulness,
            row.relevance,
            row.hallucination_rate,
        ]
        line = "".join(
            f"{value:<{width}}" if fmt == "<" else f"{value:<{width}{fmt}}"
            for value, (_, width, fmt) in zip(values, columns, strict=True)
        )
        print(line)
        if row.false_refusals or row.judge_failures:
            print(
                f"{'':<20}({row.false_refusals} false refusal(s), "
                f"{row.judge_failures} judge parse failure(s) excluded from the averages above)"
            )


def main() -> None:
    golden_set = _load_golden_set()
    retrieval_items = [item for item in golden_set if not item["expect_refusal"]]
    print(
        f"Loaded {len(golden_set)} golden items "
        f"({len(retrieval_items)} usable for retrieval/generation metrics -- "
        f"the rest are refusal-only, no expected_sources to score against)."
    )

    chunks = [chunk for filename in _CORPUS_FILES for chunk in _ingest_one(filename)]
    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])

    vector_store = get_vector_store(collection_name=_COLLECTION)
    vector_store.create_collection(dimension=vectors[0].dimension, indexing_threshold=0)
    keyword_store = get_keyword_store(index_name=_COLLECTION)
    keyword_store.create_index()
    HybridIndexer(vector_store, keyword_store).index(chunks, vectors)
    vector_store.publish()

    retriever = Retriever(vector_store, keyword_store, embedder)

    rows = [
        _score_method(retriever, retrieval_items, method, rerank=False)
        for method in RetrievalMethod
    ]

    try:
        reranker = get_reranker()
    except NotImplementedError as exc:
        print(f"\nSkipping hybrid_rrf +rerank row: {exc}")
    else:
        reranked_retriever = Retriever(vector_store, keyword_store, embedder, reranker=reranker)
        rows.append(
            _score_method(
                reranked_retriever, retrieval_items, RetrievalMethod.HYBRID_RRF, rerank=True
            )
        )

    _print_table(rows)


if __name__ == "__main__":
    main()
