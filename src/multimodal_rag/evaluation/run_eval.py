"""Retrieval-method comparison against the golden set (data/golden_set.json).

Ingests chunking_demo.md + sample.md into a scratch collection, then for
every RetrievalMethod (plus one hybrid_rrf+rerank row, if a Reranker is
configured for this profile) runs every non-refusal golden question and
prints recall@k / MRR / nDCG@k per method -- so "hybrid+rerank beat plain
cosine" is a number from this table, not a guess.

Generation-layer columns (faithfulness/relevance/hallucination) land in a
later step; this one is retrieval-only, deliberately -- no LLM calls, fast
to run and easy to trust in isolation.

Run: `uv run python -m multimodal_rag.evaluation.run_eval`
"""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

from ..chunking.parent_child import ParentChildChunker
from ..chunking.schema import Chunk
from ..ingestion import parse_document
from ..providers.factory import get_embedder, get_reranker
from ..retrieval.retriever import Retriever
from ..retrieval.schema import RetrievalMethod
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from .retrieval_metrics import mrr, ndcg_at_k, recall_at_k

_GOLDEN_SET_PATH = Path(__file__).resolve().parents[3] / "data" / "golden_set.json"
_SAMPLE_DOCS_DIR = Path(__file__).resolve().parents[3] / "data" / "samples"
_CORPUS_FILES = ["chunking_demo.md", "sample.md"]
_COLLECTION = "retrieval_eval"
_TOP_K = 5


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
) -> tuple[str, float, float, float]:
    recalls: list[float] = []
    mrrs: list[float] = []
    ndcgs: list[float] = []
    for item in items:
        results = retriever.retrieve(item["question"], method=method, top_k=_TOP_K, rerank=rerank)
        recalls.append(recall_at_k(results, item["expected_sources"]))
        mrrs.append(mrr(results, item["expected_sources"]))
        ndcgs.append(ndcg_at_k(results, item["expected_sources"]))
    label = f"{method.value}{' +rerank' if rerank else ''}"
    n = len(items)
    return label, sum(recalls) / n, sum(mrrs) / n, sum(ndcgs) / n


def _print_table(rows: list[tuple[str, float, float, float]]) -> None:
    header = f"{'Method':<20}{'Recall@' + str(_TOP_K):<12}{'MRR':<10}{'nDCG@' + str(_TOP_K):<10}"
    print(f"\n{header}")
    print("-" * len(header))
    for label, recall, mrr_score, ndcg in rows:
        print(f"{label:<20}{recall:<12.3f}{mrr_score:<10.3f}{ndcg:<10.3f}")


def main() -> None:
    golden_set = _load_golden_set()
    retrieval_items = [item for item in golden_set if not item["expect_refusal"]]
    print(
        f"Loaded {len(golden_set)} golden items "
        f"({len(retrieval_items)} usable for retrieval metrics -- "
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
