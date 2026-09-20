"""Runs the golden set as native Langfuse Experiments -- one dataset run
per retrieval method, so "hybrid+rerank beat cosine on faithfulness" is a
comparison Langfuse's own UI renders natively (per-run averages, run-vs-
run diffing over time), not just a table run_eval.py prints once and
forgets.

Reuses this project's own scoring functions (retrieval_metrics.py,
judge.py) as the actual evaluation logic -- Langfuse's run_experiment()
supplies the harness (dataset sync, per-item tracing, score attachment,
run comparison), not the scoring itself. A no-op, loudly, if Langfuse
isn't configured (see tracing.py) -- this script's whole point is
sending data to Langfuse, so unlike every other tracing call site in
this codebase, there's nothing useful left to do if it isn't.

Run: `uv run python -m multimodal_rag.evaluation.langfuse_experiment`
"""

from datetime import UTC, datetime
from typing import Any

from langfuse import Evaluation, Langfuse
from langfuse.experiment import EvaluatorFunction

from ..generation.chain import RagChain
from ..generation.context import format_context_block
from ..providers.factory import get_embedder, get_reranker
from ..retrieval.retriever import Retriever
from ..retrieval.schema import RetrievalMethod
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from ..tracing import get_langfuse_client
from .judge import JudgeParseError, score_faithfulness, score_relevance
from .retrieval_metrics import mrr, ndcg_at_k, recall_at_k
from .run_eval import (
    _CORPUS_FILES,
    _HALLUCINATION_THRESHOLD,
    _SAMPLE_DOCS_DIR,
    _TOP_K,
    _ingest_document,
    _load_golden_set,
)

_DATASET_NAME = "rag-golden-set"
_COLLECTION = "langfuse_experiment"
_EXPERIMENT_NAME = "Multimodal RAG retrieval comparison"


def _run_name(label: str) -> str:
    # Timestamped so re-running this script shows up as a new,
    # distinguishable Dataset Run each time -- see
    # run_expert_eval.py's identical _run_name for the live-caught
    # bug this fixes (a bare method name meant every run silently reused
    # the same run identity).
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{label}-{timestamp}"


class _SourceOnly:
    """The lightweight stand-in retrieval_metrics.py's HasSource Protocol
    exists for -- reconstructed from a task's JSON-serializable output
    (see _make_task) rather than passing real SearchResult objects
    through, so a trace's `output` never carries SearchResult.elements'
    possible base64 image data.
    """

    def __init__(self, source: str) -> None:
        self.source = source


def _sync_dataset(client: Langfuse, golden_set: list[dict[str, Any]]) -> None:
    """Both create_dataset and create_dataset_item upsert by name/id --
    verified live, calling either twice with the same name/id updates
    the existing one in place rather than duplicating it -- so this is
    safe to call on every run. Each golden item's own `id` is reused as
    the dataset item id, so an edit to golden_set.json (a new question,
    a corrected expected_sources list) shows up here on the next run
    too, not just a one-time upload.
    """
    client.create_dataset(
        name=_DATASET_NAME,
        description="Golden set for retrieval/generation comparison (data/golden_set.json).",
    )
    for item in golden_set:
        client.create_dataset_item(
            dataset_name=_DATASET_NAME,
            id=item["id"],
            input=item["question"],
            expected_output={
                "expected_sources": item["expected_sources"],
                "expect_refusal": item["expect_refusal"],
            },
        )


def _make_task(chain: RagChain) -> Any:
    def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        rag_answer = chain.answer(item.input)
        context_text = "\n\n".join(
            format_context_block(i, r) for i, r in enumerate(rag_answer.retrieved_chunks, start=1)
        )
        return {
            "answer": rag_answer.answer,
            "refused": rag_answer.refused,
            "retrieved_sources": [r.source for r in rag_answer.retrieved_chunks],
            "context_text": context_text,
        }

    return task


def _retrieval_evaluator(
    *, output: dict[str, Any], expected_output: dict[str, Any], **kwargs: Any
) -> list[Evaluation]:
    if expected_output.get("expect_refusal"):
        # Undefined without expected_sources -- see retrieval_metrics.py.
        return []
    expected_sources = expected_output["expected_sources"]
    retrieved = [_SourceOnly(s) for s in output["retrieved_sources"]]
    return [
        Evaluation(name="recall_at_k", value=recall_at_k(retrieved, expected_sources)),
        Evaluation(name="mrr", value=mrr(retrieved, expected_sources)),
        Evaluation(name="ndcg_at_k", value=ndcg_at_k(retrieved, expected_sources)),
    ]


def _refusal_accuracy_evaluator(
    *, output: dict[str, Any], expected_output: dict[str, Any], **kwargs: Any
) -> list[Evaluation]:
    if not expected_output.get("expect_refusal"):
        return []
    correct = bool(output["refused"])
    return [Evaluation(name="refusal_accuracy", value=1.0 if correct else 0.0)]


def _faithfulness_evaluator(
    *, output: dict[str, Any], expected_output: dict[str, Any], **kwargs: Any
) -> list[Evaluation]:
    if expected_output.get("expect_refusal") or output["refused"]:
        # A refusal makes no factual claims -- judging its "faithfulness"
        # would be meaningless, not honest. See refusal_accuracy_evaluator
        # for the check that actually applies to this case.
        return []
    try:
        score = score_faithfulness(output["answer"], output["context_text"])
    except JudgeParseError:
        return []
    return [
        Evaluation(name="faithfulness", value=score),
        Evaluation(
            name="hallucinated", value=score < _HALLUCINATION_THRESHOLD, data_type="BOOLEAN"
        ),
    ]


def _relevance_evaluator(
    *, input: Any, output: dict[str, Any], expected_output: dict[str, Any], **kwargs: Any
) -> list[Evaluation]:
    if expected_output.get("expect_refusal") or output["refused"]:
        return []
    try:
        score = score_relevance(input, output["answer"])
    except JudgeParseError:
        return []
    return [Evaluation(name="relevance", value=score)]


_EVALUATORS: list[EvaluatorFunction] = [
    _retrieval_evaluator,
    _refusal_accuracy_evaluator,
    _faithfulness_evaluator,
    _relevance_evaluator,
]


def main() -> None:
    client = get_langfuse_client()
    if client is None:
        print(
            "Langfuse isn't configured (see tracing.py) -- nothing to send. Set "
            "LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST in .env.local, or run "
            "`uv run python -m multimodal_rag.evaluation.run_eval` for the "
            "console-only comparison instead."
        )
        return

    golden_set = _load_golden_set()
    _sync_dataset(client, golden_set)
    dataset = client.get_dataset(_DATASET_NAME)

    chunks = [
        chunk
        for filename in _CORPUS_FILES
        for chunk in _ingest_document(_SAMPLE_DOCS_DIR / filename)
    ]
    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])

    vector_store = get_vector_store(collection_name=_COLLECTION)
    vector_store.create_collection(dimension=vectors[0].dimension, indexing_threshold=0)
    keyword_store = get_keyword_store(index_name=_COLLECTION)
    keyword_store.create_index()
    HybridIndexer(vector_store, keyword_store).index(chunks, vectors)
    vector_store.publish()

    retriever = Retriever(vector_store, keyword_store, embedder)

    for method in RetrievalMethod:
        chain = RagChain(retriever, method=method, top_k=_TOP_K, resolve_parent_context=True)
        result = dataset.run_experiment(
            name=_EXPERIMENT_NAME,
            run_name=_run_name(method.value),
            task=_make_task(chain),
            evaluators=_EVALUATORS,
        )
        print(result.format())

    try:
        reranker = get_reranker()
    except NotImplementedError as exc:
        print(f"\nSkipping hybrid_rrf +rerank run: {exc}")
    else:
        reranked_retriever = Retriever(vector_store, keyword_store, embedder, reranker=reranker)
        chain = RagChain(
            reranked_retriever,
            method=RetrievalMethod.HYBRID_RRF,
            top_k=_TOP_K,
            rerank=True,
            resolve_parent_context=True,
        )
        result = dataset.run_experiment(
            name=_EXPERIMENT_NAME,
            run_name=_run_name("hybrid_rrf+rerank"),
            task=_make_task(chain),
            evaluators=_EVALUATORS,
        )
        print(result.format())

    client.flush()


if __name__ == "__main__":
    main()
