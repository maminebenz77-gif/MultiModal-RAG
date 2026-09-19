"""Runs each expertise's real documents and expert-authored Q&A through the
REAL production AgentChain (the same construction routers/query.py uses,
not RagChain -- see run_eval.py/langfuse_experiment.py for the controlled
retrieval-method comparison that deliberately holds generation fixed
instead), and judges the agent's actual answer against a human expert's
reference answer.

Different purpose from data/golden_set.json's synthetic comparison: that
one asks "which retrieval method is best?" (relative, synthetic corpus).
This one asks "is the agent a real user actually gets correct, on real
domain content?" (absolute, real corpus, production config).

Convention -- see data/eval/README.md -- lets a new expertise or question
be added as a pure data change, no code:

    data/eval/<expertise-name>/
      documents/   # real source documents (PDF, DOCX, PPTX, or Markdown)
      qa.json       # [{"id": ..., "question": ..., "expert_answer": ...}, ...]

Run: `uv run python -m multimodal_rag.evaluation.run_expert_eval`
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ..generation.agent import AgentChain
from ..providers.factory import get_embedder
from ..retrieval.retriever import Retriever
from ..retrieval.schema import RetrievalMethod
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from .judge import JudgeParseError, score_answer_correctness
from .run_eval import _ingest_document

_EVAL_ROOT = Path(__file__).resolve().parents[3] / "data" / "eval"
_TOP_K = 5
"""Matches QueryRequest's real default (api/schemas.py), unlike
run_eval.py's _TOP_K=3 -- that one is deliberately tuned to make recall@k
discriminate between methods; this eval is about the agent a real user
actually gets, so it uses exactly the config routers/query.py defaults to."""


@dataclass
class _ExpertiseResult:
    name: str
    average_correctness: float
    question_count: int
    judge_failures: int
    """Questions skipped because the judge's own output couldn't be
    parsed (see judge.JudgeParseError) -- excluded from the average
    rather than silently corrupting it, same discipline as run_eval.py."""


def _discover_expertise_dirs() -> list[Path]:
    if not _EVAL_ROOT.is_dir():
        return []
    return sorted(
        p
        for p in _EVAL_ROOT.iterdir()
        if p.is_dir() and (p / "qa.json").is_file() and (p / "documents").is_dir()
    )


def _load_qa(expertise_dir: Path) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.loads((expertise_dir / "qa.json").read_text()))


def _run_expertise(expertise_dir: Path) -> _ExpertiseResult:
    name = expertise_dir.name
    qa_items = _load_qa(expertise_dir)
    documents_dir = expertise_dir / "documents"
    chunks = [
        chunk
        for path in sorted(documents_dir.iterdir())
        if path.is_file() and not path.name.startswith(".")
        for chunk in _ingest_document(path)
    ]

    # Scoped per expertise, deliberately -- a question about one expertise
    # must not be able to accidentally retrieve another expertise's
    # documents just because they happen to share a collection.
    collection = f"expert_eval_{name}"
    embedder = get_embedder()
    vectors = embedder.embed([c.text for c in chunks])

    vector_store = get_vector_store(collection_name=collection)
    vector_store.create_collection(dimension=vectors[0].dimension, indexing_threshold=0)
    keyword_store = get_keyword_store(index_name=collection)
    keyword_store.create_index()
    HybridIndexer(vector_store, keyword_store).index(chunks, vectors)
    vector_store.publish()

    retriever = Retriever(vector_store, keyword_store, embedder)
    agent = AgentChain(
        retriever,
        method=RetrievalMethod.HYBRID_RRF,
        top_k=_TOP_K,
        rerank=False,
        resolve_parent_context=True,
    )

    scores: list[float] = []
    judge_failures = 0
    for item in qa_items:
        rag_answer = agent.answer(item["question"])
        try:
            score = score_answer_correctness(
                item["question"], rag_answer.answer, item["expert_answer"]
            )
        except JudgeParseError:
            judge_failures += 1
            continue
        scores.append(score)

    return _ExpertiseResult(
        name=name,
        average_correctness=sum(scores) / len(scores) if scores else 0.0,
        question_count=len(qa_items),
        judge_failures=judge_failures,
    )


def _print_results(results: list[_ExpertiseResult]) -> None:
    columns = [("Expertise", 25, "<"), ("Correctness", 13, ".3f"), ("Questions", 10, "d")]
    header = "".join(f"{name:<{width}}" for name, width, _ in columns)
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        values = [r.name, r.average_correctness, r.question_count]
        print(
            "".join(
                f"{value:<{width}}" if fmt in ("<", "d") else f"{value:<{width}{fmt}}"
                for value, (_, width, fmt) in zip(values, columns, strict=True)
            )
        )
        if r.judge_failures:
            print(f"{'':<25}({r.judge_failures} judge parse failure(s) excluded from the average)")

    total_questions = sum(r.question_count for r in results)
    if total_questions:
        overall = sum(r.average_correctness * r.question_count for r in results) / total_questions
        print("-" * len(header))
        print(f"{'Overall':<25}{overall:<13.3f}{total_questions:<10}")


def main() -> None:
    expertise_dirs = _discover_expertise_dirs()
    if not expertise_dirs:
        print(
            f"No expertise folders found under {_EVAL_ROOT}. See "
            f"{_EVAL_ROOT / 'README.md'} for the folder convention -- no code changes "
            "needed to add a new expertise or a new question."
        )
        return

    results = [_run_expertise(d) for d in expertise_dirs]
    _print_results(results)


if __name__ == "__main__":
    main()
