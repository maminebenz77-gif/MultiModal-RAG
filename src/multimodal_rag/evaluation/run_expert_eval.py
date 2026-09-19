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
      qa.json       # [{"id": ..., "question": ..., "expert_answer": ...,
                     #   "expect_refusal": false}, ...]  -- expect_refusal
                     #   optional, defaults to false

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
    answerable_count: int
    refusal_accuracy: float | None
    """None when this expertise has no expect_refusal items at all --
    distinct from 0.0 (every refusal item was answered wrong), which
    run_eval.py's analogous field conflates since its golden set always
    has at least one expect_refusal item in practice. Per-expertise data
    here won't reliably have one, so the print layer needs to tell "not
    applicable" apart from "failed every one"."""
    refusal_count: int
    false_refusals: int
    """Answerable items (expect_refusal=false) the agent refused anyway --
    a real failure (it found nothing usable even though an answer
    exists), excluded from average_correctness rather than silently
    scoring "I don't know" against a substantive reference answer. See
    run_eval.py's _MethodScores.false_refusals for the same reasoning."""
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


def build_expertise_agent(expertise_dir: Path) -> AgentChain:
    """Ingests an expertise folder's documents into a collection scoped to
    it alone -- deliberately, so a question about one expertise can't
    accidentally retrieve another expertise's content just because they
    happen to share a collection -- and returns the real production
    AgentChain over it (routers/query.py's exact construction). Shared by
    run_expert_eval.py and langfuse_expert_eval.py so both eval every
    expertise against the same agent, not two subtly different ones.
    """
    name = expertise_dir.name
    documents_dir = expertise_dir / "documents"
    chunks = [
        chunk
        for path in sorted(documents_dir.iterdir())
        if path.is_file() and not path.name.startswith(".")
        for chunk in _ingest_document(path)
    ]

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
    return AgentChain(
        retriever,
        method=RetrievalMethod.HYBRID_RRF,
        top_k=_TOP_K,
        rerank=False,
        resolve_parent_context=True,
    )


def _run_expertise(expertise_dir: Path) -> _ExpertiseResult:
    name = expertise_dir.name
    qa_items = _load_qa(expertise_dir)
    agent = build_expertise_agent(expertise_dir)

    answerable_items = [item for item in qa_items if not item.get("expect_refusal", False)]
    refusal_items = [item for item in qa_items if item.get("expect_refusal", False)]

    scores: list[float] = []
    false_refusals = 0
    judge_failures = 0
    for item in answerable_items:
        rag_answer = agent.answer(item["question"])
        if rag_answer.refused:
            # A refusal makes no substantive claims -- grading "I don't
            # know" against a substantive expert_answer with a free-text
            # judge is exactly the failure mode that showed up live: the
            # judge marks a correct-sounding refusal wrong because it
            # doesn't restate the reference's reasoning. Track it as its
            # own, real failure instead.
            false_refusals += 1
            continue
        try:
            score = score_answer_correctness(
                item["question"], rag_answer.answer, item["expert_answer"]
            )
        except JudgeParseError:
            judge_failures += 1
            continue
        scores.append(score)

    correct_refusals = sum(1 for item in refusal_items if agent.answer(item["question"]).refused)

    return _ExpertiseResult(
        name=name,
        average_correctness=sum(scores) / len(scores) if scores else 0.0,
        answerable_count=len(answerable_items),
        refusal_accuracy=correct_refusals / len(refusal_items) if refusal_items else None,
        refusal_count=len(refusal_items),
        false_refusals=false_refusals,
        judge_failures=judge_failures,
    )


def _fmt_refusal_accuracy(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


_COLUMN_WIDTHS = {"Expertise": 25, "Correctness": 13, "Answerable": 12, "RefusalAcc": 12}


def _print_results(results: list[_ExpertiseResult]) -> None:
    header = "".join(f"{name:<{width}}" for name, width in _COLUMN_WIDTHS.items())
    print(f"\n{header}")
    print("-" * len(header))
    for r in results:
        row = {
            "Expertise": r.name,
            "Correctness": f"{r.average_correctness:.3f}",
            "Answerable": str(r.answerable_count),
            "RefusalAcc": _fmt_refusal_accuracy(r.refusal_accuracy),
        }
        print("".join(f"{row[name]:<{width}}" for name, width in _COLUMN_WIDTHS.items()))
        if r.false_refusals or r.judge_failures:
            print(
                f"{'':<25}({r.false_refusals} false refusal(s), "
                f"{r.judge_failures} judge parse failure(s) excluded from the average above)"
            )

    total_answerable = sum(r.answerable_count for r in results)
    if total_answerable:
        overall = (
            sum(r.average_correctness * r.answerable_count for r in results) / total_answerable
        )
        print("-" * len(header))
        print(f"{'Overall':<25}{overall:<13.3f}{total_answerable:<12}")


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
