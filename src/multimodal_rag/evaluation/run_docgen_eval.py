"""Runs each docgen eval set's real two-source documents and expert-authored
Q&A through the REAL docgen graph (the same graph the CLI/API drive, not a
stand-in), and judges each question's final answer against a human expert's
reference answer -- the docgen analogue of run_expert_eval.py, extended for
docgen's two-source-set, multi-question-per-run shape.

Convention -- see data/eval/README.md for the general shape this follows,
and this module's own docstring differences below:

    data/eval/<name>/
      documents/task/        # required -- task_docs source
      documents/reference/   # optional -- reference_kb source, for
                              # comparison-style questions
      docgen_qa.json          # [{"id", "question", "sources_required",
                               #   "expert_answer", "expect_refusal"}, ...]
                               # sources_required defaults to ["task_docs"].
                               # A DIFFERENT filename from the plain agent
                               # eval's qa.json, deliberately -- so the same
                               # data/eval/ root can hold both kinds of eval
                               # set without one's discovery picking up the
                               # other's folder by accident.

TWO separate runs happen per eval set, for two different reasons:

1. **The full pipeline, once, locally** -- the real pass/fail signal. Builds
   a CONFIRMED DocGenState directly from docgen_qa.json (skipping
   interpret_request/reformulate_for_confirmation entirely -- this eval is
   about per-question answer quality and the harmonize/review/escalation
   machinery, not about whether free text gets parsed into the right
   question list, which is this module's business, not this one's), then
   drives the compiled graph to completion with a FIXED, unattended resume
   policy: ask_human always gets "skip" (there is no human; a question
   that exhausts its retries is exactly what should happen for a genuinely
   unanswerable question, and exactly what SHOULDN'T happen for an
   answerable one -- scored as a false refusal when it does), human_review
   always gets "approve" (never "edit" -- this measures the FIRST-PASS
   answer quality, not a human-assisted second pass). This is what the
   CLI/CI pass/fail and the printed table's numbers are based on.

2. **One question at a time, via Langfuse's run_experiment()** -- for
   per-question tracing/visibility in the Langfuse UI, mirroring
   run_expert_eval.py's one-task-call-per-dataset-item model exactly. This
   does NOT run the full graph (harmonize_answers and human_review are
   necessarily whole-run concepts, not single-question ones) -- it runs
   exactly one retrieve -> generate_answer -> validate cycle per question
   (no retry), reusing docgen's own node functions directly. Its scoring
   uses the exact same _score_item() as the full-pipeline run, so
   "correct"/"a false refusal" mean the same thing in both places; it's a
   narrower, visibility-only view, not a second source of truth. The
   printed table's numbers come from run 1, never run 2.

Reuses run_expert_eval.py's _ItemOutcome/_ExpertiseResult dataclasses,
_aggregate/_print_results, _connect_langfuse, and _expected_output
directly -- these are private (underscore) names, but they're generic
plumbing with nothing AgentChain-specific baked in, and importing the
real functions keeps the two harnesses' aggregation/printing from
quietly drifting apart.

Deliberately does NOT reuse run_expert_eval.py's own
_correctness_evaluator/_refusal_accuracy_evaluator, even though they
were the obvious first instinct -- a live run against
data/eval/docgen-compliance-report caught why that's wrong before this
module ever shipped: those two evaluators are built around AgentChain's
explicit `refused: bool` field, which docgen simply has no equivalent
of. Confirmed empirically, not just in theory -- docgen's own
validate_answer judges an HONEST "that isn't covered by the documents"
answer as perfectly grounded (it doesn't claim anything the context
doesn't support), so it passes validation and the question's status
becomes "answered" exactly like any other accepted answer. A question
only ends up "escalated"/"skipped" when the model couldn't even produce
a GROUNDED admission of absence after retrying -- a rarer, worse
outcome, not the normal shape of a correct refusal. So an
expect_refusal item is scored the SAME way as any other question --
_score_item() below compares the final answer's substance against the
expert's own explanation of why there's no answer via
score_answer_correctness(), which naturally rewards an honest admission
and penalizes a fabricated one. Only a question that never got a
validated answer at all (escalated/skipped) skips the judge call
outright -- no fabrication is possible if nothing was ever accepted,
so that's scored as a correct refusal directly, and as a false refusal
for a question that WAS supposed to be answerable.

Run: `uv run python -m multimodal_rag.evaluation.run_docgen_eval`
"""

import json
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from langfuse import Evaluation, Langfuse
from langfuse.experiment import EvaluatorFunction
from langgraph.types import Command, RunnableConfig

from ..docgen.checkpointer import build_checkpointer
from ..docgen.graph import build_graph
from ..docgen.ingest import ingest_folder
from ..docgen.nodes.answering import formulate_query, generate_answer_text
from ..docgen.nodes.retrieval import retrieve_for_question
from ..docgen.sources import SourceRole, SourceSpec
from ..docgen.stack import DocgenStack, build_stack
from ..docgen.state import DocGenState, Question
from ..docgen.validation import ValidationParseError, validate_answer
from ..providers.factory import get_llm
from ..stores.factory import get_keyword_store
from .judge import JudgeParseError, score_answer_correctness
from .run_expert_eval import (
    _aggregate,
    _connect_langfuse,
    _expected_output,
    _ExpertiseResult,
    _ItemOutcome,
    _print_results,
)

_EVAL_ROOT = Path(__file__).resolve().parents[3] / "data" / "eval"

_DATASET_PREFIX = "docgen-eval-"
_EXPERIMENT_NAME_PREFIX = "Docgen eval: "


def _discover_docgen_eval_dirs() -> list[Path]:
    if not _EVAL_ROOT.is_dir():
        return []
    return sorted(
        p
        for p in _EVAL_ROOT.iterdir()
        if p.is_dir() and (p / "docgen_qa.json").is_file() and (p / "documents" / "task").is_dir()
    )


def _load_qa(eval_dir: Path) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], json.loads((eval_dir / "docgen_qa.json").read_text()))


def _sources_required(item: dict[str, Any]) -> list[SourceRole]:
    return cast(list[SourceRole], item.get("sources_required", ["task_docs"]))


# -- stack + source setup ----------------------------------------------------


def build_eval_stack_and_sources(
    eval_dir: Path, tmp_dir: Path
) -> tuple[DocgenStack, str, dict[SourceRole, SourceSpec]]:
    """Ingests documents/task/ (required) and documents/reference/
    (optional) into a collection scoped to this eval set alone -- same
    per-set isolation reasoning as run_expert_eval.py's
    build_expertise_agent, so a question about one eval set can't
    accidentally retrieve another's content. Returns collection_name
    alongside the stack since no store object exposes it as an
    attribute -- _cleanup_stack needs the literal name it was built
    with, not something recovered from the stack afterward."""
    name = eval_dir.name
    collection_name = f"docgen_eval_{name}_{uuid.uuid4().hex[:8]}"
    stack = build_stack(collection_name=collection_name, db_path=tmp_dir / "docgen_state.db")

    sources: dict[SourceRole, SourceSpec] = {}
    task_dir = eval_dir / "documents" / "task"
    task_tag = f"docgen-eval:{name}:task"
    ingest_folder(task_dir, task_tag, "public", stack)
    sources["task_docs"] = SourceSpec(role="task_docs", tag=task_tag, required=True)

    reference_dir = eval_dir / "documents" / "reference"
    if reference_dir.is_dir():
        reference_tag = f"docgen-eval:{name}:reference"
        ingest_folder(reference_dir, reference_tag, "public", stack)
        sources["reference_kb"] = SourceSpec(role="reference_kb", tag=reference_tag, required=False)

    return stack, collection_name, sources


def _cleanup_stack(collection_name: str) -> None:
    """Same getattr-based duck-typing as api/routers/ingest.py's
    embedder-override compatibility check -- get_keyword_store()'s
    return type is the abstract KeywordStore interface, which doesn't
    declare _current_alias_target/_client (only the concrete
    Elasticsearch-backed implementation does)."""
    backend = get_keyword_store(index_name=collection_name)
    physical = getattr(backend, "_current_alias_target", lambda: None)()
    client = getattr(backend, "_client", None)
    if physical is not None and client is not None:
        client.indices.delete(index=physical, ignore_unavailable=True)


# -- run 1: the full pipeline, once, locally --------------------------------


def _build_confirmed_state(
    qa_items: list[dict[str, Any]], sources: dict[SourceRole, SourceSpec]
) -> DocGenState:
    questions: list[Question] = [
        {
            "id": item["id"],
            "text": item["question"],
            "sources_required": _sources_required(item),
            "status": "pending",
        }
        for item in qa_items
    ]
    return {
        "sources": list(sources.values()),
        "questions": questions,
        "answers": {},
        "configuration": {"template": "", "format": "docx", "confirmed": True, "max_retries": 3},
        "review": {"decision": "pending", "flagged_question_ids": [], "guidance": None},
        "current": None,
        "usage": {"llm_calls": 0},
        "request": {"text": "", "corrections": []},
        "output_path": None,
    }


def _run_full_pipeline(
    stack: DocgenStack, initial_state: DocGenState, checkpoint_path: Path
) -> dict[str, Any]:
    """Drives the compiled graph to completion unattended -- ask_human
    always gets "skip" (no human exists; see module docstring for why
    that's the right default here, not a workaround), human_review
    always gets "approve" (measuring first-pass quality, not a
    human-assisted second pass)."""
    config: RunnableConfig = {"configurable": {"thread_id": f"eval-{uuid.uuid4().hex[:8]}"}}
    with build_checkpointer(checkpoint_path) as checkpointer:
        graph = build_graph(stack, checkpointer=checkpointer)
        result: dict[str, Any] = graph.invoke(initial_state, config)
        while "__interrupt__" in result:
            kind = result["__interrupt__"][0].value["kind"]
            if kind == "ask_human":
                resume: dict[str, Any] = {"action": "skip", "text": ""}
            elif kind == "human_review":
                resume = {"action": "approve", "question_ids": [], "text": ""}
            else:
                raise AssertionError(
                    f"Unexpected pending interrupt kind in an automated eval run: {kind!r} -- "
                    "confirmed=True in the initial state should make confirm_configuration "
                    "unreachable."
                )
            result = graph.invoke(Command(resume=resume), config)
    return result


def _score_item(
    question: str, answer_text: str | None, expert_answer: str, expect_refusal: bool
) -> list[Evaluation]:
    """The one place docgen's own notion of "correctly handled" is
    decided -- see this module's docstring for why an expect_refusal
    item is scored by SUBSTANCE against the expert's own explanation,
    not by a `refused` flag docgen doesn't have. `answer_text` is None
    when the question never got a validated answer at all (escalated or
    skipped) -- no fabrication is possible if nothing was ever accepted,
    so that's a correct refusal outright for an expect_refusal item, and
    a false refusal (a real failure) for one that was supposed to be
    answerable."""
    evaluation_name = "refusal_accuracy" if expect_refusal else "correctness"
    if answer_text is None:
        if expect_refusal:
            return [Evaluation(name=evaluation_name, value=1.0)]
        return [Evaluation(name="false_refusal", value=True, data_type="BOOLEAN")]
    try:
        score = score_answer_correctness(question, answer_text, expert_answer)
    except JudgeParseError:
        return []
    return [Evaluation(name=evaluation_name, value=score)]


def _full_pipeline_outcomes(
    qa_items: list[dict[str, Any]], final_state: dict[str, Any]
) -> list[_ItemOutcome]:
    questions_by_id = {q["id"]: q for q in final_state["questions"]}
    answers = final_state["answers"]
    outcomes = []
    for item in qa_items:
        question = questions_by_id[item["id"]]
        expect_refusal = bool(item.get("expect_refusal", False))
        answered = question["status"] == "answered"
        answer_text = answers[item["id"]]["text"] if answered else None
        evaluations = _score_item(
            item["question"], answer_text, item["expert_answer"], expect_refusal
        )
        outcomes.append(_ItemOutcome(expect_refusal, evaluations))
    return outcomes


# -- run 2: one question at a time, via Langfuse's run_experiment() ---------


def _input_payload(item: dict[str, Any]) -> dict[str, Any]:
    return {"question": item["question"], "sources_required": _sources_required(item)}


def _make_single_question_task(stack: DocgenStack, sources: dict[SourceRole, SourceSpec]) -> Any:
    """One retrieve -> generate_answer -> validate cycle, no retry --
    `answer` is None (same "never got a validated answer" meaning
    _score_item gives it in the full-pipeline run) when there are no
    chunks to work with, or validate_answer calls it ungrounded, or the
    judge's own output can't be parsed at all."""
    llm = get_llm()

    def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        question = item.input["question"]
        required_roles = item.input["sources_required"]
        active_sources = [sources[role] for role in required_roles if role in sources]
        query = formulate_query(question, [], None, None, llm)
        chunks = retrieve_for_question(query, active_sources, stack.retriever)
        if not chunks:
            return {"answer": None}
        answer = generate_answer_text(question, chunks, None, None, llm)
        try:
            result = validate_answer(question, answer, chunks, llm)
        except ValidationParseError:
            return {"answer": None}
        return {"answer": answer if result.valid else None}

    return task


def _single_question_evaluator(
    *, input: Any, output: dict[str, Any], expected_output: dict[str, Any], **kwargs: Any
) -> list[Evaluation]:
    return _score_item(
        input["question"],
        output["answer"],
        expected_output["expert_answer"],
        bool(expected_output.get("expect_refusal", False)),
    )


_SINGLE_QUESTION_EVALUATORS: list[EvaluatorFunction] = [_single_question_evaluator]


def _dataset_name(name: str) -> str:
    return f"{_DATASET_PREFIX}{name}"


def _run_name(name: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{name}-{timestamp}"


def _sync_dataset(client: Langfuse, name: str, qa_items: list[dict[str, Any]]) -> None:
    dataset_name = _dataset_name(name)
    client.create_dataset(
        name=dataset_name,
        description=(
            f"Docgen eval Q&A for '{name}' (data/eval/{name}/docgen_qa.json) -- per-question "
            "tracing only; the authoritative pass/fail comes from the full pipeline run "
            "printed to the console, see run_docgen_eval.py's module docstring."
        ),
    )
    for item in qa_items:
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item["id"],
            input=_input_payload(item),
            expected_output=_expected_output(item),
        )


def _run_on_langfuse(
    client: Langfuse,
    name: str,
    stack: DocgenStack,
    sources: dict[SourceRole, SourceSpec],
    qa_items: list[dict[str, Any]],
) -> str | None:
    _sync_dataset(client, name, qa_items)
    dataset = client.get_dataset(_dataset_name(name))
    current_ids = {item["id"] for item in qa_items}
    dataset.items = [item for item in dataset.items if item.id in current_ids]

    result = dataset.run_experiment(
        name=f"{_EXPERIMENT_NAME_PREFIX}{name}",
        run_name=_run_name(name),
        task=_make_single_question_task(stack, sources),
        evaluators=_SINGLE_QUESTION_EVALUATORS,
    )
    return result.dataset_run_url


# -- driver -------------------------------------------------------------


def _run_eval_set(eval_dir: Path, client: Langfuse | None) -> _ExpertiseResult:
    name = eval_dir.name
    qa_items = _load_qa(eval_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        stack, collection_name, sources = build_eval_stack_and_sources(eval_dir, tmp_dir)
        try:
            initial_state = _build_confirmed_state(qa_items, sources)
            final_state = _run_full_pipeline(stack, initial_state, tmp_dir / "checkpoints.sqlite")
            outcomes = _full_pipeline_outcomes(qa_items, final_state)

            langfuse_url = None
            if client is not None:
                langfuse_url = _run_on_langfuse(client, name, stack, sources, qa_items)
        finally:
            _cleanup_stack(collection_name)

    return _aggregate(name, outcomes, langfuse_url)


def main() -> None:
    eval_dirs = _discover_docgen_eval_dirs()
    if not eval_dirs:
        print(
            f"No docgen eval sets found under {_EVAL_ROOT}. See {_EVAL_ROOT / 'README.md'} for "
            "the folder convention (documents/task/, optional documents/reference/, "
            "docgen_qa.json) -- no code changes needed to add a new one."
        )
        return

    client, langfuse_status = _connect_langfuse()
    print(langfuse_status)

    results = [_run_eval_set(d, client) for d in eval_dirs]
    _print_results(results, langfuse_status)

    if client is not None:
        client.flush()


if __name__ == "__main__":
    main()
