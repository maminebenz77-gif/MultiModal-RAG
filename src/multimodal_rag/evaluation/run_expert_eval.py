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

ONE script, two outputs. Results always print to the console. It also
tries to reach Langfuse first; if it can't (not configured, blocked by the
privacy guard, or unreachable) it says so and carries on console-only --
never fails because Langfuse is missing -- and if it can, it ALSO syncs
each expertise's qa.json to its own Langfuse Dataset and runs it as a
native Experiment, which is what groups every call one question triggers
(embed, retrieve, generate, judge) under one shared trace per item.

Both paths score with the exact same evaluator functions and feed the
same aggregation, so the console table means the same thing either way --
the only difference is who drives the loop (Langfuse's run_experiment()
when connected, a plain local loop when not).

Convention -- see data/eval/README.md -- lets a new expertise or question
be added as a pure data change, no code:

    data/eval/<expertise-name>/
      documents/   # real source documents (PDF, DOCX, PPTX, Markdown, CSV, Excel)
      qa.json       # [{"id": ..., "question": ..., "expert_answer": ...,
                     #   "expect_refusal": false}, ...]  -- expect_refusal
                     #   optional, defaults to false

Run: `uv run python -m multimodal_rag.evaluation.run_expert_eval`
"""

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from langfuse import Evaluation, Langfuse
from langfuse.experiment import EvaluatorFunction

from ..generation.agent import AgentChain
from ..metadata import DocumentMetadata
from ..providers.factory import get_embedder
from ..retrieval.retriever import Retriever
from ..retrieval.schema import RetrievalMethod
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from ..tracing import get_langfuse_client
from .judge import JudgeParseError, score_answer_correctness
from .run_eval import _ingest_document

_EVAL_ROOT = Path(__file__).resolve().parents[3] / "data" / "eval"
_TOP_K = 5
"""Matches QueryRequest's real default (api/schemas.py), unlike
run_eval.py's _TOP_K=3 -- that one is deliberately tuned to make recall@k
discriminate between methods; this eval is about the agent a real user
actually gets, so it uses exactly the config routers/query.py defaults to."""

_DATASET_PREFIX = "expert-eval-"
_EXPERIMENT_NAME_PREFIX = "Expert eval: "


@dataclass
class _ItemOutcome:
    """What aggregation needs from one scored question -- deliberately
    just this, not the agent's answer text: both runners (the local loop
    and Langfuse's run_experiment()) can produce it, which is what lets
    them share one aggregation."""

    expect_refusal: bool
    evaluations: list[Evaluation]


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
    langfuse_url: str | None = None
    """This expertise's Dataset Run in Langfuse, or None when the run was
    console-only."""


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


_DEFAULT_DOC_METADATA = DocumentMetadata(classification="public")
"""What every document got before documents_metadata.json existed: no
DocumentMetadata passed to the indexer at all, i.e. no classification,
no lineage. A bare classification is the closest equivalent that's
still a real DocumentMetadata -- needed now because every document is
indexed through one, not because any existing expertise needs the
other fields."""


def _load_document_metadata(expertise_dir: Path) -> dict[str, DocumentMetadata]:
    """Optional data/eval/<expertise>/documents_metadata.json: filename ->
    DocumentMetadata fields, for an expertise set that needs to exercise
    real lineage (doc_family_id/version/status/effective_from) or a
    non-default classification -- see company-spreadsheets' budget-cap
    pair (declared supersession -- same doc_family_id, versions 1 and 2)
    and vendor-approval pair (two independently dated documents, no
    family relationship at all, that simply disagree) for why. Before
    this, every document here was ingested with NO DocumentMetadata at
    all, so neither collapse_families (Phase 6) nor the recency tilt
    could ever be exercised by this eval -- both key off fields that
    path never set. A filename with no entry here keeps the old
    default."""
    path = expertise_dir / "documents_metadata.json"
    if not path.is_file():
        return {}
    raw = cast(dict[str, dict[str, Any]], json.loads(path.read_text()))
    return {filename: DocumentMetadata(**fields) for filename, fields in raw.items()}


def build_expertise_agent(expertise_dir: Path) -> AgentChain:
    """Ingests an expertise folder's documents into a collection scoped to
    it alone -- deliberately, so a question about one expertise can't
    accidentally retrieve another expertise's content just because they
    happen to share a collection -- and returns the real production
    AgentChain over it (routers/query.py's exact construction).

    Indexes one document at a time (each with its own DocumentMetadata),
    not all documents' chunks in one shared call -- metadata is
    per-document, and HybridIndexer.index()'s doc_metadata parameter is
    a single value applied to every chunk in that call. Embedding stays
    batched across all documents for one round-trip to the embedder.
    """
    name = expertise_dir.name
    documents_dir = expertise_dir / "documents"
    doc_metadata_by_filename = _load_document_metadata(expertise_dir)
    # rglob, not iterdir -- documents/ may itself group files into real
    # subfolders (e.g. a "runbooks/" folder meant to be bulk-ingested via
    # the frontend's folder picker in a live demo); iterdir() only lists
    # immediate children, so anything nested was silently never ingested
    # at all. _load_document_metadata keys on bare filename, not the
    # relative path, so filenames still need to be unique within one
    # expertise folder regardless of which subfolder they live in.
    paths = sorted(
        p for p in documents_dir.rglob("*") if p.is_file() and not p.name.startswith(".")
    )
    chunks_by_path = {path: _ingest_document(path) for path in paths}
    all_chunks = [chunk for chunks in chunks_by_path.values() for chunk in chunks]

    collection = f"expert_eval_{name}"
    embedder = get_embedder()
    all_vectors = embedder.embed([c.text for c in all_chunks])

    vector_store = get_vector_store(collection_name=collection)
    vector_store.create_collection(dimension=all_vectors[0].dimension, indexing_threshold=0)
    keyword_store = get_keyword_store(index_name=collection)
    keyword_store.create_index()

    indexer = HybridIndexer(vector_store, keyword_store)
    offset = 0
    for path, chunks in chunks_by_path.items():
        vectors = all_vectors[offset : offset + len(chunks)]
        offset += len(chunks)
        metadata = doc_metadata_by_filename.get(path.name, _DEFAULT_DOC_METADATA)
        indexer.index(chunks, vectors, metadata)
    vector_store.publish()

    retriever = Retriever(vector_store, keyword_store, embedder)
    return AgentChain(
        retriever,
        method=RetrievalMethod.HYBRID_RRF,
        top_k=_TOP_K,
        rerank=False,
        resolve_parent_context=True,
    )


# -- scoring: shared by both runners ---------------------------------------


def _expected_output(item: dict[str, Any]) -> dict[str, Any]:
    """The one place a qa.json item becomes the expected_output both
    runners hand to the evaluators -- synced to Langfuse as the dataset
    item's expected_output, and passed directly by the local loop -- so
    the two paths can't drift apart on what "expected" means."""
    return {
        "expert_answer": item["expert_answer"],
        "expect_refusal": item.get("expect_refusal", False),
    }


def _make_task(agent: AgentChain) -> Any:
    def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        rag_answer = agent.answer(item.input)
        return {"answer": rag_answer.answer, "refused": rag_answer.refused}

    return task


def _correctness_evaluator(
    *, input: Any, output: dict[str, Any], expected_output: dict[str, Any], **kwargs: Any
) -> list[Evaluation]:
    if expected_output.get("expect_refusal"):
        # Scored by _refusal_accuracy_evaluator instead -- see its
        # docstring for why a free-text judge is the wrong tool here.
        return []
    if output["refused"]:
        # A false refusal: an answerable item the agent declined anyway.
        # A real failure, but not one a free-text correctness judge can
        # honestly score -- there's no substantive answer to compare.
        return [Evaluation(name="false_refusal", value=True, data_type="BOOLEAN")]
    try:
        score = score_answer_correctness(input, output["answer"], expected_output["expert_answer"])
    except JudgeParseError:
        return []
    return [Evaluation(name="correctness", value=score)]


def _refusal_accuracy_evaluator(
    *, output: dict[str, Any], expected_output: dict[str, Any], **kwargs: Any
) -> list[Evaluation]:
    """Deliberately separate from correctness -- grading a refusal's
    ANSWER TEXT against expert_answer with a free-text judge is exactly
    the bug caught live in this project: a correct, terse "I don't know"
    scored 0 against a longer reference explanation. This only checks
    whether the agent refused at all, never what it said.
    """
    if not expected_output.get("expect_refusal"):
        return []
    correct = bool(output["refused"])
    return [Evaluation(name="refusal_accuracy", value=1.0 if correct else 0.0)]


_EVALUATOR_FUNCTIONS = (_correctness_evaluator, _refusal_accuracy_evaluator)
_EVALUATORS: list[EvaluatorFunction] = list(_EVALUATOR_FUNCTIONS)


def _aggregate(
    name: str, outcomes: list[_ItemOutcome], langfuse_url: str | None = None
) -> _ExpertiseResult:
    def numeric_values(evaluation_name: str) -> list[float]:
        return [
            float(evaluation.value)
            for outcome in outcomes
            for evaluation in outcome.evaluations
            if evaluation.name == evaluation_name and isinstance(evaluation.value, int | float)
        ]

    answerable_count = sum(1 for o in outcomes if not o.expect_refusal)
    refusal_count = sum(1 for o in outcomes if o.expect_refusal)

    correctness_scores = numeric_values("correctness")
    false_refusals = len(numeric_values("false_refusal"))
    return _ExpertiseResult(
        name=name,
        average_correctness=(
            sum(correctness_scores) / len(correctness_scores) if correctness_scores else 0.0
        ),
        answerable_count=answerable_count,
        refusal_accuracy=(
            sum(numeric_values("refusal_accuracy")) / refusal_count if refusal_count else None
        ),
        refusal_count=refusal_count,
        false_refusals=false_refusals,
        # An answerable item that was neither scored nor a false refusal
        # can only have hit a JudgeParseError (its evaluator returns no
        # evaluation at all) -- derived rather than tracked separately, so
        # it holds identically for both runners.
        judge_failures=answerable_count - len(correctness_scores) - false_refusals,
        langfuse_url=langfuse_url,
    )


# -- runner 1: plain local loop (no Langfuse) ------------------------------


def _run_locally(agent: AgentChain, qa_items: list[dict[str, Any]]) -> list[_ItemOutcome]:
    task = _make_task(agent)
    outcomes: list[_ItemOutcome] = []
    for item in qa_items:
        expected = _expected_output(item)
        output = task(item=SimpleNamespace(input=item["question"]))
        evaluations = [
            evaluation
            for evaluator in _EVALUATOR_FUNCTIONS
            for evaluation in evaluator(
                input=item["question"], output=output, expected_output=expected
            )
        ]
        outcomes.append(_ItemOutcome(bool(expected["expect_refusal"]), evaluations))
    return outcomes


# -- runner 2: Langfuse Experiment -----------------------------------------


def _dataset_name(expertise: str) -> str:
    return f"{_DATASET_PREFIX}{expertise}"


def _run_name(expertise: str) -> str:
    # A bare expertise name as run_name meant every invocation reused the
    # exact same run identity -- re-running after editing a qa.json (or
    # just to re-check after a code change) silently landed on the SAME
    # Dataset Run in Langfuse's UI instead of creating a new,
    # distinguishable one; found live when a second run produced fresh
    # trace data but the printed Dataset Run URL was identical to the
    # first run's. Timestamped so every run is its own entry, sortable by
    # when it happened.
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{expertise}-{timestamp}"


def _sync_dataset(client: Langfuse, expertise: str, qa_items: list[dict[str, Any]]) -> None:
    """Both create_dataset and create_dataset_item upsert by name/id (see
    langfuse_experiment.py's _sync_dataset) -- safe to call on every run,
    and an edit to qa.json (a new question, a corrected expert_answer)
    shows up here on the next run too, not just a one-time upload.
    """
    dataset_name = _dataset_name(expertise)
    client.create_dataset(
        name=dataset_name,
        description=(
            f"Expert-authored Q&A for the '{expertise}' expertise (data/eval/{expertise}/qa.json)."
        ),
    )
    for item in qa_items:
        client.create_dataset_item(
            dataset_name=dataset_name,
            id=item["id"],
            input=item["question"],
            expected_output=_expected_output(item),
        )


def _run_on_langfuse(
    client: Langfuse, name: str, agent: AgentChain, qa_items: list[dict[str, Any]]
) -> tuple[list[_ItemOutcome], str | None]:
    _sync_dataset(client, name, qa_items)
    dataset = client.get_dataset(_dataset_name(name))
    # create_dataset_item upserts but never deletes, so a question removed
    # or renamed in qa.json would still be sitting in the Langfuse dataset
    # -- and run_experiment() runs whatever the dataset holds, so it would
    # silently keep being asked and counted. Restricted to the ids qa.json
    # has right now, so both runners always evaluate the same questions.
    current_ids = {item["id"] for item in qa_items}
    dataset.items = [item for item in dataset.items if item.id in current_ids]

    result = dataset.run_experiment(
        name=f"{_EXPERIMENT_NAME_PREFIX}{name}",
        run_name=_run_name(name),
        task=_make_task(agent),
        evaluators=_EVALUATORS,
    )
    outcomes = [
        _ItemOutcome(
            expect_refusal=bool(
                (getattr(item_result.item, "expected_output", None) or {}).get("expect_refusal")
            ),
            evaluations=item_result.evaluations,
        )
        for item_result in result.item_results
    ]
    return outcomes, result.dataset_run_url


def _connect_langfuse() -> tuple[Langfuse | None, str]:
    """Returns (client, status message). client is None whenever Langfuse
    can't be used, for any reason -- the message says which -- and the
    caller just carries on console-only. auth_check() is what actually
    proves reachability and valid keys; get_langfuse_client() alone only
    proves the settings looked usable, since constructing the client
    doesn't touch the network.
    """
    client = get_langfuse_client()
    if client is None:
        return None, (
            "Langfuse: not connected -- not configured, or disabled by the privacy guard "
            "(see tracing.py and the LANGFUSE_* settings in .env.local). "
            "Results are printed here only."
        )
    try:
        reachable = client.auth_check()
    except Exception as exc:
        return None, f"Langfuse: not connected -- {exc}. Results are printed here only."
    if not reachable:
        return None, (
            "Langfuse: not connected -- the auth check failed (bad keys or wrong host). "
            "Results are printed here only."
        )
    return client, "Langfuse: connected -- results are also being sent as Experiments."


# -- driver -----------------------------------------------------------------


def _run_expertise(expertise_dir: Path, client: Langfuse | None = None) -> _ExpertiseResult:
    name = expertise_dir.name
    qa_items = _load_qa(expertise_dir)
    agent = build_expertise_agent(expertise_dir)

    if client is not None:
        outcomes, langfuse_url = _run_on_langfuse(client, name, agent, qa_items)
    else:
        outcomes, langfuse_url = _run_locally(agent, qa_items), None
    return _aggregate(name, outcomes, langfuse_url)


def _fmt_refusal_accuracy(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


_COLUMN_WIDTHS = {"Expertise": 25, "Correctness": 13, "Answerable": 12, "RefusalAcc": 12}


def _print_results(results: list[_ExpertiseResult], langfuse_status: str | None = None) -> None:
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

    # Repeated at the end, not just printed up front: a run makes many real
    # LLM calls and logs plenty of its own noise, so a status line from the
    # very start has usually scrolled out of view by the time the table
    # appears.
    if langfuse_status is not None:
        print(f"\n{langfuse_status}")
    for r in results:
        if r.langfuse_url:
            print(f"  {r.name}: {r.langfuse_url}")


def main() -> None:
    expertise_dirs = _discover_expertise_dirs()
    if not expertise_dirs:
        print(
            f"No expertise folders found under {_EVAL_ROOT}. See "
            f"{_EVAL_ROOT / 'README.md'} for the folder convention -- no code changes "
            "needed to add a new expertise or a new question."
        )
        return

    client, langfuse_status = _connect_langfuse()
    print(langfuse_status)

    results = [_run_expertise(d, client) for d in expertise_dirs]
    _print_results(results, langfuse_status)

    if client is not None:
        client.flush()


if __name__ == "__main__":
    main()
