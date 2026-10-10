"""Unit tests for run_docgen_eval.py -- mocked throughout, no real
Elasticsearch/docgen-graph/Langfuse calls, same discipline as
test_run_expert_eval.py. Focuses on this module's OWN logic (discovery,
the docgen-specific _score_item, state building, dataset sync) -- the
compiled graph itself is already covered extensively under
tests/docgen/, so _run_full_pipeline isn't re-tested here beyond its
own pure interrupt-kind dispatch, which test_score_item's siblings
exercise indirectly through _full_pipeline_outcomes.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langfuse import Evaluation

from multimodal_rag.evaluation import run_docgen_eval as rde
from multimodal_rag.evaluation.judge import JudgeParseError


def _write_eval_set(
    root: Path, name: str, qa_items: list[dict[str, object]], with_reference: bool = False
) -> Path:
    eval_dir = root / name
    (eval_dir / "documents" / "task").mkdir(parents=True)
    (eval_dir / "documents" / "task" / "doc.md").write_text("content")
    if with_reference:
        (eval_dir / "documents" / "reference").mkdir(parents=True)
        (eval_dir / "documents" / "reference" / "doc.md").write_text("reference content")
    (eval_dir / "docgen_qa.json").write_text(json.dumps(qa_items))
    return eval_dir


# -- discovery ---------------------------------------------------------------


def test_discover_docgen_eval_dirs_returns_empty_list_when_eval_root_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rde, "_EVAL_ROOT", tmp_path / "does-not-exist")

    assert rde._discover_docgen_eval_dirs() == []


def test_discover_docgen_eval_dirs_skips_folders_missing_the_right_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rde, "_EVAL_ROOT", tmp_path)
    _write_eval_set(tmp_path, "compliance", [{"id": "q1", "question": "?", "expert_answer": "!"}])
    # A plain agent-eval folder (qa.json + documents/, no task/ subfolder)
    # must NOT be picked up by the docgen discovery -- the two eval kinds
    # share data/eval/ but are distinguished by filename and shape.
    agent_eval_dir = tmp_path / "agent-style"
    (agent_eval_dir / "documents").mkdir(parents=True)
    (agent_eval_dir / "qa.json").write_text("[]")

    dirs = rde._discover_docgen_eval_dirs()

    assert [d.name for d in dirs] == ["compliance"]


def test_sources_required_defaults_to_task_docs_only() -> None:
    assert rde._sources_required({"question": "?"}) == ["task_docs"]
    assert rde._sources_required({"sources_required": ["task_docs", "reference_kb"]}) == [
        "task_docs",
        "reference_kb",
    ]


# -- _score_item: docgen's own notion of "correctly handled" ----------------


def test_score_item_scores_a_substantive_answerable_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rde, "score_answer_correctness", lambda q, a, e: 0.9)

    [evaluation] = rde._score_item("Q?", "an answer", "expert answer", expect_refusal=False)

    assert evaluation.name == "correctness"
    assert evaluation.value == 0.9


def test_score_item_flags_a_false_refusal_without_calling_the_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = MagicMock(side_effect=AssertionError("must not be called"))
    monkeypatch.setattr(rde, "score_answer_correctness", judge)

    [evaluation] = rde._score_item("Q?", None, "expert answer", expect_refusal=False)

    judge.assert_not_called()
    assert evaluation.name == "false_refusal"
    assert evaluation.value is True


def test_score_item_scores_a_correct_refusal_without_an_answer_as_1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = MagicMock(side_effect=AssertionError("must not be called"))
    monkeypatch.setattr(rde, "score_answer_correctness", judge)

    [evaluation] = rde._score_item("Q?", None, "expert answer", expect_refusal=True)

    judge.assert_not_called()
    assert evaluation.name == "refusal_accuracy"
    assert evaluation.value == 1.0


def test_score_item_scores_an_honest_refusal_answer_by_substance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core regression this module exists to prevent: docgen has no
    AgentChain-style `refused` flag -- an honest "that isn't covered"
    answer is just a normal, validated answer. An expect_refusal item
    that DID get an answer must still be judged by substance against
    the expert's own explanation, not auto-scored as wrong just because
    something was answered."""
    monkeypatch.setattr(rde, "score_answer_correctness", lambda q, a, e: 0.95)

    [evaluation] = rde._score_item(
        "Q?", "That isn't stated anywhere in the documents.", "Not covered.", expect_refusal=True
    )

    assert evaluation.name == "refusal_accuracy"
    assert evaluation.value == 0.95


def test_score_item_swallows_a_judge_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(question: str, answer: str, reference_answer: str) -> float:
        raise JudgeParseError("malformed")

    monkeypatch.setattr(rde, "score_answer_correctness", _raise)

    assert rde._score_item("Q?", "an answer", "expert answer", expect_refusal=False) == []


# -- _full_pipeline_outcomes --------------------------------------------------


def test_full_pipeline_outcomes_maps_question_status_to_answer_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rde, "score_answer_correctness", lambda q, a, e: 0.7)
    qa_items = [
        {"id": "q1", "question": "Q1?", "expert_answer": "A1"},
        {"id": "q2", "question": "Q2?", "expert_answer": "A2"},
    ]
    final_state = {
        "questions": [
            {"id": "q1", "status": "answered"},
            {"id": "q2", "status": "escalated"},
        ],
        "answers": {"q1": {"text": "the real answer"}},
    }

    outcomes = rde._full_pipeline_outcomes(qa_items, final_state)

    assert outcomes[0].evaluations[0].name == "correctness"
    assert outcomes[0].evaluations[0].value == 0.7
    assert outcomes[1].evaluations[0].name == "false_refusal"


# -- state building ------------------------------------------------------


def test_build_confirmed_state_is_already_confirmed_with_pending_questions() -> None:
    qa_items = [
        {"id": "q1", "question": "Q1?", "sources_required": ["task_docs"]},
        {"id": "q2", "question": "Q2?", "sources_required": ["task_docs", "reference_kb"]},
    ]
    sources = {"task_docs": rde.SourceSpec(role="task_docs", tag="t", required=True)}

    state = rde._build_confirmed_state(qa_items, sources)

    assert state["configuration"]["confirmed"] is True
    assert [q["status"] for q in state["questions"]] == ["pending", "pending"]
    assert state["questions"][1]["sources_required"] == ["task_docs", "reference_kb"]
    assert state["sources"] == list(sources.values())


# -- Langfuse sync --------------------------------------------------------


def test_dataset_name_uses_the_docgen_prefix() -> None:
    assert rde._dataset_name("compliance") == "docgen-eval-compliance"


def test_input_payload_carries_sources_required() -> None:
    payload = rde._input_payload({"question": "Q?", "sources_required": ["reference_kb"]})
    assert payload == {"question": "Q?", "sources_required": ["reference_kb"]}


def test_sync_dataset_creates_the_dataset_and_upserts_each_item() -> None:
    client = MagicMock()
    qa_items = [{"id": "q1", "question": "Q?", "expert_answer": "A."}]

    rde._sync_dataset(client, "compliance", qa_items)

    client.create_dataset.assert_called_once()
    assert client.create_dataset.call_args.kwargs["name"] == "docgen-eval-compliance"
    call = client.create_dataset_item.call_args.kwargs
    assert call["dataset_name"] == "docgen-eval-compliance"
    assert call["input"] == {"question": "Q?", "sources_required": ["task_docs"]}


# -- single-question evaluator (Langfuse per-item visibility run) -----------


def test_single_question_evaluator_delegates_to_score_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rde, "score_answer_correctness", lambda q, a, e: 0.6)

    [evaluation] = rde._single_question_evaluator(
        input={"question": "Q?"},
        output={"answer": "an answer"},
        expected_output={"expert_answer": "A.", "expect_refusal": False},
    )

    assert evaluation.name == "correctness"
    assert evaluation.value == 0.6


def test_main_prints_a_helpful_message_when_no_eval_dirs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(rde, "_EVAL_ROOT", tmp_path / "empty")

    rde.main()

    out = capsys.readouterr().out
    assert "No docgen eval sets found" in out
    assert "no code changes needed" in out


def test_aggregate_reuses_run_expert_evals_shape() -> None:
    """Not a real behavior test -- just confirms the reused _aggregate
    still accepts docgen's _ItemOutcome objects directly, i.e. the two
    modules haven't drifted apart on that shared contract."""
    outcomes = [
        rde._ItemOutcome(False, [Evaluation(name="correctness", value=1.0)]),
        rde._ItemOutcome(True, [Evaluation(name="refusal_accuracy", value=1.0)]),
    ]

    result = rde._aggregate("compliance", outcomes)

    assert result.answerable_count == 1
    assert result.refusal_count == 1
