"""Unit tests for langfuse_experiment.py -- mocked throughout, no real
Qdrant/Elasticsearch/Langfuse calls (tests/conftest.py's autouse fixture
also blanks real Langfuse credentials for the whole suite regardless).
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from multimodal_rag.evaluation import langfuse_experiment as lx
from multimodal_rag.evaluation.judge import JudgeParseError


def test_run_name_includes_the_label_and_a_timestamp() -> None:
    # Regex, not two real calls compared for inequality -- see
    # test_run_expert_eval.py's identical helper for why that would
    # be flaky (this is second-precision, which is plenty for real usage).
    name = lx._run_name("hybrid_rrf")
    assert re.fullmatch(r"hybrid_rrf-\d{8}T\d{6}Z", name), name


def test_sync_dataset_creates_the_dataset_and_upserts_each_item() -> None:
    client = MagicMock()
    golden_set = [
        {
            "id": "q1",
            "question": "a question",
            "expected_sources": ["doc.md"],
            "expect_refusal": False,
        },
        {"id": "q2", "question": "p99?", "expected_sources": [], "expect_refusal": True},
    ]

    lx._sync_dataset(client, golden_set)

    client.create_dataset.assert_called_once()
    assert client.create_dataset.call_args.kwargs["name"] == lx._DATASET_NAME
    assert client.create_dataset_item.call_count == 2
    first_call = client.create_dataset_item.call_args_list[0].kwargs
    assert first_call["dataset_name"] == lx._DATASET_NAME
    assert first_call["id"] == "q1"
    assert first_call["input"] == "a question"
    assert first_call["expected_output"] == {
        "expected_sources": ["doc.md"],
        "expect_refusal": False,
    }


def test_make_task_returns_answer_refused_sources_and_context() -> None:
    chunk = SimpleNamespace(
        source="doc.md",
        pages=[1],
        slides=[],
        text="the chunk text",
        elements=[],
        # format_context_block() now also reads lineage fields (§10) --
        # a real SearchResult defaults these; this hand-rolled fake has to
        # set them explicitly to stand in for an untagged, current, v1
        # document (the common case, and the one that produces a citation
        # identical to before these fields existed).
        version=1,
        status="current",
        effective_from=None,
    )
    fake_answer = SimpleNamespace(answer="the answer", refused=False, retrieved_chunks=[chunk])
    chain = MagicMock()
    chain.answer.return_value = fake_answer
    task = lx._make_task(chain)
    item = SimpleNamespace(input="a question")

    result = task(item=item)

    chain.answer.assert_called_once_with("a question")
    assert result["answer"] == "the answer"
    assert result["refused"] is False
    assert result["retrieved_sources"] == ["doc.md"]
    assert "the chunk text" in result["context_text"]


def test_retrieval_evaluator_returns_empty_for_refusal_items() -> None:
    result = lx._retrieval_evaluator(
        output={"retrieved_sources": []}, expected_output={"expect_refusal": True}
    )
    assert result == []


def test_retrieval_evaluator_computes_recall_mrr_ndcg() -> None:
    result = lx._retrieval_evaluator(
        output={"retrieved_sources": ["doc.md"]},
        expected_output={"expected_sources": ["doc.md"], "expect_refusal": False},
    )
    by_name = {e.name: e.value for e in result}
    assert by_name.keys() == {"recall_at_k", "mrr", "ndcg_at_k"}
    assert all(v == pytest.approx(1.0) for v in by_name.values())


def test_refusal_accuracy_evaluator_returns_empty_for_answerable_items() -> None:
    result = lx._refusal_accuracy_evaluator(
        output={"refused": False}, expected_output={"expect_refusal": False}
    )
    assert result == []


def test_refusal_accuracy_evaluator_scores_a_correct_refusal() -> None:
    [evaluation] = lx._refusal_accuracy_evaluator(
        output={"refused": True}, expected_output={"expect_refusal": True}
    )
    assert evaluation.name == "refusal_accuracy"
    assert evaluation.value == 1.0


def test_refusal_accuracy_evaluator_scores_an_incorrect_non_refusal() -> None:
    [evaluation] = lx._refusal_accuracy_evaluator(
        output={"refused": False}, expected_output={"expect_refusal": True}
    )
    assert evaluation.value == 0.0


def test_faithfulness_evaluator_skips_expect_refusal_golden_items() -> None:
    result = lx._faithfulness_evaluator(
        output={"refused": False, "answer": "a", "context_text": "b"},
        expected_output={"expect_refusal": True},
    )
    assert result == []


def test_faithfulness_evaluator_skips_when_the_chain_itself_refused() -> None:
    result = lx._faithfulness_evaluator(
        output={"refused": True, "answer": "...", "context_text": "..."},
        expected_output={"expect_refusal": False},
    )
    assert result == []


def test_faithfulness_evaluator_scores_and_flags_hallucination_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lx, "score_faithfulness", lambda answer, context: 0.5)

    evaluations = lx._faithfulness_evaluator(
        output={"refused": False, "answer": "a", "context_text": "b"},
        expected_output={"expect_refusal": False},
    )

    by_name = {e.name: e for e in evaluations}
    assert by_name["faithfulness"].value == 0.5
    assert by_name["hallucinated"].value is True  # 0.5 < _HALLUCINATION_THRESHOLD (0.7)


def test_faithfulness_evaluator_does_not_flag_hallucination_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lx, "score_faithfulness", lambda answer, context: 0.9)

    evaluations = lx._faithfulness_evaluator(
        output={"refused": False, "answer": "a", "context_text": "b"},
        expected_output={"expect_refusal": False},
    )

    by_name = {e.name: e for e in evaluations}
    assert by_name["hallucinated"].value is False


def test_faithfulness_evaluator_swallows_a_judge_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(answer: str, context: str) -> float:
        raise JudgeParseError("malformed judge output")

    monkeypatch.setattr(lx, "score_faithfulness", _raise)

    result = lx._faithfulness_evaluator(
        output={"refused": False, "answer": "a", "context_text": "b"},
        expected_output={"expect_refusal": False},
    )
    assert result == []


def test_relevance_evaluator_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lx, "score_relevance", lambda question, answer: 0.8)

    [evaluation] = lx._relevance_evaluator(
        input="a question",
        output={"refused": False, "answer": "a"},
        expected_output={"expect_refusal": False},
    )

    assert evaluation.name == "relevance"
    assert evaluation.value == 0.8


def test_relevance_evaluator_skips_expect_refusal_golden_items() -> None:
    result = lx._relevance_evaluator(
        input="q",
        output={"refused": False, "answer": "a"},
        expected_output={"expect_refusal": True},
    )
    assert result == []


def test_relevance_evaluator_swallows_a_judge_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(question: str, answer: str) -> float:
        raise JudgeParseError("malformed judge output")

    monkeypatch.setattr(lx, "score_relevance", _raise)

    result = lx._relevance_evaluator(
        input="q",
        output={"refused": False, "answer": "a"},
        expected_output={"expect_refusal": False},
    )
    assert result == []


def test_main_prints_a_message_and_returns_early_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(lx, "get_langfuse_client", lambda: None)

    lx.main()

    assert "Langfuse isn't configured" in capsys.readouterr().out
