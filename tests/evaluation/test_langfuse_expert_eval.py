"""Unit tests for langfuse_expert_eval.py -- mocked throughout, no real
Qdrant/Elasticsearch/Langfuse calls (tests/conftest.py's autouse fixture
also blanks real Langfuse credentials for the whole suite regardless).
Mirrors test_langfuse_experiment.py's mocking style.
"""

import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from multimodal_rag.evaluation import langfuse_expert_eval as lxe
from multimodal_rag.evaluation.judge import JudgeParseError


def test_run_name_includes_the_expertise_and_a_timestamp() -> None:
    # Regex, not a second real call compared for inequality -- two calls
    # within the same real second would otherwise produce an identical
    # name (this helper is second-precision, plenty for real usage: an
    # eval run takes far longer than a second), which would make an
    # inequality-based test flaky for no real reason.
    name = lxe._run_name("legal")
    assert re.fullmatch(r"legal-\d{8}T\d{6}Z", name), name


def test_sync_dataset_creates_the_dataset_and_upserts_each_item() -> None:
    client = MagicMock()
    qa_items = [
        {"id": "q1", "question": "What is X?", "expert_answer": "X is Y."},
        {
            "id": "q2",
            "question": "What is unknowable?",
            "expert_answer": "Not covered.",
            "expect_refusal": True,
        },
    ]

    lxe._sync_dataset(client, "legal", qa_items)

    client.create_dataset.assert_called_once()
    assert client.create_dataset.call_args.kwargs["name"] == "expert-eval-legal"
    assert client.create_dataset_item.call_count == 2
    first_call = client.create_dataset_item.call_args_list[0].kwargs
    assert first_call["dataset_name"] == "expert-eval-legal"
    assert first_call["id"] == "q1"
    assert first_call["input"] == "What is X?"
    assert first_call["expected_output"] == {
        "expert_answer": "X is Y.",
        "expect_refusal": False,
    }
    second_call = client.create_dataset_item.call_args_list[1].kwargs
    assert second_call["expected_output"]["expect_refusal"] is True


def test_make_task_returns_answer_and_refused() -> None:
    fake_answer = SimpleNamespace(answer="the answer", refused=False)
    agent = MagicMock()
    agent.answer.return_value = fake_answer
    task = lxe._make_task(agent)
    item = SimpleNamespace(input="a question")

    result = task(item=item)

    agent.answer.assert_called_once_with("a question")
    assert result == {"answer": "the answer", "refused": False}


def test_correctness_evaluator_skips_expect_refusal_items() -> None:
    result = lxe._correctness_evaluator(
        input="q",
        output={"answer": "a", "refused": False},
        expected_output={"expect_refusal": True, "expert_answer": "e"},
    )
    assert result == []


def test_correctness_evaluator_flags_a_false_refusal_without_calling_the_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = MagicMock(side_effect=AssertionError("must not be called on a refusal"))
    monkeypatch.setattr(lxe, "score_answer_correctness", judge)

    [evaluation] = lxe._correctness_evaluator(
        input="q",
        output={"answer": "I don't know.", "refused": True},
        expected_output={"expect_refusal": False, "expert_answer": "e"},
    )

    judge.assert_not_called()
    assert evaluation.name == "false_refusal"
    assert evaluation.value is True


def test_correctness_evaluator_scores_a_substantive_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lxe, "score_answer_correctness", lambda q, a, e: 0.9)

    [evaluation] = lxe._correctness_evaluator(
        input="q",
        output={"answer": "a", "refused": False},
        expected_output={"expect_refusal": False, "expert_answer": "e"},
    )

    assert evaluation.name == "correctness"
    assert evaluation.value == 0.9


def test_correctness_evaluator_swallows_a_judge_parse_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(question: str, answer: str, reference_answer: str) -> float:
        raise JudgeParseError("malformed")

    monkeypatch.setattr(lxe, "score_answer_correctness", _raise)

    result = lxe._correctness_evaluator(
        input="q",
        output={"answer": "a", "refused": False},
        expected_output={"expect_refusal": False, "expert_answer": "e"},
    )
    assert result == []


def test_refusal_accuracy_evaluator_returns_empty_for_answerable_items() -> None:
    result = lxe._refusal_accuracy_evaluator(
        output={"refused": False}, expected_output={"expect_refusal": False}
    )
    assert result == []


def test_refusal_accuracy_evaluator_scores_a_correct_refusal() -> None:
    [evaluation] = lxe._refusal_accuracy_evaluator(
        output={"refused": True}, expected_output={"expect_refusal": True}
    )
    assert evaluation.name == "refusal_accuracy"
    assert evaluation.value == 1.0


def test_refusal_accuracy_evaluator_scores_an_incorrect_non_refusal() -> None:
    [evaluation] = lxe._refusal_accuracy_evaluator(
        output={"refused": False}, expected_output={"expect_refusal": True}
    )
    assert evaluation.value == 0.0


def test_main_prints_a_message_and_returns_early_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(lxe, "get_langfuse_client", lambda: None)

    lxe.main()

    assert "Langfuse isn't configured" in capsys.readouterr().out


def test_main_prints_a_message_when_no_expertise_dirs_exist(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(lxe, "get_langfuse_client", lambda: MagicMock())
    monkeypatch.setattr(lxe, "_discover_expertise_dirs", lambda: [])

    lxe.main()

    assert "No expertise folders found" in capsys.readouterr().out
