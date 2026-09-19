"""Unit tests for run_expert_eval.py -- mocked throughout, no real
Qdrant/Elasticsearch/AgentChain calls. Mirrors the mocking style
test_langfuse_experiment.py already uses for the same reason: this
script's own logic (discovery, scoring, averaging) is what's under test,
not the stores/generation stack it drives.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from multimodal_rag.evaluation import run_expert_eval as ree
from multimodal_rag.evaluation.judge import JudgeParseError


def _write_expertise(root: Path, name: str, qa_items: list[dict[str, object]]) -> Path:
    expertise_dir = root / name
    (expertise_dir / "documents").mkdir(parents=True)
    (expertise_dir / "documents" / "doc.md").write_text("content")
    (expertise_dir / "qa.json").write_text(json.dumps(qa_items))
    return expertise_dir


def _mock_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ree, "_ingest_document", lambda path: [SimpleNamespace(text="t")])
    monkeypatch.setattr(ree, "get_embedder", lambda: MagicMock(embed=lambda texts: [MagicMock()]))
    monkeypatch.setattr(ree, "get_vector_store", lambda **kwargs: MagicMock())
    monkeypatch.setattr(ree, "get_keyword_store", lambda **kwargs: MagicMock())
    monkeypatch.setattr(ree, "HybridIndexer", lambda *a, **k: MagicMock())
    monkeypatch.setattr(ree, "Retriever", lambda *a, **k: MagicMock())


def test_discover_expertise_dirs_returns_empty_list_when_eval_root_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ree, "_EVAL_ROOT", tmp_path / "does-not-exist")

    assert ree._discover_expertise_dirs() == []


def test_discover_expertise_dirs_skips_folders_missing_qa_or_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ree, "_EVAL_ROOT", tmp_path)
    _write_expertise(tmp_path, "legal", [{"id": "q1", "question": "?", "expert_answer": "!"}])
    (tmp_path / "incomplete").mkdir()  # no documents/, no qa.json
    (tmp_path / "README.md").write_text("not a directory's problem, but a file isn't a dir")

    dirs = ree._discover_expertise_dirs()

    assert [d.name for d in dirs] == ["legal"]


def test_run_expertise_ingests_documents_and_scores_each_question(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expertise_dir = _write_expertise(
        tmp_path,
        "legal",
        [
            {"id": "q1", "question": "What is X?", "expert_answer": "X is Y."},
            {"id": "q2", "question": "What is Z?", "expert_answer": "Z is W."},
        ],
    )
    _mock_stores(monkeypatch)

    fake_agent = MagicMock()
    fake_agent.answer.side_effect = [
        SimpleNamespace(answer="Y answer", refused=False),
        SimpleNamespace(answer="W answer", refused=False),
    ]
    monkeypatch.setattr(ree, "AgentChain", lambda *a, **k: fake_agent)
    monkeypatch.setattr(ree, "score_answer_correctness", lambda q, a, e: 0.8)

    result = ree._run_expertise(expertise_dir)

    assert result.name == "legal"
    assert result.answerable_count == 2
    assert result.average_correctness == pytest.approx(0.8)
    assert result.judge_failures == 0
    assert result.false_refusals == 0
    assert result.refusal_accuracy is None  # no expect_refusal items in this expertise
    assert fake_agent.answer.call_args_list[0].args == ("What is X?",)


def test_run_expertise_excludes_judge_parse_failures_from_the_average(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expertise_dir = _write_expertise(
        tmp_path,
        "legal",
        [
            {"id": "q1", "question": "?", "expert_answer": "!"},
            {"id": "q2", "question": "?", "expert_answer": "!"},
        ],
    )
    _mock_stores(monkeypatch)

    fake_agent = MagicMock()
    fake_agent.answer.return_value = SimpleNamespace(answer="an answer", refused=False)
    monkeypatch.setattr(ree, "AgentChain", lambda *a, **k: fake_agent)

    def _raise(question: str, answer: str, reference_answer: str) -> float:
        raise JudgeParseError("malformed")

    monkeypatch.setattr(ree, "score_answer_correctness", _raise)

    result = ree._run_expertise(expertise_dir)

    assert result.judge_failures == 2
    assert result.average_correctness == 0.0


def test_run_expertise_excludes_an_unexpected_refusal_from_the_average(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression test: this failure mode showed up live -- the agent
    # correctly refused an unanswerable question, but the old code fed
    # "I don't know based on the available documents." straight into
    # score_answer_correctness, which scored it 0 against a substantive
    # expert_answer. A false refusal (an ANSWERABLE item the agent
    # refused) is a real failure, but it's a DIFFERENT failure -- it must
    # never reach the correctness judge at all.
    expertise_dir = _write_expertise(
        tmp_path,
        "legal",
        [{"id": "q1", "question": "What is X?", "expert_answer": "X is Y."}],
    )
    _mock_stores(monkeypatch)

    fake_agent = MagicMock()
    fake_agent.answer.return_value = SimpleNamespace(
        answer="I don't know based on the available documents.", refused=True
    )
    monkeypatch.setattr(ree, "AgentChain", lambda *a, **k: fake_agent)

    judge = MagicMock(side_effect=AssertionError("the judge must not be called on a refusal"))
    monkeypatch.setattr(ree, "score_answer_correctness", judge)

    result = ree._run_expertise(expertise_dir)

    judge.assert_not_called()
    assert result.false_refusals == 1
    assert result.average_correctness == 0.0


def test_run_expertise_scores_refusal_accuracy_separately_from_correctness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expertise_dir = _write_expertise(
        tmp_path,
        "legal",
        [
            {"id": "q1", "question": "What is X?", "expert_answer": "X is Y."},
            {
                "id": "q2",
                "question": "What is unknowable?",
                "expert_answer": "Not covered by the documents.",
                "expect_refusal": True,
            },
        ],
    )
    _mock_stores(monkeypatch)

    fake_agent = MagicMock()
    fake_agent.answer.side_effect = [
        SimpleNamespace(answer="Y answer", refused=False),  # the answerable item
        SimpleNamespace(answer="I don't know.", refused=True),  # correctly refused
    ]
    monkeypatch.setattr(ree, "AgentChain", lambda *a, **k: fake_agent)
    monkeypatch.setattr(ree, "score_answer_correctness", lambda q, a, e: 1.0)

    result = ree._run_expertise(expertise_dir)

    assert result.answerable_count == 1
    assert result.average_correctness == pytest.approx(1.0)
    assert result.refusal_count == 1
    assert result.refusal_accuracy == pytest.approx(1.0)


def test_print_results_shows_n_a_for_refusal_accuracy_when_no_refusal_items(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = ree._ExpertiseResult(
        name="legal",
        average_correctness=0.75,
        answerable_count=4,
        refusal_accuracy=None,
        refusal_count=0,
        false_refusals=0,
        judge_failures=0,
    )

    ree._print_results([result])

    assert "n/a" in capsys.readouterr().out


def test_main_prints_a_helpful_message_when_no_expertise_dirs_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(ree, "_EVAL_ROOT", tmp_path / "empty")

    ree.main()

    out = capsys.readouterr().out
    assert "No expertise folders found" in out
    assert "no code changes needed" in out
