"""Unit tests for run_expert_eval.py -- mocked throughout, no real
Qdrant/Elasticsearch/AgentChain/Langfuse calls (tests/conftest.py's
autouse fixture also blanks real Langfuse credentials for the whole suite
regardless). Mirrors the mocking style test_langfuse_experiment.py already
uses for the same reason: this script's own logic (discovery, scoring,
aggregation, the connect-or-fall-back decision) is what's under test, not
the stores/generation stack or the Langfuse SDK it drives.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from langfuse import Evaluation

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


def test_load_document_metadata_returns_empty_dict_when_no_sidecar_file(tmp_path: Path) -> None:
    expertise_dir = tmp_path / "legal"
    expertise_dir.mkdir()

    assert ree._load_document_metadata(expertise_dir) == {}


def test_load_document_metadata_parses_the_sidecar_file(tmp_path: Path) -> None:
    expertise_dir = tmp_path / "legal"
    expertise_dir.mkdir()
    (expertise_dir / "documents_metadata.json").write_text(
        json.dumps(
            {
                "old.xlsx": {
                    "classification": "public",
                    "doc_family_id": "fam",
                    "version": 1,
                    "status": "superseded",
                },
                "new.xlsx": {"classification": "public", "doc_family_id": "fam", "version": 2},
            }
        )
    )

    result = ree._load_document_metadata(expertise_dir)

    assert result["old.xlsx"].status == "superseded"
    assert result["new.xlsx"].version == 2
    assert result["old.xlsx"].doc_family_id == result["new.xlsx"].doc_family_id == "fam"


def test_build_expertise_agent_indexes_each_document_with_its_own_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: build_expertise_agent used to batch every document's
    # chunks into ONE indexer call with no DocumentMetadata at all --
    # collapse_families and the recency tilt both key off fields that
    # path never set, so this eval could never exercise Phase 5/6.
    # Proves two things at once: a document WITH a sidecar entry gets
    # exactly its own declared fields, and one WITHOUT an entry still
    # gets indexed (not skipped) with the plain default -- not the
    # other document's metadata leaking across.
    expertise_dir = tmp_path / "legal"
    documents_dir = expertise_dir / "documents"
    documents_dir.mkdir(parents=True)
    (documents_dir / "a.md").write_text("a")
    (documents_dir / "b.md").write_text("b")
    (expertise_dir / "documents_metadata.json").write_text(
        json.dumps({"a.md": {"classification": "c2", "author": "Alice"}})
    )

    monkeypatch.setattr(ree, "_ingest_document", lambda path: [SimpleNamespace(text=path.name)])
    monkeypatch.setattr(
        ree, "get_embedder", lambda: MagicMock(embed=lambda texts: [MagicMock() for _ in texts])
    )
    monkeypatch.setattr(ree, "get_vector_store", lambda **kwargs: MagicMock())
    monkeypatch.setattr(ree, "get_keyword_store", lambda **kwargs: MagicMock())
    monkeypatch.setattr(ree, "Retriever", lambda *a, **k: MagicMock())
    monkeypatch.setattr(ree, "AgentChain", lambda *a, **k: MagicMock())

    index_calls: list[tuple[list[Any], Any]] = []

    class _SpyIndexer:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def index(self, chunks: list[Any], vectors: list[Any], doc_metadata: Any) -> None:
            index_calls.append((chunks, doc_metadata))

    monkeypatch.setattr(ree, "HybridIndexer", _SpyIndexer)

    ree.build_expertise_agent(expertise_dir)

    assert len(index_calls) == 2
    by_source = {chunks[0].text: metadata for chunks, metadata in index_calls}
    assert by_source["a.md"].classification == "c2"
    assert by_source["a.md"].author == "Alice"
    assert by_source["b.md"].classification == "public"  # default -- no sidecar entry
    assert by_source["b.md"].author is None


def test_build_expertise_agent_discovers_documents_in_subfolders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # documents/ may group files into real subfolders -- e.g. a
    # "runbooks/" folder meant to be bulk-ingested via the frontend's
    # native folder picker in a live demo. iterdir() only lists
    # immediate children, so anything nested was silently never
    # discovered at all; this is the regression for that.
    expertise_dir = tmp_path / "legal"
    documents_dir = expertise_dir / "documents"
    (documents_dir / "runbooks").mkdir(parents=True)
    (documents_dir / "top-level.md").write_text("top")
    (documents_dir / "runbooks" / "nested.md").write_text("nested")

    monkeypatch.setattr(ree, "_ingest_document", lambda path: [SimpleNamespace(text=path.name)])
    monkeypatch.setattr(
        ree, "get_embedder", lambda: MagicMock(embed=lambda texts: [MagicMock() for _ in texts])
    )
    monkeypatch.setattr(ree, "get_vector_store", lambda **kwargs: MagicMock())
    monkeypatch.setattr(ree, "get_keyword_store", lambda **kwargs: MagicMock())
    monkeypatch.setattr(ree, "Retriever", lambda *a, **k: MagicMock())
    monkeypatch.setattr(ree, "AgentChain", lambda *a, **k: MagicMock())

    index_calls: list[tuple[list[Any], Any]] = []

    class _SpyIndexer:
        def __init__(self, *a: object, **k: object) -> None:
            pass

        def index(self, chunks: list[Any], vectors: list[Any], doc_metadata: Any) -> None:
            index_calls.append((chunks, doc_metadata))

    monkeypatch.setattr(ree, "HybridIndexer", _SpyIndexer)

    ree.build_expertise_agent(expertise_dir)

    indexed_names = {chunks[0].text for chunks, _ in index_calls}
    assert indexed_names == {"top-level.md", "nested.md"}


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


# -- evaluators (shared by both runners) -----------------------------------


def test_correctness_evaluator_skips_expect_refusal_items() -> None:
    result = ree._correctness_evaluator(
        input="q",
        output={"answer": "a", "refused": False},
        expected_output={"expect_refusal": True, "expert_answer": "e"},
    )
    assert result == []


def test_correctness_evaluator_flags_a_false_refusal_without_calling_the_judge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judge = MagicMock(side_effect=AssertionError("must not be called on a refusal"))
    monkeypatch.setattr(ree, "score_answer_correctness", judge)

    [evaluation] = ree._correctness_evaluator(
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
    monkeypatch.setattr(ree, "score_answer_correctness", lambda q, a, e: 0.9)

    [evaluation] = ree._correctness_evaluator(
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

    monkeypatch.setattr(ree, "score_answer_correctness", _raise)

    result = ree._correctness_evaluator(
        input="q",
        output={"answer": "a", "refused": False},
        expected_output={"expect_refusal": False, "expert_answer": "e"},
    )
    assert result == []


def test_refusal_accuracy_evaluator_returns_empty_for_answerable_items() -> None:
    result = ree._refusal_accuracy_evaluator(
        output={"refused": False}, expected_output={"expect_refusal": False}
    )
    assert result == []


def test_refusal_accuracy_evaluator_scores_a_correct_refusal() -> None:
    [evaluation] = ree._refusal_accuracy_evaluator(
        output={"refused": True}, expected_output={"expect_refusal": True}
    )
    assert evaluation.name == "refusal_accuracy"
    assert evaluation.value == 1.0


def test_refusal_accuracy_evaluator_scores_an_incorrect_non_refusal() -> None:
    [evaluation] = ree._refusal_accuracy_evaluator(
        output={"refused": False}, expected_output={"expect_refusal": True}
    )
    assert evaluation.value == 0.0


def test_make_task_returns_answer_and_refused() -> None:
    agent = MagicMock()
    agent.answer.return_value = SimpleNamespace(answer="the answer", refused=False)

    result = ree._make_task(agent)(item=SimpleNamespace(input="a question"))

    agent.answer.assert_called_once_with("a question")
    assert result == {"answer": "the answer", "refused": False}


# -- aggregation (the one place both runners meet) -------------------------


def test_aggregate_derives_every_count_from_the_evaluations() -> None:
    outcomes = [
        ree._ItemOutcome(False, [Evaluation(name="correctness", value=1.0)]),
        ree._ItemOutcome(False, [Evaluation(name="correctness", value=0.5)]),
        ree._ItemOutcome(
            False, [Evaluation(name="false_refusal", value=True, data_type="BOOLEAN")]
        ),
        # An answerable item whose evaluator returned nothing at all can
        # only have been a judge parse failure.
        ree._ItemOutcome(False, []),
        ree._ItemOutcome(True, [Evaluation(name="refusal_accuracy", value=1.0)]),
        ree._ItemOutcome(True, [Evaluation(name="refusal_accuracy", value=0.0)]),
    ]

    result = ree._aggregate("legal", outcomes, "https://langfuse.example/run")

    assert result.answerable_count == 4
    assert result.average_correctness == pytest.approx(0.75)
    assert result.false_refusals == 1
    assert result.judge_failures == 1
    assert result.refusal_count == 2
    assert result.refusal_accuracy == pytest.approx(0.5)
    assert result.langfuse_url == "https://langfuse.example/run"


def test_aggregate_reports_no_refusal_accuracy_when_there_are_no_refusal_items() -> None:
    result = ree._aggregate(
        "legal", [ree._ItemOutcome(False, [Evaluation(name="correctness", value=1.0)])]
    )
    assert result.refusal_accuracy is None
    assert result.langfuse_url is None


# -- Langfuse runner --------------------------------------------------------


def test_run_name_includes_the_expertise_and_a_timestamp() -> None:
    # Regex, not a second real call compared for inequality -- two calls
    # within the same real second would otherwise produce an identical
    # name (this helper is second-precision, plenty for real usage: an
    # eval run takes far longer than a second), which would make an
    # inequality-based test flaky for no real reason.
    name = ree._run_name("legal")
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

    ree._sync_dataset(client, "legal", qa_items)

    client.create_dataset.assert_called_once()
    assert client.create_dataset.call_args.kwargs["name"] == "expert-eval-legal"
    assert client.create_dataset_item.call_count == 2
    first_call = client.create_dataset_item.call_args_list[0].kwargs
    assert first_call["dataset_name"] == "expert-eval-legal"
    assert first_call["id"] == "q1"
    assert first_call["input"] == "What is X?"
    assert first_call["expected_output"] == {"expert_answer": "X is Y.", "expect_refusal": False}
    second_call = client.create_dataset_item.call_args_list[1].kwargs
    assert second_call["expected_output"]["expect_refusal"] is True


def _fake_experiment_result(
    per_item: list[tuple[dict[str, object], list[Evaluation]]], url: str | None
) -> SimpleNamespace:
    return SimpleNamespace(
        item_results=[
            SimpleNamespace(item=SimpleNamespace(expected_output=expected), evaluations=evals)
            for expected, evals in per_item
        ],
        dataset_run_url=url,
    )


def test_run_on_langfuse_syncs_drops_stale_items_and_maps_the_results() -> None:
    dataset = SimpleNamespace(
        items=[SimpleNamespace(id="q1"), SimpleNamespace(id="removed-from-qa-json")]
    )
    calls: list[dict[str, object]] = []

    def run_experiment(**kwargs: object) -> SimpleNamespace:
        # Captured at call time: the dataset's items as run_experiment()
        # would actually see them.
        calls.append({**kwargs, "item_ids": [i.id for i in dataset.items]})
        return _fake_experiment_result(
            [
                ({"expect_refusal": False}, [Evaluation(name="correctness", value=0.9)]),
                ({"expect_refusal": True}, [Evaluation(name="refusal_accuracy", value=1.0)]),
            ],
            "https://langfuse.example/run/1",
        )

    dataset.run_experiment = run_experiment  # type: ignore[attr-defined]
    client = MagicMock()
    client.get_dataset.return_value = dataset
    qa_items = [{"id": "q1", "question": "Q?", "expert_answer": "A."}]

    outcomes, url = ree._run_on_langfuse(client, "legal", MagicMock(), qa_items)

    client.create_dataset.assert_called_once()  # synced before running
    [call] = calls
    # Langfuse upserts dataset items but never deletes them -- without this
    # filter a question removed from qa.json would keep being asked/counted.
    assert call["item_ids"] == ["q1"]
    assert call["name"] == "Expert eval: legal"
    assert str(call["run_name"]).startswith("legal-")
    assert call["evaluators"] is ree._EVALUATORS
    assert url == "https://langfuse.example/run/1"
    assert [o.expect_refusal for o in outcomes] == [False, True]
    assert outcomes[0].evaluations[0].value == 0.9


# -- connect-or-fall-back ---------------------------------------------------


def test_connect_langfuse_says_so_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ree, "get_langfuse_client", lambda: None)

    client, status = ree._connect_langfuse()

    assert client is None
    assert "not connected" in status
    assert "printed here only" in status


def test_connect_langfuse_says_so_when_the_auth_check_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.auth_check.side_effect = RuntimeError("host unreachable")
    monkeypatch.setattr(ree, "get_langfuse_client", lambda: fake)

    client, status = ree._connect_langfuse()

    assert client is None
    assert "not connected" in status
    assert "host unreachable" in status


def test_connect_langfuse_says_so_when_the_auth_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.auth_check.return_value = False
    monkeypatch.setattr(ree, "get_langfuse_client", lambda: fake)

    client, status = ree._connect_langfuse()

    assert client is None
    assert "auth check failed" in status


def test_connect_langfuse_returns_the_client_when_reachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = MagicMock()
    fake.auth_check.return_value = True
    monkeypatch.setattr(ree, "get_langfuse_client", lambda: fake)

    client, status = ree._connect_langfuse()

    assert client is fake
    assert "connected" in status
    assert "not connected" not in status


def test_run_expertise_uses_the_langfuse_runner_when_given_a_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expertise_dir = _write_expertise(
        tmp_path, "legal", [{"id": "q1", "question": "?", "expert_answer": "!"}]
    )
    _mock_stores(monkeypatch)
    monkeypatch.setattr(ree, "AgentChain", lambda *a, **k: MagicMock())
    monkeypatch.setattr(
        ree,
        "_run_on_langfuse",
        lambda client, name, agent, qa_items: (
            [ree._ItemOutcome(False, [Evaluation(name="correctness", value=0.8)])],
            "https://langfuse.example/run/2",
        ),
    )
    monkeypatch.setattr(
        ree,
        "_run_locally",
        MagicMock(side_effect=AssertionError("the local runner must not run when connected")),
    )

    result = ree._run_expertise(expertise_dir, client=MagicMock())

    assert result.langfuse_url == "https://langfuse.example/run/2"
    assert result.average_correctness == pytest.approx(0.8)


def _canned_result(url: str | None = None) -> "ree._ExpertiseResult":
    return ree._ExpertiseResult(
        name="legal",
        average_correctness=0.9,
        answerable_count=2,
        refusal_accuracy=None,
        refusal_count=0,
        false_refusals=0,
        judge_failures=0,
        langfuse_url=url,
    )


def test_main_prints_results_and_says_so_when_langfuse_is_not_connected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    status = "Langfuse: not connected -- test reason. Results are printed here only."
    seen_clients: list[object] = []

    def fake_run(expertise_dir: Path, client: object = None) -> "ree._ExpertiseResult":
        seen_clients.append(client)
        return _canned_result()

    monkeypatch.setattr(ree, "_discover_expertise_dirs", lambda: [tmp_path / "legal"])
    monkeypatch.setattr(ree, "_connect_langfuse", lambda: (None, status))
    monkeypatch.setattr(ree, "_run_expertise", fake_run)

    ree.main()

    out = capsys.readouterr().out
    assert seen_clients == [None]
    assert "0.900" in out  # the table is still printed
    # Once up front, once again under the table (where it won't have
    # scrolled away behind a long run's log noise).
    assert out.count(status) == 2


def test_main_also_reports_langfuse_run_urls_when_connected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_client = MagicMock()
    seen_clients: list[object] = []

    def fake_run(expertise_dir: Path, client: object = None) -> "ree._ExpertiseResult":
        seen_clients.append(client)
        return _canned_result("https://langfuse.example/run/3")

    monkeypatch.setattr(ree, "_discover_expertise_dirs", lambda: [tmp_path / "legal"])
    monkeypatch.setattr(ree, "_connect_langfuse", lambda: (fake_client, "Langfuse: connected"))
    monkeypatch.setattr(ree, "_run_expertise", fake_run)

    ree.main()

    out = capsys.readouterr().out
    assert seen_clients == [fake_client]
    assert "0.900" in out
    assert "legal: https://langfuse.example/run/3" in out
    fake_client.flush.assert_called_once()


def test_print_results_omits_langfuse_lines_when_console_only(
    capsys: pytest.CaptureFixture[str],
) -> None:
    ree._print_results([_canned_result()])

    assert "Langfuse" not in capsys.readouterr().out
