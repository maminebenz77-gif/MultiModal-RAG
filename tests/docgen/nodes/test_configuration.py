import json

import pytest

from multimodal_rag.docgen.nodes.configuration import (
    InterpretedQuestion,
    InterpretedRequest,
    InterpretParseError,
    _format_confirmation_summary,
    build_questions,
    freeze_configuration,
    interpret_request_text,
    route_after_confirmation,
)
from multimodal_rag.docgen.sources import SourceSpec
from multimodal_rag.docgen.state import Configuration, DocGenState, Question


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


def _question(id_: str, sources_required: list[str] | None = None) -> Question:
    return {
        "id": id_,
        "text": f"Question {id_}?",
        "sources_required": sources_required or ["task_docs"],  # type: ignore[typeddict-item]
        "status": "pending",
    }


def _state(questions: list[Question], sources: list[SourceSpec]) -> DocGenState:
    return {
        "sources": sources,
        "questions": questions,
        "answers": {},
        "configuration": {"template": "", "format": "pptx", "confirmed": False, "max_retries": 3},
        "review": {"decision": "pending", "flagged_question_ids": []},
        "current": None,
        "usage": {"llm_calls": 0},
        "request": {"text": "", "corrections": []},
    }


def test_interpret_request_text_parses_a_well_formed_response() -> None:
    reply = json.dumps(
        {
            "questions": [
                {"text": "What was the latency?", "sources_required": ["task_docs"]},
                {
                    "text": "How does it compare to the standard?",
                    "sources_required": ["task_docs", "reference_kb"],
                },
            ],
            "format": "pptx",
            "template": "",
        }
    )
    llm = _FakeLLM(reply)

    result = interpret_request_text(
        "Ask about latency and compare to the standard.", [], ["task_docs", "reference_kb"], llm
    )

    assert len(result.questions) == 2
    assert result.questions[0].text == "What was the latency?"
    assert result.questions[1].sources_required == ["task_docs", "reference_kb"]
    assert result.format == "pptx"


def test_interpret_request_text_includes_available_sources_and_corrections_in_the_prompt() -> (
    None
):
    reply = json.dumps(
        {"questions": [{"text": "Q?", "sources_required": []}], "format": "docx", "template": ""}
    )
    llm = _FakeLLM(reply)

    interpret_request_text(
        "Original request.", ["Also ask about cost."], ["task_docs"], llm
    )

    prompt = llm.calls[0][0]["content"]
    assert "task_docs" in prompt
    assert "Original request." in prompt
    assert "Also ask about cost." in prompt


def test_interpret_request_text_raises_on_non_json_output() -> None:
    llm = _FakeLLM("not json at all")

    with pytest.raises(InterpretParseError):
        interpret_request_text("Ask something.", [], ["task_docs"], llm)


def test_interpret_request_text_raises_on_an_empty_question_list() -> None:
    reply = json.dumps({"questions": [], "format": "pptx", "template": ""})
    llm = _FakeLLM(reply)

    with pytest.raises(InterpretParseError):
        interpret_request_text("Ask something.", [], ["task_docs"], llm)


def test_interpret_request_text_raises_on_an_invalid_format_value() -> None:
    reply = json.dumps(
        {"questions": [{"text": "Q?", "sources_required": []}], "format": "pdf", "template": ""}
    )
    llm = _FakeLLM(reply)

    with pytest.raises(InterpretParseError):
        interpret_request_text("Ask something.", [], ["task_docs"], llm)


def test_build_questions_assigns_sequential_ids_and_pending_status() -> None:
    interpreted = InterpretedRequest(
        questions=[
            InterpretedQuestion(text="First?", sources_required=["task_docs"]),
            InterpretedQuestion(text="Second?", sources_required=["reference_kb"]),
        ],
        format="pptx",
        template="",
    )

    questions = build_questions(interpreted)

    assert [q["id"] for q in questions] == ["q1", "q2"]
    assert [q["status"] for q in questions] == ["pending", "pending"]
    assert questions[1]["sources_required"] == ["reference_kb"]


def test_format_confirmation_summary_reports_no_questions_when_the_draft_is_empty() -> None:
    configuration: Configuration = {
        "template": "",
        "format": "pptx",
        "confirmed": False,
        "max_retries": 3,
    }

    summary = _format_confirmation_summary([], configuration)

    assert "revise" in summary.lower()


def test_format_confirmation_summary_lists_each_question_and_its_sources() -> None:
    questions = [_question("q1", ["task_docs"]), _question("q2", ["task_docs", "reference_kb"])]
    configuration: Configuration = {
        "template": "report.pptx",
        "format": "pptx",
        "confirmed": False,
        "max_retries": 3,
    }

    summary = _format_confirmation_summary(questions, configuration)

    assert "Question q1?" in summary
    assert "task_docs, reference_kb" in summary
    assert "report.pptx" in summary


def test_route_after_confirmation_freezes_when_confirmed() -> None:
    state = _state([_question("q1")], [])
    state["configuration"]["confirmed"] = True

    assert route_after_confirmation(state) == "frozen"


def test_route_after_confirmation_revises_when_not_confirmed() -> None:
    state = _state([_question("q1")], [])

    assert route_after_confirmation(state) == "revise"


def test_freeze_configuration_drops_a_hallucinated_source_role() -> None:
    questions = [_question("q1", ["task_docs", "reference_kb"])]
    sources = [SourceSpec(role="task_docs", tag="docgen:task:demo", required=True)]
    state = _state(questions, sources)

    update = freeze_configuration(state)

    assert update["questions"][0]["sources_required"] == ["task_docs"]


def test_freeze_configuration_keeps_a_question_with_no_valid_sources_but_empties_its_list() -> (
    None
):
    questions = [_question("q1", ["reference_kb"])]
    sources = [SourceSpec(role="task_docs", tag="docgen:task:demo", required=True)]
    state = _state(questions, sources)

    update = freeze_configuration(state)

    assert update["questions"][0]["sources_required"] == []
    assert len(update["questions"]) == 1  # the question itself is not dropped
