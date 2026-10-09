import json

import pytest

from multimodal_rag.docgen.nodes.harmonize import (
    HarmonizeParseError,
    _format_qa_pairs,
    harmonize_answers_text,
)
from multimodal_rag.docgen.state import Answer, Question


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


def _question(id_: str, text: str) -> Question:
    return {"id": id_, "text": text, "sources_required": ["task_docs"], "status": "answered"}


def _answer(text: str) -> Answer:
    return {"text": text, "attempts": [], "accepted_by": "validation"}


def test_format_qa_pairs_includes_every_answered_question() -> None:
    questions = [_question("q1", "What was the latency?"), _question("q2", "What was the cost?")]
    answers = {"q1": _answer("120ms."), "q2": _answer("$5.")}

    pairs = _format_qa_pairs(questions, answers)

    assert "What was the latency?" in pairs
    assert "120ms." in pairs
    assert "What was the cost?" in pairs
    assert "$5." in pairs


def test_format_qa_pairs_skips_a_question_with_no_answer() -> None:
    questions = [_question("q1", "Skipped question?")]

    pairs = _format_qa_pairs(questions, {})

    assert pairs == ""


def test_harmonize_answers_text_returns_the_rewritten_mapping() -> None:
    reply = json.dumps({"q1": "The average latency was 120ms.", "q2": "It cost $5."})
    llm = _FakeLLM(reply)
    questions = [_question("q1", "What was the latency?"), _question("q2", "What was the cost?")]
    answers = {"q1": _answer("120ms."), "q2": _answer("Five dollars.")}

    result = harmonize_answers_text(questions, answers, llm=llm)

    assert result == {"q1": "The average latency was 120ms.", "q2": "It cost $5."}
    assert len(llm.calls) == 1


def test_harmonize_answers_text_raises_on_non_json_output() -> None:
    llm = _FakeLLM("not json at all")

    with pytest.raises(HarmonizeParseError):
        harmonize_answers_text([_question("q1", "Q?")], {"q1": _answer("A.")}, llm=llm)


def test_harmonize_answers_text_raises_when_a_value_is_not_a_string() -> None:
    llm = _FakeLLM(json.dumps({"q1": {"nested": "object"}}))

    with pytest.raises(HarmonizeParseError):
        harmonize_answers_text([_question("q1", "Q?")], {"q1": _answer("A.")}, llm=llm)
