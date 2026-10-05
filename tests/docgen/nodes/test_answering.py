from typing import Literal

import pytest

from multimodal_rag.docgen.nodes.answering import (
    MAX_RETRIES,
    accept_answer,
    escalate,
    generate_answer_text,
    question_by_id,
    route_after_select,
    route_after_validation,
    select_next_question,
)
from multimodal_rag.docgen.nodes.retrieval import RetrievedChunk
from multimodal_rag.docgen.state import Attempt, DocGenState, InProgress, Question
from multimodal_rag.stores.schema import SearchResult


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


def _question(
    id_: str, status: Literal["pending", "answered", "escalated"] = "pending"
) -> Question:
    return {
        "id": id_,
        "text": f"Question {id_}?",
        "sources_required": ["task_docs"],
        "status": status,
    }


def _chunk(text: str = "evidence") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=SearchResult(
            chunk_id="c1",
            score=1.0,
            text=text,
            source="doc.md",
            doc_id="doc-1",
            element_types=["paragraph"],
        ),
        source_role="task_docs",
    )


def _attempt() -> Attempt:
    return {"query": "q", "chunks": [], "answer": "a", "reason": "r"}


def _in_progress(valid: bool | None, attempt_count: int = 0) -> InProgress:
    return {
        "question_id": "q1",
        "query": "q",
        "chunks": [],
        "answer": "a",
        "attempts": [_attempt() for _ in range(attempt_count)],
        "valid": valid,
    }


def _state(questions: list[Question], current: InProgress | None) -> DocGenState:
    return {
        "sources": [],
        "questions": questions,
        "answers": {},
        "configuration": {"template": "", "format": "pptx", "confirmed": False},
        "review": {"decision": "pending", "flagged_question_ids": []},
        "current": current,
    }


def test_select_next_question_picks_the_first_pending_question() -> None:
    state = _state([_question("q1", "answered"), _question("q2", "pending")], current=None)

    update = select_next_question(state)

    current = update["current"]
    assert current["question_id"] == "q2"
    assert current["query"] == "Question q2?"
    assert current["chunks"] == []
    assert current["attempts"] == []
    assert current["valid"] is None


def test_select_next_question_returns_none_when_nothing_is_pending() -> None:
    state = _state([_question("q1", "answered")], current=None)

    assert select_next_question(state) == {"current": None}


def test_route_after_select_goes_to_retrieve_when_a_question_was_selected() -> None:
    state = _state([_question("q1")], current=_in_progress(valid=None))

    assert route_after_select(state) == "retrieve"


def test_route_after_select_goes_to_end_when_nothing_was_selected() -> None:
    state = _state([], current=None)

    assert route_after_select(state) == "end"


def test_route_after_validation_accepts_when_valid() -> None:
    state = _state([_question("q1")], current=_in_progress(valid=True))

    assert route_after_validation(state) == "accept"


def test_route_after_validation_retries_when_invalid_and_retries_remain() -> None:
    current = _in_progress(valid=False, attempt_count=MAX_RETRIES - 1)
    state = _state([_question("q1")], current=current)

    assert route_after_validation(state) == "retry"


def test_route_after_validation_escalates_when_retries_are_exhausted() -> None:
    state = _state([_question("q1")], current=_in_progress(valid=False, attempt_count=MAX_RETRIES))

    assert route_after_validation(state) == "escalate"


def test_accept_answer_records_the_answer_and_marks_the_question_answered() -> None:
    current = _in_progress(valid=True, attempt_count=1)
    current["answer"] = "final answer"
    state = _state([_question("q1", "pending")], current=current)

    update = accept_answer(state)

    assert update["answers"]["q1"]["text"] == "final answer"
    assert update["answers"]["q1"]["accepted_by"] == "validation"
    assert len(update["answers"]["q1"]["attempts"]) == 1
    assert update["questions"][0]["status"] == "answered"
    assert update["current"] is None


def test_escalate_marks_the_question_escalated_and_clears_current() -> None:
    state = _state(
        [_question("q1", "pending")], current=_in_progress(valid=False, attempt_count=MAX_RETRIES)
    )

    update = escalate(state)

    assert update["questions"][0]["status"] == "escalated"
    assert update["current"] is None


def test_question_by_id_raises_for_an_unknown_id() -> None:
    with pytest.raises(KeyError):
        question_by_id([_question("q1")], "does-not-exist")


def test_generate_answer_text_includes_prior_rejected_attempts_in_the_prompt() -> None:
    llm = _FakeLLM("final answer")
    attempts: list[Attempt] = [
        {"query": "first try", "chunks": [], "answer": "wrong answer", "reason": "not grounded"}
    ]

    generate_answer_text("What is the latency?", [_chunk()], attempts, llm=llm)

    prompt = llm.calls[0][0]["content"]
    assert "first try" in prompt
    assert "wrong answer" in prompt
    assert "not grounded" in prompt


def test_generate_answer_text_omits_the_history_section_on_a_first_attempt() -> None:
    llm = _FakeLLM("final answer")

    generate_answer_text("What is the latency?", [_chunk()], None, llm=llm)

    prompt = llm.calls[0][0]["content"]
    assert "PREVIOUS ATTEMPTS" not in prompt
