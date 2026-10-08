from typing import Literal

import pytest

from multimodal_rag.docgen.nodes.answering import (
    DEFAULT_MAX_RETRIES,
    escalate,
    formulate_query,
    generate_answer_text,
    prior_qa_pairs,
    question_by_id,
    record_answer,
    route_after_human,
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
    id_: str, status: Literal["pending", "answered", "escalated", "skipped"] = "pending"
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


def _in_progress(
    valid: bool | None, attempt_count: int = 0, human_guidance: str | None = None
) -> InProgress:
    return {
        "question_id": "q1",
        "query": "q",
        "chunks": [],
        "answer": "a",
        "attempts": [_attempt() for _ in range(attempt_count)],
        "valid": valid,
        "human_guidance": human_guidance,
    }


def _state(questions: list[Question], current: InProgress | None) -> DocGenState:
    return {
        "sources": [],
        "questions": questions,
        "answers": {},
        "configuration": {
            "template": "",
            "format": "pptx",
            "confirmed": False,
            "max_retries": DEFAULT_MAX_RETRIES,
        },
        "review": {"decision": "pending", "flagged_question_ids": []},
        "current": current,
        "usage": {"llm_calls": 0},
    }


def test_select_next_question_picks_the_first_pending_question() -> None:
    state = _state([_question("q1", "answered"), _question("q2", "pending")], current=None)

    update = select_next_question(state)

    current = update["current"]
    assert current["question_id"] == "q2"
    assert current["query"] == ""  # formulate_query fills this in once the graph runs
    assert current["chunks"] == []
    assert current["attempts"] == []
    assert current["valid"] is None
    assert current["human_guidance"] is None


def test_select_next_question_returns_none_when_nothing_is_pending() -> None:
    state = _state([_question("q1", "answered")], current=None)

    assert select_next_question(state) == {"current": None}


def test_route_after_select_goes_to_attempt_when_a_question_was_selected() -> None:
    state = _state([_question("q1")], current=_in_progress(valid=None))

    assert route_after_select(state) == "attempt"


def test_route_after_select_goes_to_end_when_nothing_was_selected() -> None:
    state = _state([], current=None)

    assert route_after_select(state) == "end"


def test_route_after_validation_selects_next_when_current_was_cleared() -> None:
    """validate_node (docgen/graph.py) calls record_answer directly on a
    valid verdict -- that already clears `current`, same signal
    route_after_human uses, so there's no separate "accept" outcome to
    route to."""
    state = _state([_question("q1", "answered")], current=None)

    assert route_after_validation(state) == "select_next"


def test_route_after_validation_retries_when_invalid_and_retries_remain() -> None:
    current = _in_progress(valid=False, attempt_count=DEFAULT_MAX_RETRIES - 1)
    state = _state([_question("q1")], current=current)

    assert route_after_validation(state) == "retry"


def test_route_after_validation_escalates_when_retries_are_exhausted() -> None:
    current = _in_progress(valid=False, attempt_count=DEFAULT_MAX_RETRIES)
    state = _state([_question("q1")], current=current)

    assert route_after_validation(state) == "escalate"


def test_route_after_validation_honors_a_per_run_max_retries() -> None:
    """max_retries is per-run configuration, not a fixed constant -- a
    run configured for fewer attempts escalates sooner."""
    current = _in_progress(valid=False, attempt_count=1)
    state = _state([_question("q1")], current=current)
    state["configuration"]["max_retries"] = 1

    assert route_after_validation(state) == "escalate"


def test_record_answer_marks_the_question_answered() -> None:
    current = _in_progress(valid=True, attempt_count=1)
    state = _state([_question("q1", "pending")], current=current)

    update = record_answer(state, current, "final answer", "validation")

    assert update["answers"]["q1"]["text"] == "final answer"
    assert update["answers"]["q1"]["accepted_by"] == "validation"
    assert len(update["answers"]["q1"]["attempts"]) == 1
    assert update["questions"][0]["status"] == "answered"
    assert update["current"] is None


def test_escalate_marks_the_question_escalated_and_preserves_current() -> None:
    """current is deliberately left in place -- ask_human
    (nodes/escalation.py) needs to show a human the chunks/attempts
    that led to escalation, which this placeholder must not discard
    first."""
    current = _in_progress(valid=False, attempt_count=DEFAULT_MAX_RETRIES)
    state = _state([_question("q1", "pending")], current=current)

    update = escalate(state)

    assert update["questions"][0]["status"] == "escalated"
    assert "current" not in update  # unchanged, not cleared


def test_route_after_human_retries_when_current_is_still_populated() -> None:
    """ask_human's "reformulate" action leaves `current` populated --
    no separate action flag needed, its presence alone says what to do."""
    state = _state(
        [_question("q1", "escalated")],
        current=_in_progress(valid=None, human_guidance="Check the hosted API row specifically."),
    )

    assert route_after_human(state) == "retry"


def test_route_after_human_selects_next_when_current_was_cleared() -> None:
    """Both "answer" (via record_answer) and "skip" clear `current` --
    either way, there's nothing left to retry."""
    state = _state([_question("q1", "answered")], current=None)

    assert route_after_human(state) == "select_next"


def test_question_by_id_raises_for_an_unknown_id() -> None:
    with pytest.raises(KeyError):
        question_by_id([_question("q1")], "does-not-exist")


def test_prior_qa_pairs_includes_only_already_answered_questions_in_order() -> None:
    state = _state(
        [_question("q1", "answered"), _question("q2", "pending"), _question("q3", "answered")],
        current=None,
    )
    state["answers"] = {
        "q1": {"text": "Answer 1.", "attempts": [], "accepted_by": "validation"},
        "q3": {"text": "Answer 3.", "attempts": [], "accepted_by": "validation"},
    }

    pairs = prior_qa_pairs(state)

    assert pairs == [("Question q1?", "Answer 1."), ("Question q3?", "Answer 3.")]


def test_formulate_query_includes_prior_qa_for_resolving_references() -> None:
    llm = _FakeLLM("resolved search query")
    prior_qa = [("What is the hosted API's latency?", "120ms on average.")]

    formulate_query("How does it compare to the standard?", prior_qa, llm=llm)

    prompt = llm.calls[0][0]["content"]
    assert "What is the hosted API's latency?" in prompt
    assert "120ms on average." in prompt


def test_formulate_query_includes_rejected_attempts_on_a_retry() -> None:
    llm = _FakeLLM("a different search query")
    attempts: list[Attempt] = [
        {"query": "latency benchmark", "chunks": [], "answer": "wrong", "reason": "not grounded"}
    ]

    formulate_query("What was the latency?", [], attempts, llm=llm)

    prompt = llm.calls[0][0]["content"]
    assert "latency benchmark" in prompt
    assert "not grounded" in prompt


def test_formulate_query_omits_both_optional_sections_on_a_bare_first_attempt() -> None:
    llm = _FakeLLM("search query")

    formulate_query("What was the latency?", [], None, llm=llm)

    prompt = llm.calls[0][0]["content"]
    # The task instructions mention "PRIOR QUESTIONS AND ANSWERS" in
    # passing regardless -- check for the actual section HEADING, which
    # only appears when prior_qa is non-empty.
    assert "## PRIOR QUESTIONS AND ANSWERS (" not in prompt
    assert "PREVIOUS SEARCH ATTEMPTS" not in prompt


def test_formulate_query_includes_human_guidance_after_a_reformulate() -> None:
    llm = _FakeLLM("a steered search query")

    formulate_query(
        "What was the latency?",
        [],
        None,
        "Check the hosted API row specifically, not the gateway.",
        llm=llm,
    )

    prompt = llm.calls[0][0]["content"]
    assert "GUIDANCE FROM A HUMAN" in prompt
    assert "Check the hosted API row specifically, not the gateway." in prompt


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


def test_generate_answer_text_includes_human_guidance_after_a_reformulate() -> None:
    llm = _FakeLLM("final answer")

    generate_answer_text(
        "What is the latency?",
        [_chunk()],
        None,
        "By API, the human means the hosted one specifically.",
        llm=llm,
    )

    prompt = llm.calls[0][0]["content"]
    assert "GUIDANCE FROM A HUMAN" in prompt
    assert "By API, the human means the hosted one specifically." in prompt
