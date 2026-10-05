"""The per-question subgraph's remaining pieces: picking the next
question, drafting an answer, routing after validation, and accepting
or escalating the result.

Functions here that only touch `state` (select_next_question,
accept_answer, escalate, the two routers, question_by_id) are plain,
graph-independent and directly testable. generate_answer_text needs an
LLMProvider, injected the same optional way as validate_answer, for
the same testability reason. The three nodes that ALSO need a
Retriever (retrieve) or an LLMProvider bound ahead of time
(generate_answer, validate_answer) are built as closures in
docgen/graph.py instead of living here, so this module stays about
WHAT each step does, not how its dependencies get wired in.
"""

from __future__ import annotations

from typing import Any, Literal

from ...providers.base import LLMProvider
from ...providers.factory import get_llm
from ...tracing import traced_span, update_span_output
from ..state import Attempt, DocGenState, Question
from .retrieval import RetrievedChunk, format_chunks

MAX_RETRIES = 3
"""Caps the retrieve -> generate_answer -> validate_answer loop per
question before escalating to a human -- distinct from LangGraph's own
per-node retry_policy, which is for transient failures (a network
blip), not a node succeeding and deliberately judging the result
invalid."""

_ANSWER_PROMPT = """## ROLE
You are a careful research assistant. Answer the QUESTION using only the CONTEXT given -- do \
not use outside knowledge, and do not guess at anything the CONTEXT doesn't support.
{history}
## QUESTION
{question}

## CONTEXT
{context}

## ANSWER
Write the answer directly, in plain prose. Do not repeat the question."""

_HISTORY_HEADER = """
## PREVIOUS ATTEMPTS (rejected -- do not repeat the same mistake)
{attempts}
"""


def question_by_id(questions: list[Question], question_id: str) -> Question:
    for question in questions:
        if question["id"] == question_id:
            return question
    raise KeyError(f"No question with id={question_id!r}")


def _format_attempts(attempts: list[Attempt]) -> str:
    return "\n".join(
        f"- Searched {attempt['query']!r}, answered {attempt['answer']!r} -- rejected because: "
        f"{attempt['reason']}"
        for attempt in attempts
    )


def generate_answer_text(
    question: str,
    chunks: list[RetrievedChunk],
    attempts: list[Attempt] | None = None,
    llm: LLMProvider | None = None,
) -> str:
    """Drafts an answer from the retrieved chunks. Shows prior REJECTED
    attempts (query/answer/reason) when retrying, so the model doesn't
    just repeat the same search and the same wrong answer."""
    llm = llm or get_llm()
    history = _HISTORY_HEADER.format(attempts=_format_attempts(attempts)) if attempts else ""
    prompt = _ANSWER_PROMPT.format(
        history=history, question=question, context=format_chunks(chunks)
    )
    with traced_span("docgen_generate_answer", as_type="generation", input=prompt) as span:
        answer = llm.generate([{"role": "user", "content": prompt}])
        update_span_output(span, answer)
    return answer


def select_next_question(state: DocGenState) -> dict[str, Any]:
    """Picks the first still-pending question and opens a fresh
    InProgress record for it. Returns `{"current": None}` when none are
    left -- route_after_select is what actually decides where that
    sends the graph."""
    for question in state["questions"]:
        if question["status"] == "pending":
            return {
                "current": {
                    "question_id": question["id"],
                    "query": question["text"],
                    "chunks": [],
                    "answer": "",
                    "attempts": [],
                    "valid": None,
                }
            }
    return {"current": None}


def route_after_select(state: DocGenState) -> Literal["retrieve", "end"]:
    return "retrieve" if state["current"] is not None else "end"


def route_after_validation(state: DocGenState) -> Literal["accept", "retry", "escalate"]:
    current = state["current"]
    assert current is not None
    if current["valid"]:
        return "accept"
    return "retry" if len(current["attempts"]) < MAX_RETRIES else "escalate"


def accept_answer(state: DocGenState) -> dict[str, Any]:
    current = state["current"]
    assert current is not None
    answers = dict(state["answers"])
    answers[current["question_id"]] = {
        "text": current["answer"],
        "attempts": current["attempts"],
        "accepted_by": "validation",
    }
    questions = [
        {**q, "status": "answered"} if q["id"] == current["question_id"] else q
        for q in state["questions"]
    ]
    return {"answers": answers, "questions": questions, "current": None}


def escalate(state: DocGenState) -> dict[str, Any]:
    """Stand-in for nodes/escalation.py's real, interrupt-based
    ask_human (a later phase) -- for now, just marks the question
    escalated and stops, so this subgraph is fully testable before that
    human-in-the-loop machinery exists."""
    current = state["current"]
    assert current is not None
    questions = [
        {**q, "status": "escalated"} if q["id"] == current["question_id"] else q
        for q in state["questions"]
    ]
    return {"questions": questions, "current": None}
