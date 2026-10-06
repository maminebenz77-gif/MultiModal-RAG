"""The per-question subgraph's remaining pieces: picking the next
question, formulating a search query, drafting an answer, routing
after validation, and accepting or escalating the result.

formulate_query runs before EVERY retrieval, not just retries -- a
question's own text can be a bad standalone search query (it can refer
back to an earlier question, "it"/"that"), and a retry that reused the
same query against the same corpus would deterministically get back
the exact same chunks every time, which can never fix a retrieval
problem no matter how many times the answer gets rewritten.

Functions here that only touch `state` (select_next_question,
accept_answer, escalate, the two routers, question_by_id,
prior_qa_pairs) are plain, graph-independent and directly testable.
formulate_query/generate_answer_text need an LLMProvider, injected the
same optional way as validate_answer, for the same testability reason.
The actual graph nodes that call formulate_query/retrieve_for_question/
generate_answer_text/validate_answer, with a bound Retriever and
LLMProvider and the try/except that turns a persistent failure into a
rejected attempt instead of crashing the whole run, are built as
closures in docgen/graph.py instead of living here, so this module
stays about WHAT each step does, not how its dependencies get wired in
or how a failure gets handled.
"""

from __future__ import annotations

from typing import Any, Literal

from ...providers.base import LLMProvider
from ...providers.factory import get_llm
from ...tracing import traced_span, update_span_output
from ..state import Attempt, DocGenState, InProgress, Question
from .retrieval import RetrievedChunk, format_chunks

MAX_RETRIES = 3
"""Caps the attempt_answer -> validate_answer loop per question before
escalating to a human -- distinct from LangGraph's own per-node
retry_policy (docgen/graph.py), which is for a genuinely transient
failure (a dropped connection), not a node succeeding and deliberately
judging the result invalid, or a persistent failure that already
survived those automatic attempts."""

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

_QUERY_PROMPT = """## ROLE
You turn a QUESTION into a short, standalone search query for a document retrieval system.

## TASK
Resolve any pronouns or references in the QUESTION ("it", "that", "the above", ...) using the \
PRIOR QUESTIONS AND ANSWERS below, then write a short search query focused on the concrete \
terms someone would search a document store for. Output ONLY the search query text -- no \
quotes, no explanation, nothing else.
{prior_qa}{history}
## QUESTION
{question}

## SEARCH QUERY"""

_PRIOR_QA_SECTION = """
## PRIOR QUESTIONS AND ANSWERS (for resolving references like "it")
{pairs}
"""

_RETRY_SECTION = """
## PREVIOUS SEARCH ATTEMPTS FOR THIS QUESTION (rejected -- try something meaningfully different)
{attempts}
"""

_GUIDANCE_SECTION = """
## GUIDANCE FROM A HUMAN (follow this -- it was given specifically to help with this question)
{guidance}
"""


def question_by_id(questions: list[Question], question_id: str) -> Question:
    for question in questions:
        if question["id"] == question_id:
            return question
    raise KeyError(f"No question with id={question_id!r}")


def prior_qa_pairs(state: DocGenState) -> list[tuple[str, str]]:
    """Every already-accepted question/answer pair, in question order --
    context formulate_query needs to resolve a later question's
    reference back to an earlier one ("it", "that")."""
    return [
        (question["text"], state["answers"][question["id"]]["text"])
        for question in state["questions"]
        if question["id"] in state["answers"]
    ]


def _format_attempts(attempts: list[Attempt]) -> str:
    return "\n".join(
        f"- Searched {attempt['query']!r}, answered {attempt['answer']!r} -- rejected because: "
        f"{attempt['reason']}"
        for attempt in attempts
    )


def formulate_query(
    question: str,
    prior_qa: list[tuple[str, str]],
    attempts: list[Attempt] | None = None,
    human_guidance: str | None = None,
    llm: LLMProvider | None = None,
) -> str:
    """Turns a question into a search query -- run before EVERY
    retrieval, first attempt included, not just retries: a question's
    own text can be a bad standalone query the moment it refers back to
    an earlier one ("How does it compare to the standard?"), so prior
    Q&A context is needed even the first time, not only when retrying.
    On a retry, `attempts` is also shown, so the new query is actually
    steered away from what already failed, not a blind re-ask of the
    same text against the same corpus (which would just return the
    exact same chunks every time). `human_guidance` is set only after
    ask_human's "reformulate" action -- a human's own steering context,
    separate from (and shown ahead of) automated rejection history."""
    llm = llm or get_llm()
    prior_qa_section = (
        _PRIOR_QA_SECTION.format(pairs=_format_prior_qa(prior_qa)) if prior_qa else ""
    )
    guidance_section = (
        _GUIDANCE_SECTION.format(guidance=human_guidance) if human_guidance else ""
    )
    history_section = (
        _RETRY_SECTION.format(attempts=_format_attempts(attempts)) if attempts else ""
    )
    prompt = _QUERY_PROMPT.format(
        prior_qa=prior_qa_section,
        history=guidance_section + history_section,
        question=question,
    )
    with traced_span("docgen_formulate_query", as_type="generation", input=prompt) as span:
        query = llm.generate([{"role": "user", "content": prompt}]).strip()
        update_span_output(span, query)
    return query


def _format_prior_qa(prior_qa: list[tuple[str, str]]) -> str:
    return "\n".join(f"- Q: {question}\n  A: {answer}" for question, answer in prior_qa)


def generate_answer_text(
    question: str,
    chunks: list[RetrievedChunk],
    attempts: list[Attempt] | None = None,
    human_guidance: str | None = None,
    llm: LLMProvider | None = None,
) -> str:
    """Drafts an answer from the retrieved chunks. Shows prior REJECTED
    attempts (query/answer/reason) when retrying, so the model doesn't
    just repeat the same search and the same wrong answer, plus a
    human's own guidance (set only after ask_human's "reformulate"
    action), shown ahead of that automated history."""
    llm = llm or get_llm()
    guidance_section = (
        _GUIDANCE_SECTION.format(guidance=human_guidance) if human_guidance else ""
    )
    history_section = (
        _HISTORY_HEADER.format(attempts=_format_attempts(attempts)) if attempts else ""
    )
    prompt = _ANSWER_PROMPT.format(
        history=guidance_section + history_section,
        question=question,
        context=format_chunks(chunks),
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
                    "query": "",
                    "chunks": [],
                    "answer": "",
                    "attempts": [],
                    "valid": None,
                    "human_guidance": None,
                }
            }
    return {"current": None}


def route_after_select(state: DocGenState) -> Literal["attempt", "end"]:
    return "attempt" if state["current"] is not None else "end"


def route_after_validation(state: DocGenState) -> Literal["accept", "retry", "escalate"]:
    current = state["current"]
    assert current is not None
    if current["valid"]:
        return "accept"
    return "retry" if len(current["attempts"]) < MAX_RETRIES else "escalate"


def route_after_human(state: DocGenState) -> Literal["retry", "select_next"]:
    """ask_human leaves `current` populated (non-None) only for its
    "reformulate" action -- "answer" and "skip" both clear it via
    record_answer/escalate()'s own return, same as a normal acceptance.
    No separate action flag needed: current's presence already says
    which happened."""
    return "retry" if state["current"] is not None else "select_next"


def record_answer(
    state: DocGenState, current: InProgress, text: str, accepted_by: Literal["validation", "human"]
) -> dict[str, Any]:
    """Writes the final Answer and marks the question done -- shared by
    accept_answer (a validated answer) and nodes/escalation.py's
    ask_human (a human-provided one), so there is one place that does
    this instead of two near-identical copies."""
    answers = dict(state["answers"])
    answers[current["question_id"]] = {
        "text": text,
        "attempts": current["attempts"],
        "accepted_by": accepted_by,
    }
    questions = [
        {**q, "status": "answered"} if q["id"] == current["question_id"] else q
        for q in state["questions"]
    ]
    return {"answers": answers, "questions": questions, "current": None}


def accept_answer(state: DocGenState) -> dict[str, Any]:
    current = state["current"]
    assert current is not None
    return record_answer(state, current, current["answer"], "validation")


def escalate(state: DocGenState) -> dict[str, Any]:
    """Stand-in for nodes/escalation.py's real, interrupt-based
    ask_human (a later phase) -- for now, just marks the question
    escalated and stops, so this subgraph is fully testable before that
    human-in-the-loop machinery exists.

    Deliberately does NOT clear `current`: the spec calls for surfacing
    "the question, the retrieved chunks, and the last failed attempt"
    to a human, which means ask_human needs current.chunks/attempts
    still sitting in state when it eventually reads them -- clearing it
    here would discard exactly the information escalation exists to
    show someone."""
    current = state["current"]
    assert current is not None
    questions = [
        {**q, "status": "escalated"} if q["id"] == current["question_id"] else q
        for q in state["questions"]
    ]
    return {"questions": questions}
