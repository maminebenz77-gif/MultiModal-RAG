"""Builds and compiles the per-question subgraph: select a pending
question, attempt an answer (formulate a search query, retrieve, draft
an answer), validate it, and either accept it, loop back with the
rejection reason, or -- once retries run out -- escalate to a real
human via nodes/escalation.py's ask_human, pausing with
LangGraph's interrupt() ("no questions left" still goes straight to
END instead of harmonize_answers, a later phase -- nothing needs to
pause for that one).

A checkpointer is only required to RESUME an interrupt (via
Command(resume=...)), not to reach one -- so build_graph's
checkpointer parameter defaults to None, and only real runs (and tests
that actually resume) need to pass one.

attempt_answer and validate_answer are each wrapped in a try/except AND
given a LangGraph retry_policy: the retry_policy handles a genuinely
transient failure automatically (a dropped connection, a 5xx from the
LLM provider -- see langgraph.types.default_retry_on, which already
excludes ValueError/TypeError/OSError-family exceptions, including our
own ValidationParseError, from automatic retry, since those mean "the
model responded badly," not "the network blipped"); the try/except is
what stops a PERSISTENT failure -- one that survives those automatic
attempts too -- from crashing the whole run and losing every other
question's completed work. It's converted into a rejected attempt with
the technical reason recorded, so it flows through the exact same
retry/escalate logic a substantive rejection would.

Nodes that need real dependencies (a Retriever, an LLMProvider) are
built as closures inside build_graph() -- the simplest way to hand
them in without threading LangGraph's own config/context mechanism
through, and it keeps nodes/answering.py's functions free of any
graph-building or error-handling concern.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from ..providers.base import LLMProvider
from ..providers.factory import get_llm
from .nodes.answering import (
    accept_answer,
    escalate,
    formulate_query,
    generate_answer_text,
    prior_qa_pairs,
    question_by_id,
    route_after_select,
    route_after_validation,
    select_next_question,
)
from .nodes.escalation import ask_human
from .nodes.retrieval import retrieve_for_question
from .stack import DocgenStack
from .state import Attempt, DocGenState
from .validation import validate_answer

_TRANSIENT_RETRY = RetryPolicy(max_attempts=3)


def build_graph(
    stack: DocgenStack,
    llm: LLMProvider | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    llm = llm or get_llm()

    def attempt_answer_node(state: DocGenState) -> dict[str, Any]:
        current = state["current"]
        assert current is not None
        question = question_by_id(state["questions"], current["question_id"])
        try:
            query = formulate_query(
                question["text"], prior_qa_pairs(state), current["attempts"], llm
            )
            sources = [s for s in state["sources"] if s.role in question["sources_required"]]
            chunks = retrieve_for_question(query, sources, stack.retriever)
            answer = generate_answer_text(question["text"], chunks, current["attempts"], llm)
        except Exception as exc:
            attempt: Attempt = {
                "query": current["query"],
                "chunks": current["chunks"],
                "answer": "",
                "reason": f"Technical failure while attempting an answer: {exc}",
            }
            return {
                "current": {
                    **current,
                    "valid": False,
                    "attempts": [*current["attempts"], attempt],
                }
            }
        return {
            "current": {
                **current,
                "query": query,
                "chunks": chunks,
                "answer": answer,
                "valid": None,  # a fresh attempt, not yet judged -- clears any stale
                # False left over from the cycle that just got retried.
            }
        }

    def validate_node(state: DocGenState) -> dict[str, Any]:
        current = state["current"]
        assert current is not None
        if current["valid"] is False:
            # attempt_answer_node already failed and recorded its own
            # rejected attempt above -- there is no answer to judge.
            return {}
        question = question_by_id(state["questions"], current["question_id"])
        try:
            result = validate_answer(question["text"], current["answer"], current["chunks"], llm)
        except Exception as exc:
            attempt: Attempt = {
                "query": current["query"],
                "chunks": current["chunks"],
                "answer": current["answer"],
                "reason": f"Technical failure while validating the answer: {exc}",
            }
            return {
                "current": {
                    **current,
                    "valid": False,
                    "attempts": [*current["attempts"], attempt],
                }
            }
        if result.valid:
            return {"current": {**current, "valid": True}}
        attempt = {
            "query": current["query"],
            "chunks": current["chunks"],
            "answer": current["answer"],
            "reason": result.reason,
        }
        return {
            "current": {**current, "valid": False, "attempts": [*current["attempts"], attempt]}
        }

    builder = StateGraph(DocGenState)
    builder.add_node("select_next_question", select_next_question)
    builder.add_node("attempt_answer", attempt_answer_node, retry_policy=_TRANSIENT_RETRY)
    builder.add_node("validate_answer", validate_node, retry_policy=_TRANSIENT_RETRY)
    builder.add_node("accept_answer", accept_answer)
    builder.add_node("escalate", escalate)
    builder.add_node("ask_human", ask_human)

    builder.add_edge(START, "select_next_question")
    builder.add_conditional_edges(
        "select_next_question", route_after_select, {"attempt": "attempt_answer", "end": END}
    )
    builder.add_edge("attempt_answer", "validate_answer")
    builder.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {"accept": "accept_answer", "retry": "attempt_answer", "escalate": "escalate"},
    )
    builder.add_edge("accept_answer", "select_next_question")
    builder.add_edge("escalate", "ask_human")
    builder.add_edge("ask_human", "select_next_question")

    return builder.compile(checkpointer=checkpointer)
