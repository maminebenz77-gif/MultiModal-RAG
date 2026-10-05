"""Builds and compiles the per-question subgraph: select a pending
question, retrieve evidence, draft an answer, validate it, and either
accept it, loop back with the rejection reason, or -- once retries run
out -- escalate (nodes/answering.py's escalate() is a stand-in here;
nodes/escalation.py's real, interrupt-based ask_human replaces it in a
later phase, same for "no questions left" currently going straight to
END instead of harmonize_answers).

Nodes that need real dependencies (a Retriever, an LLMProvider) are
built as closures inside build_graph() -- the simplest way to hand
them in without threading LangGraph's own config/context mechanism
through, and it keeps nodes/answering.py's functions free of any
graph-building concern.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from ..providers.base import LLMProvider
from ..providers.factory import get_llm
from .nodes.answering import (
    accept_answer,
    escalate,
    generate_answer_text,
    question_by_id,
    route_after_select,
    route_after_validation,
    select_next_question,
)
from .nodes.retrieval import retrieve_for_question
from .stack import DocgenStack
from .state import Attempt, DocGenState
from .validation import validate_answer


def build_graph(stack: DocgenStack, llm: LLMProvider | None = None) -> CompiledStateGraph:
    llm = llm or get_llm()

    def retrieve_node(state: DocGenState) -> dict[str, Any]:
        current = state["current"]
        assert current is not None
        question = question_by_id(state["questions"], current["question_id"])
        sources = [s for s in state["sources"] if s.role in question["sources_required"]]
        chunks = retrieve_for_question(current["query"], sources, stack.retriever)
        return {"current": {**current, "chunks": chunks}}

    def generate_answer_node(state: DocGenState) -> dict[str, Any]:
        current = state["current"]
        assert current is not None
        question = question_by_id(state["questions"], current["question_id"])
        answer = generate_answer_text(
            question["text"], current["chunks"], current["attempts"], llm
        )
        return {"current": {**current, "answer": answer}}

    def validate_node(state: DocGenState) -> dict[str, Any]:
        current = state["current"]
        assert current is not None
        question = question_by_id(state["questions"], current["question_id"])
        result = validate_answer(question["text"], current["answer"], current["chunks"], llm)
        if result.valid:
            return {"current": {**current, "valid": True}}
        attempt: Attempt = {
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
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate_answer", generate_answer_node)
    builder.add_node("validate_answer", validate_node)
    builder.add_node("accept_answer", accept_answer)
    builder.add_node("escalate", escalate)

    builder.add_edge(START, "select_next_question")
    builder.add_conditional_edges(
        "select_next_question", route_after_select, {"retrieve": "retrieve", "end": END}
    )
    builder.add_edge("retrieve", "generate_answer")
    builder.add_edge("generate_answer", "validate_answer")
    builder.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {"accept": "accept_answer", "retry": "retrieve", "escalate": "escalate"},
    )
    builder.add_edge("accept_answer", "select_next_question")
    builder.add_edge("escalate", END)

    return builder.compile()
