"""Builds and compiles the per-question subgraph: select a pending
question, attempt an answer (formulate a search query, retrieve, draft
an answer), validate it, and either accept it, loop back with the
rejection reason, or -- once retries run out -- escalate to a real
human via nodes/escalation.py's ask_human, pausing with
LangGraph's interrupt(). Once no questions are left pending,
harmonize_answers runs once over the full set, then goes straight to
END for now -- human_review (a later phase) replaces that edge once it
exists, the same placeholder pattern escalate()/ask_human followed
before Phase 7 landed.

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

human_review (after harmonize_answers) pauses for a final approve/edit
decision -- a third, different use of the same interrupt()/
response_schema mechanism ask_human and reformulate_for_confirmation
already use. Approving goes to END for now (export, a later phase,
replaces that edge). Requesting edits only flips the FLAGGED
questions' status back to "pending" -- they flow through the exact
same select_next_question -> attempt_answer -> validate_answer loop
everything else already went through (select_next_question seeds
their InProgress.human_guidance from the review's own feedback), and
once nothing is pending again, harmonize_answers and human_review
simply run again over the full, now-updated set -- nothing needed to
special-case "this is a second pass."

validate_node calls record_answer directly on a valid verdict rather
than routing to a separate "accept" node -- unlike ask_human (which
MUST stay separate from escalate(), since interrupt() raises instead
of returning), validate_node always returns normally, so there's no
structural reason to keep "judge" and "finalize" in two node hops.

route_at_start decides whether a run needs the configuration loop at
all: state["configuration"]["confirmed"] already being True (every
test from earlier phases sets this directly, with questions already
populated) skips interpret_request/reformulate_for_confirmation/
freeze_configuration entirely and goes straight to
select_next_question, unchanged -- a real run starts with confirmed
False and a request.text to interpret instead.

Both LLM-calling nodes track how many real calls they made in
state["usage"]["llm_calls"] (validate_answer's ValidationResult.llm_called
says whether its own zero-chunks short-circuit skipped the call) -- a
running total a caller (docgen/cli.py) can show live via graph.stream()
rather than only knowing it after the whole run finishes.

Nodes that need real dependencies (a Retriever, an LLMProvider) are
built as closures inside build_graph() -- the simplest way to hand
them in without threading LangGraph's own config/context mechanism
through, and it keeps nodes/answering.py's functions free of any
graph-building, error-handling, or usage-tracking concern.
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from ..providers.base import LLMProvider
from ..providers.factory import get_llm
from .nodes.answering import (
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
from .nodes.configuration import (
    InterpretParseError,
    build_questions,
    freeze_configuration,
    interpret_request_text,
    reformulate_for_confirmation,
    route_after_confirmation,
)
from .nodes.escalation import ask_human
from .nodes.harmonize import HarmonizeParseError, harmonize_answers_text
from .nodes.retrieval import retrieve_for_question
from .nodes.review import human_review, route_after_review
from .stack import DocgenStack
from .state import Attempt, DocGenState
from .validation import validate_answer

_TRANSIENT_RETRY = RetryPolicy(max_attempts=3)


def _route_at_start(state: DocGenState) -> Literal["answer", "configure"]:
    return "answer" if state["configuration"]["confirmed"] else "configure"


def build_graph(
    stack: DocgenStack,
    llm: LLMProvider | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    llm = llm or get_llm()

    def interpret_request_node(state: DocGenState) -> dict[str, Any]:
        available_sources = sorted({source.role for source in state["sources"]})
        try:
            interpreted = interpret_request_text(
                state["request"]["text"], state["request"]["corrections"], available_sources, llm
            )
        except InterpretParseError:
            # The call succeeded; its output just didn't parse -- still
            # a real, billable call.
            return {"questions": [], "usage": {"llm_calls": state["usage"]["llm_calls"] + 1}}
        except Exception:
            # A provider/network failure -- no response ever came back.
            return {"questions": [], "usage": {"llm_calls": state["usage"]["llm_calls"]}}
        return {
            "questions": build_questions(interpreted),
            "configuration": {
                **state["configuration"],
                "format": interpreted.format,
                "template": interpreted.template,
            },
            "usage": {"llm_calls": state["usage"]["llm_calls"] + 1},
        }

    def harmonize_node(state: DocGenState) -> dict[str, Any]:
        try:
            harmonized = harmonize_answers_text(state["questions"], state["answers"], llm)
        except HarmonizeParseError:
            # The call succeeded; its output just didn't parse -- fail
            # soft and keep the original answers, but still a real,
            # billable call.
            return {"usage": {"llm_calls": state["usage"]["llm_calls"] + 1}}
        except Exception:
            # A provider/network failure -- no response ever came back.
            return {"usage": {"llm_calls": state["usage"]["llm_calls"]}}
        answers = dict(state["answers"])
        for question_id, text in harmonized.items():
            if question_id in answers:
                answers[question_id] = {**answers[question_id], "text": text}
        return {"answers": answers, "usage": {"llm_calls": state["usage"]["llm_calls"] + 1}}

    def attempt_answer_node(state: DocGenState) -> dict[str, Any]:
        current = state["current"]
        assert current is not None
        question = question_by_id(state["questions"], current["question_id"])
        calls = 0
        try:
            query = formulate_query(
                question["text"],
                prior_qa_pairs(state),
                current["attempts"],
                current["human_guidance"],
                llm,
            )
            calls += 1
            sources = [s for s in state["sources"] if s.role in question["sources_required"]]
            chunks = retrieve_for_question(query, sources, stack.retriever)
            answer = generate_answer_text(
                question["text"], chunks, current["attempts"], current["human_guidance"], llm
            )
            calls += 1
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
                },
                "usage": {"llm_calls": state["usage"]["llm_calls"] + calls},
            }
        return {
            "current": {
                **current,
                "query": query,
                "chunks": chunks,
                "answer": answer,
                "valid": None,  # a fresh attempt, not yet judged -- clears any stale
                # False left over from the cycle that just got retried.
            },
            "usage": {"llm_calls": state["usage"]["llm_calls"] + calls},
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
            # Reaching the except block means validate_answer did NOT
            # take its own zero-chunks short-circuit (that path never
            # raises) -- a real call was attempted before this failure.
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
                },
                "usage": {"llm_calls": state["usage"]["llm_calls"] + 1},
            }
        usage = {"llm_calls": state["usage"]["llm_calls"] + (1 if result.llm_called else 0)}
        if result.valid:
            return {
                **record_answer(state, current, current["answer"], "validation"),
                "usage": usage,
            }
        attempt = {
            "query": current["query"],
            "chunks": current["chunks"],
            "answer": current["answer"],
            "reason": result.reason,
        }
        return {
            "usage": usage,
            "current": {**current, "valid": False, "attempts": [*current["attempts"], attempt]},
        }

    builder = StateGraph(DocGenState)
    builder.add_node("interpret_request", interpret_request_node, retry_policy=_TRANSIENT_RETRY)
    builder.add_node("reformulate_for_confirmation", reformulate_for_confirmation)
    builder.add_node("freeze_configuration", freeze_configuration)
    builder.add_node("select_next_question", select_next_question)
    builder.add_node("attempt_answer", attempt_answer_node, retry_policy=_TRANSIENT_RETRY)
    builder.add_node("validate_answer", validate_node, retry_policy=_TRANSIENT_RETRY)
    builder.add_node("escalate", escalate)
    builder.add_node("ask_human", ask_human)
    builder.add_node("harmonize_answers", harmonize_node, retry_policy=_TRANSIENT_RETRY)
    builder.add_node("human_review", human_review)

    builder.add_conditional_edges(
        START,
        _route_at_start,
        {"configure": "interpret_request", "answer": "select_next_question"},
    )
    builder.add_edge("interpret_request", "reformulate_for_confirmation")
    builder.add_conditional_edges(
        "reformulate_for_confirmation",
        route_after_confirmation,
        {"frozen": "freeze_configuration", "revise": "interpret_request"},
    )
    builder.add_edge("freeze_configuration", "select_next_question")
    builder.add_conditional_edges(
        "select_next_question",
        route_after_select,
        {"attempt": "attempt_answer", "end": "harmonize_answers"},
    )
    builder.add_edge("harmonize_answers", "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {"approved": END, "edit_requested": "select_next_question"},
    )
    builder.add_edge("attempt_answer", "validate_answer")
    builder.add_conditional_edges(
        "validate_answer",
        route_after_validation,
        {
            "select_next": "select_next_question",
            "retry": "attempt_answer",
            "escalate": "escalate",
        },
    )
    builder.add_edge("escalate", "ask_human")
    builder.add_conditional_edges(
        "ask_human",
        route_after_human,
        {"retry": "attempt_answer", "select_next": "select_next_question"},
    )

    return builder.compile(checkpointer=checkpointer)
