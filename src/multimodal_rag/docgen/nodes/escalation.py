"""ask_human: the real human-in-the-loop escalation, replacing
nodes/answering.py's escalate() as the subgraph's terminal step on the
"escalate" branch -- escalate() still runs first (it marks the
question's status, visible in state WHILE this node is paused, since
interrupt() halts before this node returns anything of its own).

A human gets three choices (HumanResponse.action), not just "supply
the answer": answer it directly, skip it entirely, or reformulate --
give steering guidance without writing the answer themselves or
touching the original question's own wording. interrupt()'s
response_schema is what makes this a validated, structured choice
rather than a bare string the caller has to parse by hand -- a
malformed resume value raises pydantic.ValidationError and leaves the
graph exactly where it was, so the human (or the CLI) can just retry
with a valid one.

LangGraph re-runs a node that calls interrupt() from its own top on
resume -- everything here before the interrupt() call must therefore
be side-effect-free (it IS: just reading state and building a plain
dict to surface), and the actual branching only happens AFTER
interrupt() returns, which is only true once a human has actually
responded.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.types import interrupt

from ..state import DocGenState
from .answering import question_by_id, record_answer


class HumanResponse(TypedDict):
    action: Literal["answer", "reformulate", "skip"]
    text: str
    """The final answer (action="answer"), steering guidance for the
    next automated attempt (action="reformulate"), or unused
    (action="skip")."""


def ask_human(state: DocGenState) -> dict[str, Any]:
    current = state["current"]
    assert current is not None
    question = question_by_id(state["questions"], current["question_id"])
    payload: dict[str, Any] = {
        "kind": "ask_human",
        "question": question["text"],
        "chunks": current["chunks"],
        "attempts": current["attempts"],
    }
    if current["human_guidance"]:
        # A re-escalation after a reformulate that still didn't work --
        # show the human what they already told it, so they're not
        # repeating themselves blind.
        payload["previous_guidance"] = current["human_guidance"]

    response = interrupt(payload, response_schema=HumanResponse)

    if response["action"] == "answer":
        return record_answer(state, current, response["text"], "human")

    if response["action"] == "skip":
        questions = [
            {**q, "status": "skipped"} if q["id"] == current["question_id"] else q
            for q in state["questions"]
        ]
        return {"questions": questions, "current": None}

    # "reformulate" -- a fresh attempt budget, steered by the human's
    # guidance, for the SAME question (Question.text is never rewritten).
    return {
        "current": {
            "question_id": current["question_id"],
            "query": "",
            "chunks": [],
            "answer": "",
            "attempts": [],
            "valid": None,
            "human_guidance": response["text"],
        }
    }
