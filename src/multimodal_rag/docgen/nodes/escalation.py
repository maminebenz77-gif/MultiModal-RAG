"""ask_human: the real human-in-the-loop escalation, replacing
nodes/answering.py's escalate() as the subgraph's terminal step on the
"escalate" branch -- escalate() still runs first (it marks the
question's status, visible in state WHILE this node is paused, since
interrupt() halts before this node returns anything of its own).

LangGraph re-runs a node that calls interrupt() from its own top on
resume -- everything here before the interrupt() call must therefore
be side-effect-free (it IS: just reading state and building a plain
dict to surface), and the actual answer-recording only happens AFTER
interrupt() returns, which is only true once a human has actually
answered.
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from ..state import DocGenState
from .answering import question_by_id, record_answer


def ask_human(state: DocGenState) -> dict[str, Any]:
    current = state["current"]
    assert current is not None
    question = question_by_id(state["questions"], current["question_id"])
    human_answer = interrupt(
        {
            "question": question["text"],
            "chunks": current["chunks"],
            "attempts": current["attempts"],
        }
    )
    return record_answer(state, current, human_answer, "human")
