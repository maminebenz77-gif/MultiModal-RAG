"""human_review: the last human checkpoint before export (a later
phase) -- shows the full set of harmonized answers and pauses,
reusing the exact same interrupt()/response_schema mechanism Phase 7
(ask_human) and Phase 8 (reformulate_for_confirmation) already built,
just for a third, different decision.

Approving clears review state and moves on. Requesting edits does NOT
touch any already-accepted answer -- it only flips the FLAGGED
questions' status back to "pending", so they flow through the exact
same per-question loop (with its own retry/escalation safety net)
everything else already went through, rather than needing a second,
parallel way to answer a question. Once those are redone,
select_next_question finding nothing left pending sends the FULL set
through harmonize_answers and human_review again, automatically --
nothing here needs to special-case "this is a second pass."
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.types import interrupt

from ..state import Answer, DocGenState, Question


class ReviewResponse(TypedDict):
    action: Literal["approve", "edit"]
    question_ids: list[str]
    """Only used when action == "edit" -- which questions to send back
    through the per-question loop."""
    text: str
    """Only used when action == "edit" -- guidance for redoing them,
    shared across every flagged question (not one note per question;
    flagging questions one at a time across separate review rounds is
    how a human gives each its own distinct guidance)."""


def _format_review_summary(questions: list[Question], answers: dict[str, Answer]) -> str:
    lines = ["Here are the harmonized answers, ready for review:"]
    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            continue
        lines.append(f"- [{question['id']}] {question['text']}")
        lines.append(f"  {answer['text']}")
    return "\n".join(lines)


def human_review(state: DocGenState) -> dict[str, Any]:
    summary = _format_review_summary(state["questions"], state["answers"])
    payload = {"kind": "human_review", "summary": summary}
    response = interrupt(payload, response_schema=ReviewResponse)
    if response["action"] == "approve":
        return {"review": {"decision": "approved", "flagged_question_ids": [], "guidance": None}}
    questions = [
        {**q, "status": "pending"} if q["id"] in response["question_ids"] else q
        for q in state["questions"]
    ]
    return {
        "questions": questions,
        "review": {
            "decision": "edit_requested",
            "flagged_question_ids": response["question_ids"],
            "guidance": response["text"],
        },
    }


def route_after_review(state: DocGenState) -> Literal["approved", "edit_requested"]:
    return "approved" if state["review"]["decision"] == "approved" else "edit_requested"
