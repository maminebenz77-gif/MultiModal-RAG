from multimodal_rag.docgen.nodes.review import _format_review_summary, route_after_review
from multimodal_rag.docgen.state import Answer, DocGenState, Question


def _question(id_: str, text: str) -> Question:
    return {"id": id_, "text": text, "sources_required": ["task_docs"], "status": "answered"}


def _answer(text: str) -> Answer:
    return {"text": text, "attempts": [], "accepted_by": "validation"}


def _state(decision: str) -> DocGenState:
    return {
        "sources": [],
        "questions": [],
        "answers": {},
        "configuration": {"template": "", "format": "pptx", "confirmed": True, "max_retries": 3},
        "review": {"decision": decision, "flagged_question_ids": [], "guidance": None},  # type: ignore[typeddict-item]
        "current": None,
        "usage": {"llm_calls": 0},
        "request": {"text": "", "corrections": []},
    }


def test_format_review_summary_lists_every_answered_question() -> None:
    questions = [_question("q1", "What was the latency?"), _question("q2", "What was the cost?")]
    answers = {"q1": _answer("120ms."), "q2": _answer("$5.")}

    summary = _format_review_summary(questions, answers)

    assert "What was the latency?" in summary
    assert "120ms." in summary
    assert "What was the cost?" in summary
    assert "$5." in summary


def test_route_after_review_approves() -> None:
    assert route_after_review(_state("approved")) == "approved"


def test_route_after_review_treats_anything_else_as_edit_requested() -> None:
    assert route_after_review(_state("pending")) == "edit_requested"
    assert route_after_review(_state("edit_requested")) == "edit_requested"
