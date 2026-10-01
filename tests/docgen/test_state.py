import copy

from multimodal_rag.docgen.nodes.retrieval import RetrievedChunk
from multimodal_rag.docgen.sources import SourceSpec
from multimodal_rag.docgen.state import (
    Answer,
    Attempt,
    Configuration,
    DocGenState,
    Question,
    Review,
)
from multimodal_rag.stores.schema import SearchResult


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk=SearchResult(
            chunk_id="c1",
            score=1.0,
            text="evidence",
            source="doc.md",
            doc_id="doc-1",
            element_types=["paragraph"],
        ),
        source_role="task_docs",
    )


def _sample_state() -> DocGenState:
    question: Question = {
        "id": "q1",
        "text": "What is the latency?",
        "sources_required": ["task_docs"],
        "status": "answered",
    }
    attempt: Attempt = {
        "query": "latency benchmark",
        "chunks": [_chunk()],
        "answer": "Unsupported guess.",
        "reason": "Not grounded in the retrieved chunks.",
    }
    answer: Answer = {
        "text": "120ms on average.",
        "attempts": [attempt],
        "accepted_by": "validation",
    }
    configuration: Configuration = {
        "template": "templates/report.pptx",
        "format": "pptx",
        "confirmed": True,
    }
    review: Review = {"decision": "pending", "flagged_question_ids": []}
    return {
        "sources": [SourceSpec(role="task_docs", tag="docgen:task:demo", required=True)],
        "questions": [question],
        "answers": {"q1": answer},
        "configuration": configuration,
        "review": review,
    }


def test_docgen_state_is_a_plain_dict_at_runtime() -> None:
    state = _sample_state()

    assert isinstance(state, dict)
    assert state["questions"][0]["status"] == "answered"


def test_docgen_state_survives_a_deep_copy_unchanged() -> None:
    """Stand-in for what a checkpointer does -- copies the whole state
    between steps/processes. A field that couldn't survive this (e.g. an
    open file handle or a live connection) would be a bad fit here."""
    state = _sample_state()

    copied = copy.deepcopy(state)

    assert copied == state
    assert copied is not state
    assert copied["answers"]["q1"]["attempts"][0]["chunks"][0] == _chunk()


def test_retry_count_is_derived_from_attempts_length_not_a_separate_counter() -> None:
    state = _sample_state()

    assert len(state["answers"]["q1"]["attempts"]) == 1
