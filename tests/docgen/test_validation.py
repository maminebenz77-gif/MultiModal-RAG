import json

import pytest

from multimodal_rag.docgen.nodes.retrieval import RetrievedChunk
from multimodal_rag.docgen.validation import ValidationParseError, validate_answer
from multimodal_rag.stores.schema import SearchResult


class _FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        return self.reply


class _ExplodingLLM:
    def generate(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("the LLM should not have been called")


def _chunk(text: str, source_role: str, chunk_id: str = "c1") -> RetrievedChunk:
    return RetrievedChunk(
        chunk=SearchResult(
            chunk_id=chunk_id,
            score=1.0,
            text=text,
            source="doc.md",
            doc_id="doc-1",
            element_types=["paragraph"],
        ),
        source_role=source_role,
    )


def test_validate_answer_short_circuits_to_invalid_when_no_chunks_were_retrieved() -> None:
    result = validate_answer("What is the latency?", "It is 50ms.", [], llm=_ExplodingLLM())

    assert result.valid is False
    assert "No chunks were retrieved" in result.reason


def test_validate_answer_returns_valid_when_the_judge_says_so() -> None:
    llm = _FakeLLM(json.dumps({"valid": True, "reason": "Fully supported."}))
    chunks = [_chunk("The hosted API had 120ms average latency.", "task_docs")]

    result = validate_answer("What was the hosted API's latency?", "120ms.", chunks, llm=llm)

    assert result.valid is True
    assert result.reason == "Fully supported."
    assert len(llm.calls) == 1


def test_validate_answer_returns_invalid_with_the_judges_reason() -> None:
    llm = _FakeLLM(json.dumps({"valid": False, "reason": "The answer claims 80ms, not 120ms."}))
    chunks = [_chunk("The hosted API had 120ms average latency.", "task_docs")]

    result = validate_answer("What was the hosted API's latency?", "80ms.", chunks, llm=llm)

    assert result.valid is False
    assert result.reason == "The answer claims 80ms, not 120ms."


def test_validate_answer_labels_each_chunk_with_its_source_role_in_the_prompt() -> None:
    llm = _FakeLLM(json.dumps({"valid": True, "reason": "ok"}))
    chunks = [
        _chunk("Task content.", "task_docs", chunk_id="t1"),
        _chunk("Reference content.", "reference_kb", chunk_id="r1"),
    ]

    validate_answer("q", "a", chunks, llm=llm)

    prompt = llm.calls[0][0]["content"]
    assert "[task_docs] Task content." in prompt
    assert "[reference_kb] Reference content." in prompt


def test_validate_answer_raises_on_malformed_judge_output() -> None:
    llm = _FakeLLM("not json at all")
    chunks = [_chunk("Some evidence.", "task_docs")]

    with pytest.raises(ValidationParseError):
        validate_answer("q", "a", chunks, llm=llm)


def test_validate_answer_raises_when_the_judge_omits_a_required_field() -> None:
    llm = _FakeLLM(json.dumps({"valid": True}))  # missing "reason"
    chunks = [_chunk("Some evidence.", "task_docs")]

    with pytest.raises(ValidationParseError):
        validate_answer("q", "a", chunks, llm=llm)
