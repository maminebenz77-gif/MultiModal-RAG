import pytest

from multimodal_rag.generation.agent import AgentChain
from multimodal_rag.providers.base import LLMProvider
from multimodal_rag.providers.schema import ToolCall, ToolResponse
from multimodal_rag.stores.schema import SearchResult


class FakeLLM(LLMProvider):
    """`responses` is popped one-per-round from generate_with_tools() --
    scripting the exact sequence of tool-calls/final-answer a test wants
    the model to produce. `final_text`, if set, is what plain generate()
    returns for the "max_tool_rounds exhausted" forced-final path;
    calling generate() without it set is a test bug (the loop shouldn't
    fall back to it unless rounds were actually exhausted)."""

    def __init__(self, responses: list[ToolResponse], final_text: str | None = None) -> None:
        self._responses = list(responses)
        self._final_text = final_text
        self.last_messages: list[dict] | None = None
        self.tool_choices: list[str | dict | None] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        assert self._final_text is not None, "generate() called without max_tool_rounds exhausted"
        return self._final_text

    def generate_with_tools(self, messages, tools, tool_choice=None) -> ToolResponse:
        self.last_messages = messages
        self.tool_choices.append(tool_choice)
        return self._responses.pop(0)


class FakeRetriever:
    def __init__(self, results_by_query: dict[str, list[SearchResult]]) -> None:
        self._results_by_query = results_by_query
        self.calls: list[dict] = []

    def retrieve(
        self,
        query: str,
        method,
        top_k: int,
        rerank: bool = False,
        resolve_parent_context: bool = False,
        doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "method": method,
                "top_k": top_k,
                "rerank": rerank,
                "resolve_parent_context": resolve_parent_context,
                "doc_ids": doc_ids,
            }
        )
        return self._results_by_query.get(query, [])


def _result(chunk_id: str, text: str, source: str = "doc.md") -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=1.0,
        text=text,
        source=source,
        doc_id=source,
        element_types=["title"],
    )


def _tool_call(call_id: str, query: str) -> ToolCall:
    return ToolCall(id=call_id, name="search_knowledge_base", arguments={"query": query})


def test_no_tool_call_returns_direct_answer_without_retrieving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM([ToolResponse(content="Hi there!", tool_calls=[])])
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({})
    agent = AgentChain(retriever)

    result = agent.answer("hi")

    assert result.answer == "Hi there!"
    assert result.citations == []
    assert result.needs_clarification is False
    assert retriever.calls == []


def test_first_round_refusal_falls_back_to_searching_the_original_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    question = "What was the latency?"
    fake_llm = FakeLLM(
        [
            ToolResponse(content="I don't know based on the available documents."),
            ToolResponse(content="The latency was 220ms ⟦1⟧."),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({question: [_result("a", "220ms latency")]})
    result = AgentChain(retriever).answer(question)

    assert result.answer == "The latency was 220ms ⟦1⟧."
    assert [citation.chunk_id for citation in result.citations] == ["a"]
    assert retriever.calls[0]["query"] == question


def test_first_round_refusal_with_an_explanation_still_falls_back_to_searching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test: is_refusal must recognize a refusal even when the model
    # appends a reason (e.g. "...documents. The context has P95, not P99."),
    # not just a bare exact match -- otherwise this safety net silently stops
    # firing the moment the model explains itself.
    question = "What was the latency?"
    fake_llm = FakeLLM(
        [
            ToolResponse(
                content="I don't know based on the available documents. Nothing retrieved yet."
            ),
            ToolResponse(content="The latency was 220ms ⟦1⟧."),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({question: [_result("a", "220ms latency")]})
    result = AgentChain(retriever).answer(question)

    assert result.answer == "The latency was 220ms ⟦1⟧."
    assert retriever.calls[0]["query"] == question


def test_refusal_with_an_explanation_after_evidence_is_accepted_with_the_explanation_intact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusal_with_reason = (
        "I don't know based on the available documents. The context describes a 30-day return "
        "window, but does not mention a warranty period."
    )
    fake_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "warranty")]),
            ToolResponse(content=refusal_with_reason),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"warranty": [_result("a", "30-day return window")]})
    result = AgentChain(retriever).answer("What is the warranty period?")

    assert result.answer == refusal_with_reason
    assert result.refused is True
    assert len(retriever.calls) == 1


def test_refusal_after_empty_search_forces_one_reformulated_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "latency")]),
            ToolResponse(content="I don't know based on the available documents."),
            ToolResponse(content=None, tool_calls=[_tool_call("call_2", "response time")]),
            ToolResponse(content="The latency was 220ms ⟦1⟧."),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"latency": [], "response time": [_result("a", "220ms latency")]})
    result = AgentChain(retriever).answer("What was the latency?")

    assert result.answer == "The latency was 220ms ⟦1⟧."
    assert [call["query"] for call in retriever.calls] == ["latency", "response time"]
    assert fake_llm.tool_choices == [None, None, "required", None]


def test_refusal_after_nonempty_evidence_is_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusal = "I don't know based on the available documents."
    fake_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "warranty")]),
            ToolResponse(content=refusal),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"warranty": [_result("a", "30-day return window")]})
    result = AgentChain(retriever).answer("What is the warranty period?")

    assert result.answer == refusal
    assert result.refused is True
    assert len(retriever.calls) == 1


def test_empty_reformulation_is_attempted_only_once_before_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refusal = "I don't know based on the available documents."
    fake_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "latency")]),
            ToolResponse(content=refusal),
            ToolResponse(content=None, tool_calls=[_tool_call("call_2", "response time")]),
            ToolResponse(content=refusal),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"latency": [], "response time": []})
    result = AgentChain(retriever).answer("What was the latency?")

    assert result.refused is True
    assert [call["query"] for call in retriever.calls] == ["latency", "response time"]
    assert fake_llm.tool_choices.count("required") == 1


def test_single_tool_call_then_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "latency")]),
            ToolResponse(content="The latency was 220ms ⟦1⟧.", tool_calls=[]),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"latency": [_result("a", "220ms latency")]})
    agent = AgentChain(retriever)

    result = agent.answer("What was the latency?")

    assert result.answer == "The latency was 220ms ⟦1⟧."
    assert [c.chunk_id for c in result.citations] == ["a"]
    assert len(retriever.calls) == 1


def test_two_sequential_tool_calls_produce_stable_non_overlapping_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "latency")]),
            ToolResponse(content=None, tool_calls=[_tool_call("call_2", "embedding model")]),
            ToolResponse(
                content="Latency was 220ms ⟦1⟧, embedding model was gpt-4o-mini ⟦2⟧.",
                tool_calls=[],
            ),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever(
        {
            "latency": [_result("a", "220ms latency")],
            "embedding model": [_result("b", "gpt-4o-mini embeddings")],
        }
    )
    agent = AgentChain(retriever)

    result = agent.answer("How does latency compare, and which embedding model was used?")

    assert [c.chunk_id for c in result.citations] == ["a", "b"]
    assert len(retriever.calls) == 2


def test_duplicate_chunk_across_calls_keeps_its_original_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared = _result("a", "shared chunk")
    fake_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "q1")]),
            ToolResponse(content=None, tool_calls=[_tool_call("call_2", "q2")]),
            ToolResponse(content="Answer ⟦1⟧.", tool_calls=[]),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"q1": [shared], "q2": [shared]})
    agent = AgentChain(retriever)

    result = agent.answer("a question needing the same chunk twice")

    assert [c.marker for c in result.citations] == [1]
    assert len(result.retrieved_chunks) == 1


def test_max_tool_rounds_exhausted_forces_final_answer_via_plain_generate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        ToolResponse(content=None, tool_calls=[_tool_call(f"call_{i}", "q")]) for i in range(2)
    ]
    fake_llm = FakeLLM(responses, final_text="Best answer with what I have ⟦1⟧.")
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"q": [_result("a", "text")]})
    agent = AgentChain(retriever, max_tool_rounds=2)

    result = agent.answer("a question the model keeps searching for")

    assert result.answer == "Best answer with what I have ⟦1⟧."
    assert fake_llm.last_messages is not None
    assert fake_llm.last_messages[-1]["role"] == "system"


def test_clarifying_question_response_sets_needs_clarification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM(
        [ToolResponse(content="CLARIFYING QUESTION: Do you mean X or Y?", tool_calls=[])]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({})
    agent = AgentChain(retriever)

    result = agent.answer("how does it compare?")

    assert result.needs_clarification is True
    assert result.answer == "Do you mean X or Y?"
    assert result.citations == []
    assert result.refused is False
    assert retriever.calls == []


def test_history_is_sent_as_real_chat_messages(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = FakeLLM([ToolResponse(content="An answer.", tool_calls=[])])
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({})
    agent = AgentChain(retriever)

    agent.answer("What about that?", history=[("What model was used?", "gpt-4o-mini")])

    assert fake_llm.last_messages is not None
    assert {"role": "user", "content": "What model was used?"} in fake_llm.last_messages
    assert {"role": "assistant", "content": "gpt-4o-mini"} in fake_llm.last_messages
    assert {"role": "user", "content": "What about that?"} in fake_llm.last_messages


def test_doc_ids_are_passed_through_to_every_retrieve_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "q")]),
            ToolResponse(content="Answer ⟦1⟧.", tool_calls=[]),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"q": [_result("a", "text")]})
    agent = AgentChain(retriever)

    agent.answer("a question", doc_ids=["doc-a"])

    assert retriever.calls[0]["doc_ids"] == ["doc-a"]
