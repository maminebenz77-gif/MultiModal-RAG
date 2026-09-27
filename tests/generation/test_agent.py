import pytest

from multimodal_rag.generation.agent import AgentChain, _build_system_prompt
from multimodal_rag.generation.prompt import CONFLICT_RESOLUTION_RULE
from multimodal_rag.generation.schema import AgentDone, AgentToken, AgentToolCall
from multimodal_rag.providers.base import LLMProvider
from multimodal_rag.providers.schema import TokenChunk, ToolCall, ToolCallsChunk, ToolResponse
from multimodal_rag.retrieval.schema import RetrievalMethod
from multimodal_rag.stores.filters import SearchFilter
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


class FakeStreamingLLM(LLMProvider):
    """`rounds` is popped one-per-round from generate_with_tools_stream()
    as a (tokens, tool_calls) pair -- tokens stream first, then the tool
    call (if any) arrives as the terminal ToolCallsChunk, mirroring how a
    real OpenAI-compatible stream behaves. `final_tokens`, if set, is
    what generate_stream() yields for the "max_tool_rounds exhausted"
    forced-final path."""

    def __init__(
        self,
        rounds: list[tuple[list[str], list[ToolCall]]],
        final_tokens: list[str] | None = None,
    ) -> None:
        self._rounds = list(rounds)
        self._final_tokens = final_tokens
        self.tool_choices: list[str | dict | None] = []

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("answer_stream() should never call the non-streaming generate()")

    def generate_with_tools(self, messages, tools, tool_choice=None) -> ToolResponse:
        raise AssertionError(
            "answer_stream() should never call the non-streaming generate_with_tools()"
        )

    def generate_stream(self, messages: list[dict[str, str]]):
        assert self._final_tokens is not None, "generate_stream() called without final_tokens set"
        yield from self._final_tokens

    def generate_with_tools_stream(self, messages, tools, tool_choice=None):
        self.tool_choices.append(tool_choice)
        tokens, tool_calls = self._rounds.pop(0)
        for text in tokens:
            yield TokenChunk(text=text)
        if tool_calls:
            yield ToolCallsChunk(tool_calls=tool_calls)


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
        search_filter: SearchFilter | None = None,
    ) -> list[SearchResult]:
        self.calls.append(
            {
                "query": query,
                "method": method,
                "top_k": top_k,
                "rerank": rerank,
                "resolve_parent_context": resolve_parent_context,
                "doc_ids": doc_ids,
                "search_filter": search_filter,
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


def test_search_filter_is_passed_through_to_every_retrieve_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same guarantee as doc_ids above, for the frontend's tags/author/
    # date-range "Filters" panel: it's a fixed argument to answer(), set
    # once by the human asking the question -- never something the
    # model's own tool-call arguments could add, remove, or widen.
    fake_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "q")]),
            ToolResponse(content="Answer ⟦1⟧.", tool_calls=[]),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"q": [_result("a", "text")]})
    agent = AgentChain(retriever)
    search_filter = SearchFilter(any_of={"tags": ["runbook"]})

    agent.answer("a question", search_filter=search_filter)

    assert retriever.calls[0]["search_filter"] == search_filter


def test_agent_system_prompt_includes_the_same_conflict_resolution_rule_as_the_chain() -> None:
    """Both prompts embed the identical CONFLICT_RESOLUTION_RULE string --
    the whole point being that this rule can't drift between them the way
    the module docstring names as a real risk for the OTHER, hand-typed
    rules they share."""
    prompt = _build_system_prompt(RetrievalMethod.HYBRID_RRF, max_tool_rounds=4)
    assert CONFLICT_RESOLUTION_RULE in prompt


def test_answer_stream_yields_tool_call_then_tokens_then_a_terminal_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeStreamingLLM(
        [
            ([], [_tool_call("call_1", "latency")]),
            (["The latency was 220ms ", "⟦1⟧."], []),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"latency": [_result("a", "220ms latency")]})
    events = list(AgentChain(retriever).answer_stream("What was the latency?"))

    assert isinstance(events[-1], AgentDone)  # AgentDone is always last
    tool_calls = [e for e in events if isinstance(e, AgentToolCall)]
    tokens = [e for e in events if isinstance(e, AgentToken)]
    assert len(tool_calls) == 1
    assert tool_calls[0].round_index == 1
    assert tool_calls[0].query == "latency"
    assert [c.chunk_id for c in tool_calls[0].results] == ["a"]
    assert "".join(t.text for t in tokens) == "The latency was 220ms ⟦1⟧."
    assert all(t.round_index == 2 for t in tokens)

    done = events[-1]
    assert done.result.answer == "The latency was 220ms ⟦1⟧."
    assert [c.chunk_id for c in done.result.citations] == ["a"]


def test_answer_stream_matches_answer_for_an_equivalent_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """answer() and answer_stream() run the exact same round loop (see
    _run_rounds) -- this pins down that an equivalent script through
    either entry point produces the identical final RagAnswer, not just
    superficially similar output."""
    question = "What was the latency?"

    blocking_llm = FakeLLM(
        [
            ToolResponse(content=None, tool_calls=[_tool_call("call_1", "latency")]),
            ToolResponse(content="The latency was 220ms ⟦1⟧.", tool_calls=[]),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: blocking_llm)
    blocking_result = AgentChain(
        FakeRetriever({"latency": [_result("a", "220ms latency")]})
    ).answer(question)

    streaming_llm = FakeStreamingLLM(
        [
            ([], [_tool_call("call_1", "latency")]),
            (["The latency was 220ms ⟦1⟧."], []),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: streaming_llm)
    stream_events = list(
        AgentChain(FakeRetriever({"latency": [_result("a", "220ms latency")]})).answer_stream(
            question
        )
    )
    streamed_result = next(e for e in stream_events if isinstance(e, AgentDone)).result

    assert streamed_result == blocking_result


def test_answer_stream_forced_final_after_rounds_exhausted_streams_via_generate_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeStreamingLLM(
        [([], [_tool_call(f"call_{i}", "q")]) for i in range(2)],
        final_tokens=["Best answer with what I have ", "⟦1⟧."],
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"q": [_result("a", "text")]})
    agent = AgentChain(retriever, max_tool_rounds=2)

    events = list(agent.answer_stream("a question the model keeps searching for"))

    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.result.answer == "Best answer with what I have ⟦1⟧."
    forced_round_tokens = [e for e in events if isinstance(e, AgentToken)]
    assert "".join(t.text for t in forced_round_tokens) == "Best answer with what I have ⟦1⟧."
    # max_tool_rounds=2, so the forced-final round is round 3 -- distinct
    # from the two real tool rounds' own indices (1 and 2).
    assert {t.round_index for t in forced_round_tokens} == {3}


def test_answer_stream_narration_before_a_tool_call_is_streamed_but_excluded_from_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A round can carry both content and a tool call (see ToolResponse's
    # docstring) -- answer()'s non-streaming path already receives and
    # discards that narration when tool_calls is present. answer_stream()
    # still surfaces it live as AgentTokens (a caller may want to show
    # "thinking" text), but it must not leak into the final answer.
    fake_llm = FakeStreamingLLM(
        [
            (["Let me check that."], [_tool_call("call_1", "latency")]),
            (["The latency was 220ms ⟦1⟧."], []),
        ]
    )
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    retriever = FakeRetriever({"latency": [_result("a", "220ms latency")]})
    events = list(AgentChain(retriever).answer_stream("What was the latency?"))

    round_1_tokens = [e.text for e in events if isinstance(e, AgentToken) and e.round_index == 1]
    assert round_1_tokens == ["Let me check that."]
    done = next(e for e in events if isinstance(e, AgentDone))
    assert done.result.answer == "The latency was 220ms ⟦1⟧."


def test_answer_stream_requires_a_provider_that_supports_streaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM([ToolResponse(content="Hi there!", tool_calls=[])])
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)

    agent = AgentChain(FakeRetriever({}))

    with pytest.raises(NotImplementedError):
        list(agent.answer_stream("hi"))
