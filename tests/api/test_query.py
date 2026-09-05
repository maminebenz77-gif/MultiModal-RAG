"""The LLM call is faked here (deterministic, free, fast) -- generation
quality itself is already covered by generation/test_chain.py's
FakeLLM-based tests and by the live demo. What's new and worth testing
at this layer is the API's OWN plumbing: request/response shape, the
doc_ids filter, and query logging -- not "can an LLM answer from
context" again.
"""

from pathlib import Path

import httpx
import pytest

from multimodal_rag.providers.base import LLMProvider, Reranker
from multimodal_rag.providers.schema import ToolCall, ToolResponse

from .conftest import ingest_sample_doc, make_client


class _FakeLLM(LLMProvider):
    """Simulates the minimal one-search agent turn: a tool call (echoing
    the latest user message as the search query) followed by a fixed
    final answer. That's enough to exercise real retrieval (and so real
    citation resolution, and real doc_ids/rerank plumbing) without
    re-testing the agent's own decomposition/looping behavior -- see
    generation/test_agent.py for that."""

    def __init__(self, response: str = "Fixed answer ⟦1⟧.") -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] | None = None
        self._searched = False

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response

    def generate_with_tools(self, messages: list[dict[str, str]], tools) -> ToolResponse:
        self.last_messages = messages
        if not self._searched:
            self._searched = True
            latest_message = messages[-1]["content"]
            return ToolResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="search_knowledge_base",
                        arguments={"query": latest_message},
                    )
                ],
            )
        return ToolResponse(content=self._response, tool_calls=[])


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: _FakeLLM())


async def test_query_returns_an_answer_with_citations(client: httpx.AsyncClient) -> None:
    await ingest_sample_doc(client)

    response = await client.post(
        "/query",
        json={
            "question": "How does local inference latency compare to the internal gateway?",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Fixed answer ⟦1⟧."
    assert body["refused"] is False
    assert len(body["citations"]) == 1
    assert body["citations"][0]["marker"] == 1
    assert "query_id" in body
    assert len(body["retrieved_chunks"]) >= 1
    assert "text" in body["retrieved_chunks"][0]


async def test_query_without_conversation_id_creates_a_new_conversation(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/query", json={"question": "anything"})

    assert response.status_code == 200
    assert response.json()["conversation_id"]


async def test_query_with_unknown_conversation_id_returns_404(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/query", json={"question": "anything", "conversation_id": "nonexistent"}
    )

    assert response.status_code == 404


async def test_query_continues_history_across_requests_via_conversation_id(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_llm = _FakeLLM("It is 220ms ⟦1⟧.")
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: first_llm)
    first_response = await client.post(
        "/query", json={"question": "What is the local inference latency?"}
    )
    assert first_response.status_code == 200
    conversation_id = first_response.json()["conversation_id"]

    second_llm = _FakeLLM("Fixed answer ⟦1⟧.")
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: second_llm)
    second_response = await client.post(
        "/query",
        json={"question": "How does it compare?", "conversation_id": conversation_id},
    )

    assert second_response.status_code == 200
    assert second_response.json()["conversation_id"] == conversation_id
    messages = second_llm.last_messages
    assert messages is not None
    assert {"role": "user", "content": "What is the local inference latency?"} in messages
    assert {"role": "assistant", "content": "It is 220ms ⟦1⟧."} in messages
    assert {"role": "user", "content": "How does it compare?"} in messages


async def test_query_response_includes_needs_clarification_field(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    clarifying_llm = _FakeLLM("CLARIFYING QUESTION: Which deployment do you mean?")
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: clarifying_llm)

    response = await client.post("/query", json={"question": "How does it compare?"})

    assert response.status_code == 200
    body = response.json()
    assert body["needs_clarification"] is True
    assert body["answer"] == "Which deployment do you mean?"
    assert body["citations"] == []


async def test_query_with_no_ingested_documents_still_returns_a_response(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post("/query", json={"question": "anything"})
    assert response.status_code == 200


async def test_query_doc_ids_filter_excludes_non_matching_documents(
    client: httpx.AsyncClient,
) -> None:
    await ingest_sample_doc(client)

    response = await client.post(
        "/query",
        json={
            "question": "How does local inference latency compare to the internal gateway?",
            "doc_ids": ["some-other-document.pdf"],
        },
    )

    assert response.status_code == 200
    # No context chunk belonged to the requested doc_ids, so context is
    # empty and the fake LLM's ⟦1⟧ marker has nothing valid to resolve to.
    assert response.json()["citations"] == []


async def test_query_defaults_to_hybrid_rrf_retrieval_method(client: httpx.AsyncClient) -> None:
    response = await client.post("/query", json={"question": "anything"})
    assert response.json()["retrieval_method"] == "hybrid_rrf"


async def test_query_accepts_an_explicit_retrieval_method(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/query", json={"question": "anything", "retrieval_method": "bm25"}
    )
    assert response.json()["retrieval_method"] == "bm25"


async def test_query_rejects_top_k_out_of_range(client: httpx.AsyncClient) -> None:
    response = await client.post("/query", json={"question": "anything", "top_k": 0})
    assert response.status_code == 422


async def test_query_with_rerank_true_succeeds_when_a_reranker_is_configured(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeReranker(Reranker):
        def rerank(self, query: str, documents: list[str]) -> list[int]:
            return list(range(len(documents)))

    monkeypatch.setattr("multimodal_rag.api.main.get_reranker", lambda settings=None: _FakeReranker())

    async with make_client(tmp_path) as client:
        await ingest_sample_doc(client)

        response = await client.post(
            "/query",
            json={
                "question": "How does local inference latency compare to the internal gateway?",
                "rerank": True,
                "top_k": 3,
            },
        )

        assert response.status_code == 200


async def test_query_with_rerank_true_returns_400_when_no_reranker_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_not_configured(settings: object = None) -> Reranker:
        raise NotImplementedError("Reranker is not configured for this profile")

    monkeypatch.setattr("multimodal_rag.api.main.get_reranker", _raise_not_configured)

    async with make_client(tmp_path) as ac:
        response = await ac.post("/query", json={"question": "anything", "rerank": True})
        assert response.status_code == 400


async def test_query_returns_503_when_llm_provider_fails(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FailingLLM(LLMProvider):
        def generate(self, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("upstream model unavailable")

        def generate_with_tools(self, messages: list[dict[str, str]], tools) -> ToolResponse:
            raise RuntimeError("upstream model unavailable")

    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: _FailingLLM())

    response = await client.post("/query", json={"question": "anything"})
    assert response.status_code == 503
    assert "Query generation failed" in response.json()["detail"]


async def test_query_runtime_overrides_use_llm_provider_from_request(
    client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await ingest_sample_doc(client)

    class _OverrideLLM(LLMProvider):
        def generate(self, messages: list[dict[str, str]]) -> str:
            return "Override answer ⟦1⟧."

        def generate_with_tools(self, messages: list[dict[str, str]], tools) -> ToolResponse:
            return ToolResponse(content="Override answer ⟦1⟧.", tool_calls=[])

    monkeypatch.setattr(
        "multimodal_rag.api.routers.query.llm_from_override",
        lambda **_: _OverrideLLM(),
    )

    response = await client.post(
        "/query",
        json={
            "question": "How does local inference latency compare to the internal gateway?",
            "runtime_overrides": {
                "llm": {
                    "provider": "litellm",
                    "model": "gpt-4o-mini",
                    "base_url": "http://localhost:1234",
                    "api_key": "x",
                }
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Override answer ⟦1⟧."
