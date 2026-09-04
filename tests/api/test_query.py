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

from .conftest import ingest_sample_doc, make_client


class _FakeLLM(LLMProvider):
    def __init__(self, response: str = "Fixed answer ⟦1⟧.") -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] | None = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: _FakeLLM())


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


async def test_query_passes_history_to_the_rewrite_step(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await ingest_sample_doc(client)

    rewrite_llm = _FakeLLM("How does local inference latency compare to the internal gateway?")
    monkeypatch.setattr("multimodal_rag.generation.rewrite.get_llm", lambda: rewrite_llm)

    response = await client.post(
        "/query",
        json={
            "question": "How does it compare?",
            "history": [
                {
                    "question": "What is the local inference latency?",
                    "answer": "It is 220ms.",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert rewrite_llm.last_messages is not None
    assert "What is the local inference latency?" in rewrite_llm.last_messages[1]["content"]
    assert "It is 220ms." in rewrite_llm.last_messages[1]["content"]


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

    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: _FailingLLM())

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
