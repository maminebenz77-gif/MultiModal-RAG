import httpx
import pytest

from multimodal_rag.providers.base import LLMProvider

from .conftest import ingest_sample_doc


class _FakeLLM(LLMProvider):
    def __init__(self, response: str = "Fixed answer ⟦1⟧.") -> None:
        self._response = response

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self._response


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: _FakeLLM())


async def test_metrics_on_a_fresh_service_are_all_zero(client: httpx.AsyncClient) -> None:
    response = await client.get("/metrics")

    body = response.json()
    assert body == {
        "total_documents": 0,
        "total_chunks": 0,
        "total_queries": 0,
        "refusal_rate": 0.0,
        "feedback_up": 0,
        "feedback_down": 0,
    }


async def test_metrics_reflect_ingestion_queries_and_feedback(client: httpx.AsyncClient) -> None:
    await ingest_sample_doc(client)

    query_response = await client.post("/query", json={"question": "anything"})
    query_id = query_response.json()["query_id"]
    await client.post("/feedback", json={"query_id": query_id, "rating": "up"})

    response = await client.get("/metrics")

    body = response.json()
    assert body["total_documents"] == 1
    assert body["total_chunks"] > 0
    assert body["total_queries"] == 1
    assert body["feedback_up"] == 1
    assert body["feedback_down"] == 0


async def test_refusal_rate_reflects_refused_queries(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from multimodal_rag.generation.prompt import REFUSAL_TEXT

    monkeypatch.setattr(
        "multimodal_rag.generation.chain.get_llm", lambda: _FakeLLM(REFUSAL_TEXT)
    )

    await client.post("/query", json={"question": "anything"})

    response = await client.get("/metrics")
    assert response.json()["refusal_rate"] == 1.0
