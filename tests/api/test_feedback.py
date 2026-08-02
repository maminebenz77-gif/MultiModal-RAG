import httpx
import pytest

from multimodal_rag.providers.base import LLMProvider


class _FakeLLM(LLMProvider):
    def generate(self, messages: list[dict[str, str]]) -> str:
        return "Fixed answer ⟦1⟧."


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.generation.chain.get_llm", lambda: _FakeLLM())


async def _run_a_query(client: httpx.AsyncClient) -> str:
    response = await client.post("/query", json={"question": "anything"})
    query_id: str = response.json()["query_id"]
    return query_id


async def test_feedback_on_an_existing_query_is_recorded(client: httpx.AsyncClient) -> None:
    query_id = await _run_a_query(client)

    response = await client.post(
        "/feedback", json={"query_id": query_id, "rating": "up", "comment": "nice"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "recorded"


async def test_feedback_without_a_comment_is_optional(client: httpx.AsyncClient) -> None:
    query_id = await _run_a_query(client)

    response = await client.post("/feedback", json={"query_id": query_id, "rating": "down"})

    assert response.status_code == 200


async def test_feedback_on_an_unknown_query_id_returns_404(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/feedback", json={"query_id": "nonexistent", "rating": "up"}
    )

    assert response.status_code == 404


async def test_feedback_rejects_an_invalid_rating(client: httpx.AsyncClient) -> None:
    query_id = await _run_a_query(client)

    response = await client.post(
        "/feedback", json={"query_id": query_id, "rating": "sideways"}
    )

    assert response.status_code == 422
