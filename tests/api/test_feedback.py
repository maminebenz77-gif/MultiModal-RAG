import httpx
import pytest

from multimodal_rag.providers.base import LLMProvider
from multimodal_rag.providers.schema import ToolResponse


class _FakeLLM(LLMProvider):
    def generate(self, messages: list[dict[str, str]]) -> str:
        return "Fixed answer ⟦1⟧."

    def generate_with_tools(self, messages: list[dict[str, str]], tools) -> ToolResponse:
        return ToolResponse(content="Fixed answer ⟦1⟧.", tool_calls=[])


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: _FakeLLM())
    # Every /query call here creates a new conversation, which now also
    # triggers a title-generation call (see routers/query.py) -- without
    # this, that call would fall through to the real provider factory.
    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", lambda: _FakeLLM())


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
