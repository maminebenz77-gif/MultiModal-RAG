"""GET /conversations/{conversation_id} -- covers what test_query.py
doesn't: reading a conversation's full turn history (with citations)
back out, and the 404 for an unknown id.
"""

import httpx
import pytest

from multimodal_rag.providers.base import LLMProvider
from multimodal_rag.providers.schema import ToolCall, ToolResponse

from .conftest import ingest_sample_doc


class _FakeLLM(LLMProvider):
    """Same one-search-then-answer shape as test_query.py's fake -- see
    that file for why."""

    def __init__(self, response: str = "Fixed answer ⟦1⟧.") -> None:
        self._response = response
        self._searched = False

    def generate(self, messages: list[dict[str, str]]) -> str:
        return self._response

    def generate_with_tools(self, messages: list[dict[str, str]], tools) -> ToolResponse:
        if not self._searched:
            self._searched = True
            return ToolResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="search_knowledge_base",
                        arguments={"query": messages[-1]["content"]},
                    )
                ],
            )
        return ToolResponse(content=self._response, tool_calls=[])


async def test_get_conversation_returns_the_turn_with_its_citations(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await ingest_sample_doc(client)
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: _FakeLLM())

    query_response = await client.post(
        "/query",
        json={"question": "How does local inference latency compare to the internal gateway?"},
    )
    conversation_id = query_response.json()["conversation_id"]

    response = await client.get(f"/conversations/{conversation_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["conversation_id"] == conversation_id
    assert len(body["messages"]) == 1
    message = body["messages"][0]
    assert message["question"] == (
        "How does local inference latency compare to the internal gateway?"
    )
    assert message["answer"] == "Fixed answer ⟦1⟧."
    assert message["refused"] is False
    assert message["needs_clarification"] is False
    assert len(message["citations"]) == 1
    assert message["citations"][0]["marker"] == 1


async def test_get_conversation_returns_404_for_an_unknown_id(client: httpx.AsyncClient) -> None:
    response = await client.get("/conversations/nonexistent")

    assert response.status_code == 404
