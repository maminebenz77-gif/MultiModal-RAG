"""GET /conversations and GET /conversations/{conversation_id} -- covers
what test_query.py doesn't: listing recent conversations for a picker,
reading a conversation's full turn history (with citations) back out,
and the 404 for an unknown id.
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

    def generate_with_tools(
        self, messages: list[dict[str, str]], tools, tool_choice=None
    ) -> ToolResponse:
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


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: _FakeLLM())
    # Every new conversation created here also triggers a title-generation
    # call (see routers/query.py) -- without this, that call would fall
    # through to the real, unmocked provider factory.
    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", lambda: _FakeLLM())


async def test_get_conversation_returns_the_turn_with_its_citations(
    client: httpx.AsyncClient,
) -> None:
    await ingest_sample_doc(client)

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
    citation = message["citations"][0]
    assert citation["marker"] == 1
    # The whole point: a reloaded conversation's citation carries enough
    # to show real chunk detail, not just marker/source/location.
    assert citation["text"]
    assert citation["elements"]


async def test_get_conversation_returns_404_for_an_unknown_id(client: httpx.AsyncClient) -> None:
    response = await client.get("/conversations/nonexistent")

    assert response.status_code == 404


async def test_list_conversations_includes_a_newly_created_one_with_preview(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Override the autouse fixture's title mock specifically: no title
    # provider configured -- generate_title() fails soft (see
    # generation/test_title.py), so preview stays the raw first question,
    # which is what this test is actually about; title generation itself
    # is covered separately.

    def _fail_get_llm() -> LLMProvider:
        raise RuntimeError("no title provider for this test")

    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", _fail_get_llm)

    query_response = await client.post("/query", json={"question": "a fresh question"})
    conversation_id = query_response.json()["conversation_id"]

    response = await client.get("/conversations")

    assert response.status_code == 200
    conversations = response.json()["conversations"]
    match = next(c for c in conversations if c["conversation_id"] == conversation_id)
    assert match["preview"] == "a fresh question"
    assert match["message_count"] == 1


async def test_list_conversations_is_empty_on_a_fresh_service(client: httpx.AsyncClient) -> None:
    response = await client.get("/conversations")

    assert response.status_code == 200
    assert response.json()["conversations"] == []


async def test_delete_conversation_removes_it_from_the_list(client: httpx.AsyncClient) -> None:
    query_response = await client.post("/query", json={"question": "a question to delete"})
    conversation_id = query_response.json()["conversation_id"]

    delete_response = await client.delete(f"/conversations/{conversation_id}")

    assert delete_response.status_code == 200
    assert delete_response.json() == {"status": "deleted", "conversation_id": conversation_id}
    assert (await client.get(f"/conversations/{conversation_id}")).status_code == 404
    remaining = (await client.get("/conversations")).json()["conversations"]
    assert conversation_id not in [c["conversation_id"] for c in remaining]


async def test_delete_conversation_returns_404_for_an_unknown_id(
    client: httpx.AsyncClient,
) -> None:
    response = await client.delete("/conversations/nonexistent")

    assert response.status_code == 404
