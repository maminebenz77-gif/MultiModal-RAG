"""End-to-end proof that ScopedRetriever is actually wired into
POST /query, not just correct in isolation (see
tests/retrieval/test_scoped.py for the unit-level coverage of the
mechanism itself). Swaps identity via FastAPI's own
`dependency_overrides` -- the standard way to control auth in a test
without constructing real signed tokens, and it exercises the exact
same `Depends(get_principal)` wiring a real request would.
"""

import httpx
import pytest

from multimodal_rag.api.identity import get_principal
from multimodal_rag.identity import Principal
from multimodal_rag.providers.base import LLMProvider
from multimodal_rag.providers.schema import ToolCall, ToolResponse

from .conftest import SAMPLE_DOC


class _FakeLLM(LLMProvider):
    """Same one-search-then-answer script as test_query.py's _FakeLLM."""

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
    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", lambda: _FakeLLM())


def _as(client: httpx.AsyncClient, principal: Principal) -> None:
    client.app.dependency_overrides[get_principal] = lambda: principal  # type: ignore[attr-defined]


async def _ingest_private(client: httpx.AsyncClient, owner: str) -> str:
    with open(SAMPLE_DOC, "rb") as f:
        response = await client.post(
            "/ingest",
            files={"file": ("chunking_demo.md", f, "text/markdown")},
            data={
                "metadata_json": (
                    f'{{"classification": "public", "private": true, "owner": "{owner}"}}'
                )
            },
        )
    assert response.status_code == 200, response.text
    doc_id: str = response.json()["doc_id"]
    return doc_id


async def test_a_strangers_query_never_retrieves_another_users_private_document(
    client: httpx.AsyncClient,
) -> None:
    # /ingest isn't principal-scoped yet (that's the next phase) -- only
    # WHO can later SEARCH it is what's under test here.
    await _ingest_private(client, owner="user:alice")

    _as(client, Principal(principal_id="user:mallory", clearance="c3"))
    response = await client.post(
        "/query",
        json={"question": "How does local inference latency compare to the internal gateway?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieved_chunks"] == []
    assert body["citations"] == []


async def test_the_owners_own_query_does_retrieve_their_private_document(
    client: httpx.AsyncClient,
) -> None:
    await _ingest_private(client, owner="user:alice")

    _as(client, Principal(principal_id="user:alice", clearance="public"))
    response = await client.post(
        "/query",
        json={"question": "How does local inference latency compare to the internal gateway?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["retrieved_chunks"] != []


async def test_an_admin_principal_can_still_see_a_strangers_private_document(
    client: httpx.AsyncClient,
) -> None:
    await _ingest_private(client, owner="user:alice")

    _as(client, Principal.unrestricted())
    response = await client.post(
        "/query",
        json={"question": "How does local inference latency compare to the internal gateway?"},
    )

    assert response.status_code == 200
    assert response.json()["retrieved_chunks"] != []
