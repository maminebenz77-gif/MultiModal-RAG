"""Access control across the WHOLE API, not just search (search itself is
covered by test_query_scoping.py).

Two kinds of test live here:

1. Behavior: what user B actually gets back when they touch user A's data,
   endpoint by endpoint. The rule under test is always the same -- if you
   can't see it, the answer is "not found" (indistinguishable from it not
   existing); if you can see it but don't own it, editing/deleting is a 403.

2. A route-table walk: every endpoint the app exposes must (a) sit behind
   the identity dependency and (b) be registered below. Add an endpoint
   without protecting it and these fail on their own -- nobody has to
   remember to write a test.

Identity is swapped with FastAPI's dependency_overrides, the standard way
to control auth in a test without minting real tokens; it exercises the
same Depends(get_principal) wiring a real request goes through.
"""

import json

import httpx
import pytest
from fastapi.routing import APIRoute

from multimodal_rag.api.identity import get_principal
from multimodal_rag.identity import Principal
from multimodal_rag.providers.base import LLMProvider
from multimodal_rag.providers.schema import ToolCall, ToolResponse

from .conftest import SAMPLE_DOC

_ALICE = Principal(principal_id="user:alice", clearance="c2")
_BOB = Principal(principal_id="user:bob", clearance="c2")
_ADMIN = Principal.unrestricted()

_QUESTION = "How does local inference latency compare to the internal gateway?"


class _FakeLLM(LLMProvider):
    """One search, then a fixed answer -- enough to create real
    conversations and queries without re-testing the agent."""

    def __init__(self) -> None:
        self._searched = False

    def generate(self, messages: list[dict[str, str]]) -> str:
        return "Fixed answer."

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
        return ToolResponse(content="Fixed answer.", tool_calls=[])


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: _FakeLLM())
    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", lambda: _FakeLLM())


def _as(client: httpx.AsyncClient, principal: Principal) -> None:
    client.app.dependency_overrides[get_principal] = lambda: principal  # type: ignore[attr-defined]


async def _upload(
    client: httpx.AsyncClient,
    filename: str = "notes.md",
    content: bytes | None = None,
    **metadata: object,
) -> httpx.Response:
    return await client.post(
        "/ingest",
        files={"file": (filename, content or SAMPLE_DOC.read_bytes(), "text/markdown")},
        data={"metadata_json": json.dumps({"classification": "public", **metadata})},
    )


async def _upload_as(client: httpx.AsyncClient, who: Principal, **kwargs: object) -> str:
    _as(client, who)
    response = await _upload(client, **kwargs)  # type: ignore[arg-type]
    assert response.status_code == 200, response.text
    doc_id: str = response.json()["doc_id"]
    return doc_id


async def _ask_as(client: httpx.AsyncClient, who: Principal) -> dict:
    _as(client, who)
    response = await client.post("/query", json={"question": _QUESTION})
    assert response.status_code == 200, response.text
    body: dict = response.json()
    return body


# ---------------------------------------------------------------- ingest


async def test_two_users_uploading_the_same_filename_get_two_separate_documents(
    client: httpx.AsyncClient,
) -> None:
    """The takeover this prevents: a document's identity used to be the
    filename alone, so the second person to upload "notes.md" silently
    overwrote the first person's document -- content, classification and
    owner included."""
    alice_doc = await _upload_as(client, _ALICE, filename="notes.md", private=True)
    bob_doc = await _upload_as(
        client, _BOB, filename="notes.md", content=b"# Bob\n\nbob's own notes about gateways"
    )

    assert alice_doc != bob_doc
    _as(client, _ADMIN)
    documents = {d["doc_id"]: d for d in (await client.get("/documents")).json()["documents"]}
    assert documents[alice_doc]["metadata"]["owner"] == "user:alice"
    assert documents[bob_doc]["metadata"]["owner"] == "user:bob"


async def test_the_owner_comes_from_the_caller_not_from_the_request(
    client: httpx.AsyncClient,
) -> None:
    doc_id = await _upload_as(client, _BOB, owner="user:alice")  # bob claims alice

    _as(client, _ADMIN)
    documents = {d["doc_id"]: d for d in (await client.get("/documents")).json()["documents"]}
    assert documents[doc_id]["metadata"]["owner"] == "user:bob"


async def test_uploading_a_file_someone_else_privately_holds_does_not_reveal_them(
    client: httpx.AsyncClient,
) -> None:
    """The duplicate-upload oracle: "duplicate_of" used to name whoever
    already had this exact file, whether or not you could see it."""
    await _upload_as(client, _ALICE, filename="alices-private.md", private=True)

    _as(client, _BOB)
    response = await _upload(client, filename="bobs-copy.md")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["duplicate_of"] is None


async def test_uploading_a_file_that_is_shared_and_visible_is_still_a_duplicate(
    client: httpx.AsyncClient,
) -> None:
    await _upload_as(client, _ALICE, filename="shared.md")  # private defaults to false

    _as(client, _BOB)
    response = await _upload(client, filename="bobs-copy.md")

    assert response.json()["status"] == "duplicate_content"


# ------------------------------------------------------------- documents


async def test_a_private_document_is_missing_from_someone_elses_document_list(
    client: httpx.AsyncClient,
) -> None:
    private_doc = await _upload_as(client, _ALICE, filename="private.md", private=True)
    shared_doc = await _upload_as(
        client, _ALICE, filename="shared.md", content=b"# Shared\n\nsome shared content"
    )

    _as(client, _BOB)
    listed = [d["doc_id"] for d in (await client.get("/documents")).json()["documents"]]

    assert shared_doc in listed
    assert private_doc not in listed


@pytest.mark.parametrize("method", ["PATCH", "DELETE"])
async def test_changing_a_document_you_cannot_see_is_a_404(
    client: httpx.AsyncClient, method: str
) -> None:
    doc_id = await _upload_as(client, _ALICE, private=True)

    _as(client, _BOB)
    response = await client.request(method, f"/documents/{doc_id}", json={"tags": ["x"]})
    missing = await client.request(method, "/documents/does-not-exist", json={"tags": ["x"]})

    assert response.status_code == 404
    # Same status AND same shape as a document that genuinely isn't there:
    # nothing to tell "hidden" from "absent".
    assert response.json().keys() == missing.json().keys()


@pytest.mark.parametrize("method", ["PATCH", "DELETE"])
async def test_changing_a_document_you_can_see_but_do_not_own_is_a_403(
    client: httpx.AsyncClient, method: str
) -> None:
    doc_id = await _upload_as(client, _ALICE)  # shared, so bob can see it

    _as(client, _BOB)
    response = await client.request(method, f"/documents/{doc_id}", json={"tags": ["x"]})

    assert response.status_code == 403
    _as(client, _ALICE)
    assert doc_id in [d["doc_id"] for d in (await client.get("/documents")).json()["documents"]]


async def test_a_refused_delete_leaves_the_documents_chunks_in_place(
    client: httpx.AsyncClient,
) -> None:
    """The permission check must happen BEFORE any chunk is removed --
    otherwise a 403 would still have destroyed the data."""
    doc_id = await _upload_as(client, _ALICE)
    vector_store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    before = {c for c in vector_store.list_chunk_ids() if c.startswith(doc_id)}
    assert before

    _as(client, _BOB)
    assert (await client.delete(f"/documents/{doc_id}")).status_code == 403

    assert {c for c in vector_store.list_chunk_ids() if c.startswith(doc_id)} == before


async def test_the_owner_can_edit_and_delete_their_own_document(
    client: httpx.AsyncClient,
) -> None:
    doc_id = await _upload_as(client, _ALICE, private=True)

    _as(client, _ALICE)
    assert (await client.patch(f"/documents/{doc_id}", json={"tags": ["mine"]})).status_code == 200
    assert (await client.delete(f"/documents/{doc_id}")).status_code == 200


async def test_a_non_admin_cannot_reassign_a_documents_owner(
    client: httpx.AsyncClient,
) -> None:
    doc_id = await _upload_as(client, _ALICE)

    _as(client, _ALICE)
    response = await client.patch(f"/documents/{doc_id}", json={"owner": "user:bob"})

    assert response.status_code == 403


async def test_only_an_admin_can_wipe_the_corpus(client: httpx.AsyncClient) -> None:
    await _upload_as(client, _ALICE)
    vector_store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    chunks_before = set(vector_store.list_chunk_ids())

    _as(client, _BOB)
    assert (await client.delete("/documents")).status_code == 403
    assert set(vector_store.list_chunk_ids()) == chunks_before

    _as(client, _ADMIN)
    assert (await client.delete("/documents")).status_code == 200
    assert vector_store.list_chunk_ids() == []


# ---------------------------------------------------------- conversations


async def test_someone_elses_conversation_is_not_found_by_any_route(
    client: httpx.AsyncClient,
) -> None:
    conversation_id = (await _ask_as(client, _ALICE))["conversation_id"]

    _as(client, _BOB)
    assert (await client.get("/conversations")).json()["conversations"] == []
    assert (await client.get(f"/conversations/{conversation_id}")).status_code == 404
    assert (await client.delete(f"/conversations/{conversation_id}")).status_code == 404

    _as(client, _ALICE)
    assert (await client.get(f"/conversations/{conversation_id}")).status_code == 200


async def test_you_cannot_append_a_turn_to_someone_elses_conversation(
    client: httpx.AsyncClient,
) -> None:
    conversation_id = (await _ask_as(client, _ALICE))["conversation_id"]

    _as(client, _BOB)
    response = await client.post(
        "/query", json={"question": _QUESTION, "conversation_id": conversation_id}
    )

    assert response.status_code == 404
    _as(client, _ALICE)
    messages = (await client.get(f"/conversations/{conversation_id}")).json()["messages"]
    assert len(messages) == 1


async def test_feedback_on_someone_elses_query_is_a_404(client: httpx.AsyncClient) -> None:
    query_id = (await _ask_as(client, _ALICE))["query_id"]

    _as(client, _BOB)
    response = await client.post("/feedback", json={"query_id": query_id, "rating": "up"})
    unknown = await client.post("/feedback", json={"query_id": "no-such-query", "rating": "up"})

    # Same status and same shape as a query that genuinely doesn't exist.
    assert response.status_code == unknown.status_code == 404
    assert response.json().keys() == unknown.json().keys()
    _as(client, _ALICE)
    assert (
        await client.post("/feedback", json={"query_id": query_id, "rating": "up"})
    ).status_code == 200


async def test_metrics_describe_only_the_callers_own_activity(
    client: httpx.AsyncClient,
) -> None:
    await _ask_as(client, _ALICE)

    _as(client, _BOB)
    assert (await client.get("/metrics")).json()["total_queries"] == 0
    _as(client, _ALICE)
    assert (await client.get("/metrics")).json()["total_queries"] == 1
    _as(client, _ADMIN)
    assert (await client.get("/metrics")).json()["total_queries"] == 1


# ------------------------------------------------------------ route table

# Every endpoint the app exposes, and the one deliberate exemption. A new
# route that isn't listed here fails test_every_endpoint_is_accounted_for --
# the point is to force a person to decide how it's scoped, and to add a
# test above for it.
_PROTECTED = {
    ("POST", "/ingest"),
    ("GET", "/documents"),
    ("DELETE", "/documents"),
    ("PATCH", "/documents/{doc_id}"),
    ("DELETE", "/documents/{doc_id}"),
    ("POST", "/query"),
    ("GET", "/conversations"),
    ("GET", "/conversations/{conversation_id}"),
    ("DELETE", "/conversations/{conversation_id}"),
    ("POST", "/feedback"),
    ("GET", "/metrics"),
}
_EXEMPT = {("GET", "/health")}


_WALKER_BROKEN = "route walk found (almost) nothing -- the walker itself is broken"


def _walk(routes, inherited=()):
    """Yield (APIRoute, dependencies inherited from include_router) for
    every endpoint, however this FastAPI version nests them.

    Recent FastAPI wraps each included router in an _IncludedRouter (its
    routes in `original_router`, the include_router(dependencies=...) in
    `include_context`) instead of copying routes into the app. That shape
    is internal, so this is deliberately tolerant -- and both tests below
    assert they found routes at all, so a future FastAPI that changes the
    shape again fails loudly instead of letting the walk pass on nothing.
    """
    for route in routes:
        if isinstance(route, APIRoute):
            yield route, inherited
        elif hasattr(route, "original_router"):
            context = getattr(route, "include_context", None)
            deps = tuple(getattr(context, "dependencies", None) or ())
            yield from _walk(route.original_router.routes, inherited + deps)
        elif hasattr(route, "routes"):
            yield from _walk(route.routes, inherited)


def _all_endpoints(client: httpx.AsyncClient):
    return list(_walk(client.app.routes))  # type: ignore[attr-defined]


def _depends_on(dependant, target) -> bool:
    return any(d.call is target or _depends_on(d, target) for d in dependant.dependencies)


def _is_identified(route: APIRoute, inherited) -> bool:
    return any(getattr(d, "dependency", None) is get_principal for d in inherited) or _depends_on(
        route.dependant, get_principal
    )


async def test_every_endpoint_is_accounted_for(client: httpx.AsyncClient) -> None:
    endpoints = _all_endpoints(client)
    assert len(endpoints) >= len(_PROTECTED), _WALKER_BROKEN
    exposed = {(m, r.path) for r, _ in endpoints for m in r.methods}

    unaccounted = exposed - _PROTECTED - _EXEMPT
    assert not unaccounted, (
        f"New endpoint(s) {sorted(unaccounted)}: decide how they are scoped to the caller, "
        "add tests for it in this file, then register them in _PROTECTED."
    )
    stale = (_PROTECTED | _EXEMPT) - exposed
    assert not stale, f"Registered but no longer exposed: {sorted(stale)}"


async def test_every_endpoint_except_health_sits_behind_the_identity_dependency(
    client: httpx.AsyncClient,
) -> None:
    endpoints = _all_endpoints(client)
    assert len(endpoints) >= len(_PROTECTED), _WALKER_BROKEN

    unprotected = [
        f"{sorted(r.methods)} {r.path}"
        for r, inherited in endpoints
        if r.path != "/health" and not _is_identified(r, inherited)
    ]
    assert not unprotected, f"Endpoint(s) not behind get_principal: {unprotected}"


async def test_the_identity_check_is_not_vacuously_true(client: httpx.AsyncClient) -> None:
    """The check above only means something if it CAN fail. Run the same
    predicate over every endpoint: it must flag exactly the one route
    that genuinely has no identity dependency (/health) -- so it is
    telling protected routes from unprotected ones, not answering "yes"
    to everything."""
    endpoints = _all_endpoints(client)
    assert len(endpoints) >= len(_PROTECTED), _WALKER_BROKEN

    flagged = {r.path for r, inherited in endpoints if not _is_identified(r, inherited)}

    assert flagged == {"/health"}
