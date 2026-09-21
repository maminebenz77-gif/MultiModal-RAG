"""Version lineage through the real API: replacing a document, and what
search does about it. Identity is swapped with dependency_overrides, as in
test_scoping.py.
"""

import json

import httpx
import pytest

from multimodal_rag.api.identity import get_principal
from multimodal_rag.identity import Principal
from multimodal_rag.providers.base import LLMProvider
from multimodal_rag.providers.schema import ToolCall, ToolResponse

from .conftest import SAMPLE_DOC

_ALICE = Principal(principal_id="user:alice", clearance="c2")
_BOB = Principal(principal_id="user:bob", clearance="c2")
_ADMIN = Principal.unrestricted()


def _as(client: httpx.AsyncClient, principal: Principal) -> None:
    client.app.dependency_overrides[get_principal] = lambda: principal  # type: ignore[attr-defined]


async def _upload(
    client: httpx.AsyncClient,
    who: Principal,
    filename: str,
    content: bytes | None = None,
    supersedes: str | None = None,
    **metadata: object,
) -> httpx.Response:
    _as(client, who)
    data = {"metadata_json": json.dumps({"classification": "public", **metadata})}
    if supersedes is not None:
        data["supersedes_doc_id"] = supersedes
    return await client.post(
        "/ingest",
        files={"file": (filename, content or SAMPLE_DOC.read_bytes(), "text/markdown")},
        data=data,
    )


async def _document(client: httpx.AsyncClient, who: Principal, doc_id: str) -> dict:
    _as(client, who)
    documents = (await client.get("/documents")).json()["documents"]
    found = [d for d in documents if d["doc_id"] == doc_id]
    assert found, f"{doc_id} is not visible to {who.principal_id}"
    document: dict = found[0]
    return document


async def test_a_caller_cannot_forge_lifecycle_fields_in_the_upload(
    client: httpx.AsyncClient,
) -> None:
    """status/version/family are decided by the server. A caller who could
    say version=9 or status=superseded could forge their place in a
    document's history."""
    response = await _upload(
        client,
        _ALICE,
        "a.md",
        status="superseded",
        version=9,
        doc_family_id="someone-elses-family",
        effective_to="2020-01-01",
    )

    assert response.status_code == 200
    metadata = response.json()["metadata"]
    assert metadata["status"] == "current"
    assert metadata["version"] == 1
    assert metadata["doc_family_id"] == response.json()["doc_id"]
    assert metadata["effective_to"] is None


# --------------------------------------------------------- helpers


def _chunk_ids(client: httpx.AsyncClient, doc_id: str) -> set[str]:
    store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    return {c for c in store.list_chunk_ids() if c.startswith(doc_id)}


def _store_statuses(client: httpx.AsyncClient, doc_id: str) -> dict[str, set[str]]:
    """The `status` each search store holds for a document's chunks."""
    state = client.app.state.app_state  # type: ignore[attr-defined]
    vector, keyword = state.vector_store, state.keyword_store
    points, _ = vector._client.scroll(collection_name=vector._alias, limit=500, with_payload=True)
    hits = keyword._client.search(
        index=keyword._index_name, query={"term": {"doc_id": doc_id}}, size=500
    )["hits"]["hits"]
    return {
        "qdrant": {p.payload["status"] for p in points if p.payload["doc_id"] == doc_id},
        "elasticsearch": {h["_source"]["status"] for h in hits},
    }


# --------------------------------------------------------- replacing


async def test_replacing_a_document_makes_the_new_one_the_next_version_and_retires_the_old(
    client: httpx.AsyncClient,
) -> None:
    old = (await _upload(client, _ALICE, "policy-2025.md")).json()
    new = (
        await _upload(
            client,
            _ALICE,
            "policy-2026.md",
            b"# Policy 2026\n\nThe limit is 20.",
            supersedes=old["doc_id"],
        )
    ).json()

    assert new["metadata"]["version"] == 2
    assert new["metadata"]["status"] == "current"
    assert new["metadata"]["doc_family_id"] == old["doc_id"]  # the FIRST version's id

    retired = await _document(client, _ALICE, old["doc_id"])
    assert retired["metadata"]["status"] == "superseded"
    assert retired["metadata"]["effective_to"] is not None
    assert retired["metadata"]["doc_family_id"] == old["doc_id"]


async def test_retiring_the_old_version_re_embeds_nothing_and_updates_both_stores(
    client: httpx.AsyncClient,
) -> None:
    old = (await _upload(client, _ALICE, "policy-2025.md")).json()
    chunks_before = _chunk_ids(client, old["doc_id"])
    assert chunks_before
    assert _store_statuses(client, old["doc_id"]) == {
        "qdrant": {"current"},
        "elasticsearch": {"current"},
    }

    new = (
        await _upload(
            client, _ALICE, "policy-2026.md", b"# Policy 2026\n\nnew text", supersedes=old["doc_id"]
        )
    ).json()

    assert _chunk_ids(client, old["doc_id"]) == chunks_before  # same chunks, none re-made
    assert _store_statuses(client, old["doc_id"]) == {
        "qdrant": {"superseded"},
        "elasticsearch": {"superseded"},
    }
    assert _store_statuses(client, new["doc_id"]) == {
        "qdrant": {"current"},
        "elasticsearch": {"current"},
    }


async def test_a_re_upload_of_the_same_filename_edits_in_place_and_keeps_its_version(
    client: httpx.AsyncClient,
) -> None:
    old = (await _upload(client, _ALICE, "a.md")).json()
    v2 = (
        await _upload(client, _ALICE, "b.md", b"# B\n\nfirst", supersedes=old["doc_id"])
    ).json()
    assert v2["metadata"]["version"] == 2

    edited = (await _upload(client, _ALICE, "b.md", b"# B\n\nedited text")).json()

    assert edited["doc_id"] == v2["doc_id"]
    assert edited["metadata"]["version"] == 2  # an edit is not a new version
    assert edited["metadata"]["doc_family_id"] == old["doc_id"]
    assert edited["metadata"]["status"] == "current"


# --------------------------------------------------------- refusals


async def test_you_cannot_replace_a_document_you_cannot_see(client: httpx.AsyncClient) -> None:
    hidden = (await _upload(client, _ALICE, "private.md", private=True)).json()

    response = await _upload(
        client, _BOB, "bobs.md", b"# Bob\n\nbob text", supersedes=hidden["doc_id"]
    )

    assert response.status_code == 404
    _as(client, _ADMIN)
    documents = (await client.get("/documents")).json()["documents"]
    # The refusal happened BEFORE anything was ingested: Bob's file isn't
    # stored, and Alice's document is untouched.
    assert [d["filename"] for d in documents] == ["private.md"]
    assert documents[0]["metadata"]["status"] == "current"


async def test_you_cannot_replace_a_document_you_can_see_but_do_not_own(
    client: httpx.AsyncClient,
) -> None:
    """Replacing hides the old document from search, so it needs the same
    right as deleting it -- otherwise anyone could hide anyone's document."""
    shared = (await _upload(client, _ALICE, "shared.md")).json()

    response = await _upload(
        client, _BOB, "bobs.md", b"# Bob\n\nbob text", supersedes=shared["doc_id"]
    )

    assert response.status_code == 403
    assert (await _document(client, _ALICE, shared["doc_id"]))["metadata"]["status"] == "current"
    _as(client, _ADMIN)
    assert [d["filename"] for d in (await client.get("/documents")).json()["documents"]] == [
        "shared.md"
    ]


async def test_a_document_cannot_replace_itself(client: httpx.AsyncClient) -> None:
    own = (await _upload(client, _ALICE, "a.md")).json()

    response = await _upload(client, _ALICE, "a.md", b"# A\n\nedit", supersedes=own["doc_id"])

    assert response.status_code == 422


async def test_a_replacement_that_fails_to_ingest_leaves_the_old_version_current(
    client: httpx.AsyncClient,
) -> None:
    """The ordering guarantee: the old version is retired only AFTER the new
    one is fully stored. If the ingest blows up half way, hiding the old
    one would leave nothing at all."""
    old = (await _upload(client, _ALICE, "policy-2025.md")).json()

    class _BrokenEmbedder:
        def embed(self, texts: list[str]) -> list:
            raise RuntimeError("embedding service is down")

    client.app.state.app_state.embedder = _BrokenEmbedder()  # type: ignore[attr-defined]
    with pytest.raises(RuntimeError):
        await _upload(
            client, _ALICE, "policy-2026.md", b"# Policy 2026\n\nnew", supersedes=old["doc_id"]
        )

    assert (await _document(client, _ALICE, old["doc_id"]))["metadata"]["status"] == "current"
    assert _store_statuses(client, old["doc_id"]) == {
        "qdrant": {"current"},
        "elasticsearch": {"current"},
    }


async def test_a_replace_request_on_a_duplicate_upload_is_reported_and_ignored(
    client: httpx.AsyncClient,
) -> None:
    other = (await _upload(client, _ALICE, "other.md", b"# Other\n\nunrelated")).json()
    await _upload(client, _ALICE, "a.md")  # the sample content

    response = await _upload(client, _ALICE, "b.md", supersedes=other["doc_id"])

    assert response.json()["status"] == "duplicate_content"
    assert any("ignored" in w for w in response.json()["ingest_warnings"])
    assert (await _document(client, _ALICE, other["doc_id"]))["metadata"]["status"] == "current"


async def test_an_empty_supersedes_field_means_no_replacement(
    client: httpx.AsyncClient,
) -> None:
    _as(client, _ALICE)
    response = await client.post(
        "/ingest",
        files={"file": ("a.md", SAMPLE_DOC.read_bytes(), "text/markdown")},
        data={"metadata_json": '{"classification": "public"}', "supersedes_doc_id": ""},
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["version"] == 1


# --------------------------------------------------------- undoing / editing


async def test_undoing_a_replacement_restores_the_old_version_everywhere(
    client: httpx.AsyncClient,
) -> None:
    old = (await _upload(client, _ALICE, "policy-2025.md")).json()
    await _upload(
        client, _ALICE, "policy-2026.md", b"# Policy 2026\n\nnew", supersedes=old["doc_id"]
    )
    _as(client, _ALICE)

    response = await client.patch(f"/documents/{old['doc_id']}", json={"status": "current"})

    assert response.status_code == 200
    assert response.json()["metadata"]["status"] == "current"
    assert response.json()["metadata"]["effective_to"] is None
    assert _store_statuses(client, old["doc_id"]) == {
        "qdrant": {"current"},
        "elasticsearch": {"current"},
    }


async def test_editing_a_retired_documents_tags_does_not_bring_it_back_in_the_stores(
    client: httpx.AsyncClient,
) -> None:
    """Regression guard: PATCH rebuilt the metadata from a hand-written
    field list, which would silently drop `status` and reset a superseded
    document to "current" in the stores on any unrelated edit."""
    old = (await _upload(client, _ALICE, "policy-2025.md")).json()
    await _upload(
        client, _ALICE, "policy-2026.md", b"# Policy 2026\n\nnew", supersedes=old["doc_id"]
    )
    _as(client, _ALICE)

    await client.patch(f"/documents/{old['doc_id']}", json={"tags": ["archive"]})

    assert _store_statuses(client, old["doc_id"]) == {
        "qdrant": {"superseded"},
        "elasticsearch": {"superseded"},
    }


async def test_an_owner_cannot_forge_version_or_family_through_patch(
    client: httpx.AsyncClient,
) -> None:
    doc = (await _upload(client, _ALICE, "a.md")).json()
    _as(client, _ALICE)

    response = await client.patch(
        f"/documents/{doc['doc_id']}", json={"version": 9, "doc_family_id": "forged"}
    )

    assert response.status_code == 200
    assert response.json()["metadata"]["version"] == 1
    assert response.json()["metadata"]["doc_family_id"] == doc["doc_id"]


# --------------------------------------------------------- the contradiction


class _FakeLLM(LLMProvider):
    """One search, then a fixed answer -- enough to run real retrieval."""

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


_V1 = b"# Request limits\n\nThe request limit is 10 per minute."
_V2 = b"# Request limits\n\nThe request limit is 20 per minute."
_QUESTION = "What is the request limit per minute?"


async def _sources(client: httpx.AsyncClient, who: Principal, **extra: object) -> set[str]:
    _as(client, who)
    response = await client.post("/query", json={"question": _QUESTION, **extra})
    assert response.status_code == 200, response.text
    return {c["doc_id"] for c in response.json()["retrieved_chunks"]}


async def test_after_a_replacement_answers_come_from_the_new_version_only(
    client: httpx.AsyncClient,
) -> None:
    """The scenario this whole phase exists for: two versions saying
    opposite things, and only the current one may reach the model."""
    v1 = (await _upload(client, _ALICE, "limits-v1.md", _V1)).json()["doc_id"]

    # Precondition, so the assertions below can't pass by finding nothing
    # at all: before the replacement, the old version IS what search returns.
    assert await _sources(client, _ALICE) == {v1}

    v2 = (await _upload(client, _ALICE, "limits-v2.md", _V2, supersedes=v1)).json()["doc_id"]

    assert await _sources(client, _ALICE) == {v2}
    # Someone else asking gets the same current version...
    assert await _sources(client, _BOB) == {v2}
    # ...and history is still there when explicitly requested.
    assert await _sources(client, _ALICE, include_superseded=True) == {v1, v2}


async def test_undoing_a_replacement_puts_the_old_version_back_in_answers(
    client: httpx.AsyncClient,
) -> None:
    v1 = (await _upload(client, _ALICE, "limits-v1.md", _V1)).json()["doc_id"]
    v2 = (await _upload(client, _ALICE, "limits-v2.md", _V2, supersedes=v1)).json()["doc_id"]
    assert await _sources(client, _ALICE) == {v2}

    _as(client, _ALICE)
    await client.patch(f"/documents/{v1}", json={"status": "current"})

    # Both are current again -- nothing stops that (deciding between them
    # is the next phase's job), but the undo must really have taken effect.
    assert await _sources(client, _ALICE) == {v1, v2}
