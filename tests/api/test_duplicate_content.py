"""Non-regression: the SAME BYTES arriving in every way they can.

The duplicate-content check (a hash of the file's bytes, see
Database.get_document_by_content_hash) exists so the same file isn't
embedded twice under two names. It interacts with everything added for
multi-user access: per-uploader document ids, visibility-scoped lookups,
private documents. Each test here is one way identical content can show
up, and each must end in a clean, sensible result -- never a crash,
never a leak of someone else's document, never stale or orphaned data.
"""

import json

import httpx

from multimodal_rag.api.identity import get_principal
from multimodal_rag.identity import Principal

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
    **metadata: object,
) -> httpx.Response:
    _as(client, who)
    return await client.post(
        "/ingest",
        files={"file": (filename, content or SAMPLE_DOC.read_bytes(), "text/markdown")},
        data={"metadata_json": json.dumps({"classification": "public", **metadata})},
    )


def _chunk_ids(client: httpx.AsyncClient, doc_id: str) -> set[str]:
    store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    return {c for c in store.list_chunk_ids() if c.startswith(doc_id)}


async def _all_documents(client: httpx.AsyncClient) -> list[dict]:
    _as(client, _ADMIN)
    documents: list[dict] = (await client.get("/documents")).json()["documents"]
    return documents


async def test_one_person_uploading_the_same_bytes_under_a_new_name_is_a_duplicate(
    client: httpx.AsyncClient,
) -> None:
    first = await _upload(client, _ALICE, "a.md")
    second = await _upload(client, _ALICE, "b.md")

    assert second.status_code == 200
    body = second.json()
    assert body["status"] == "duplicate_content"
    assert body["duplicate_of"] == "a.md"
    assert body["doc_id"] == first.json()["doc_id"]
    # Nothing new was stored: still one document, one set of chunks.
    documents = await _all_documents(client)
    assert [d["filename"] for d in documents] == ["a.md"]


async def test_two_people_uploading_identical_bytes_both_succeed_when_the_first_is_private(
    client: httpx.AsyncClient,
) -> None:
    """Bob must neither crash nor learn Alice has this file: he simply gets
    his own copy. Everything that could collide -- document id, chunk ids,
    store points -- has to stay separate."""
    alice = await _upload(client, _ALICE, "x.md", private=True)
    bob = await _upload(client, _BOB, "y.md")

    assert alice.status_code == bob.status_code == 200
    assert bob.json()["status"] == "ingested"
    assert bob.json()["duplicate_of"] is None
    alice_doc, bob_doc = alice.json()["doc_id"], bob.json()["doc_id"]
    assert alice_doc != bob_doc

    alice_chunks, bob_chunks = _chunk_ids(client, alice_doc), _chunk_ids(client, bob_doc)
    assert alice_chunks and bob_chunks
    assert alice_chunks.isdisjoint(bob_chunks)
    assert len(await _all_documents(client)) == 2


async def test_two_people_uploading_identical_bytes_under_the_same_filename_stay_separate(
    client: httpx.AsyncClient,
) -> None:
    """The original failure mode: with a filename-only document id, Bob's
    upload found ALICE's row, saw the same content hash, and answered
    "already ingested" with Alice's document and metadata."""
    alice = await _upload(client, _ALICE, "notes.md", private=True)
    bob = await _upload(client, _BOB, "notes.md")

    assert bob.status_code == 200
    body = bob.json()
    assert body["status"] == "ingested"
    assert body["doc_id"] != alice.json()["doc_id"]
    assert body["metadata"]["owner"] == "user:bob"


async def test_deleting_one_copy_leaves_the_other_persons_identical_copy_intact(
    client: httpx.AsyncClient,
) -> None:
    alice = await _upload(client, _ALICE, "x.md", private=True)
    bob = await _upload(client, _BOB, "y.md")
    alice_doc, bob_doc = alice.json()["doc_id"], bob.json()["doc_id"]
    alice_chunks = _chunk_ids(client, alice_doc)

    _as(client, _BOB)
    assert (await client.delete(f"/documents/{bob_doc}")).status_code == 200

    assert _chunk_ids(client, bob_doc) == set()
    assert _chunk_ids(client, alice_doc) == alice_chunks


async def test_a_duplicate_of_a_shared_document_is_reported_and_stores_nothing(
    client: httpx.AsyncClient,
) -> None:
    alice = await _upload(client, _ALICE, "shared.md")
    bob = await _upload(client, _BOB, "bobs-copy.md")

    assert bob.json()["status"] == "duplicate_content"
    assert bob.json()["duplicate_of"] == "shared.md"
    assert bob.json()["doc_id"] == alice.json()["doc_id"]
    assert [d["filename"] for d in await _all_documents(client)] == ["shared.md"]


async def test_a_duplicate_does_not_dangle_when_the_original_is_later_deleted(
    client: httpx.AsyncClient,
) -> None:
    """Bob was told his upload duplicated Alice's shared document. If Alice
    deletes it, Bob uploading again must simply ingest -- nothing still
    pointing at the deleted document."""
    alice = await _upload(client, _ALICE, "shared.md")
    assert (await _upload(client, _BOB, "bobs-copy.md")).json()["status"] == "duplicate_content"

    _as(client, _ALICE)
    assert (await client.delete(f"/documents/{alice.json()['doc_id']}")).status_code == 200

    again = await _upload(client, _BOB, "bobs-copy.md")
    assert again.status_code == 200
    assert again.json()["status"] == "ingested"
    assert _chunk_ids(client, again.json()["doc_id"])


async def test_reuploading_a_document_you_own_but_can_no_longer_see_replaces_it_cleanly(
    client: httpx.AsyncClient,
) -> None:
    """Regression for a bug scoping introduced: an owner can raise their own
    document's classification above their own clearance, after which they
    can't SEE it. Re-uploading an edited version then looked like a brand
    new document, so the diff never ran and the old version's chunks were
    left behind in the stores -- still searchable (34 chunks became 39)."""
    alice = Principal(principal_id="user:alice", clearance="c1")
    original = SAMPLE_DOC.read_bytes()
    edited = original.replace(b"latency", b"latencys", 1)
    assert edited != original

    # The control goes in FIRST and PRIVATE. First, so Alice's later upload
    # can't be swallowed as a duplicate of it; private, so Alice can't see
    # it anyway. (An earlier version of this test ingested the control
    # afterwards, got "duplicate_content" pointing at Alice's own document,
    # and compared her chunks with themselves -- passing while proving
    # nothing.)
    control = await _upload(client, _BOB, "control.md", edited, private=True)
    control_doc = control.json()["doc_id"]
    assert control.json()["status"] == "ingested"

    first = await _upload(client, alice, "notes.md", original, classification="c1")
    doc_id = first.json()["doc_id"]
    assert doc_id != control_doc

    _as(client, alice)
    assert (
        await client.patch(f"/documents/{doc_id}", json={"classification": "c3"})
    ).status_code == 200
    # Precondition: she really can't see it any more.
    assert doc_id not in [d["doc_id"] for d in (await client.get("/documents")).json()["documents"]]

    second = await _upload(client, alice, "notes.md", edited, classification="c1")
    assert second.status_code == 200
    assert second.json()["status"] == "ingested"
    assert second.json()["doc_id"] == doc_id

    def body_hashes(document_id: str) -> set[str]:
        return {c.split("::", 1)[1] for c in _chunk_ids(client, document_id)}

    clean, actual = body_hashes(control_doc), body_hashes(doc_id)
    assert clean and actual  # neither side may be empty
    assert actual == clean, (
        f"{len(actual - clean)} stale chunk(s) from the previous version were left behind"
    )
