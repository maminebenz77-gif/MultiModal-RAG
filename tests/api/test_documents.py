from pathlib import Path

import httpx

from .conftest import MINIMAL_METADATA_JSON, ingest_sample_doc

_OTHER_SAMPLE_DOC = Path(__file__).resolve().parents[2] / "data" / "samples" / "sample.md"


async def test_no_documents_ingested_yet_returns_empty_list(client: httpx.AsyncClient) -> None:
    response = await client.get("/documents")

    assert response.status_code == 200
    assert response.json()["documents"] == []


async def test_lists_a_document_after_ingestion(client: httpx.AsyncClient) -> None:
    doc_id = await ingest_sample_doc(client)

    response = await client.get("/documents")

    documents = response.json()["documents"]
    assert len(documents) == 1
    assert documents[0]["doc_id"] == doc_id
    assert documents[0]["filename"] == "chunking_demo.md"
    assert documents[0]["metadata"]["classification"] == "public"


async def test_patch_updates_only_the_fields_sent(client: httpx.AsyncClient) -> None:
    doc_id = await ingest_sample_doc(client)

    response = await client.patch(
        f"/documents/{doc_id}", json={"classification": "c2", "tags": ["policy"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["metadata"]["classification"] == "c2"
    assert body["metadata"]["tags"] == ["policy"]
    # author was never sent -- must be untouched (still its original
    # default), not wiped by the patch.
    assert body["metadata"]["author"] is None


async def test_patch_is_visible_in_a_later_get(client: httpx.AsyncClient) -> None:
    doc_id = await ingest_sample_doc(client)
    await client.patch(f"/documents/{doc_id}", json={"private": True})

    documents = (await client.get("/documents")).json()["documents"]
    assert documents[0]["metadata"]["private"] is True


async def test_patch_with_an_empty_body_changes_nothing(client: httpx.AsyncClient) -> None:
    doc_id = await ingest_sample_doc(client)

    response = await client.patch(f"/documents/{doc_id}", json={})

    assert response.status_code == 200
    assert response.json()["metadata"]["classification"] == "public"


async def test_patch_returns_404_for_an_unknown_doc_id(client: httpx.AsyncClient) -> None:
    response = await client.patch("/documents/nonexistent", json={"private": True})

    assert response.status_code == 404


async def test_patch_does_not_require_re_embedding(client: httpx.AsyncClient) -> None:
    """The whole point of PATCH -- a metadata-only change never touches
    chunk_ids, since it's a payload patch, not a re-ingest."""
    doc_id = await ingest_sample_doc(client)
    vector_store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    chunk_ids_before = set(vector_store.list_chunk_ids())

    await client.patch(f"/documents/{doc_id}", json={"classification": "c1"})

    assert set(vector_store.list_chunk_ids()) == chunk_ids_before


async def test_patch_recomputes_acl_allow_when_owner_changes_without_private(
    client: httpx.AsyncClient,
) -> None:
    """Regression test: acl_allow (stores/*) is derived from private AND
    owner together (see DocumentMetadata.to_payload()). A patch touching
    only `owner`, with `private` already true from a prior patch, must
    still recompute acl_allow in the store -- not leave it pointing at
    the OLD owner."""
    doc_id = await ingest_sample_doc(client)
    await client.patch(f"/documents/{doc_id}", json={"private": True, "owner": "user:alice"})

    await client.patch(f"/documents/{doc_id}", json={"owner": "user:bob"})

    vector_store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    points, _ = vector_store._client.scroll(
        collection_name=vector_store._alias, limit=10, with_payload=True
    )
    for point in points:
        assert point.payload is not None
        assert point.payload["acl_allow"] == ["user:bob"]


async def test_wipe_removes_all_documents_and_their_chunks(client: httpx.AsyncClient) -> None:
    await ingest_sample_doc(client)

    response = await client.delete("/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "wiped"
    assert body["documents_deleted"] == 1
    assert body["chunks_deleted"] > 0

    assert (await client.get("/documents")).json()["documents"] == []

    vector_store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    assert vector_store.list_chunk_ids() == []


async def test_wipe_with_no_documents_is_a_harmless_no_op(client: httpx.AsyncClient) -> None:
    response = await client.delete("/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["documents_deleted"] == 0
    assert body["chunks_deleted"] == 0


async def test_delete_one_document_leaves_the_other_untouched(
    client: httpx.AsyncClient,
) -> None:
    doc_id_to_delete = await ingest_sample_doc(client)
    with open(_OTHER_SAMPLE_DOC, "rb") as f:
        other_response = await client.post(
            "/ingest",
            files={"file": ("sample.md", f, "text/markdown")},
            data={"metadata_json": MINIMAL_METADATA_JSON},
        )
    other_doc_id = other_response.json()["doc_id"]

    response = await client.delete(f"/documents/{doc_id_to_delete}")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "deleted"
    assert body["doc_id"] == doc_id_to_delete
    assert body["chunks_deleted"] > 0

    remaining_ids = [d["doc_id"] for d in (await client.get("/documents")).json()["documents"]]
    assert remaining_ids == [other_doc_id]

    vector_store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    remaining_chunk_ids = vector_store.list_chunk_ids()
    assert remaining_chunk_ids
    assert all(cid.startswith(other_doc_id) for cid in remaining_chunk_ids)


async def test_delete_document_returns_404_for_an_unknown_doc_id(
    client: httpx.AsyncClient,
) -> None:
    response = await client.delete("/documents/nonexistent")

    assert response.status_code == 404


async def test_an_old_row_without_a_classification_does_not_take_down_list_or_metrics(
    client: httpx.AsyncClient,
) -> None:
    """The real-world failure this pins: a database from before
    classification existed has rows with classification ''. Reading one
    used to raise, so GET /documents and GET /metrics both answered 500 --
    for every user, whether or not their own documents were fine."""
    import sqlite3

    good = await ingest_sample_doc(client)
    db = client.app.state.app_state.db  # type: ignore[attr-defined]
    with sqlite3.connect(db._db_path) as conn:
        conn.execute(
            "INSERT INTO documents (doc_id, filename, content_hash, num_parent_chunks, "
            "num_child_chunks, ingested_at, classification) VALUES ('old', 'old.md', 'h', 1, 1, "
            "'2026-09-06T00:00:00+00:00', '')"
        )

    listed = await client.get("/documents")
    metrics = await client.get("/metrics")

    assert listed.status_code == 200
    assert [d["doc_id"] for d in listed.json()["documents"]] == [good]
    assert metrics.status_code == 200
    assert metrics.json()["total_documents"] == 1
