from pathlib import Path

import httpx

from .conftest import ingest_sample_doc

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
            "/ingest", files={"file": ("sample.md", f, "text/markdown")}
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
