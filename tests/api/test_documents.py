import httpx

from .conftest import ingest_sample_doc


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
