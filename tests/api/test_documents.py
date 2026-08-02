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
