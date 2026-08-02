import httpx

from .conftest import SAMPLE_DOC, ingest_sample_doc


async def test_ingest_returns_doc_id_and_chunk_counts(client: httpx.AsyncClient) -> None:
    with open(SAMPLE_DOC, "rb") as f:
        response = await client.post(
            "/ingest", files={"file": ("chunking_demo.md", f, "text/markdown")}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["filename"] == "chunking_demo.md"
    assert body["num_parent_chunks"] == 4
    assert body["num_child_chunks"] > 0
    assert len(body["doc_id"]) == 64  # sha256 hex digest


async def test_reingesting_identical_content_returns_the_same_doc_id(
    client: httpx.AsyncClient,
) -> None:
    doc_id_1 = await ingest_sample_doc(client)
    doc_id_2 = await ingest_sample_doc(client)

    assert doc_id_1 == doc_id_2


async def test_reingesting_identical_content_does_not_duplicate_documents(
    client: httpx.AsyncClient,
) -> None:
    await ingest_sample_doc(client)
    await ingest_sample_doc(client)

    response = await client.get("/documents")
    assert len(response.json()["documents"]) == 1


async def test_ingest_rejects_empty_file(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/ingest", files={"file": ("empty.md", b"", "text/markdown")}
    )
    assert response.status_code == 400
