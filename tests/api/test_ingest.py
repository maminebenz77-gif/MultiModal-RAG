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


async def test_reingesting_identical_content_is_marked_already_ingested_and_skips_work(
    client: httpx.AsyncClient,
) -> None:
    """The short-circuit this guards: chunk_id() being content-addressed
    already prevented duplicate chunks on a re-upload, but the pipeline
    still re-parsed/re-chunked/re-embedded every time -- slow, and
    indistinguishable from a genuinely new ingest from the response
    alone. A known doc_id should now skip straight to a cheap DB lookup."""
    with open(SAMPLE_DOC, "rb") as f:
        first = await client.post(
            "/ingest", files={"file": ("chunking_demo.md", f, "text/markdown")}
        )
    assert first.json()["status"] == "ingested"

    with open(SAMPLE_DOC, "rb") as f:
        second = await client.post(
            "/ingest", files={"file": ("chunking_demo.md", f, "text/markdown")}
        )
    body = second.json()
    assert body["status"] == "already_ingested"
    assert body["doc_id"] == first.json()["doc_id"]
    assert body["num_parent_chunks"] == first.json()["num_parent_chunks"]
    assert body["num_child_chunks"] == first.json()["num_child_chunks"]
    assert body["ingested_at"] == first.json()["ingested_at"]


async def test_ingest_rejects_empty_file(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/ingest", files={"file": ("empty.md", b"", "text/markdown")}
    )
    assert response.status_code == 400


async def test_renaming_and_reingesting_identical_content_does_not_duplicate_chunks(
    client: httpx.AsyncClient,
) -> None:
    """Regression test: chunk_id() used to hash in the uploaded FILENAME,
    so renaming a.md -> b.md (byte-identical content) minted a whole new
    set of chunk_ids and left the old set orphaned in the stores, even
    though the /documents list correctly showed only one document."""
    with open(SAMPLE_DOC, "rb") as f:
        response_a = await client.post("/ingest", files={"file": ("a.md", f, "text/markdown")})
    assert response_a.status_code == 200
    doc_id_a = response_a.json()["doc_id"]

    with open(SAMPLE_DOC, "rb") as f:
        response_b = await client.post("/ingest", files={"file": ("b.md", f, "text/markdown")})
    assert response_b.status_code == 200
    body_b = response_b.json()

    assert body_b["doc_id"] == doc_id_a
    assert body_b["status"] == "already_ingested"

    vector_store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    chunk_ids = vector_store.list_chunk_ids()
    matching = [cid for cid in chunk_ids if cid.startswith(doc_id_a)]
    assert len(matching) == body_b["num_parent_chunks"] + body_b["num_child_chunks"]

    documents = (await client.get("/documents")).json()["documents"]
    assert len(documents) == 1
    assert documents[0]["filename"] == "b.md"
