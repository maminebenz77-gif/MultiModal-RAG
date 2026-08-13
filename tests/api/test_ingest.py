import httpx
import warnings

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


async def test_ingest_surfaces_non_fatal_parser_warnings(
    client: httpx.AsyncClient,
    monkeypatch,
) -> None:
    from multimodal_rag.ingestion.schema import Element, ElementMetadata, ElementType

    def _fake_parse_document(_path, summarize_tables: bool = False):
        warnings.warn("PDF hi_res parsing failed; falling back to fast mode", RuntimeWarning)
        return [
            Element(
                type=ElementType.PARAGRAPH,
                text="hello",
                metadata=ElementMetadata(source_file="tmp", position=0),
            )
        ]

    monkeypatch.setattr("multimodal_rag.api.routers.ingest.parse_document", _fake_parse_document)

    response = await client.post(
        "/ingest", files={"file": ("warn.pdf", b"not-empty", "application/pdf")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert any("falling back to fast mode" in w for w in body["ingest_warnings"])


async def test_renaming_identical_content_is_recognized_as_a_duplicate_not_reingested(
    client: httpx.AsyncClient,
) -> None:
    """doc_id is derived from the FILENAME (schemas.IngestResponse
    explains why: it's what makes chunk-level diffing on edits possible
    with a single identity concept), which on its own would mean a
    rename looks like an unrelated new document. The global
    content_hash check (checked across ALL documents, not scoped to one
    doc_id) catches this: the same bytes under a new name are recognized
    as a duplicate and not re-ingested, without needing doc_id itself to
    survive the rename."""
    with open(SAMPLE_DOC, "rb") as f:
        response_a = await client.post("/ingest", files={"file": ("a.md", f, "text/markdown")})
    assert response_a.status_code == 200
    body_a = response_a.json()
    assert body_a["status"] == "ingested"

    with open(SAMPLE_DOC, "rb") as f:
        response_b = await client.post("/ingest", files={"file": ("b.md", f, "text/markdown")})
    assert response_b.status_code == 200
    body_b = response_b.json()

    assert body_b["status"] == "duplicate_content"
    assert body_b["duplicate_of"] == "a.md"
    assert body_b["doc_id"] == body_a["doc_id"]

    # Nothing new was actually ingested -- still just the one document,
    # under its original name.
    documents = (await client.get("/documents")).json()["documents"]
    assert len(documents) == 1
    assert documents[0]["filename"] == "a.md"


async def test_duplicate_content_check_does_not_trigger_for_genuinely_new_content(
    client: httpx.AsyncClient,
) -> None:
    with open(SAMPLE_DOC, "rb") as f:
        await client.post("/ingest", files={"file": ("a.md", f, "text/markdown")})

    response = await client.post(
        "/ingest", files={"file": ("b.md", b"completely different content", "text/markdown")}
    )

    assert response.json()["status"] == "ingested"
    documents = (await client.get("/documents")).json()["documents"]
    assert {d["filename"] for d in documents} == {"a.md", "b.md"}


async def test_editing_content_under_the_same_filename_only_touches_changed_chunks(
    client: httpx.AsyncClient,
) -> None:
    """The actual bug this guards: editing one word in a document and
    re-ingesting it under the SAME filename used to invalidate every
    chunk_id, not just the changed one -- doc_id (a content hash of the
    WHOLE file) was baked into every chunk_id, so any edit anywhere
    changed doc_id, which changed every chunk_id, even for chunks whose
    text was completely unchanged. That orphaned the entire previous
    version's chunks and re-embedded the entire document for a one-word
    change."""
    original = SAMPLE_DOC.read_bytes()
    response_1 = await client.post(
        "/ingest", files={"file": ("doc.md", original, "text/markdown")}
    )
    assert response_1.status_code == 200
    body_1 = response_1.json()

    vector_store = client.app.state.app_state.vector_store  # type: ignore[attr-defined]
    chunk_ids_1 = {
        cid for cid in vector_store.list_chunk_ids() if cid.startswith(body_1["doc_id"])
    }

    modified = original.replace(b"latency", b"latencys", 1)
    response_2 = await client.post(
        "/ingest", files={"file": ("doc.md", modified, "text/markdown")}
    )
    assert response_2.status_code == 200
    body_2 = response_2.json()

    # Same document identity (an edit, not a new document), but real
    # work happened -- this is NOT the "nothing changed" short-circuit.
    assert body_2["doc_id"] == body_1["doc_id"]
    assert body_2["status"] == "ingested"

    chunk_ids_2 = {
        cid for cid in vector_store.list_chunk_ids() if cid.startswith(body_1["doc_id"])
    }
    unchanged = chunk_ids_1 & chunk_ids_2
    removed = chunk_ids_1 - chunk_ids_2
    added = chunk_ids_2 - chunk_ids_1

    assert len(unchanged) > 0, "most of the document's chunks should be untouched"
    assert len(removed) > 0, "the old chunk(s) covering the edited text should be gone"
    assert len(added) > 0, "the new chunk(s) covering the edited text should be embedded"
    # The single strongest check: what's actually in the store now is
    # EXACTLY the unchanged chunks plus the new ones -- nothing orphaned.
    assert chunk_ids_2 == unchanged | added

    documents = (await client.get("/documents")).json()["documents"]
    assert len(documents) == 1
