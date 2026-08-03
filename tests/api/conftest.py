"""Shared fixture: a genuine httpx.AsyncClient talking to the FastAPI
app in-process (via ASGITransport -- no real server/port needed), with
the app's own lifespan (store bootstrapping) run manually since
ASGITransport doesn't trigger it automatically the way a real deployed
server would.

Each test gets its own temp sqlite file AND its own uniquely-named
Qdrant collection / Elasticsearch index, torn down after -- otherwise
every test run would keep ingesting into (and never cleaning up) one
shared "api_corpus" collection, the exact kind of orphaned-collection
cruft the rest of this project's store-layer tests are careful to avoid.
"""

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from multimodal_rag.api.main import create_app
from multimodal_rag.stores.factory import get_keyword_store, get_vector_store

SAMPLE_DOC = Path(__file__).resolve().parents[2] / "data" / "samples" / "chunking_demo.md"


@asynccontextmanager
async def make_client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    """Reusable outside the `client` fixture too, for tests that need to
    monkeypatch something BEFORE the app's lifespan runs (e.g. simulating
    an unconfigured reranker) -- the shared fixture's app is already
    constructed and its lifespan already entered by the time a test body
    runs, too late for that kind of setup."""
    collection_name = f"test_api_{uuid.uuid4().hex[:8]}"
    app = create_app(db_path=tmp_path / "api_state.db", collection_name=collection_name)
    try:
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
                ac.app = app  # type: ignore[attr-defined]
                yield ac
    finally:
        vector_store = get_vector_store(collection_name=collection_name)
        physical = vector_store._current_alias_target()
        if physical is not None:
            vector_store._client.delete_collection(physical)
        keyword_store = get_keyword_store(index_name=collection_name)
        keyword_store._client.indices.delete(index=collection_name, ignore_unavailable=True)


@pytest.fixture
async def client(tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    async with make_client(tmp_path) as ac:
        yield ac


async def ingest_sample_doc(client: httpx.AsyncClient) -> str:
    """Ingests the shared sample corpus doc and returns its doc_id."""
    with open(SAMPLE_DOC, "rb") as f:
        response = await client.post(
            "/ingest", files={"file": ("chunking_demo.md", f, "text/markdown")}
        )
    assert response.status_code == 200, response.text
    doc_id: str = response.json()["doc_id"]
    return doc_id
