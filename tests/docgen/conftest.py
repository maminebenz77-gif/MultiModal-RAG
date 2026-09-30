"""Shared fixture: a DocgenStack pointed at a fresh, uniquely-named
Elasticsearch index and a temp sqlite file -- the same isolation
tests/api/conftest.py uses for its own store-backed tests, just built
directly via docgen.stack.build_stack() instead of through a FastAPI
app's lifespan, since docgen never goes through the API.
"""

import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from multimodal_rag.docgen.stack import DocgenStack, build_stack
from multimodal_rag.stores.factory import get_keyword_store


@pytest.fixture
def stack(tmp_path: Path) -> Iterator[DocgenStack]:
    collection_name = f"test_docgen_{uuid.uuid4().hex[:8]}"
    docgen_stack = build_stack(
        collection_name=collection_name, db_path=tmp_path / "docgen_state.db"
    )
    try:
        yield docgen_stack
    finally:
        backend = get_keyword_store(index_name=collection_name)
        physical = backend._current_alias_target()
        if physical is not None:
            backend._client.indices.delete(index=physical, ignore_unavailable=True)
