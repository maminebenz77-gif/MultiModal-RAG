"""Standalone construction of the store/embedder/indexer/db bundle
docgen needs -- the same pieces api/dependencies.py's AppState bundles,
but built here directly rather than by FastAPI's lifespan, since docgen
has no running app: it's a script (see docgen/cli.py), not a request
handler.

Defaults to the exact same collection name and sqlite path the live API
uses (config.COLLECTION_NAME / DEFAULT_DB_PATH) so a real docgen run
reads and writes the SAME corpus AgentChain searches, not a separate
one -- the whole point of ingestion being "into the existing vector
database". Tests override both to a fresh, uniquely-named index and a
temp file instead (see tests/docgen/conftest.py), the same isolation
tests/api/conftest.py already relies on for its own store-backed tests.

These two constants live in config.py, not api/main.py, specifically so
this import doesn't have to reach into api/ at all -- api/ is the outer
layer that imports from the rest of this package, never the reverse
(see identity.py's own docstring for the same rule). api/main.py itself
depends on docgen/ now too (Phase 12's new router), so the old
docgen -> api.main direction would have been a real circular import,
not just a style violation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..api.db import Database
from ..config import COLLECTION_NAME, DEFAULT_DB_PATH, get_settings
from ..providers.base import EmbeddingProvider
from ..providers.factory import get_embedder
from ..retrieval.retriever import Retriever
from ..stores.base import KeywordStore, VectorStore
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer


@dataclass
class DocgenStack:
    vector_store: VectorStore
    keyword_store: KeywordStore
    embedder: EmbeddingProvider
    indexer: HybridIndexer
    retriever: Retriever
    db: Database


def build_stack(
    *, collection_name: str = COLLECTION_NAME, db_path: Path = DEFAULT_DB_PATH
) -> DocgenStack:
    settings = get_settings()
    vector_store = get_vector_store(settings, collection_name=collection_name)
    keyword_store = get_keyword_store(settings, index_name=collection_name)
    embedder = get_embedder(settings)

    # Bootstrapping the store needs a vector dimension, which only the
    # embedder actually knows -- same reasoning as api/main.py's lifespan,
    # which this mirrors. ensure_ready() is idempotent (create if
    # missing, otherwise just connect), unlike create_collection(), which
    # assumes a fresh index -- wrong here, since this must be able to
    # attach to the corpus's EXISTING data every time, not recreate it.
    probe = embedder.embed(["dimension probe"])
    vector_store.ensure_ready(probe[0].dimension)
    keyword_store.ensure_ready()

    return DocgenStack(
        vector_store=vector_store,
        keyword_store=keyword_store,
        embedder=embedder,
        indexer=HybridIndexer(vector_store, keyword_store),
        # No reranker -- nothing in docgen asks for one, and the
        # hybrid_rrf method's own rank fusion across vector + keyword
        # search is doing the equivalent job inside Elasticsearch
        # already; see Retriever.retrieve()'s rerank=False default.
        retriever=Retriever(vector_store, keyword_store, embedder),
        db=Database(db_path),
    )
