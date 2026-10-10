"""FastAPI app assembly.

Everything expensive (store connections, the embedder, the retriever,
the sqlite Database) is built exactly once in the lifespan, not per
request. Bootstrapping the stores needs a vector dimension, which only
the embedder actually knows -- a single cheap probe embed call at
startup answers that, rather than hardcoding a dimension that would
silently drift out of sync if the configured embedding model changed.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI
from starlette.concurrency import run_in_threadpool

from ..config import COLLECTION_NAME, DEFAULT_DB_PATH, get_settings
from ..docgen.checkpointer import DEFAULT_CHECKPOINT_PATH
from ..docgen.runs_db import DEFAULT_RUNS_DB_PATH, DocgenRunsDB
from ..docgen.stack import DocgenStack
from ..providers.factory import get_embedder, get_reranker
from ..retrieval.retriever import Retriever
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from .db import Database
from .dependencies import AppState
from .identity import get_principal
from .routers import conversations, docgen, documents, feedback, health, ingest, metrics, query

__all__ = ["COLLECTION_NAME", "DEFAULT_DB_PATH", "app", "create_app"]
"""COLLECTION_NAME/DEFAULT_DB_PATH now LIVE in config.py (so docgen/
can depend on them without depending on api/ -- see config.py's own
docstring) but are re-exported here unchanged for existing callers
(tests/live/wipe_db.py) that import them from this module."""

_logger = logging.getLogger(__name__)


def _load_optional_reranker(settings):
    try:
        return get_reranker(settings)
    except (NotImplementedError, OSError, httpx.HTTPError) as exc:
        # Reranking is optional; a missing cache or temporary model-download
        # failure must not prevent hybrid search from starting.
        _logger.warning("Reranker unavailable; starting without reranking: %s", exc)
        return None


def create_app(
    db_path: Path = DEFAULT_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    docgen_runs_db_path: Path = DEFAULT_RUNS_DB_PATH,
    docgen_checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        settings = get_settings()
        vector_store = get_vector_store(settings, collection_name=collection_name)
        keyword_store = get_keyword_store(settings, index_name=collection_name)
        embedder = get_embedder(settings)

        probe = await run_in_threadpool(embedder.embed, ["dimension probe"])
        await run_in_threadpool(vector_store.ensure_ready, probe[0].dimension)
        await run_in_threadpool(keyword_store.ensure_ready)

        reranker = await run_in_threadpool(_load_optional_reranker, settings)

        app.state.app_state = AppState(
            vector_store=vector_store,
            keyword_store=keyword_store,
            embedder=embedder,
            indexer=HybridIndexer(vector_store, keyword_store),
            retriever=Retriever(
                vector_store,
                keyword_store,
                embedder,
                reranker=reranker,
                recency_tilt_weight=settings.recency_tilt_weight,
                recency_half_life_days=settings.recency_half_life_days,
            ),
            db=Database(db_path),
        )
        # Reuses the SAME store/embedder/db instances just built above --
        # not a second build_stack() call, which would construct its own
        # independent embedder/stores for no benefit (identical data,
        # double the memory). docgen is a separate module from the main
        # app's own AppState, so it gets its own app.state attribute
        # rather than being folded into that bundle.
        app.state.docgen_stack = DocgenStack(
            vector_store=vector_store,
            keyword_store=keyword_store,
            embedder=embedder,
            indexer=HybridIndexer(vector_store, keyword_store),
            retriever=Retriever(vector_store, keyword_store, embedder),
            db=Database(db_path),
        )
        app.state.docgen_runs_db = DocgenRunsDB(docgen_runs_db_path)
        app.state.docgen_checkpoint_path = docgen_checkpoint_path
        yield

    app = FastAPI(title="Multimodal RAG API", lifespan=lifespan)
    # Identity is attached where routers are INCLUDED, not on individual
    # routes -- so a router (or a route inside one) added later can't
    # forget it. /health is the one deliberate exemption: a load
    # balancer's liveness probe has no user, and it touches no
    # application data.
    identified = [Depends(get_principal)]
    app.include_router(docgen.router, dependencies=identified)
    app.include_router(ingest.router, dependencies=identified)
    app.include_router(documents.router, dependencies=identified)
    app.include_router(query.router, dependencies=identified)
    app.include_router(conversations.router, dependencies=identified)
    app.include_router(feedback.router, dependencies=identified)
    app.include_router(metrics.router, dependencies=identified)
    app.include_router(health.router)
    return app


app = create_app()
