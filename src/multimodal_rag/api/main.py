"""FastAPI app assembly.

Everything expensive (store connections, the embedder, the retriever,
the sqlite Database) is built exactly once in the lifespan, not per
request. Bootstrapping the stores needs a vector dimension, which only
the embedder actually knows -- a single cheap probe embed call at
startup answers that, rather than hardcoding a dimension that would
silently drift out of sync if the configured embedding model changed.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.concurrency import run_in_threadpool

from ..config import PROJECT_ROOT, get_settings
from ..providers.factory import get_embedder
from ..retrieval.retriever import Retriever
from ..stores.factory import get_keyword_store, get_vector_store
from ..stores.indexer import HybridIndexer
from .db import Database
from .dependencies import AppState
from .routers import documents, feedback, health, ingest, metrics, query

_DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "api_state.db"
_COLLECTION_NAME = "api_corpus"


def create_app(
    db_path: Path = _DEFAULT_DB_PATH, collection_name: str = _COLLECTION_NAME
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

        app.state.app_state = AppState(
            vector_store=vector_store,
            keyword_store=keyword_store,
            embedder=embedder,
            indexer=HybridIndexer(vector_store, keyword_store),
            retriever=Retriever(vector_store, keyword_store, embedder),
            db=Database(db_path),
        )
        yield

    app = FastAPI(title="Multimodal RAG API", lifespan=lifespan)
    app.include_router(ingest.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(feedback.router)
    app.include_router(metrics.router)
    app.include_router(health.router)
    return app


app = create_app()
