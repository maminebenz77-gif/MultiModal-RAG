"""FastAPI dependency providers.

Everything expensive to construct (store clients, the embedder, the
retriever, the sqlite Database) is built exactly once, in main.py's
lifespan, and stashed on app.state. These Depends() functions just hand
pieces of that singleton bundle to route handlers — routers never
construct a store or provider themselves, matching the same
factory-only-construction rule the rest of the codebase follows.

No RagChain singleton here: /query needs a fresh method/top_k per
request (see routers/query.py), and constructing a RagChain is cheap
(references + a LangChain pipeline object, no I/O), so it builds one
per call from the shared Retriever rather than this module holding a
fixed one.
"""

from dataclasses import dataclass

from fastapi import Request

from ..providers.base import EmbeddingProvider
from ..retrieval.retriever import Retriever
from ..stores.base import KeywordStore, VectorStore
from ..stores.indexer import HybridIndexer
from .db import Database


@dataclass
class AppState:
    vector_store: VectorStore
    keyword_store: KeywordStore
    embedder: EmbeddingProvider
    indexer: HybridIndexer
    retriever: Retriever
    db: Database


def get_app_state(request: Request) -> AppState:
    state: AppState = request.app.state.app_state
    return state


def get_vector_store(request: Request) -> VectorStore:
    return get_app_state(request).vector_store


def get_keyword_store(request: Request) -> KeywordStore:
    return get_app_state(request).keyword_store


def get_embedder(request: Request) -> EmbeddingProvider:
    return get_app_state(request).embedder


def get_indexer(request: Request) -> HybridIndexer:
    return get_app_state(request).indexer


def get_retriever(request: Request) -> Retriever:
    return get_app_state(request).retriever


def get_db(request: Request) -> Database:
    return get_app_state(request).db
