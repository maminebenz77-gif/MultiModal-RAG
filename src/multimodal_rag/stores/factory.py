"""Factory: the ONE place allowed to import concrete store classes.

Same rule as providers/factory.py — get_vector_store() is the only
sanctioned way to obtain a VectorStore, and the only place the privacy
guard actually runs for the store's connection URL. A misconfigured
qdrant_url pointing at a public address would leak the entire corpus's
text and vectors, not just one call's worth — at least as sensitive as
anything the provider guards already cover.
"""

from ..config import Settings, get_settings
from ..privacy_guard import enforce_privacy_guard
from .base import VectorStore
from .qdrant_store import QdrantStore

_DEFAULT_COLLECTION_NAME = "chunks"


def get_vector_store(
    settings: Settings | None = None, collection_name: str = _DEFAULT_COLLECTION_NAME
) -> VectorStore:
    settings = settings or get_settings()
    enforce_privacy_guard(settings.qdrant_url, settings.allow_external)
    return QdrantStore(url=settings.qdrant_url, collection_name=collection_name)
