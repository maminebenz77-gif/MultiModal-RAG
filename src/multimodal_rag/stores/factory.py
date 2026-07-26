"""Factory: the ONE place allowed to import concrete store classes.

Same rule as providers/factory.py — get_vector_store()/get_keyword_store()
are the only sanctioned way to obtain a store, and the only place the
privacy guard actually runs for each store's connection URL. A
misconfigured qdrant_url or elastic_url pointing at a public address
would leak the entire corpus's text (and vectors, for Qdrant) — not just
one call's worth — at least as sensitive as anything the provider guards
already cover.
"""

from ..config import Settings, get_settings
from ..privacy_guard import enforce_privacy_guard
from .base import KeywordStore, VectorStore
from .elasticsearch_store import ElasticsearchStore
from .qdrant_store import QdrantStore

_DEFAULT_COLLECTION_NAME = "chunks"
_DEFAULT_INDEX_NAME = "chunks"


def get_vector_store(
    settings: Settings | None = None, collection_name: str = _DEFAULT_COLLECTION_NAME
) -> VectorStore:
    settings = settings or get_settings()
    enforce_privacy_guard(settings.qdrant_url, settings.allow_external)
    return QdrantStore(url=settings.qdrant_url, collection_name=collection_name)


def get_keyword_store(
    settings: Settings | None = None, index_name: str = _DEFAULT_INDEX_NAME
) -> KeywordStore:
    settings = settings or get_settings()
    enforce_privacy_guard(settings.elastic_url, settings.allow_external)
    return ElasticsearchStore(url=settings.elastic_url, index_name=index_name)
