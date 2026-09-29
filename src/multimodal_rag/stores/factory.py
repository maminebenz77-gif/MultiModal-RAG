"""Factory: the ONE place allowed to import concrete store classes.

Same rule as providers/factory.py — get_vector_store()/get_keyword_store()
are the only sanctioned way to obtain a store, and the only place the
privacy guard actually runs for each store's connection URL. A
misconfigured elastic_url pointing at a public address would leak the
entire corpus's text AND vectors — not just one call's worth.

Elasticsearch serves both roles now (see elasticsearch_store.py's module
docstring for why that's two classes, not one) against the SAME
physical index/alias, so get_vector_store() and get_keyword_store()
must return two objects that share one ElasticsearchStore's client and
alias/blue-green state, not two independent ones each pointed at the
same name -- otherwise a blue-green rebuild started via one role's
object would be invisible to the other's writes. `_shared_backend`
below is a small per-process cache, keyed by (url, name), that makes
calling the two factory functions independently (as every real call
site already does -- see api/main.py's lifespan) still hand back views
onto the one real backend.
"""

from ..config import Settings, get_settings
from ..privacy_guard import enforce_privacy_guard
from .base import KeywordStore, VectorStore
from .elasticsearch_store import ElasticsearchStore, ElasticsearchVectorStore

_DEFAULT_COLLECTION_NAME = "chunks"

_backends: dict[tuple[str, str], ElasticsearchStore] = {}


def _shared_backend(url: str, name: str) -> ElasticsearchStore:
    key = (url, name)
    if key not in _backends:
        _backends[key] = ElasticsearchStore(url=url, index_name=name)
    return _backends[key]


def get_vector_store(
    settings: Settings | None = None, collection_name: str = _DEFAULT_COLLECTION_NAME
) -> VectorStore:
    settings = settings or get_settings()
    enforce_privacy_guard(settings.elastic_url, settings.allow_external)
    return ElasticsearchVectorStore(_shared_backend(settings.elastic_url, collection_name))


def get_keyword_store(
    settings: Settings | None = None, index_name: str = _DEFAULT_COLLECTION_NAME
) -> KeywordStore:
    settings = settings or get_settings()
    enforce_privacy_guard(settings.elastic_url, settings.allow_external)
    return _shared_backend(settings.elastic_url, index_name)
