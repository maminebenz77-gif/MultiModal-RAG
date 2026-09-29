import socket

import pytest

from multimodal_rag.config import RagEnv, Settings
from multimodal_rag.privacy_guard import ExternalCallBlockedError
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore, ElasticsearchVectorStore
from multimodal_rag.stores.factory import get_keyword_store, get_vector_store


def _make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "rag_env": RagEnv.SERVER,
        "llm_provider": "litellm",
        "llm_model": "internal-model",
        "embed_provider": "sentence_transformers",
        "embed_model": "all-MiniLM-L6-v2",
        "elastic_url": "http://10.0.0.1:9200",
        "allow_external": False,
    }
    base.update(overrides)
    return Settings.model_validate(base)


class TestGetVectorStore:
    def test_blocks_external_elastic_url_on_server_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "1.2.3.4")
        settings = _make_settings(elastic_url="https://public-elastic.example.com:9200")
        with pytest.raises(ExternalCallBlockedError):
            get_vector_store(settings)

    def test_allows_internal_elastic_url_on_server_profile(self) -> None:
        settings = _make_settings(elastic_url="http://10.0.0.6:9200")
        store = get_vector_store(settings)
        assert isinstance(store, ElasticsearchVectorStore)

    def test_allows_external_elastic_url_on_local_profile(self) -> None:
        settings = _make_settings(
            rag_env=RagEnv.LOCAL,
            allow_external=True,
            elastic_url="https://public-elastic.example.com:9200",
        )
        store = get_vector_store(settings)
        assert isinstance(store, ElasticsearchVectorStore)

    def test_uses_provided_collection_name(self) -> None:
        settings = _make_settings(elastic_url="http://localhost:9200")
        store = get_vector_store(settings, collection_name="custom_collection")
        assert isinstance(store, ElasticsearchVectorStore)
        assert store._store._alias == "custom_collection"


class TestGetKeywordStore:
    def test_blocks_external_elastic_url_on_server_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "1.2.3.4")
        settings = _make_settings(elastic_url="https://public-elastic.example.com:9200")
        with pytest.raises(ExternalCallBlockedError):
            get_keyword_store(settings)

    def test_allows_internal_elastic_url_on_server_profile(self) -> None:
        settings = _make_settings(elastic_url="http://10.0.0.7:9200")
        store = get_keyword_store(settings)
        assert isinstance(store, ElasticsearchStore)

    def test_allows_external_elastic_url_on_local_profile(self) -> None:
        settings = _make_settings(
            rag_env=RagEnv.LOCAL,
            allow_external=True,
            elastic_url="https://public-elastic.example.com:9200",
        )
        store = get_keyword_store(settings)
        assert isinstance(store, ElasticsearchStore)

    def test_uses_provided_index_name(self) -> None:
        settings = _make_settings(elastic_url="http://localhost:9200")
        store = get_keyword_store(settings, index_name="custom_index")
        assert isinstance(store, ElasticsearchStore)
        assert store._alias == "custom_index"


class TestSharedBackend:
    """The whole point of merging the two stores: get_vector_store() and
    get_keyword_store() called with the same (url, name) must hand back
    views onto ONE real backend, not two independent ones that happen to
    share a name -- otherwise a blue-green rebuild started through one
    role would be invisible to the other's writes (see
    elasticsearch_store.py's module docstring)."""

    def test_vector_and_keyword_store_share_the_same_backend_for_the_same_name(self) -> None:
        settings = _make_settings(elastic_url="http://localhost:9200")
        vector_store = get_vector_store(settings, collection_name="shared_name")
        keyword_store = get_keyword_store(settings, index_name="shared_name")

        assert vector_store._store is keyword_store

    def test_different_names_get_independent_backends(self) -> None:
        settings = _make_settings(elastic_url="http://localhost:9200")
        a = get_keyword_store(settings, index_name="name_a")
        b = get_keyword_store(settings, index_name="name_b")

        assert a is not b
