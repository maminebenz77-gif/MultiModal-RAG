import socket

import pytest

from multimodal_rag.config import RagEnv, Settings
from multimodal_rag.privacy_guard import ExternalCallBlockedError
from multimodal_rag.stores.factory import get_vector_store
from multimodal_rag.stores.qdrant_store import QdrantStore


def _make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "rag_env": RagEnv.SERVER,
        "llm_provider": "litellm",
        "llm_model": "internal-model",
        "embed_provider": "sentence_transformers",
        "embed_model": "all-MiniLM-L6-v2",
        "qdrant_url": "http://10.0.0.1:6333",
        "elastic_url": "http://10.0.0.1:9200",
        "allow_external": False,
    }
    base.update(overrides)
    return Settings.model_validate(base)


class TestGetVectorStore:
    def test_blocks_external_qdrant_url_on_server_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "1.2.3.4")
        settings = _make_settings(qdrant_url="https://public-qdrant.example.com:6333")
        with pytest.raises(ExternalCallBlockedError):
            get_vector_store(settings)

    def test_allows_internal_qdrant_url_on_server_profile(self) -> None:
        settings = _make_settings(qdrant_url="http://10.0.0.6:6333")
        store = get_vector_store(settings)
        assert isinstance(store, QdrantStore)

    def test_allows_external_qdrant_url_on_local_profile(self) -> None:
        settings = _make_settings(
            rag_env=RagEnv.LOCAL,
            allow_external=True,
            qdrant_url="https://public-qdrant.example.com:6333",
        )
        store = get_vector_store(settings)
        assert isinstance(store, QdrantStore)

    def test_uses_provided_collection_name(self) -> None:
        settings = _make_settings(qdrant_url="http://localhost:6333")
        store = get_vector_store(settings, collection_name="custom_collection")
        assert isinstance(store, QdrantStore)
        assert store._collection_name == "custom_collection"
