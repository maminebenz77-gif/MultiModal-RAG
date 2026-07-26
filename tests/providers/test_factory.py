import socket

import pytest

from multimodal_rag.config import RagEnv, Settings
from multimodal_rag.privacy_guard import ExternalCallBlockedError
from multimodal_rag.providers.embeddings import LiteLLMEmbeddingProvider
from multimodal_rag.providers.factory import get_embedder, get_llm, get_reranker, get_vision
from multimodal_rag.providers.llm import LiteLLMProvider
from multimodal_rag.providers.rerank import CrossEncoderReranker
from multimodal_rag.providers.vision import LiteLLMVisionProvider


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


class TestGetLLM:
    def test_blocks_external_llm_on_server_profile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "1.2.3.4")
        settings = _make_settings(llm_base_url="https://api.openai.com/v1")
        with pytest.raises(ExternalCallBlockedError):
            get_llm(settings)

    def test_allows_internal_llm_on_server_profile(self) -> None:
        settings = _make_settings(llm_base_url="http://10.0.0.5:8080/v1")
        provider = get_llm(settings)
        assert isinstance(provider, LiteLLMProvider)

    def test_allows_external_llm_on_local_profile(self) -> None:
        settings = _make_settings(
            rag_env=RagEnv.LOCAL, allow_external=True, llm_base_url=None
        )
        provider = get_llm(settings)
        assert isinstance(provider, LiteLLMProvider)


class TestGetEmbedder:
    def test_blocks_external_embed_litellm_on_server_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "1.2.3.4")
        settings = _make_settings(
            embed_provider="litellm", embed_base_url="https://api.openai.com/v1"
        )
        with pytest.raises(ExternalCallBlockedError):
            get_embedder(settings)

    def test_allows_internal_embed_litellm_on_server_profile(self) -> None:
        settings = _make_settings(embed_provider="litellm", embed_base_url="http://10.0.0.5:9000")
        provider = get_embedder(settings)
        assert isinstance(provider, LiteLLMEmbeddingProvider)


class TestGetVision:
    def test_unconfigured_vision_raises_not_implemented(self) -> None:
        # Continuity with ingestion's graceful degradation: a profile that
        # simply never set VISION_PROVIDER must keep raising
        # NotImplementedError, the exact exception ImageDescriber catches.
        settings = _make_settings(vision_provider=None)
        with pytest.raises(NotImplementedError):
            get_vision(settings)

    def test_blocks_external_vision_on_server_profile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "1.2.3.4")
        settings = _make_settings(
            vision_provider="litellm",
            vision_model="internal-vision-model",
            vision_base_url="https://api.openai.com/v1",
        )
        with pytest.raises(ExternalCallBlockedError):
            get_vision(settings)

    def test_allows_internal_vision_on_server_profile(self) -> None:
        settings = _make_settings(
            vision_provider="litellm",
            vision_model="internal-vision-model",
            vision_base_url="http://10.0.0.5:8080/v1",
        )
        provider = get_vision(settings)
        assert isinstance(provider, LiteLLMVisionProvider)

    def test_allows_external_vision_on_local_profile(self) -> None:
        settings = _make_settings(
            rag_env=RagEnv.LOCAL,
            allow_external=True,
            vision_provider="litellm",
            vision_model="gpt-4o-mini",
            vision_base_url="https://api.openai.com/v1",
        )
        provider = get_vision(settings)
        assert isinstance(provider, LiteLLMVisionProvider)

    def test_litellm_provider_without_vision_model_raises_value_error(self) -> None:
        settings = _make_settings(
            rag_env=RagEnv.LOCAL,
            allow_external=True,
            vision_provider="litellm",
            vision_model=None,
        )
        with pytest.raises(ValueError, match="VISION_MODEL"):
            get_vision(settings)


class TestGetReranker:
    def test_unconfigured_reranker_raises_not_implemented(self) -> None:
        settings = _make_settings(reranker_provider=None)
        with pytest.raises(NotImplementedError):
            get_reranker(settings)

    def test_cross_encoder_without_model_raises_value_error(self) -> None:
        settings = _make_settings(
            rag_env=RagEnv.LOCAL,
            allow_external=True,
            reranker_provider="cross_encoder",
            reranker_model=None,
        )
        with pytest.raises(ValueError, match="RERANKER_MODEL"):
            get_reranker(settings)

    def test_unknown_reranker_provider_raises_value_error(self) -> None:
        settings = _make_settings(
            rag_env=RagEnv.LOCAL,
            allow_external=True,
            reranker_provider="not-a-real-provider",
        )
        with pytest.raises(ValueError, match="Unknown reranker_provider"):
            get_reranker(settings)

    def test_cross_encoder_is_constructed_locally_regardless_of_allow_external(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "multimodal_rag.providers.rerank.CrossEncoder", lambda model_name, device: object()
        )
        settings = _make_settings(
            reranker_provider="cross_encoder",
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        reranker = get_reranker(settings)
        assert isinstance(reranker, CrossEncoderReranker)
