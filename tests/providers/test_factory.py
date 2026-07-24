import os
import socket

import pytest

from multimodal_rag.config import RagEnv, Settings
from multimodal_rag.providers.embeddings import LiteLLMEmbeddingProvider
from multimodal_rag.providers.factory import (
    ExternalCallBlockedError,
    _enforce_privacy_guard,
    _is_internal_host,
    get_embedder,
    get_llm,
    get_vision,
)
from multimodal_rag.providers.llm import LiteLLMProvider
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


class TestIsInternalHost:
    def test_localhost_is_internal(self) -> None:
        assert _is_internal_host("localhost") is True

    def test_loopback_ip_is_internal(self) -> None:
        assert _is_internal_host("127.0.0.1") is True

    def test_private_ip_is_internal(self) -> None:
        assert _is_internal_host("10.0.0.5") is True
        assert _is_internal_host("192.168.1.10") is True

    def test_public_ip_is_not_internal(self) -> None:
        assert _is_internal_host("8.8.8.8") is False

    def test_unresolvable_hostname_is_not_internal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_gethostbyname(host: str) -> str:
            raise socket.gaierror("not found")

        monkeypatch.setattr(socket, "gethostbyname", fake_gethostbyname)
        assert _is_internal_host("nonexistent.example") is False

    def test_hostname_resolving_to_public_ip_is_not_internal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "8.8.8.8")
        assert _is_internal_host("api.openai.com") is False


class TestEnforcePrivacyGuard:
    def test_allow_external_true_is_a_noop(self) -> None:
        _enforce_privacy_guard("https://api.openai.com", allow_external=True)

    def test_none_base_url_is_allowed_but_forces_offline_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        _enforce_privacy_guard(None, allow_external=False)
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    def test_internal_base_url_is_allowed(self) -> None:
        _enforce_privacy_guard("http://10.0.0.5:8080/v1", allow_external=False)

    def test_external_base_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "1.2.3.4")
        with pytest.raises(ExternalCallBlockedError):
            _enforce_privacy_guard("https://api.openai.com/v1", allow_external=False)


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
