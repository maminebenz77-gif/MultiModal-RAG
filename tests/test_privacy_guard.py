import os
import socket

import pytest

from multimodal_rag.privacy_guard import (
    ExternalCallBlockedError,
    enforce_privacy_guard,
    is_internal_host,
)


class TestIsInternalHost:
    def test_localhost_is_internal(self) -> None:
        assert is_internal_host("localhost") is True

    def test_loopback_ip_is_internal(self) -> None:
        assert is_internal_host("127.0.0.1") is True

    def test_private_ip_is_internal(self) -> None:
        assert is_internal_host("10.0.0.5") is True
        assert is_internal_host("192.168.1.10") is True

    def test_public_ip_is_not_internal(self) -> None:
        assert is_internal_host("8.8.8.8") is False

    def test_unresolvable_hostname_is_not_internal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_gethostbyname(host: str) -> str:
            raise socket.gaierror("not found")

        monkeypatch.setattr(socket, "gethostbyname", fake_gethostbyname)
        assert is_internal_host("nonexistent.example") is False

    def test_hostname_resolving_to_public_ip_is_not_internal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "8.8.8.8")
        assert is_internal_host("api.openai.com") is False


class TestEnforcePrivacyGuard:
    def test_allow_external_true_is_a_noop(self) -> None:
        enforce_privacy_guard("https://api.openai.com", allow_external=True)

    def test_none_base_url_is_allowed_but_forces_offline_mode(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        enforce_privacy_guard(None, allow_external=False)
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    def test_internal_base_url_is_allowed(self) -> None:
        enforce_privacy_guard("http://10.0.0.5:8080/v1", allow_external=False)

    def test_external_base_url_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(socket, "gethostbyname", lambda host: "1.2.3.4")
        with pytest.raises(ExternalCallBlockedError):
            enforce_privacy_guard("https://api.openai.com/v1", allow_external=False)
