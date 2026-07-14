import litellm
import pytest

from multimodal_rag.providers.llm import InternalServerLLM, LiteLLMProvider


def test_litellm_provider_disables_telemetry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "telemetry", True)
    LiteLLMProvider(model="gpt-4o-mini")
    assert litellm.telemetry is False


def test_litellm_provider_extracts_text_from_response(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_kwargs = {}

    class FakeMessage:
        content = "hello from the model"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_completion(**kwargs: object) -> FakeResponse:
        captured_kwargs.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("multimodal_rag.providers.llm.litellm.completion", fake_completion)

    provider = LiteLLMProvider(model="gpt-4o-mini", base_url="http://localhost:11434", api_key="k")
    result = provider.generate([{"role": "user", "content": "hi"}])

    assert result == "hello from the model"
    assert captured_kwargs["model"] == "gpt-4o-mini"
    assert captured_kwargs["base_url"] == "http://localhost:11434"


def test_internal_server_llm_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        InternalServerLLM(base_url=None)


def test_internal_server_llm_generate_is_unimplemented() -> None:
    provider = InternalServerLLM(base_url="http://10.0.0.5:8080")
    with pytest.raises(NotImplementedError):
        provider.generate([{"role": "user", "content": "hi"}])
