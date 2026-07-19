from typing import Any

import pytest

from multimodal_rag.providers.vision import InternalServerVisionProvider, LiteLLMVisionProvider

_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a4944415478da6360000002000155007c0000000049454e44ae426082"
)


def test_litellm_vision_provider_builds_multimodal_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    class FakeMessage:
        content = "a red one-pixel image"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_completion(**kwargs: object) -> FakeResponse:
        captured_kwargs.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("multimodal_rag.providers.vision.litellm.completion", fake_completion)

    provider = LiteLLMVisionProvider(model="gpt-4o-mini", base_url="https://api.openai.com/v1")
    result = provider.describe(_TINY_PNG)

    assert result == "a red one-pixel image"
    assert captured_kwargs["model"] == "gpt-4o-mini"

    messages = captured_kwargs["messages"]
    content_blocks = messages[0]["content"]
    text_block = next(b for b in content_blocks if b["type"] == "text")
    image_block = next(b for b in content_blocks if b["type"] == "image_url")

    assert "Describe this image" in text_block["text"]
    assert image_block["image_url"]["url"].startswith("data:image/png;base64,")


def test_litellm_vision_provider_uses_custom_prompt_when_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_kwargs: dict[str, Any] = {}

    class FakeMessage:
        content = "ok"

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_completion(**kwargs: object) -> FakeResponse:
        captured_kwargs.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("multimodal_rag.providers.vision.litellm.completion", fake_completion)

    provider = LiteLLMVisionProvider(model="gpt-4o-mini")
    provider.describe(_TINY_PNG, prompt="What color is this?")

    text_block = next(
        b for b in captured_kwargs["messages"][0]["content"] if b["type"] == "text"
    )
    assert text_block["text"] == "What color is this?"


def test_internal_server_vision_provider_requires_base_url() -> None:
    with pytest.raises(ValueError, match="base_url"):
        InternalServerVisionProvider(base_url=None)


def test_internal_server_vision_provider_describe_is_unimplemented() -> None:
    provider = InternalServerVisionProvider(base_url="http://10.0.0.5:8080")
    with pytest.raises(NotImplementedError):
        provider.describe(_TINY_PNG)
