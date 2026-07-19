import base64
import io
from typing import Any

import pytest
from PIL import Image

from multimodal_rag.providers.vision import (
    InternalServerVisionProvider,
    LiteLLMVisionProvider,
    _downscale_if_needed,
)

_TINY_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a4944415478da6360000002000155007c0000000049454e44ae426082"
)


def _make_png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color="blue").save(buffer, format="PNG")
    return buffer.getvalue()


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


class TestDownscaleIfNeeded:
    def test_small_image_is_returned_unchanged(self) -> None:
        small = _make_png(100, 100)
        assert _downscale_if_needed(small, max_dimension=1024) == small

    def test_large_image_is_shrunk_to_max_dimension(self) -> None:
        large = _make_png(3000, 1500)
        result = _downscale_if_needed(large, max_dimension=1024)

        assert result != large
        resized = Image.open(io.BytesIO(result))
        assert max(resized.width, resized.height) == 1024

    def test_aspect_ratio_is_preserved(self) -> None:
        large = _make_png(3000, 1500)  # 2:1
        result = _downscale_if_needed(large, max_dimension=1024)
        resized = Image.open(io.BytesIO(result))
        assert resized.width / resized.height == pytest.approx(2.0, rel=0.02)

    def test_undecodable_bytes_are_returned_unchanged(self) -> None:
        garbage = b"not an image at all"
        assert _downscale_if_needed(garbage, max_dimension=1024) == garbage


class TestLiteLLMVisionProviderDownscaling:
    def test_large_image_is_downscaled_before_being_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured_kwargs: dict[str, Any] = {}

        class FakeMessage:
            content = "described"

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices = [FakeChoice()]

        def fake_completion(**kwargs: object) -> FakeResponse:
            captured_kwargs.update(kwargs)
            return FakeResponse()

        monkeypatch.setattr(
            "multimodal_rag.providers.vision.litellm.completion", fake_completion
        )

        provider = LiteLLMVisionProvider(model="gpt-4o-mini", max_dimension=256)
        provider.describe(_make_png(2000, 1000))

        image_block = next(
            b for b in captured_kwargs["messages"][0]["content"] if b["type"] == "image_url"
        )
        sent_data_url = image_block["image_url"]["url"]
        sent_bytes = base64.b64decode(sent_data_url.split(",", 1)[1])
        sent_image = Image.open(io.BytesIO(sent_bytes))
        assert max(sent_image.width, sent_image.height) == 256
