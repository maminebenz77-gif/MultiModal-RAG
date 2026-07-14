import pytest

from multimodal_rag.ingestion.vision import ImageDescriber
from multimodal_rag.providers.base import VisionProvider


class FakeVisionProvider(VisionProvider):
    def __init__(self, result: str | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def describe(self, image_bytes: bytes, prompt: str | None = None) -> str:
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def test_degrades_to_placeholder_when_no_vision_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_not_implemented() -> VisionProvider:
        raise NotImplementedError("No concrete VisionProvider implemented yet.")

    monkeypatch.setattr("multimodal_rag.ingestion.vision.get_vision", raise_not_implemented)

    describer = ImageDescriber()
    description, status = describer.describe(b"fake-bytes")

    assert status == "placeholder"
    assert "unavailable" in description


def test_uses_real_provider_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeVisionProvider(result="a bar chart showing throughput over time")
    monkeypatch.setattr("multimodal_rag.ingestion.vision.get_vision", lambda: fake)

    describer = ImageDescriber()
    description, status = describer.describe(b"fake-bytes")

    assert status == "generated"
    assert description == "a bar chart showing throughput over time"


def test_provider_construction_only_happens_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def build_provider() -> VisionProvider:
        calls["count"] += 1
        return FakeVisionProvider(result="ok")

    monkeypatch.setattr("multimodal_rag.ingestion.vision.get_vision", build_provider)

    describer = ImageDescriber()
    describer.describe(b"a")
    describer.describe(b"b")
    describer.describe(b"c")

    assert calls["count"] == 1


def test_degrades_to_placeholder_when_provider_call_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeVisionProvider(error=RuntimeError("provider timed out"))
    monkeypatch.setattr("multimodal_rag.ingestion.vision.get_vision", lambda: fake)

    describer = ImageDescriber()
    description, status = describer.describe(b"fake-bytes")

    assert status == "placeholder"
    assert "failed" in description
