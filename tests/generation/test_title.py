import pytest

from multimodal_rag.generation.title import generate_title
from multimodal_rag.providers.base import LLMProvider


class _FakeLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] | None = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response


def test_returns_the_llm_title_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = _FakeLLM("  Local Inference Latency Comparison  ")
    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", lambda: fake_llm)

    title = generate_title("How fast is local inference?", "It's 180ms.")

    assert title == "Local Inference Latency Comparison"
    assert fake_llm.last_messages is not None
    assert "How fast is local inference?" in fake_llm.last_messages[1]["content"]
    assert "It's 180ms." in fake_llm.last_messages[1]["content"]


def test_returns_none_when_the_llm_call_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FailingLLM(LLMProvider):
        def generate(self, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", lambda: _FailingLLM())

    assert generate_title("a question", "an answer") is None


def test_returns_none_when_no_provider_is_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> LLMProvider:
        raise ValueError("no llm_provider configured")

    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", _raise)

    assert generate_title("a question", "an answer") is None


def test_returns_none_for_a_blank_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", lambda: _FakeLLM("   "))

    assert generate_title("a question", "an answer") is None
