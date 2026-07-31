import pytest

from multimodal_rag.generation.rewrite import build_rewrite_messages, rewrite_query
from multimodal_rag.providers.base import LLMProvider


class FakeLLM(LLMProvider):
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_messages: list[dict[str, str]] | None = None

    def generate(self, messages: list[dict[str, str]]) -> str:
        self.last_messages = messages
        return self._response


def test_empty_history_returns_query_unchanged_without_calling_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_get_llm() -> FakeLLM:
        raise AssertionError("must not call get_llm() when there's no history")

    monkeypatch.setattr("multimodal_rag.generation.rewrite.get_llm", _fail_get_llm)

    result = rewrite_query("What was the latency?", [])

    assert result == "What was the latency?"


def test_with_history_calls_llm_and_returns_stripped_rewritten_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_llm = FakeLLM("  What was the internal gateway's P95 latency?  ")
    monkeypatch.setattr("multimodal_rag.generation.rewrite.get_llm", lambda: fake_llm)

    result = rewrite_query(
        "What about the P95 numbers instead?",
        history=[("What was the average latency?", "340ms for the internal gateway.")],
    )

    assert result == "What was the internal gateway's P95 latency?"
    assert fake_llm.last_messages is not None


def test_build_rewrite_messages_includes_history_and_latest_query() -> None:
    messages = build_rewrite_messages(
        history=[("What model was used?", "gpt-4o-mini")],
        latest_query="What about that?",
    )
    user_content = messages[1]["content"]
    assert "What model was used?" in user_content
    assert "gpt-4o-mini" in user_content
    assert "What about that?" in user_content
