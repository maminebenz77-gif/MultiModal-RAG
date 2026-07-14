import pytest

from multimodal_rag.ingestion.tables import rows_to_markdown, summarize_table
from multimodal_rag.providers.base import LLMProvider


def test_rows_to_markdown_empty() -> None:
    assert rows_to_markdown([]) == ""


def test_rows_to_markdown_renders_header_separator_and_body() -> None:
    result = rows_to_markdown([["Model", "Latency"], ["gpt-4o-mini", "200ms"]])
    assert result == "| Model | Latency |\n| --- | --- |\n| gpt-4o-mini | 200ms |"


def test_rows_to_markdown_strips_newlines_in_cells() -> None:
    result = rows_to_markdown([["A"], ["line1\nline2"]])
    assert "\n" not in result.split("\n")[2]


class FakeLLM(LLMProvider):
    def __init__(self, result: str | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error

    def generate(self, messages: list[dict[str, str]]) -> str:
        if self._error is not None:
            raise self._error
        assert self._result is not None
        return self._result


def test_summarize_table_returns_none_when_no_llm_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_error() -> LLMProvider:
        raise RuntimeError("no provider configured")

    monkeypatch.setattr("multimodal_rag.ingestion.tables.get_llm", raise_error)
    assert summarize_table("| a |\n| --- |\n| 1 |") is None


def test_summarize_table_returns_none_when_generate_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM(error=RuntimeError("timed out"))
    monkeypatch.setattr("multimodal_rag.ingestion.tables.get_llm", lambda: fake)
    assert summarize_table("| a |\n| --- |\n| 1 |") is None


def test_summarize_table_returns_llm_output_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeLLM(result="A one-row table of latency numbers.")
    monkeypatch.setattr("multimodal_rag.ingestion.tables.get_llm", lambda: fake)
    assert summarize_table("| a |\n| --- |\n| 1 |") == "A one-row table of latency numbers."
