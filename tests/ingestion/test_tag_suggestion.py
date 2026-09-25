"""Unit tests for tag_suggestion.py -- mocked LLM throughout, same
discipline as tables.py's summarize_table tests (a nice-to-have, LLM-
derived feature must fail soft, never raise).
"""

import pytest

from multimodal_rag.ingestion import tag_suggestion


def test_build_excerpt_joins_and_truncates_texts() -> None:
    excerpt = tag_suggestion.build_excerpt(["Title", "First paragraph."])
    assert excerpt == "Title\nFirst paragraph."


def test_build_excerpt_drops_none_entries() -> None:
    excerpt = tag_suggestion.build_excerpt(["Title", None, "Body"])  # type: ignore[list-item]
    assert excerpt == "Title\nBody"


def test_build_excerpt_truncates_long_text() -> None:
    long_text = "x" * 5000
    excerpt = tag_suggestion.build_excerpt([long_text])
    assert len(excerpt) == tag_suggestion._MAX_EXCERPT_CHARS


def test_suggest_tags_returns_empty_list_for_a_blank_excerpt() -> None:
    assert tag_suggestion.suggest_tags("   ") == []


def test_suggest_tags_returns_empty_list_when_llm_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise NotImplementedError("no provider configured")

    monkeypatch.setattr(tag_suggestion, "get_llm", _raise)
    assert tag_suggestion.suggest_tags("some excerpt") == []


def test_suggest_tags_returns_empty_list_when_the_llm_call_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FailingLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FailingLLM())
    assert tag_suggestion.suggest_tags("some excerpt") == []


def test_suggest_tags_parses_a_clean_json_array(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            return '["runbook", "q3-2026"]'

    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FakeLLM())
    assert tag_suggestion.suggest_tags("some excerpt") == ["runbook", "q3-2026"]


def test_suggest_tags_extracts_the_array_from_surrounding_prose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            return 'Here you go:\n```json\n["policy", "hr"]\n```\nHope that helps!'

    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FakeLLM())
    assert tag_suggestion.suggest_tags("some excerpt") == ["policy", "hr"]


def test_suggest_tags_returns_empty_list_for_malformed_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            return "not a json array at all"

    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FakeLLM())
    assert tag_suggestion.suggest_tags("some excerpt") == []


def test_suggest_tags_ignores_non_string_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            return '["runbook", 42, null, "q3-2026"]'

    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FakeLLM())
    assert tag_suggestion.suggest_tags("some excerpt") == ["runbook", "q3-2026"]


def test_suggest_tags_drops_blank_and_whitespace_only_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            return '["runbook", "  ", ""]'

    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FakeLLM())
    assert tag_suggestion.suggest_tags("some excerpt") == ["runbook"]


def test_suggest_tags_caps_at_five_even_when_the_llm_returns_more(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            return json_of_ten

    json_of_ten = str([f"tag{i}" for i in range(10)]).replace("'", '"')
    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FakeLLM())
    assert tag_suggestion.suggest_tags("some excerpt") == [f"tag{i}" for i in range(5)]


def test_suggest_tags_returns_empty_list_when_the_json_is_not_an_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No square brackets anywhere in this response -- unlike a wrapped
    # array (e.g. '{"tags": ["runbook"]}'), which the regex would still
    # correctly find and extract as ["runbook"].
    class _FakeLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            return '{"tags": "runbook"}'

    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FakeLLM())
    assert tag_suggestion.suggest_tags("some excerpt") == []
