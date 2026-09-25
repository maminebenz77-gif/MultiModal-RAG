"""POST /suggest-tags: preview-only tag suggestion, no write to the
corpus. The LLM is faked or explicitly simulated-unconfigured in every
test here -- unlike Langfuse (blanked globally, see tests/conftest.py),
this project has NO global fixture blocking a real LLM provider, so a
test that never touches get_llm() would call whatever's real and
configured in a developer's own .env.local. Caught live while writing
these tests: an earlier version of the "unconfigured" test below relied
on ambient environment state instead of monkeypatching, and made a
real, slow/hanging network call on a machine that happened to have a
real provider configured -- exactly the "silent real LLM call in
tests" class of bug this project has fixed before elsewhere.
"""

import httpx
import pytest

from .conftest import SAMPLE_DOC


async def test_suggest_tags_returns_empty_list_when_no_llm_is_configured(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from multimodal_rag.ingestion import tag_suggestion

    def _raise() -> None:
        raise NotImplementedError("no provider configured")

    monkeypatch.setattr(tag_suggestion, "get_llm", _raise)

    with open(SAMPLE_DOC, "rb") as f:
        response = await client.post(
            "/suggest-tags", files={"file": ("chunking_demo.md", f, "text/markdown")}
        )

    assert response.status_code == 200
    assert response.json() == {"tags": []}


async def test_suggest_tags_returns_empty_list_for_an_empty_file(
    client: httpx.AsyncClient,
) -> None:
    response = await client.post(
        "/suggest-tags", files={"file": ("empty.md", b"", "text/markdown")}
    )

    assert response.status_code == 200
    assert response.json() == {"tags": []}


async def test_suggest_tags_returns_empty_list_for_an_unparseable_file(
    client: httpx.AsyncClient,
) -> None:
    # A .md extension doesn't reliably trigger a real parse failure --
    # markdown parsing is permissive enough to "succeed" on almost any
    # byte content, which was caught live here: an earlier version of
    # this test used a .md-suffixed garbage file, which parsed fine and
    # went on to make a real, unmocked LLM call. A .xlsx extension over
    # non-zip bytes reliably fails inside openpyxl instead (garbage
    # content is never a valid zip archive), so this exercises the
    # actual except-Exception-return-[] path in _suggest_tags_sync
    # without ever reaching suggest_tags() at all -- no LLM call to mock.
    response = await client.post(
        "/suggest-tags",
        files={
            "file": (
                "garbage.xlsx",
                b"not a real zip archive",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )

    assert response.status_code == 200
    assert response.json() == {"tags": []}


async def test_suggest_tags_returns_the_llms_suggestions_for_a_real_document(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from multimodal_rag.ingestion import tag_suggestion

    class _FakeLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            return '["runbook", "latency"]'

    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FakeLLM())

    with open(SAMPLE_DOC, "rb") as f:
        response = await client.post(
            "/suggest-tags", files={"file": ("chunking_demo.md", f, "text/markdown")}
        )

    assert response.status_code == 200
    assert response.json() == {"tags": ["runbook", "latency"]}


async def test_suggest_tags_writes_nothing_to_the_corpus(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from multimodal_rag.ingestion import tag_suggestion

    class _FakeLLM:
        def generate(self, messages: list[dict[str, str]]) -> str:
            return '["runbook"]'

    monkeypatch.setattr(tag_suggestion, "get_llm", lambda: _FakeLLM())

    with open(SAMPLE_DOC, "rb") as f:
        await client.post(
            "/suggest-tags", files={"file": ("chunking_demo.md", f, "text/markdown")}
        )

    documents_response = await client.get("/documents")
    assert documents_response.json()["documents"] == []
