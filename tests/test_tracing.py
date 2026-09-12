from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from multimodal_rag import tracing


@pytest.fixture(autouse=True)
def _clear_client_cache() -> Iterator[None]:
    # get_langfuse_client() is @lru_cache'd (once per process, by design --
    # see its docstring) -- tests need a fresh cache each time or they'd
    # leak a mocked client (or None) across unrelated tests.
    tracing.get_langfuse_client.cache_clear()
    yield
    tracing.get_langfuse_client.cache_clear()


def _settings(
    public_key: str | None = None,
    secret_key: str | None = None,
    *,
    langfuse_host: str | None = None,
    allow_external: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        langfuse_public_key=public_key,
        langfuse_secret_key=secret_key,
        langfuse_host=langfuse_host,
        allow_external=allow_external,
    )


def test_get_langfuse_client_returns_none_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing, "get_settings", lambda: _settings())

    assert tracing.get_langfuse_client() is None


def test_get_langfuse_client_returns_none_when_only_one_key_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracing, "get_settings", lambda: _settings(public_key="pub"))

    assert tracing.get_langfuse_client() is None


def test_get_langfuse_client_constructs_a_client_when_host_is_explicit_and_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracing, "get_settings", lambda: _settings("pub", "sec", langfuse_host="http://localhost:3000")
    )
    fake_client = MagicMock()
    with patch.object(tracing, "Langfuse", return_value=fake_client) as mock_langfuse:
        client = tracing.get_langfuse_client()

    assert client is fake_client
    mock_langfuse.assert_called_once_with(
        public_key="pub",
        secret_key="sec",
        host="http://localhost:3000",
        timeout=tracing._TIMEOUT_SECONDS,
    )


def test_get_langfuse_client_returns_none_when_host_is_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No silent default to the public cloud -- an unset host means tracing
    # stays off, full stop, regardless of allow_external.
    monkeypatch.setattr(tracing, "get_settings", lambda: _settings("pub", "sec"))

    with patch.object(tracing, "Langfuse") as mock_langfuse:
        client = tracing.get_langfuse_client()

    assert client is None
    mock_langfuse.assert_not_called()


@pytest.mark.parametrize(
    "cloud_host",
    [
        "https://cloud.langfuse.com",
        "https://us.cloud.langfuse.com",
        "https://eu.cloud.langfuse.com",
    ],
)
def test_get_langfuse_client_refuses_the_public_langfuse_cloud_even_when_external_calls_are_allowed(
    monkeypatch: pytest.MonkeyPatch, cloud_host: str
) -> None:
    # The whole point: allow_external=True (a normal, non-air-gapped
    # profile -- e.g. a company laptop that's allowed to reach the
    # company's own external LLM gateway) must NOT be read as "also fine
    # to send real query/answer content to a third party's cloud."
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: _settings("pub", "sec", langfuse_host=cloud_host, allow_external=True),
    )

    with patch.object(tracing, "Langfuse") as mock_langfuse:
        client = tracing.get_langfuse_client()

    assert client is None
    mock_langfuse.assert_not_called()


def test_get_langfuse_client_is_blocked_when_offline_and_host_is_a_non_cloud_external_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A self-hosted instance that's still reachable only over the public
    # internet -- allow_external=False should still catch this, same as
    # every other external-facing provider in this project.
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: _settings(
            "pub", "sec", langfuse_host="https://langfuse.example.com", allow_external=False
        ),
    )

    with patch.object(tracing, "Langfuse") as mock_langfuse:
        client = tracing.get_langfuse_client()

    assert client is None
    mock_langfuse.assert_not_called()


def test_get_langfuse_client_is_allowed_offline_when_host_is_internal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: _settings(
            "pub", "sec", langfuse_host="http://localhost:3000", allow_external=False
        ),
    )

    with patch.object(tracing, "Langfuse", return_value=MagicMock()) as mock_langfuse:
        client = tracing.get_langfuse_client()

    assert client is not None
    mock_langfuse.assert_called_once()


def test_get_langfuse_client_is_only_constructed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tracing, "get_settings", lambda: _settings("pub", "sec", langfuse_host="http://localhost:3000")
    )
    with patch.object(tracing, "Langfuse", return_value=MagicMock()) as mock_langfuse:
        first = tracing.get_langfuse_client()
        second = tracing.get_langfuse_client()

    assert first is second
    mock_langfuse.assert_called_once()


def test_traced_query_is_a_noop_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: None)

    with tracing.traced_query("conv-1", "query-1"):
        pass  # must not raise


def test_traced_query_propagates_session_id_and_opens_a_span_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)
    with patch.object(tracing, "propagate_attributes") as mock_propagate:
        with tracing.traced_query("conv-1", "query-1"):
            pass

    mock_propagate.assert_called_once_with(session_id="conv-1", trace_name="query")
    fake_client.start_as_current_observation.assert_called_once_with(
        name="query", as_type="span", metadata={"query_id": "query-1"}
    )


def test_log_search_event_is_a_noop_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: None)

    tracing.log_search_event("a query", [])  # must not raise


def test_log_search_event_creates_an_event_with_result_summaries_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)
    result = SimpleNamespace(source="doc.md", score=0.9, chunk_id="doc.md::0::hash")

    tracing.log_search_event("a query", [result])

    fake_client.create_event.assert_called_once_with(
        name="search_knowledge_base",
        input="a query",
        output=[{"source": "doc.md", "score": 0.9, "chunk_id": "doc.md::0::hash"}],
    )


def test_traced_generation_yields_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: None)

    with tracing.traced_generation("generate", "gpt-4o-mini", []) as generation:
        assert generation is None


def test_traced_generation_starts_a_generation_observation_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_generation = MagicMock()
    fake_client.start_as_current_observation.return_value.__enter__.return_value = fake_generation
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)
    messages = [{"role": "user", "content": "hi"}]

    with tracing.traced_generation("generate", "gpt-4o-mini", messages) as generation:
        assert generation is fake_generation

    fake_client.start_as_current_observation.assert_called_once_with(
        name="generate", as_type="generation", model="gpt-4o-mini", input=messages
    )


def test_record_generation_result_is_a_noop_when_generation_is_none() -> None:
    tracing.record_generation_result(None, response=MagicMock(), output="answer")  # must not raise


def test_record_generation_result_attaches_output_usage_and_cost() -> None:
    generation = MagicMock()
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    )
    with patch.object(tracing.litellm, "completion_cost", return_value=0.002):
        tracing.record_generation_result(generation, response, output="the answer")

    generation.update.assert_called_once_with(
        output="the answer",
        usage_details={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        cost_details={"total": 0.002},
    )


def test_record_generation_result_tolerates_a_failing_cost_lookup() -> None:
    # An unrecognized/custom model is expected to make completion_cost()
    # raise -- must not break the trace over a missing cost figure.
    generation = MagicMock()
    response = SimpleNamespace(usage=None)
    with patch.object(tracing.litellm, "completion_cost", side_effect=Exception("unknown model")):
        tracing.record_generation_result(generation, response, output="the answer")

    generation.update.assert_called_once_with(
        output="the answer", usage_details=None, cost_details=None
    )
