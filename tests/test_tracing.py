from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from multimodal_rag import tracing


def _clear_langfuse_client_cache() -> None:
    # Some tests in this file monkeypatch tracing.get_langfuse_client
    # itself (replacing the whole function with a plain lambda) -- if
    # this runs while that replacement is still in effect, the real
    # function's .cache_clear() is gone. A no-op fallback is correct
    # either way: no attribute means nothing of ours to clear.
    cache_clear = getattr(tracing.get_langfuse_client, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()


@pytest.fixture(autouse=True)
def _clear_client_cache() -> Iterator[None]:
    # get_langfuse_client() is @lru_cache'd (once per process, by design --
    # see its docstring) -- tests need a fresh cache each time or they'd
    # leak a mocked client (or None) across unrelated tests.
    _clear_langfuse_client_cache()
    yield
    _clear_langfuse_client_cache()


def _settings(
    public_key: str | None = None,
    secret_key: str | None = None,
    *,
    langfuse_host: str | None = None,
    allow_external: bool = True,
    langfuse_allow_cloud_host: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        langfuse_public_key=public_key,
        langfuse_secret_key=secret_key,
        langfuse_host=langfuse_host,
        allow_external=allow_external,
        langfuse_allow_cloud_host=langfuse_allow_cloud_host,
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


def test_get_langfuse_client_allows_the_public_cloud_when_explicitly_opted_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The deliberate escape hatch: a profile that's both allow_external=True
    # AND has explicitly set langfuse_allow_cloud_host=True (e.g. a personal
    # dev machine with no confidential documents) is making a conscious,
    # per-environment decision -- that's allowed to go through.
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: _settings(
            "pub",
            "sec",
            langfuse_host="https://cloud.langfuse.com",
            allow_external=True,
            langfuse_allow_cloud_host=True,
        ),
    )

    with patch.object(tracing, "Langfuse", return_value=MagicMock()) as mock_langfuse:
        client = tracing.get_langfuse_client()

    assert client is not None
    mock_langfuse.assert_called_once()


def test_get_langfuse_client_still_blocks_the_cloud_opt_in_on_an_air_gapped_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cloud opt-in only lifts the CLOUD-specific block -- it must not
    # also bypass the general "air-gapped profiles call nothing external"
    # guard. allow_external=False still wins.
    monkeypatch.setattr(
        tracing,
        "get_settings",
        lambda: _settings(
            "pub",
            "sec",
            langfuse_host="https://cloud.langfuse.com",
            allow_external=False,
            langfuse_allow_cloud_host=True,
        ),
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


def test_get_langfuse_client_returns_none_when_construction_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulates Langfuse being unreachable (wrong host, service down,
    # network blip) at the moment the client is first built.
    monkeypatch.setattr(
        tracing, "get_settings", lambda: _settings("pub", "sec", langfuse_host="http://localhost:3000")
    )
    with patch.object(tracing, "Langfuse", side_effect=ConnectionError("unreachable")):
        client = tracing.get_langfuse_client()

    assert client is None


def test_get_langfuse_client_construction_failure_is_cached_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # lru_cache does NOT cache a raised exception -- get_langfuse_client()
    # must catch the failure itself and return None, so THAT gets cached
    # and construction isn't retried (and re-failed) on every request.
    monkeypatch.setattr(
        tracing, "get_settings", lambda: _settings("pub", "sec", langfuse_host="http://localhost:3000")
    )
    with patch.object(
        tracing, "Langfuse", side_effect=ConnectionError("unreachable")
    ) as mock_langfuse:
        tracing.get_langfuse_client()
        tracing.get_langfuse_client()

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

    with tracing.traced_query("conv-1", "query-1", "a question") as span:
        assert span is None


def test_traced_query_propagates_session_id_and_opens_a_span_with_the_question_as_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_span = MagicMock()
    fake_client.start_as_current_observation.return_value.__enter__.return_value = fake_span
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)
    with patch.object(tracing, "propagate_attributes") as mock_propagate:
        with tracing.traced_query("conv-1", "query-1", "a question") as span:
            assert span is fake_span

    mock_propagate.assert_called_once_with(session_id="conv-1", trace_name="query")
    fake_client.start_as_current_observation.assert_called_once_with(
        name="query",
        as_type="span",
        input="a question",
        metadata={"query_id": "query-1", "metadata_filter": None},
    )


def test_traced_query_attaches_the_metadata_filter_when_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # What makes the frontend's "Filters" panel selection visible on the
    # trace itself, at a glance, rather than only inferable from the
    # request that produced it.
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value.__enter__.return_value = MagicMock()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)
    metadata_filter = {"tags": ["runbook"], "author": None, "date_from": None, "date_to": None}

    with tracing.traced_query("conv-1", "query-1", "a question", metadata_filter):
        pass

    fake_client.start_as_current_observation.assert_called_once_with(
        name="query",
        as_type="span",
        input="a question",
        metadata={"query_id": "query-1", "metadata_filter": metadata_filter},
    )


def test_traced_query_can_attach_the_answer_as_output_via_update_span_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_span = MagicMock()
    fake_client.start_as_current_observation.return_value.__enter__.return_value = fake_span
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)

    with tracing.traced_query("conv-1", "query-1", "a question") as span:
        tracing.update_span_output(span, "the answer")

    fake_span.update.assert_called_once_with(output="the answer")


def test_traced_query_still_runs_the_body_when_opening_the_span_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value.__enter__.side_effect = ConnectionError(
        "unreachable"
    )
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)
    body_ran = False

    with tracing.traced_query("conv-1", "query-1", "a question") as span:
        body_ran = True  # must still happen -- the whole point of this fix
        assert span is None

    assert body_ran


def test_traced_query_swallows_a_failure_closing_the_span(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value.__exit__.side_effect = ConnectionError(
        "unreachable"
    )
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)

    with tracing.traced_query("conv-1", "query-1", "a question"):
        pass  # must not raise on exit either


def test_traced_query_still_propagates_a_real_exception_from_the_wrapped_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The critical safety property in the other direction: tracing must
    # never swallow a REAL error from the actual query -- only failures
    # in the tracing machinery itself.
    fake_client = MagicMock()
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)

    with pytest.raises(ValueError, match="real business error"):
        with tracing.traced_query("conv-1", "query-1", "a question"):
            raise ValueError("real business error")


def test_traced_span_yields_none_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: None)

    with tracing.traced_span("qdrant_search") as span:
        assert span is None


def test_traced_span_starts_an_observation_with_the_given_type_and_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_observation = MagicMock()
    fake_client.start_as_current_observation.return_value.__enter__.return_value = (
        fake_observation
    )
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)

    with tracing.traced_span(
        "qdrant_search", as_type="span", input="a query", metadata={"top_k": 5}
    ) as span:
        assert span is fake_observation

    fake_client.start_as_current_observation.assert_called_once_with(
        name="qdrant_search", as_type="span", input="a query", metadata={"top_k": 5}
    )


def test_traced_span_yields_none_when_opening_the_observation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value.__enter__.side_effect = ConnectionError(
        "unreachable"
    )
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)
    body_ran = False

    with tracing.traced_span("qdrant_search") as span:
        body_ran = True
        assert span is None

    assert body_ran


def test_traced_span_swallows_a_failure_closing_the_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value.__exit__.side_effect = ConnectionError(
        "unreachable"
    )
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)

    with tracing.traced_span("qdrant_search"):
        pass  # must not raise on exit


def test_update_span_output_is_a_noop_when_observation_is_none() -> None:
    tracing.update_span_output(None, output={"a": 1})  # must not raise


def test_update_span_output_attaches_output() -> None:
    observation = MagicMock()

    tracing.update_span_output(observation, output={"chunks": []})

    observation.update.assert_called_once_with(output={"chunks": []})


def test_update_span_output_swallows_a_failure_calling_update() -> None:
    observation = MagicMock()
    observation.update.side_effect = ConnectionError("unreachable")

    tracing.update_span_output(observation, output={"a": 1})  # must not raise


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


def test_traced_generation_yields_none_when_opening_the_observation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value.__enter__.side_effect = ConnectionError(
        "unreachable"
    )
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)
    body_ran = False

    with tracing.traced_generation("generate", "gpt-4o-mini", []) as generation:
        body_ran = True
        assert generation is None  # nothing to attach a result to

    assert body_ran


def test_traced_generation_swallows_a_failure_closing_the_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = MagicMock()
    fake_client.start_as_current_observation.return_value.__exit__.side_effect = ConnectionError(
        "unreachable"
    )
    monkeypatch.setattr(tracing, "get_langfuse_client", lambda: fake_client)

    with tracing.traced_generation("generate", "gpt-4o-mini", []):
        pass  # must not raise on exit


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


def test_record_generation_result_swallows_a_failure_calling_update() -> None:
    # This runs right after a real, successful LLM response comes back --
    # a Langfuse hiccup here must not be able to take that answer down.
    generation = MagicMock()
    generation.update.side_effect = ConnectionError("unreachable")
    response = SimpleNamespace(usage=None)

    tracing.record_generation_result(generation, response, output="the answer")  # must not raise
