"""POST /query/stream: same agentic turn as /query, surfaced as NDJSON
events instead of one blocking response. See routers/query.py's
_stream_response docstring for the event shapes -- this file exercises
the API's own plumbing (event framing, DB recording still happening,
the one status-code trade-off streaming makes), not the agent's
decomposition/looping behavior, which generation/test_agent.py already
covers directly.
"""

import json
import threading

import httpx
import pytest

import multimodal_rag.api.routers.query as query_module
from multimodal_rag.providers.base import LLMProvider
from multimodal_rag.providers.schema import TokenChunk, ToolCall, ToolCallsChunk, ToolResponse

from .conftest import ingest_sample_doc


class _FakeStreamingLLM(LLMProvider):
    """Simulates the minimal one-search agent turn, streamed: a tool call
    (echoing the latest user message as the search query) followed by a
    fixed final answer, delivered as text fragments -- mirrors
    test_query.py's _FakeLLM, but through the streaming provider methods
    /query/stream actually calls."""

    def __init__(self, response_tokens: list[str] | None = None) -> None:
        self._response_tokens = response_tokens or ["Fixed answer ", "⟦1⟧."]
        self._searched = False

    def generate(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("non-streaming generate() should not be called")

    def generate_with_tools(self, messages, tools, tool_choice=None) -> ToolResponse:
        raise AssertionError("non-streaming generate_with_tools() should not be called")

    def generate_with_tools_stream(self, messages, tools, tool_choice=None):
        if not self._searched:
            self._searched = True
            latest_message = messages[-1]["content"]
            yield ToolCallsChunk(
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        name="search_knowledge_base",
                        arguments={"query": latest_message},
                    )
                ]
            )
            return
        for text in self._response_tokens:
            yield TokenChunk(text=text)


class _FailingStreamingLLM(LLMProvider):
    def generate(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("non-streaming generate() should not be called")

    def generate_with_tools(self, messages, tools, tool_choice=None) -> ToolResponse:
        raise AssertionError("non-streaming generate_with_tools() should not be called")

    def generate_with_tools_stream(self, messages, tools, tool_choice=None):
        raise RuntimeError("upstream model unavailable")
        yield  # pragma: no cover -- makes this a generator function


class _TitleLLM(LLMProvider):
    def generate(self, messages: list[dict[str, str]]) -> str:
        return "A title"

    def generate_with_tools(self, messages, tools, tool_choice=None) -> ToolResponse:
        raise AssertionError("title generation should not call generate_with_tools")


@pytest.fixture(autouse=True)
def _fake_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    # ONE shared instance, not `lambda: _FakeStreamingLLM()` -- _run_rounds
    # calls get_llm() fresh every round (see agent.py), so a lambda that
    # constructs a new fake each time would reset `_searched` every round
    # instead of letting it persist across the turn.
    fake_llm = _FakeStreamingLLM()
    monkeypatch.setattr("multimodal_rag.generation.agent.get_llm", lambda: fake_llm)
    # A new conversation also triggers title generation (see
    # routers/query.py) -- generate_title() itself calls plain generate(),
    # not the streaming path, since title generation isn't streamed.
    monkeypatch.setattr("multimodal_rag.generation.title.get_llm", lambda: _TitleLLM())


async def _collect_ndjson_events(
    client: httpx.AsyncClient, payload: dict
) -> tuple[int, list[dict]]:
    async with client.stream("POST", "/query/stream", json=payload) as response:
        status_code = response.status_code
        events = [
            json.loads(line) async for line in response.aiter_lines() if line.strip()
        ]
    return status_code, events


async def test_query_stream_emits_tool_call_tokens_then_a_terminal_done(
    client: httpx.AsyncClient,
) -> None:
    await ingest_sample_doc(client)

    status_code, events = await _collect_ndjson_events(
        client,
        {
            "question": "How does local inference latency compare to the internal gateway?",
            "top_k": 3,
        },
    )

    assert status_code == 200
    assert events[-1]["type"] == "done"
    tool_call_events = [e for e in events if e["type"] == "tool_call"]
    token_events = [e for e in events if e["type"] == "token"]
    assert len(tool_call_events) == 1
    assert tool_call_events[0]["result_count"] >= 1
    assert "".join(e["text"] for e in token_events) == "Fixed answer ⟦1⟧."

    done = events[-1]
    assert done["answer"] == "Fixed answer ⟦1⟧."
    assert done["refused"] is False
    assert len(done["citations"]) == 1
    assert done["citations"][0]["marker"] == 1
    assert "query_id" in done
    assert "conversation_id" in done


async def test_query_stream_done_event_matches_query_response_shape(
    client: httpx.AsyncClient,
) -> None:
    """The "done" event should be a drop-in replacement for /query's JSON
    body -- same fields, so a client doesn't need two separate response
    parsers for the streaming vs. non-streaming endpoints."""
    from multimodal_rag.api.schemas import QueryResponse

    await ingest_sample_doc(client)

    status_code, events = await _collect_ndjson_events(
        client, {"question": "How does local inference latency compare?", "top_k": 3}
    )
    assert status_code == 200
    done = events[-1]
    assert done["type"] == "done"
    assert set(done.keys()) - {"type"} == set(QueryResponse.model_fields.keys())


async def test_query_stream_records_the_query_and_creates_a_conversation(
    client: httpx.AsyncClient,
) -> None:
    _, events = await _collect_ndjson_events(client, {"question": "anything"})
    done = events[-1]
    assert done["type"] == "done"
    conversation_id = done["conversation_id"]

    history_response = await client.get(f"/conversations/{conversation_id}")
    assert history_response.status_code == 200
    messages = history_response.json()["messages"]
    assert len(messages) == 1
    assert messages[0]["question"] == "anything"


async def test_query_stream_with_unknown_conversation_id_returns_404_before_streaming(
    client: httpx.AsyncClient,
) -> None:
    # This failure happens in _setup_query(), still inside the async
    # endpoint function before StreamingResponse exists -- so, unlike a
    # failure mid-stream, it's a normal HTTP 404 with a JSON body, not an
    # in-stream error event.
    response = await client.post(
        "/query/stream", json={"question": "anything", "conversation_id": "nonexistent"}
    )
    assert response.status_code == 404


async def test_bridge_sync_stream_runs_the_generator_on_a_single_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for a real bug: handing a plain sync generator
    straight to StreamingResponse makes Starlette dispatch EACH next()
    call separately (iterate_in_threadpool), potentially on a different
    worker thread every time -- timing/load-dependent, so not reliably
    reproducible through a single isolated HTTP request in a test (a
    from-the-endpoint version of this test passed even against the
    un-fixed code, since a quiet test run's thread pool happened to reuse
    one idle thread throughout). It WAS 100% reproducible against a real
    running server under real traffic: traced_query()/traced_generation()
    (tracing.py) hold an OpenTelemetry span open across many yields, and
    context attach/detach must happen on the exact thread it was entered
    on -- split across pool threads, every detach failed ("Token was
    created in a different Context"), confirmed live against a real
    Langfuse Cloud project.

    This tests the actual fix directly and deterministically instead:
    _bridge_sync_stream() must run whatever generator function it's
    given to completion on one dedicated background thread, full stop --
    a property of _bridge_sync_stream itself, independent of pool timing
    or how many requests happen to be in flight.
    """
    thread_ids: list[int] = []

    def _fake_stream_response(request, db, principal, setup):
        for i in range(25):
            thread_ids.append(threading.get_ident())
            yield f"line-{i}"

    monkeypatch.setattr(query_module, "_stream_response", _fake_stream_response)

    items = [
        item async for item in query_module._bridge_sync_stream(None, None, None, None)  # type: ignore[arg-type]
    ]

    assert items == [f"line-{i}" for i in range(25)]
    assert len(set(thread_ids)) == 1, (
        f"the wrapped generator ran on {len(set(thread_ids))} different threads across its "
        "yields -- exactly the condition that breaks OpenTelemetry context attach/detach"
    )


async def test_query_stream_endpoint_actually_uses_the_threading_bridge(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Closes the gap the test above leaves open: that _bridge_sync_stream
    is correct in isolation doesn't guarantee query_stream() still calls
    it -- someone could revert to wiring _stream_response directly into
    StreamingResponse (the original bug) without breaking that test at
    all. This pins down that the real endpoint still routes through it."""
    calls = []
    real_bridge = query_module._bridge_sync_stream

    def _spy_bridge(*args, **kwargs):
        calls.append(args)
        return real_bridge(*args, **kwargs)

    monkeypatch.setattr(query_module, "_bridge_sync_stream", _spy_bridge)

    status_code, events = await _collect_ndjson_events(client, {"question": "anything"})

    assert status_code == 200
    assert events[-1]["type"] == "done"
    assert len(calls) == 1


async def test_query_stream_emits_an_error_event_when_llm_provider_fails(
    client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "multimodal_rag.generation.agent.get_llm", lambda: _FailingStreamingLLM()
    )

    status_code, events = await _collect_ndjson_events(client, {"question": "anything"})

    # Headers (200 + the NDJSON content type) are already sent by the
    # time the agent turn can fail -- see _stream_response's docstring --
    # so the failure surfaces as the stream's last line, not the status
    # code /query would have used for the same failure (503).
    assert status_code == 200
    assert events[-1]["type"] == "error"
    assert "Query generation failed" in events[-1]["detail"]
