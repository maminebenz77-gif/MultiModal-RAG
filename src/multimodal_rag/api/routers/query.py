"""POST /query: agentic retrieve -> generate -> answer with citations.
POST /query/stream: the same agent turn, surfaced as newline-delimited
JSON (NDJSON) events as they happen instead of one blocking response --
see _stream_response's docstring for the event shapes and the one real
trade-off streaming makes (no HTTP error status once the stream has
started), and _bridge_sync_stream's docstring for why it isn't handed to
StreamingResponse directly.

retrieval_method/top_k are per-request here even though AgentChain
normally fixes them at construction time (see the demo) -- constructing
an AgentChain is cheap (it just wraps references, no I/O), so a fresh
one per request is the simplest way to let each call choose its own
method/top_k against the one shared Retriever singleton.
"""

import json
import queue
import threading
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from ...chunking.schema import ChunkElement
from ...config import get_settings
from ...device import resolve_device
from ...generation.agent import AgentChain
from ...generation.schema import AgentDone, AgentToken, AgentToolCall, RagAnswer
from ...generation.title import generate_title
from ...identity import Principal
from ...providers.base import LLMProvider
from ...providers.factory import embedder_from_override, llm_from_override
from ...retrieval.retriever import Retriever
from ...retrieval.scoped import ScopedRetriever
from ...tracing import traced_query, update_span_output
from ..db import Database
from ..dependencies import get_app_state, get_db, get_retriever
from ..identity import get_principal
from ..schemas import (
    ChunkElementOut,
    CitationOut,
    ProviderOverride,
    QueryRequest,
    QueryResponse,
    RetrievedChunkOut,
)

router = APIRouter()

_HISTORY_WINDOW = 10
"""How many prior turns of a conversation get fed to the agent -- was
previously a client-side max_length=10 on the request body; now that
the server owns the full conversation, it's a windowed read instead."""


@contextmanager
def _temporary_llm_provider(llm: LLMProvider):
    from ...generation import agent as agent_module
    from ...generation import title as title_module

    original_agent_llm = agent_module.get_llm
    original_title_llm = title_module.get_llm
    agent_module.get_llm = lambda: llm
    title_module.get_llm = lambda: llm
    try:
        yield
    finally:
        agent_module.get_llm = original_agent_llm
        title_module.get_llm = original_title_llm


def _to_chunk_element_out(e: ChunkElement) -> ChunkElementOut:
    return ChunkElementOut(
        type=e.type,
        text=e.text,
        image_base64=e.image_base64,
        description=e.description,
        page=e.page,
        slide=e.slide,
    )


def _to_query_response(
    query_id: str, conversation_id: str, request: QueryRequest, result: RagAnswer
) -> QueryResponse:
    return QueryResponse(
        query_id=query_id,
        conversation_id=conversation_id,
        question=request.question,
        answer=result.answer,
        citations=[
            CitationOut(
                marker=c.marker,
                chunk_id=c.chunk_id,
                source=c.source,
                doc_id=c.doc_id,
                pages=c.pages,
                slides=c.slides,
                text=c.text,
                elements=[_to_chunk_element_out(e) for e in c.elements],
            )
            for c in result.citations
        ],
        refused=result.refused,
        needs_clarification=result.needs_clarification,
        retrieval_method=request.retrieval_method,
        retrieved_chunks=[
            RetrievedChunkOut(
                chunk_id=c.chunk_id,
                score=c.score,
                text=c.text,
                source=c.source,
                doc_id=c.doc_id,
                pages=c.pages,
                slides=c.slides,
                elements=[_to_chunk_element_out(e) for e in c.elements],
            )
            for c in result.retrieved_chunks
        ],
    )


def _build_retriever_for_request(
    request: QueryRequest,
    default_retriever: Retriever,
    *,
    device: str,
    allow_external: bool,
) -> Retriever:
    overrides = request.runtime_overrides
    if overrides is None or overrides.embedder is None:
        return default_retriever

    chosen: ProviderOverride = overrides.embedder
    embedder = embedder_from_override(
        provider=chosen.provider,
        model=chosen.model,
        base_url=chosen.base_url,
        api_key=chosen.api_key,
        allow_external=allow_external,
        device=device,
    )
    return Retriever(
        default_retriever._vector_store,
        default_retriever._keyword_store,
        embedder,
        reranker=default_retriever._reranker,
    )


@dataclass
class _QuerySetup:
    """Everything both /query and /query/stream need before they can
    actually run the agent turn -- built once, by _setup_query(), so the
    two endpoints can't drift apart on how a runtime-override embedder,
    a new-vs-existing conversation, or the metadata filter get resolved."""

    agent: AgentChain
    llm_override: LLMProvider | None
    search_filter: object | None
    history: list[tuple[str, str]]
    conversation_id: str
    is_new_conversation: bool
    query_id: str
    metadata_filter_for_trace: dict | None


async def _setup_query(
    request: QueryRequest, retriever: Retriever, db: Database, principal: Principal
) -> _QuerySetup:
    settings = get_settings()
    allow_external = settings.allow_external
    device = resolve_device(settings.device)

    try:
        retriever_for_request = _build_retriever_for_request(
            request, retriever, device=device, allow_external=allow_external
        )
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ScopedRetriever wraps whichever Retriever this request ends up
    # using (default, or one rebuilt for an embedder override above) --
    # built fresh per request, from THIS caller's principal, never
    # reused across requests the way the underlying Retriever singleton
    # is. Handed to AgentChain in place of the plain Retriever below: it
    # satisfies the same RetrieverLike shape, so nothing about AgentChain
    # itself needs to know scoping exists.
    scoped_retriever = ScopedRetriever(
        retriever_for_request, principal, db, include_superseded=request.include_superseded
    )

    if request.conversation_id is None:
        conversation_id = await run_in_threadpool(db.create_conversation, principal)
        is_new_conversation = True
    else:
        exists = await run_in_threadpool(
            db.conversation_exists, principal, request.conversation_id
        )
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"No conversation with id={request.conversation_id!r}",
            )
        conversation_id = request.conversation_id
        is_new_conversation = False
    history = await run_in_threadpool(
        db.get_recent_turns, principal, conversation_id, _HISTORY_WINDOW
    )

    agent = AgentChain(
        scoped_retriever,
        method=request.retrieval_method,
        top_k=request.top_k,
        rerank=request.rerank,
        resolve_parent_context=True,
    )

    llm_override: LLMProvider | None = None
    overrides = request.runtime_overrides
    if overrides is not None and overrides.llm is not None:
        chosen: ProviderOverride = overrides.llm
        llm_override = llm_from_override(
            provider=chosen.provider,
            model=chosen.model,
            base_url=chosen.base_url,
            api_key=chosen.api_key,
            allow_external=allow_external,
        )

    search_filter = (
        request.metadata_filter.to_search_filter() if request.metadata_filter is not None else None
    )
    metadata_filter_for_trace = (
        request.metadata_filter.model_dump() if request.metadata_filter is not None else None
    )

    return _QuerySetup(
        agent=agent,
        llm_override=llm_override,
        search_filter=search_filter,
        history=history,
        conversation_id=conversation_id,
        is_new_conversation=is_new_conversation,
        query_id=str(uuid.uuid4()),
        metadata_filter_for_trace=metadata_filter_for_trace,
    )


def _record_query_and_maybe_title(
    db: Database,
    principal: Principal,
    request: QueryRequest,
    setup: _QuerySetup,
    result: RagAnswer,
    latency_ms: float,
) -> None:
    """The bookkeeping that follows a successful agent turn, shared
    verbatim by /query (via a threadpool -- see its caller) and
    /query/stream (called directly -- its generator already runs off the
    event loop thread, courtesy of StreamingResponse)."""
    db.record_query(
        principal,
        setup.query_id,
        request.question,
        result.answer,
        result.refused,
        request.retrieval_method.value,
        conversation_id=setup.conversation_id,
        needs_clarification=result.needs_clarification,
        citations=result.citations,
        latency_ms=latency_ms,
    )

    if not setup.is_new_conversation:
        return
    # Nice-to-have, not core to the response -- generate_title() is
    # fail-soft (returns None rather than raising) and a missing title
    # just leaves the picker showing the raw first question. Reuses this
    # request's LLM override, if any, so the title comes from the same
    # model the answer did rather than silently falling back to the .env
    # default.
    llm_context: AbstractContextManager = (
        _temporary_llm_provider(setup.llm_override)
        if setup.llm_override is not None
        else nullcontext()
    )
    with llm_context:
        title = generate_title(request.question, result.answer)
    if title:
        db.set_conversation_title(principal, setup.conversation_id, title)


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    db: Database = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> QueryResponse:
    setup = await _setup_query(request, retriever, db, principal)

    start_time = time.perf_counter()
    try:
        with traced_query(
            setup.conversation_id, setup.query_id, request.question, setup.metadata_filter_for_trace
        ) as query_span:
            if setup.llm_override is not None:

                def _answer_with_override():
                    with _temporary_llm_provider(setup.llm_override):
                        return setup.agent.answer(
                            request.question, setup.history, request.doc_ids, setup.search_filter
                        )

                result = await run_in_threadpool(_answer_with_override)
            else:
                result = await run_in_threadpool(
                    setup.agent.answer,
                    request.question,
                    setup.history,
                    request.doc_ids,
                    setup.search_filter,
                )
            update_span_output(query_span, result.answer)
    except ValueError as exc:
        # Retriever._rerank raises this when rerank=True but no Reranker
        # is configured for this deployment -- a config gap, not a bad
        # request shape, but still the client's rerank=True that triggered it.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Model/provider/network failures should not leak as raw 500s.
        raise HTTPException(status_code=503, detail=f"Query generation failed: {exc}") from exc
    latency_ms = (time.perf_counter() - start_time) * 1000
    # Measured around the WHOLE agent.answer() call, not a single LLM
    # generation -- a compound question can trigger multiple search/
    # generate rounds (see agent.py's max_tool_rounds), and "latency of
    # this query" means the full round-trip the caller actually waited
    # through, not just its last completion call.

    await run_in_threadpool(
        _record_query_and_maybe_title, db, principal, request, setup, result, latency_ms
    )

    return _to_query_response(setup.query_id, setup.conversation_id, request, result)


def _ndjson(payload: dict) -> str:
    return json.dumps(payload) + "\n"


def _stream_response(
    request: QueryRequest, db: Database, principal: Principal, setup: _QuerySetup
) -> Iterator[str]:
    """The /query/stream generator: one NDJSON object per line --
    {"type": "tool_call", "round": int, "query": str, "result_count": int}
    each time a search executes, {"type": "token", "round": int, "text":
    str} for each fragment of generated text, and exactly one final
    {"type": "done", ...} carrying the same fields /query's JSON body
    would (query_id, answer, citations, retrieved_chunks, ...) -- or, if
    the turn fails partway through, {"type": "error", "detail": str}.

    Runs as a plain sync generator: Starlette's StreamingResponse already
    iterates a sync generator off the event loop thread (the same
    guarantee run_in_threadpool gives /query above), so the blocking
    agent/DB calls in here are safe without wrapping each one
    individually.

    The one real trade-off of streaming: by the time anything below can
    fail, the 200 response and its headers are already sent, so unlike
    /query, a failure here can never become an HTTP 4xx/5xx -- it can
    only become an in-stream {"type": "error", ...} line for the caller
    to check for. There's no way around that once the first byte of a
    streamed body is on the wire.
    """
    start_time = time.perf_counter()
    llm_context: AbstractContextManager = (
        _temporary_llm_provider(setup.llm_override)
        if setup.llm_override is not None
        else nullcontext()
    )
    try:
        with (
            traced_query(
                setup.conversation_id,
                setup.query_id,
                request.question,
                setup.metadata_filter_for_trace,
            ) as query_span,
            llm_context,
        ):
            result: RagAnswer | None = None
            for event in setup.agent.answer_stream(
                request.question, setup.history, request.doc_ids, setup.search_filter
            ):
                if isinstance(event, AgentToolCall):
                    yield _ndjson(
                        {
                            "type": "tool_call",
                            "round": event.round_index,
                            "query": event.query,
                            "result_count": len(event.results),
                        }
                    )
                elif isinstance(event, AgentToken):
                    yield _ndjson(
                        {"type": "token", "round": event.round_index, "text": event.text}
                    )
                else:
                    assert isinstance(event, AgentDone)
                    result = event.result
            assert result is not None, "answer_stream() ended without yielding AgentDone"
            update_span_output(query_span, result.answer)
    except Exception as exc:
        # Mirrors /query's ValueError-> 400 / Exception -> 503 split in
        # spirit (a config gap like rerank=True with no Reranker
        # configured vs. a genuine model/provider/network failure), but
        # as a single event type -- see the trade-off in this function's
        # docstring for why a real status code isn't available here.
        yield _ndjson({"type": "error", "detail": f"Query generation failed: {exc}"})
        return

    latency_ms = (time.perf_counter() - start_time) * 1000
    _record_query_and_maybe_title(db, principal, request, setup, result, latency_ms)

    response = _to_query_response(setup.query_id, setup.conversation_id, request, result)
    yield _ndjson({"type": "done", **response.model_dump(mode="json")})


_STREAM_DONE = object()
"""Sentinel telling _bridge_sync_stream's consumer loop the producer
thread has finished -- a plain None can't be used since it's also a
value queue.Queue can legitimately hold."""


def _drain_into_queue(
    request: QueryRequest,
    db: Database,
    principal: Principal,
    setup: _QuerySetup,
    out_queue: "queue.Queue[str | BaseException | object]",
) -> None:
    """Runs _stream_response() to completion on the thread this function
    is given -- see _bridge_sync_stream's docstring for why. A `finally`
    guarantees the sentinel always lands even if the generator raises
    something _stream_response's own try/except didn't already turn into
    an {"type": "error", ...} line -- otherwise the consumer below would
    block on out_queue.get() forever."""
    try:
        for line in _stream_response(request, db, principal, setup):
            out_queue.put(line)
    except BaseException as exc:  # noqa: BLE001 -- relayed to the consumer, not swallowed
        out_queue.put(exc)
    finally:
        out_queue.put(_STREAM_DONE)


async def _bridge_sync_stream(
    request: QueryRequest, db: Database, principal: Principal, setup: _QuerySetup
) -> AsyncIterator[str]:
    """Runs _stream_response() on ONE dedicated background thread instead
    of handing it to StreamingResponse directly, and relays its output
    through a queue.Queue.

    The difference matters: StreamingResponse, given a plain *sync*
    generator, iterates it via Starlette's iterate_in_threadpool() --
    which dispatches EACH individual next() call separately to a worker
    thread from its pool, not necessarily the same one twice in a row.
    _stream_response() holds traced_query()/traced_generation()
    (tracing.py) open across many `yield` points (once per tool call,
    once per token) -- OpenTelemetry's context attach/detach uses a
    Token that MUST be detached on the exact same thread (technically
    the same contextvars.Context) it was attached on. Split across pool
    threads, every detach silently failed ("Failed to detach context...
    Token was created in a different Context", caught and logged by
    tracing.py's _safe_exit, never raised into the request) -- harmless
    to the response itself, but it meant a streaming query's Langfuse
    trace closed incorrectly, if it reported at all. Confirmed live
    against a real Langfuse Cloud project: every /query/stream call
    logged that failure twice (one traced_generation span per round);
    /query never has this problem because run_in_threadpool() already
    runs its single blocking agent.answer() call start-to-finish on one
    assigned thread.

    Running the whole generator on one thread here gives /query/stream
    that identical guarantee: every attach/detach pair happens on the
    thread that entered it, because nothing about the generator's own
    execution is ever paused and resumed on a different one. The queue
    hand-off itself never touches OpenTelemetry, so it's exempt from the
    problem it exists to route around.
    """
    q: "queue.Queue[str | BaseException | object]" = queue.Queue()
    thread = threading.Thread(
        target=_drain_into_queue, args=(request, db, principal, setup, q), daemon=True
    )
    thread.start()
    while True:
        item = await run_in_threadpool(q.get)
        if item is _STREAM_DONE:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


@router.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    db: Database = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> StreamingResponse:
    setup = await _setup_query(request, retriever, db, principal)
    return StreamingResponse(
        _bridge_sync_stream(request, db, principal, setup), media_type="application/x-ndjson"
    )
