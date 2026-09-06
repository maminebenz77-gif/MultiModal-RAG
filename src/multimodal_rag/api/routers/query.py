"""POST /query: agentic retrieve -> generate -> answer with citations.

retrieval_method/top_k are per-request here even though AgentChain
normally fixes them at construction time (see the demo) -- constructing
an AgentChain is cheap (it just wraps references, no I/O), so a fresh
one per request is the simplest way to let each call choose its own
method/top_k against the one shared Retriever singleton.
"""

import uuid
from contextlib import contextmanager

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ...config import get_settings
from ...device import resolve_device
from ...generation.agent import AgentChain
from ...providers.base import LLMProvider
from ...providers.factory import embedder_from_override, llm_from_override
from ...retrieval.retriever import Retriever
from ..db import Database
from ..dependencies import get_app_state, get_db, get_retriever
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

    original_agent_llm = agent_module.get_llm
    agent_module.get_llm = lambda: llm
    try:
        yield
    finally:
        agent_module.get_llm = original_agent_llm


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


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    db: Database = Depends(get_db),
) -> QueryResponse:
    settings = get_settings()
    allow_external = settings.allow_external
    device = resolve_device(settings.device)

    try:
        retriever_for_request = _build_retriever_for_request(
            request,
            retriever,
            device=device,
            allow_external=allow_external,
        )
    except (ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.conversation_id is None:
        conversation_id = await run_in_threadpool(db.create_conversation)
    else:
        exists = await run_in_threadpool(db.conversation_exists, request.conversation_id)
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"No conversation with id={request.conversation_id!r}",
            )
        conversation_id = request.conversation_id
    history = await run_in_threadpool(db.get_recent_turns, conversation_id, _HISTORY_WINDOW)

    agent = AgentChain(
        retriever_for_request,
        method=request.retrieval_method,
        top_k=request.top_k,
        rerank=request.rerank,
        resolve_parent_context=True,
    )

    try:
        overrides = request.runtime_overrides
        if overrides is not None and overrides.llm is not None:
            chosen: ProviderOverride = overrides.llm
            llm = llm_from_override(
                provider=chosen.provider,
                model=chosen.model,
                base_url=chosen.base_url,
                api_key=chosen.api_key,
                allow_external=allow_external,
            )

            def _answer_with_override():
                with _temporary_llm_provider(llm):
                    return agent.answer(request.question, history, request.doc_ids)

            result = await run_in_threadpool(_answer_with_override)
        else:
            result = await run_in_threadpool(
                agent.answer, request.question, history, request.doc_ids
            )
    except ValueError as exc:
        # Retriever._rerank raises this when rerank=True but no Reranker
        # is configured for this deployment -- a config gap, not a bad
        # request shape, but still the client's rerank=True that triggered it.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Model/provider/network failures should not leak as raw 500s.
        raise HTTPException(status_code=503, detail=f"Query generation failed: {exc}") from exc

    query_id = str(uuid.uuid4())
    await run_in_threadpool(
        db.record_query,
        query_id,
        request.question,
        result.answer,
        result.refused,
        request.retrieval_method.value,
        conversation_id=conversation_id,
        needs_clarification=result.needs_clarification,
        citations=result.citations,
    )

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
                pages=c.pages,
                slides=c.slides,
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
                pages=c.pages,
                slides=c.slides,
                elements=[
                    ChunkElementOut(
                        type=e.type,
                        text=e.text,
                        image_base64=e.image_base64,
                        description=e.description,
                        page=e.page,
                        slide=e.slide,
                    )
                    for e in c.elements
                ],
            )
            for c in result.retrieved_chunks
        ],
    )
