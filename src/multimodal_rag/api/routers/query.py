"""POST /query: retrieve -> generate -> answer with citations.

retrieval_method/top_k are per-request here even though RagChain
normally fixes them at construction time (see the demo) -- constructing
a RagChain is cheap (it just wraps references and builds a LangChain
pipeline, no I/O), so a fresh one per request is the simplest way to
let each call choose its own method/top_k against the one shared
Retriever singleton.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ...generation.chain import RagChain
from ...retrieval.retriever import Retriever
from ..db import Database
from ..dependencies import get_db, get_retriever
from ..schemas import CitationOut, QueryRequest, QueryResponse, RetrievedChunkOut

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    retriever: Retriever = Depends(get_retriever),
    db: Database = Depends(get_db),
) -> QueryResponse:
    chain = RagChain(
        retriever,
        method=request.retrieval_method,
        top_k=request.top_k,
        rerank=request.rerank,
        resolve_parent_context=True,
    )
    try:
        result = await run_in_threadpool(chain.answer, request.question, None, request.doc_ids)
    except ValueError as exc:
        # Retriever._rerank raises this when rerank=True but no Reranker
        # is configured for this deployment -- a config gap, not a bad
        # request shape, but still the client's rerank=True that triggered it.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    query_id = str(uuid.uuid4())
    await run_in_threadpool(
        db.record_query,
        query_id,
        request.question,
        result.answer,
        result.refused,
        request.retrieval_method.value,
    )

    return QueryResponse(
        query_id=query_id,
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
        retrieval_method=request.retrieval_method,
        retrieved_chunks=[
            RetrievedChunkOut(
                chunk_id=c.chunk_id,
                score=c.score,
                text=c.text,
                source=c.source,
                pages=c.pages,
                slides=c.slides,
            )
            for c in result.retrieved_chunks
        ],
    )
