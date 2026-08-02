"""POST /query: retrieve -> generate -> answer with citations.

retrieval_method/top_k are per-request here even though RagChain
normally fixes them at construction time (see the demo) -- constructing
a RagChain is cheap (it just wraps references and builds a LangChain
pipeline, no I/O), so a fresh one per request is the simplest way to
let each call choose its own method/top_k against the one shared
Retriever singleton.
"""

import uuid

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ...generation.chain import RagChain
from ...retrieval.retriever import Retriever
from ..db import Database
from ..dependencies import get_db, get_retriever
from ..schemas import CitationOut, QueryRequest, QueryResponse

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
        resolve_parent_context=True,
    )
    result = await run_in_threadpool(chain.answer, request.question, None, request.doc_ids)

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
    )
