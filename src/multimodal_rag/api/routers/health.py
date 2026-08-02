"""GET /health: pings the two backing stores. Deliberately does NOT call
the LLM/embedding providers -- those cost real time/money per call, and
their reachability is orthogonal to "is this API process up."
"""

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ...stores.base import KeywordStore, VectorStore
from ..dependencies import get_keyword_store, get_vector_store
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    vector_store: VectorStore = Depends(get_vector_store),
    keyword_store: KeywordStore = Depends(get_keyword_store),
) -> HealthResponse:
    qdrant_up = await run_in_threadpool(vector_store.ping)
    elasticsearch_up = await run_in_threadpool(keyword_store.ping)
    status = "ok" if (qdrant_up and elasticsearch_up) else "degraded"
    return HealthResponse(
        status=status,
        qdrant="up" if qdrant_up else "down",
        elasticsearch="up" if elasticsearch_up else "down",
    )
