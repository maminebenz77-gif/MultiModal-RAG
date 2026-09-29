"""GET /health: pings the backing store. Deliberately does NOT call
the LLM/embedding providers -- those cost real time/money per call, and
their reachability is orthogonal to "is this API process up."
"""

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ...stores.base import KeywordStore
from ..dependencies import get_keyword_store
from ..schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(keyword_store: KeywordStore = Depends(get_keyword_store)) -> HealthResponse:
    # Only one ping, not two -- vector_store and keyword_store are two
    # role-views onto the SAME Elasticsearch backend (see
    # stores/elasticsearch_store.py), so pinging both would just ping
    # the identical connection twice and always agree.
    elasticsearch_up = await run_in_threadpool(keyword_store.ping)
    return HealthResponse(
        status="ok" if elasticsearch_up else "degraded",
        elasticsearch="up" if elasticsearch_up else "down",
    )
