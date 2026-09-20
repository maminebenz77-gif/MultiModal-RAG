"""GET /metrics: a plain JSON summary of documents/queries/feedback so
far, derived from the sqlite tables -- not a Prometheus exposition
endpoint, which would be real scope creep for what this project needs.

Scoped to the caller: an admin sees the whole system, everyone else sees
numbers about their own conversations and the documents they can see.
"""

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ...identity import Principal
from ..db import Database
from ..dependencies import get_db
from ..identity import get_principal
from ..schemas import MetricsResponse

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    db: Database = Depends(get_db), principal: Principal = Depends(get_principal)
) -> MetricsResponse:
    return await run_in_threadpool(db.metrics, principal)
