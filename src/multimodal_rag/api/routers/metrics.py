"""GET /metrics: a plain JSON summary of documents/queries/feedback so
far, derived from the sqlite tables -- not a Prometheus exposition
endpoint, which would be real scope creep for what this project needs.
"""

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ..db import Database
from ..dependencies import get_db
from ..schemas import MetricsResponse

router = APIRouter()


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(db: Database = Depends(get_db)) -> MetricsResponse:
    return await run_in_threadpool(db.metrics)
