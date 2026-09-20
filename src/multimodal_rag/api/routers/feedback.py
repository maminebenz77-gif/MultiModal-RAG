"""POST /feedback: thumbs up/down + optional comment on a prior query."""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ...identity import Principal
from ..db import Database, QueryNotFoundError
from ..dependencies import get_db
from ..identity import get_principal
from ..schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    db: Database = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> FeedbackResponse:
    # QueryNotFoundError covers "no such query" AND "someone else's
    # query" identically -- a 404 that differed between the two would let
    # a caller probe which query ids exist.
    try:
        feedback_id = await run_in_threadpool(
            db.record_feedback, principal, request.query_id, request.rating, request.comment
        )
    except QueryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FeedbackResponse(feedback_id=feedback_id, status="recorded")
