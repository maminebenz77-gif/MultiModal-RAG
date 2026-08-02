"""POST /feedback: thumbs up/down + optional comment on a prior query."""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ..db import Database, QueryNotFoundError
from ..dependencies import get_db
from ..schemas import FeedbackRequest, FeedbackResponse

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest, db: Database = Depends(get_db)
) -> FeedbackResponse:
    try:
        feedback_id = await run_in_threadpool(
            db.record_feedback, request.query_id, request.rating, request.comment
        )
    except QueryNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return FeedbackResponse(feedback_id=feedback_id, status="recorded")
