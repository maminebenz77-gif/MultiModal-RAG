"""GET /documents: list previously ingested documents."""

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ..db import Database
from ..dependencies import get_db
from ..schemas import DocumentsResponse

router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(db: Database = Depends(get_db)) -> DocumentsResponse:
    documents = await run_in_threadpool(db.list_documents)
    return DocumentsResponse(documents=documents)
