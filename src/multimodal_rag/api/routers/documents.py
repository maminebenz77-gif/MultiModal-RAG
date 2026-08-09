"""GET /documents: list previously ingested documents.
DELETE /documents: wipe the whole corpus -- every chunk in both stores,
every row in the documents table. Query/feedback history is untouched
(see schemas.WipeResponse)."""

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from ...stores.indexer import HybridIndexer
from ..db import Database
from ..dependencies import get_db, get_indexer
from ..schemas import DocumentsResponse, WipeResponse

router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(db: Database = Depends(get_db)) -> DocumentsResponse:
    documents = await run_in_threadpool(db.list_documents)
    return DocumentsResponse(documents=documents)


@router.delete("/documents", response_model=WipeResponse)
async def wipe_documents(
    indexer: HybridIndexer = Depends(get_indexer),
    db: Database = Depends(get_db),
) -> WipeResponse:
    chunks_deleted = await run_in_threadpool(indexer.delete_all)
    documents_deleted = await run_in_threadpool(db.wipe_documents)
    return WipeResponse(
        status="wiped",
        documents_deleted=documents_deleted,
        chunks_deleted=chunks_deleted,
    )
