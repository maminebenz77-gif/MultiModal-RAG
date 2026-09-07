"""GET /documents: list previously ingested documents.
DELETE /documents: wipe the whole corpus -- every chunk in both stores,
every row in the documents table. DELETE /documents/{doc_id}: remove
just one document. Query/feedback history is untouched either way (see
schemas.WipeResponse / DocumentDeleteResponse)."""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ...stores.indexer import HybridIndexer
from ..db import Database
from ..dependencies import get_db, get_indexer
from ..schemas import DocumentDeleteResponse, DocumentsResponse, WipeResponse

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


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    doc_id: str,
    indexer: HybridIndexer = Depends(get_indexer),
    db: Database = Depends(get_db),
) -> DocumentDeleteResponse:
    existing = await run_in_threadpool(db.get_document, doc_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No document with id={doc_id!r}")
    chunks_deleted = await run_in_threadpool(indexer.delete_document, doc_id)
    await run_in_threadpool(db.delete_document, doc_id)
    return DocumentDeleteResponse(status="deleted", doc_id=doc_id, chunks_deleted=chunks_deleted)
