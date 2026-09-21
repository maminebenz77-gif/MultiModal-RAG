"""GET /documents: list previously ingested documents.
DELETE /documents: wipe the whole corpus -- every chunk in both stores,
every row in the documents table. DELETE /documents/{doc_id}: remove
just one document. PATCH /documents/{doc_id}: edit a document's tags
(classification, author, ...) WITHOUT touching its chunks -- no
re-embedding, since a payload patch is all a metadata-only change ever
needs (see stores.indexer.HybridIndexer.set_document_metadata). Query/
feedback history is untouched by any of these (see schemas.WipeResponse
/ DocumentDeleteResponse).

Every endpoint is scoped to the caller. A document the caller cannot see
is a 404, exactly like one that doesn't exist -- never a 403, which would
confirm it's there. A 403 is reserved for documents the caller CAN see
but doesn't own (editing/deleting is owner-or-admin), and for the
admin-only wipe."""

from fastapi import APIRouter, Depends, HTTPException
from starlette.concurrency import run_in_threadpool

from ...identity import Principal
from ...metadata import DocumentMetadata
from ...stores.indexer import HybridIndexer
from ..db import AccessDeniedError, Database
from ..dependencies import get_db, get_indexer
from ..identity import get_principal
from ..schemas import (
    DocumentDeleteResponse,
    DocumentMetadataUpdate,
    DocumentsResponse,
    DocumentSummary,
    WipeResponse,
)

router = APIRouter()


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    db: Database = Depends(get_db), principal: Principal = Depends(get_principal)
) -> DocumentsResponse:
    documents = await run_in_threadpool(db.list_documents, principal)
    return DocumentsResponse(documents=documents)


@router.delete("/documents", response_model=WipeResponse)
async def wipe_documents(
    indexer: HybridIndexer = Depends(get_indexer),
    db: Database = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> WipeResponse:
    # Checked BEFORE touching either store -- the database re-enforces it
    # too, but by then the chunks would already be gone.
    if not principal.is_admin:
        raise HTTPException(status_code=403, detail="Only an admin may wipe the whole corpus")
    chunks_deleted = await run_in_threadpool(indexer.delete_all)
    documents_deleted = await run_in_threadpool(db.wipe_documents, principal)
    return WipeResponse(
        status="wiped",
        documents_deleted=documents_deleted,
        chunks_deleted=chunks_deleted,
    )


@router.patch("/documents/{doc_id}", response_model=DocumentSummary)
async def update_document_metadata(
    doc_id: str,
    patch: DocumentMetadataUpdate,
    indexer: HybridIndexer = Depends(get_indexer),
    db: Database = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> DocumentSummary:
    existing = await run_in_threadpool(db.get_document, principal, doc_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No document with id={doc_id!r}")

    # exclude_unset, not "which fields aren't None" -- a caller sending
    # private=false must actually flip it, not be indistinguishable from
    # not mentioning `private` at all.
    changed_fields = patch.model_dump(exclude_unset=True)
    if not changed_fields:
        return existing

    try:
        updated = await run_in_threadpool(
            db.update_document_metadata, principal, doc_id, changed_fields
        )
    except AccessDeniedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    assert updated is not None  # existence already confirmed above

    # NOT changed_fields -- the FULL merged row's payload. Most fields
    # are independent and a partial patch would be fine to send as-is,
    # but acl_allow (see DocumentMetadata.to_payload()) is DERIVED from
    # BOTH private and owner together: patching just one of those two
    # without recomputing acl_allow would leave the store's copy
    # granting access based on a combination that's no longer what
    # sqlite actually says. Recomputing from the merged row -- rather
    # than trying to special-case "did this patch touch private or
    # owner" -- means this can never be forgotten as a new derived field
    # is added later.
    # model_dump(), not a hand-written field list: a list silently drops
    # any field added later -- and a dropped `status` would reset a
    # superseded document to "current" in the stores on its next PATCH.
    full_metadata = DocumentMetadata(**updated.metadata.model_dump())
    await run_in_threadpool(indexer.set_document_metadata, doc_id, full_metadata.to_payload())
    return updated


@router.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(
    doc_id: str,
    indexer: HybridIndexer = Depends(get_indexer),
    db: Database = Depends(get_db),
    principal: Principal = Depends(get_principal),
) -> DocumentDeleteResponse:
    existing = await run_in_threadpool(db.get_document, principal, doc_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"No document with id={doc_id!r}")
    # Checked BEFORE deleting any chunks -- the database re-enforces it
    # too, but by then the chunks would already be gone.
    if not principal.can_modify(existing.metadata.owner):
        raise HTTPException(status_code=403, detail=f"Not allowed to delete {doc_id!r}")
    chunks_deleted = await run_in_threadpool(indexer.delete_document, doc_id)
    await run_in_threadpool(db.delete_document, principal, doc_id)
    return DocumentDeleteResponse(status="deleted", doc_id=doc_id, chunks_deleted=chunks_deleted)
