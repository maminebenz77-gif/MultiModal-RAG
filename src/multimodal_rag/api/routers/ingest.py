"""POST /ingest: file -> parse -> chunk -> embed -> index in both stores.

doc_id is sha256(filename) -- a STABLE identity across content edits,
deliberately not sha256(file bytes). See schemas.IngestResponse for the
full reasoning; in short, keying identity to the filename is what makes
"only re-embed the chunks that actually changed" possible at all,
because chunk_id (chunking/ids.py) hashes doc_id in -- an identity that
changed on every edit would invalidate every chunk_id on every edit too,
which is exactly the bug this module used to have.

content_hash (sha256 of the actual bytes) is tracked separately in the
documents table, purely to detect "is this exact content already what's
stored for this filename" without redoing any work to find out.

The file has to touch disk briefly (a temp file) because parse_document()
ultimately calls libmagic + format-specific parsers that all expect a
real path, not an in-memory buffer.
"""

import hashlib
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ...chunking.parent_child import ParentChildChunker
from ...ingestion import parse_document
from ...providers.base import EmbeddingProvider
from ...stores.base import VectorStore
from ...stores.indexer import HybridIndexer
from ..db import Database
from ..dependencies import get_db, get_embedder, get_indexer, get_vector_store
from ..schemas import IngestResponse

router = APIRouter()

_chunker = ParentChildChunker()


def _ingest_sync(
    raw_bytes: bytes,
    filename: str,
    embedder: EmbeddingProvider,
    indexer: HybridIndexer,
    vector_store: VectorStore,
    db: Database,
) -> IngestResponse:
    doc_id = hashlib.sha256(filename.encode()).hexdigest()
    content_hash = hashlib.sha256(raw_bytes).hexdigest()

    existing = db.get_document(doc_id)
    if existing is not None and existing.content_hash == content_hash:
        # Byte-identical re-upload of the same filename -- nothing to do.
        return IngestResponse(
            doc_id=doc_id,
            filename=existing.filename,
            status="already_ingested",
            num_parent_chunks=existing.num_parent_chunks,
            num_child_chunks=existing.num_child_chunks,
            ingested_at=existing.ingested_at,
        )

    # Not a match on THIS filename -- but is this exact content already
    # sitting in the corpus under a DIFFERENT filename? Checked globally
    # (not scoped to doc_id) specifically to catch the same file arriving
    # through two different upload paths that report its name
    # differently -- e.g. the folder picker's webkitRelativePath
    # ("MyFolder/report.docx") vs. the single-file picker's bare name
    # ("report.docx") for the identical bytes. Declining to duplicate the
    # embedding work here, rather than normalizing filenames to catch
    # this, also sidesteps needing to guess which of several possible
    # upload-path naming quirks (a path prefix, Unicode normalization,
    # ...) is responsible in any given case.
    existing_by_content = db.get_document_by_content_hash(content_hash)
    if existing_by_content is not None and existing_by_content.doc_id != doc_id:
        return IngestResponse(
            doc_id=existing_by_content.doc_id,
            filename=filename,
            status="duplicate_content",
            duplicate_of=existing_by_content.filename,
            num_parent_chunks=existing_by_content.num_parent_chunks,
            num_child_chunks=existing_by_content.num_child_chunks,
            ingested_at=existing_by_content.ingested_at,
        )

    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        elements = parse_document(Path(tmp.name))

    # parse_document() stamps every element with the *temp* file's path
    # (a fresh random name per call) -- overwrite it with doc_id (stable
    # per filename, see module docstring) before chunking, since
    # chunk_id() hashes source_file in.
    for element in elements:
        element.metadata.source_file = doc_id

    chunks = _chunker.chunk(elements)

    # Chunk IDs are already locked in above -- safe to swap back to the
    # human-readable filename now, purely for citation/display purposes.
    for chunk in chunks:
        chunk.metadata.source_file = filename

    if existing is None:
        # Never seen this filename before -- every chunk is new.
        chunks_to_embed = chunks
        orphaned_chunk_ids: list[str] = []
    else:
        # Re-ingesting an edited document: doc_id is stable, so any chunk
        # whose TEXT is unchanged hashes to the exact same chunk_id it
        # already has in the stores -- re-embedding it would just be
        # wasted work for an identical vector. Only chunk_ids not already
        # present need embedding; chunk_ids that existed before but don't
        # appear in the new chunking are stale and need deleting.
        current_chunk_ids = {
            cid for cid in vector_store.list_chunk_ids() if cid.startswith(doc_id)
        }
        new_chunk_ids = {c.id for c in chunks}
        chunks_to_embed = [c for c in chunks if c.id not in current_chunk_ids]
        orphaned_chunk_ids = sorted(current_chunk_ids - new_chunk_ids)

    if chunks_to_embed:
        vectors = embedder.embed([c.text for c in chunks_to_embed])
        indexer.index(chunks_to_embed, vectors)
    if orphaned_chunk_ids:
        indexer.delete(orphaned_chunk_ids)

    num_parent_chunks = sum(1 for c in chunks if c.parent_id is None)
    num_child_chunks = len(chunks) - num_parent_chunks
    return db.upsert_document(doc_id, filename, content_hash, num_parent_chunks, num_child_chunks)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    embedder: EmbeddingProvider = Depends(get_embedder),
    indexer: HybridIndexer = Depends(get_indexer),
    vector_store: VectorStore = Depends(get_vector_store),
    db: Database = Depends(get_db),
) -> IngestResponse:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    filename = file.filename or "unnamed"

    try:
        return await run_in_threadpool(
            _ingest_sync, raw_bytes, filename, embedder, indexer, vector_store, db
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
