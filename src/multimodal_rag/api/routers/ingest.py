"""POST /ingest: file -> parse -> chunk -> embed -> index in both stores.

doc_id is sha256(file_bytes) -- see schemas.IngestResponse for why. The
file has to touch disk briefly (a temp file) because parse_document()
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
from ...stores.indexer import HybridIndexer
from ..db import Database
from ..dependencies import get_db, get_embedder, get_indexer
from ..schemas import IngestResponse

router = APIRouter()

_chunker = ParentChildChunker()


def _ingest_sync(
    raw_bytes: bytes,
    filename: str,
    embedder: EmbeddingProvider,
    indexer: HybridIndexer,
    db: Database,
) -> IngestResponse:
    doc_id = hashlib.sha256(raw_bytes).hexdigest()

    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix) as tmp:
        tmp.write(raw_bytes)
        tmp.flush()
        elements = parse_document(Path(tmp.name))

    # parse_document() stamps every element with the *temp* file's path
    # (a fresh random name per call) -- overwrite it with doc_id (the
    # content hash) before chunking, since chunk_id() hashes source_file
    # in. Using doc_id here, not the uploaded filename, is what actually
    # makes re-ingestion idempotent: two uploads of byte-identical
    # content always produce the same doc_id and therefore the same
    # chunk_ids, even if the file was renamed in between. Using the
    # filename instead would silently mint a fresh set of chunk_ids on
    # every rename, orphaning the old set rather than upserting over it.
    for element in elements:
        element.metadata.source_file = doc_id

    chunks = _chunker.chunk(elements)

    # Chunk IDs are already locked in above -- safe to swap back to the
    # human-readable filename now, purely for citation/display purposes.
    for chunk in chunks:
        chunk.metadata.source_file = filename

    vectors = embedder.embed([c.text for c in chunks])
    indexer.index(chunks, vectors)

    num_parent_chunks = sum(1 for c in chunks if c.parent_id is None)
    num_child_chunks = len(chunks) - num_parent_chunks
    return db.upsert_document(doc_id, filename, num_parent_chunks, num_child_chunks)


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    embedder: EmbeddingProvider = Depends(get_embedder),
    indexer: HybridIndexer = Depends(get_indexer),
    db: Database = Depends(get_db),
) -> IngestResponse:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    filename = file.filename or "unnamed"

    try:
        return await run_in_threadpool(_ingest_sync, raw_bytes, filename, embedder, indexer, db)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
