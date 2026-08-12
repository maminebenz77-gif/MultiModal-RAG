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
import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from ...chunking.parent_child import ParentChildChunker
from ...config import get_settings
from ...device import resolve_device
from ...ingestion import parse_document
from ...providers.base import EmbeddingProvider
from ...providers.factory import embedder_from_override
from ...stores.base import VectorStore
from ...stores.indexer import HybridIndexer
from ..db import Database
from ..dependencies import get_db, get_embedder, get_indexer, get_vector_store
from ..schemas import IngestResponse, ProviderOverride, RuntimeOverrides

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

    # On Windows, NamedTemporaryFile keeps an open handle that can block
    # other readers (python-magic/libmagic) from opening the same path.
    # Create with delete=False, close it, then parse by path.
    with tempfile.NamedTemporaryFile(
        suffix=Path(filename).suffix, delete=False
    ) as tmp:
        tmp.write(raw_bytes)
        tmp_path = Path(tmp.name)

    try:
        elements = parse_document(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

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
        _validate_vector_store_compatibility(vector_store, vectors)
        indexer.index(chunks_to_embed, vectors)
    if orphaned_chunk_ids:
        indexer.delete(orphaned_chunk_ids)

    num_parent_chunks = sum(1 for c in chunks if c.parent_id is None)
    num_child_chunks = len(chunks) - num_parent_chunks
    return db.upsert_document(doc_id, filename, content_hash, num_parent_chunks, num_child_chunks)


def _validate_vector_store_compatibility(
    vector_store: VectorStore, vectors: list
) -> None:
    if not vectors:
        return

    expected_dimension = _live_vector_dimension(vector_store)
    if expected_dimension is not None and expected_dimension != vectors[0].dimension:
        raise ValueError(
            "Runtime embedder override is incompatible with the existing corpus: "
            f"stored dimension is {expected_dimension}, override produced {vectors[0].dimension}."
        )

    stored_model_id = getattr(vector_store, "_stored_model_id", lambda: None)()
    if stored_model_id is not None and stored_model_id != vectors[0].model_id:
        raise ValueError(
            "Runtime embedder override is incompatible with the existing corpus: "
            f"stored model is {stored_model_id!r}, override produced {vectors[0].model_id!r}."
        )


def _live_vector_dimension(vector_store: VectorStore) -> int | None:
    current_alias_target = getattr(vector_store, "_current_alias_target", lambda: None)()
    if current_alias_target is None:
        return None

    client = getattr(vector_store, "_client", None)
    if client is None:
        return None

    try:
        collection = client.get_collection(current_alias_target)
    except Exception:
        return None

    params = getattr(collection.config, "params", None)
    vectors = getattr(params, "vectors", None)
    size = getattr(vectors, "size", None)
    return size if isinstance(size, int) else None


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    file: UploadFile,
    runtime_overrides_json: str | None = Form(default=None),
    embedder: EmbeddingProvider = Depends(get_embedder),
    indexer: HybridIndexer = Depends(get_indexer),
    vector_store: VectorStore = Depends(get_vector_store),
    db: Database = Depends(get_db),
) -> IngestResponse:
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    filename = file.filename or "unnamed"

    embedder_to_use = embedder
    if runtime_overrides_json:
        try:
            overrides_payload = json.loads(runtime_overrides_json)
            overrides = RuntimeOverrides.model_validate(overrides_payload)
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"Invalid runtime overrides: {exc}") from exc

        if overrides.embedder is not None:
            settings = get_settings()
            chosen: ProviderOverride = overrides.embedder
            try:
                embedder_to_use = embedder_from_override(
                    provider=chosen.provider,
                    model=chosen.model,
                    base_url=chosen.base_url,
                    api_key=chosen.api_key,
                    allow_external=settings.allow_external,
                    device=resolve_device(settings.device),
                )
            except (ValueError, NotImplementedError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        return await run_in_threadpool(
            _ingest_sync, raw_bytes, filename, embedder_to_use, indexer, vector_store, db
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
