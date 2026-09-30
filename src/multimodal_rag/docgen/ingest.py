"""Folder ingestion for docgen.

Reuses the exact same pipeline POST /ingest uses -- api.routers.ingest's
_ingest_sync -- once per file, rather than reimplementing parsing,
chunking, embedding or indexing a second time. Deliberately not going
through the HTTP endpoint: docgen's ingestion input is a local folder
path (see the docgen build plan), not an upload.

_ingest_sync is reached into directly even though it's a "private"
(leading-underscore) name in another module -- unlike api.main's
DEFAULT_DB_PATH/COLLECTION_NAME, renaming it was out of scope for this
phase, since it's a business-logic function in api/routers/ingest.py
rather than a plain constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..api.routers.ingest import _ingest_sync
from ..api.schemas import IngestResponse
from ..identity import Principal
from ..metadata import Classification, DocumentMetadata
from .stack import DocgenStack


@dataclass(frozen=True)
class IngestSummary:
    folder: Path
    tag: str
    results: list[IngestResponse]

    @property
    def num_files(self) -> int:
        return len(self.results)


def _ensure_tag(
    result: IngestResponse, tag: str, principal: Principal, stack: DocgenStack
) -> IngestResponse:
    """_ingest_sync() only applies newly-submitted metadata (tags
    included) when it actually does ingestion work. A byte-identical
    re-upload ("already_ingested"), or content that matches something
    already stored under a different filename ("duplicate_content"),
    both return the EXISTING document's metadata untouched -- by
    design, documented in api/routers/ingest.py: re-upload is for
    content changes, PATCH /documents/{doc_id} is for metadata changes.

    docgen still needs that file discoverable under its new source tag
    even when nothing was re-embedded, so this adds the tag the same
    way a PATCH request would -- through Database.update_document_metadata
    + HybridIndexer.set_document_metadata -- rather than changing
    _ingest_sync's own contract, which the real /ingest endpoint and its
    tests also depend on."""
    if result.status not in ("already_ingested", "duplicate_content"):
        return result  # a fresh or changed-content ingest already applied the tag
    if tag in result.metadata.tags:
        return result  # e.g. re-running the same folder/tag combination twice

    merged_tags = [*result.metadata.tags, tag]
    updated = stack.db.update_document_metadata(principal, result.doc_id, {"tags": merged_tags})
    assert updated is not None  # result.doc_id was just resolved to an existing document above
    full_metadata = DocumentMetadata(**updated.metadata.model_dump())
    stack.indexer.set_document_metadata(result.doc_id, full_metadata.to_payload())
    return result.model_copy(update={"metadata": updated.metadata})


def ingest_folder(
    folder: Path,
    tag: str,
    classification: Classification,
    stack: DocgenStack,
    *,
    principal: Principal | None = None,
) -> IngestSummary:
    """Ingest every top-level file directly inside `folder` (no
    recursion into subfolders, dotfiles skipped), tagging every one of
    them with `tag` -- one label for the whole batch, since a docgen
    source is a single named set of documents, not per-file metadata.
    `classification` is required with no default here, same as
    DocumentMetadata itself: the caller must choose one for the batch,
    never silently assumed.

    A file already known to the corpus under a different tag keeps
    THAT tag too (see _ensure_tag) -- pointing a docgen source at
    documents that happen to already be ingested must not silently fail
    to make them discoverable under the new tag."""
    principal = principal or Principal.unrestricted()
    metadata = DocumentMetadata(classification=classification, tags=[tag])

    files = sorted(
        path for path in folder.iterdir() if path.is_file() and not path.name.startswith(".")
    )
    if not files:
        raise ValueError(f"No files found directly inside {folder}")

    results = [
        _ensure_tag(
            _ingest_sync(
                path.read_bytes(),
                path.name,
                metadata,
                principal,
                stack.embedder,
                stack.indexer,
                stack.vector_store,
                stack.db,
            ),
            tag,
            principal,
            stack,
        )
        for path in files
    ]
    return IngestSummary(folder=folder, tag=tag, results=results)
