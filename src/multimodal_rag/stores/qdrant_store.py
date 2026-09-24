"""Qdrant-backed VectorStore implementation.

Point IDs: Qdrant requires an unsigned integer or a valid UUID — our
human-readable Chunk.id strings ("doc.md::structure::0::a1b2c3d4e5")
aren't valid as one directly. We deterministically hash chunk_id into a
UUID5 (same input always produces the same UUID, so re-upserting a
chunk updates it rather than creating a duplicate), and keep the
original readable chunk_id in the payload for citation/debugging.

Collection versioning: `collection_name` (passed to __init__) is a
Qdrant *alias*, not a physical collection — the stable, logical name
search() always reads through. create_collection() creates a new,
uniquely-named physical collection without touching the alias at all;
upsert() populates it; publish() atomically repoints the alias at it
(and removes the previous physical collection). A production pipeline
that instead deleted-and-recreated the live collection in place would
have no rollback if the new version turned out broken, and no way to
diff what changed to debug it — this makes that structurally
impossible: the old version stays fully intact and queryable right up
until the atomic cutover.

Not built (deliberately, to stay scoped to fixing that specific
failure mode): retaining N previous versions for a rollback window.
publish() removes the immediately-previous version once the swap
succeeds, rather than accumulating collections indefinitely.
"""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from ..chunking.schema import Chunk, ChunkElement, ChunkMetadata
from ..metadata import DocumentMetadata
from ..providers.schema import EmbeddingVector, assert_single_model
from ..retry import retry_with_backoff
from .base import VectorStore
from .filters import SearchFilter
from .schema import SearchResult

_logger = logging.getLogger(__name__)

_DISTANCE_MAP = {
    "cosine": models.Distance.COSINE,
    "euclidean": models.Distance.EUCLID,
    "dot": models.Distance.DOT,
}

_DEFAULT_BATCH_SIZE = 100

# Qdrant filters an UNINDEXED payload field by evaluating the condition on
# each candidate as it walks the HNSW graph -- slower, and it loses results,
# because the traversal keeps wandering into regions where nothing matches.
# An indexed field instead gets extra graph links that keep the filtered
# subset connected, plus a cardinality estimate that lets Qdrant fall back to
# exact search when the filter is selective enough. So a payload index is a
# precondition for filtering being CORRECT, not just fast.
#
# is_parent is already filtered on literally every search (see
# _EXCLUDE_PARENTS_FILTER) and has never been indexed. doc_id is filtered
# from the moment document-scoped search exists.
#
# Every later filterable payload field adds its entry here alongside the
# field itself, so this list never describes a schema that isn't really
# stored.
_PAYLOAD_INDEXES: dict[str, models.PayloadSchemaType] = {
    "is_parent": models.PayloadSchemaType.BOOL,
    "doc_id": models.PayloadSchemaType.KEYWORD,
    # acl_allow/classification are filtered on EVERY search once identity
    # is enabled (see retrieval/scoped.py's mandatory security filter) --
    # unindexed here would mean the one filter that matters most for
    # correctness falls back to the slow, potentially lossy path (see
    # "Payload indexes" above).
    "acl_allow": models.PayloadSchemaType.KEYWORD,
    "classification": models.PayloadSchemaType.KEYWORD,
    # Every search filters on status == "current" unless history is asked
    # for (retrieval/scoped.py). Like the two above, unindexed would make
    # the filter that decides which version of a document you read the
    # slow, lossy kind.
    "status": models.PayloadSchemaType.KEYWORD,
    # tags/author back the user-facing "Filters" panel (frontend); no
    # security property rides on them, but the same correctness argument
    # applies -- an unindexed filter on a real corpus is a slow scan that
    # can silently lose results, not just a slow one.
    "tags": models.PayloadSchemaType.KEYWORD,
    "author": models.PayloadSchemaType.KEYWORD,
    # DATETIME, not KEYWORD -- this is the field the date-range side of
    # the filter panel narrows on (SearchFilter.date_range), and a range
    # condition needs a range-comparable index, not exact-match.
    "doc_date": models.PayloadSchemaType.DATETIME,
}

# A parent chunk (from parent-child chunking) is meant to be reached only
# by resolving up from one of its children, never matched directly --
# this condition enforces that natively, at the store level, rather than
# relying on every caller to remember to exclude them. Kept as a bare
# condition (not a whole Filter) so _build_query_filter() can always AND
# it into whatever the caller's own search_filter adds -- composed in
# exactly one place, so it can never be replaced by a caller-supplied
# filter, only added to.
_EXCLUDE_PARENTS_CONDITION = models.FieldCondition(
    key="is_parent", match=models.MatchValue(value=True)
)


def _build_query_filter(search_filter: SearchFilter | None) -> models.Filter:
    must: list[models.Condition] = []
    if search_filter is not None:
        for field, values in search_filter.any_of.items():
            # MatchAny(any=[]) matches nothing, by design -- see
            # SearchFilter.any_of's docstring on why an empty value list
            # is a deliberate "exclude everything" clause, not "no
            # constraint".
            must.append(models.FieldCondition(key=field, match=models.MatchAny(any=values)))
        for field, (from_date, to_date) in search_filter.date_range.items():
            # gte/lte on the bare "YYYY-MM-DD" strings as stored --
            # Qdrant's DATETIME payload index accepts date-only RFC 3339
            # values directly, no need to round-trip through a real
            # datetime object just to build this condition.
            must.append(
                models.FieldCondition(
                    key=field, range=models.DatetimeRange(gte=from_date, lte=to_date)
                )
            )
    return models.Filter(must=must, must_not=[_EXCLUDE_PARENTS_CONDITION])


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class UpsertBatchError(RuntimeError):
    """Raised when one or more batches failed to upsert even after
    retries. Carries the chunk_ids that DID succeed and which didn't, so
    a caller can retry just the failures rather than redoing the whole
    call — the already-succeeded batches are already durably stored in
    Qdrant, there's nothing to redo for them.
    """

    def __init__(
        self, message: str, succeeded_chunk_ids: list[str], failed_chunk_ids: list[str]
    ) -> None:
        super().__init__(message)
        self.succeeded_chunk_ids = succeeded_chunk_ids
        self.failed_chunk_ids = failed_chunk_ids


class ModelMismatchError(RuntimeError):
    """Raised when a query vector's model_id doesn't match the model_id
    of the vectors actually stored in the collection being searched —
    the same invariant assert_single_model() enforces on write, now
    enforced on read too. Dimension matching alone (which Qdrant already
    enforces) isn't enough: two different models can share a dimension
    count while encoding meaning in completely incompatible spaces, and
    a mismatched search would return confident-looking, meaningless
    results with no error at all otherwise.
    """


class QdrantStore(VectorStore):
    def __init__(
        self,
        url: str,
        collection_name: str,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        # check_compatibility=False: we run qdrant/qdrant:latest locally,
        # which can be ahead of whatever version this client was tested
        # against — the mismatch is expected, not a real problem.
        self._client = QdrantClient(url=url, check_compatibility=False)
        self._alias = collection_name
        self._pending_collection: str | None = None
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    def create_collection(
        self,
        dimension: int,
        distance: str = "cosine",
        m: int = 16,
        ef_construct: int = 100,
        indexing_threshold: int = 20000,
    ) -> None:
        if distance not in _DISTANCE_MAP:
            raise ValueError(f"Unknown distance {distance!r}; choose from {sorted(_DISTANCE_MAP)}")

        if self._pending_collection is not None:
            # An earlier create_collection() was never published — its
            # physical collection is otherwise orphaned. Not the live
            # version (never published), so nothing is lost by dropping it.
            self._client.delete_collection(self._pending_collection)

        physical_name = f"{self._alias}__{uuid.uuid4().hex[:8]}"
        self._client.create_collection(
            collection_name=physical_name,
            vectors_config=models.VectorParams(size=dimension, distance=_DISTANCE_MAP[distance]),
            hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=indexing_threshold),
        )
        # Indexed before a single point is written, so the version the alias
        # eventually cuts over to is never briefly searchable-but-unindexed.
        self._ensure_payload_indexes(physical_name)
        self._pending_collection = physical_name

    def publish(self) -> None:
        if self._pending_collection is None:
            raise RuntimeError(
                "No pending collection to publish — call create_collection() (and "
                "upsert() to populate it) first."
            )

        previous = self._current_alias_target()
        operations: list[models.AliasOperations] = []
        if previous is not None:
            operations.append(
                models.DeleteAliasOperation(delete_alias=models.DeleteAlias(alias_name=self._alias))
            )
        operations.append(
            models.CreateAliasOperation(
                create_alias=models.CreateAlias(
                    collection_name=self._pending_collection, alias_name=self._alias
                )
            )
        )
        # Single call: the alias never has zero or two targets in between —
        # search() sees either the old version or the new one, atomically.
        self._client.update_collection_aliases(change_aliases_operations=operations)

        if previous is not None and previous != self._pending_collection:
            self._client.delete_collection(previous)

        self._pending_collection = None

    def _current_alias_target(self) -> str | None:
        for alias in self._client.get_aliases().aliases:
            if alias.alias_name == self._alias:
                return alias.collection_name
        return None

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[EmbeddingVector],
        doc_metadata: DocumentMetadata | None = None,
    ) -> None:
        """Insert or update chunks in batches, into whichever collection
        version is currently pending (since the last create_collection()
        call), or the live one via the alias if no new version is
        pending. Each batch is retried (exponential backoff) before
        being considered failed — most real failures here are transient
        (network blip, momentary unavailability). If a batch still
        fails, upserting keeps going with the remaining batches;
        already-succeeded batches are already durably stored, so one bad
        batch shouldn't force redoing everything. Raises UpsertBatchError
        at the end if anything failed persistently, carrying which
        chunk_ids succeeded/failed.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must be the same length: {len(chunks)} != {len(vectors)}"
            )
        assert_single_model(vectors)

        target = self._pending_collection or self._alias
        succeeded_ids: list[str] = []
        failed_ids: list[str] = []
        first_error: Exception | None = None

        for start in range(0, len(chunks), self._batch_size):
            batch_chunks = chunks[start : start + self._batch_size]
            batch_vectors = vectors[start : start + self._batch_size]
            points = [
                self._to_point(chunk, vector, doc_metadata)
                for chunk, vector in zip(batch_chunks, batch_vectors, strict=True)
            ]
            try:
                self._upsert_batch(target, points)
                succeeded_ids.extend(chunk.id for chunk in batch_chunks)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                failed_ids.extend(chunk.id for chunk in batch_chunks)

        if failed_ids:
            detail = f" First error: {type(first_error).__name__}: {first_error}" if first_error else ""
            raise UpsertBatchError(
                f"{len(failed_ids)} of {len(chunks)} chunks failed to upsert after "
                f"{self._max_retries} attempts per batch; {len(succeeded_ids)} succeeded."
                f"{detail}",
                succeeded_chunk_ids=succeeded_ids,
                failed_chunk_ids=failed_ids,
            )

    def _upsert_batch(self, collection_name: str, points: list[models.PointStruct]) -> None:
        def call() -> None:
            self._client.upsert(collection_name=collection_name, points=points)

        retry_with_backoff(call, self._max_retries, self._retry_backoff_seconds)

    @staticmethod
    def _to_point(
        chunk: Chunk, vector: EmbeddingVector, doc_metadata: DocumentMetadata | None = None
    ) -> models.PointStruct:
        payload: dict[str, Any] = {
            "chunk_id": chunk.id,
            "text": chunk.text,
            "source": chunk.metadata.source_file,
            # NOT chunk.metadata.source_file (see ChunkMetadata.doc_id)
            # -- that's the human-readable filename, and using it here
            # was the identity bug: this payload field is meant to be
            # the STABLE id the API hands out (sha256 of the filename),
            # which is what /query's doc_ids filter and the
            # supersession flip (future phase) both key against.
            # Deliberately no `or chunk.metadata.source_file` fallback
            # -- that would silently resurrect the bug for any chunk
            # written by code that forgot to set doc_id.
            "doc_id": chunk.metadata.doc_id,
            "element_types": chunk.metadata.element_types,
            "elements": [e.model_dump() for e in chunk.metadata.elements],
            "pages": chunk.metadata.pages,
            "slides": chunk.metadata.slides,
            "model_id": vector.model_id,
            "parent_id": chunk.parent_id,
            "is_parent": chunk.is_parent,
        }
        if doc_metadata is not None:
            # Document-level tags (classification, author, tags, ...) --
            # merged in here rather than living on ChunkMetadata itself,
            # since they're a property of the DOCUMENT (mutable later via
            # set_document_metadata, no re-embed needed), not of how this
            # particular chunk was cut.
            payload.update(doc_metadata.to_payload())
        return models.PointStruct(id=_point_id(chunk.id), vector=vector.vector, payload=payload)

    def set_document_metadata(self, doc_id: str, fields: dict[str, Any]) -> None:
        if not fields:
            return
        # Same target resolution as upsert() -- the pending collection if
        # one's being built, otherwise the live alias -- so a metadata
        # patch always lands wherever a concurrent ingest's chunks
        # actually are, never silently patching the WRONG collection
        # version during a blue-green rebuild.
        target = self._pending_collection or self._alias
        self._client.set_payload(
            collection_name=target,
            payload=fields,
            points=models.Filter(
                must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))]
            ),
            wait=True,
        )

    def search(
        self,
        query_vector: EmbeddingVector,
        top_k: int = 5,
        ef_search: int | None = None,
        with_vectors: bool = False,
        search_filter: SearchFilter | None = None,
    ) -> list[SearchResult]:
        stored_model_id = self._stored_model_id()
        if stored_model_id is not None and stored_model_id != query_vector.model_id:
            raise ModelMismatchError(
                f"Query vector was produced by {query_vector.model_id!r}, but this "
                f"collection's stored vectors were produced by {stored_model_id!r}. "
                "Dimension matching alone doesn't guarantee compatibility."
            )

        search_params = models.SearchParams(hnsw_ef=ef_search) if ef_search is not None else None
        response = self._client.query_points(
            collection_name=self._alias,
            query=query_vector.vector,
            query_filter=_build_query_filter(search_filter),
            limit=top_k,
            search_params=search_params,
            with_payload=True,
            with_vectors=with_vectors,
        )
        return [
            SearchResult(
                chunk_id=point.payload["chunk_id"],
                score=point.score,
                text=point.payload["text"],
                source=point.payload["source"],
                doc_id=point.payload["doc_id"],
                element_types=point.payload["element_types"],
                elements=[ChunkElement(**e) for e in point.payload.get("elements", [])],
                pages=point.payload.get("pages", []),
                slides=point.payload.get("slides", []),
                parent_id=point.payload.get("parent_id"),
                model_id=point.payload["model_id"],
                version=point.payload.get("version", 1),
                doc_family_id=point.payload.get("doc_family_id"),
                effective_from=point.payload.get("effective_from"),
                status=point.payload.get("status", "current"),
                vector=point.vector if isinstance(point.vector, list) else None,
            )
            for point in response.points
            if point.payload is not None
        ]

    def _stored_model_id(self) -> str | None:
        """Peek at one already-stored point's payload to find which model
        produced the live collection's vectors. None if the collection
        doesn't exist yet or has no points — nothing to compare against,
        so nothing to reject."""
        try:
            points, _ = self._client.scroll(
                collection_name=self._alias, limit=1, with_payload=True
            )
        except Exception:
            return None
        if not points or points[0].payload is None:
            return None
        model_id = points[0].payload.get("model_id")
        return model_id if isinstance(model_id, str) else None

    def list_chunk_ids(self) -> list[str]:
        if not self._client.collection_exists(self._alias):
            return []

        chunk_ids: list[str] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._alias,
                limit=256,
                offset=offset,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            chunk_ids.extend(
                point.payload["chunk_id"]
                for point in points
                if point.payload is not None and "chunk_id" in point.payload
            )
            if offset is None:
                break
        return chunk_ids

    def ensure_ready(self, dimension: int) -> None:
        if self._current_alias_target() is not None:
            # A collection is already live. It may have been created before
            # this field was added to _PAYLOAD_INDEXES, so index it now --
            # otherwise an existing deployment would need a full rebuild to
            # gain an index, which is exactly what the blue-green design is
            # meant to avoid having to do for something this cheap.
            self._ensure_payload_indexes(self._alias)
            return
        self.create_collection(dimension=dimension)
        self.publish()

    def _ensure_payload_indexes(self, collection_name: str) -> None:
        """Idempotently create every _PAYLOAD_INDEXES entry. Re-creating an
        index that already exists with the same schema is a no-op in Qdrant,
        so this is safe to call on every startup.

        Failures are logged, not raised: this runs from ensure_ready(), which
        runs during service startup, and an index that conflicts with an old
        collection's existing schema would otherwise take the whole API down
        at boot. A missing payload index degrades filter performance; a
        service that won't start is an outage."""
        for field_name, field_schema in _PAYLOAD_INDEXES.items():
            try:
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema,
                )
            except Exception:
                _logger.warning(
                    "Could not create Qdrant payload index on %r in %r; filters on "
                    "that field will fall back to unindexed evaluation.",
                    field_name,
                    collection_name,
                    exc_info=True,
                )

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        self._client.delete(
            collection_name=self._alias,
            points_selector=[_point_id(chunk_id) for chunk_id in chunk_ids],
        )

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        try:
            points = self._client.retrieve(
                collection_name=self._alias, ids=[_point_id(chunk_id)], with_payload=True
            )
        except Exception:
            # Collection doesn't exist (or another retrieve-time issue) --
            # nothing to resolve, same as "not found".
            return None
        if not points or points[0].payload is None:
            return None
        payload = points[0].payload
        return Chunk(
            id=payload["chunk_id"],
            text=payload["text"],
            parent_id=payload.get("parent_id"),
            is_parent=payload.get("is_parent", False),
            metadata=ChunkMetadata(
                source_file=payload["source"],
                # .get(..., "") not payload["doc_id"] -- points written
                # before this field existed won't have it in their
                # payload, and a missing field should round-trip as
                # "unknown" (""), not raise a KeyError.
                doc_id=payload.get("doc_id", ""),
                element_types=payload["element_types"],
                elements=[ChunkElement(**e) for e in payload.get("elements", [])],
                pages=payload.get("pages", []),
                slides=payload.get("slides", []),
            ),
        )
