"""Elasticsearch-backed store — serves BOTH roles (VectorStore and
KeywordStore) against one physical index, via two classes that share
state: `ElasticsearchStore` (the KeywordStore role, and the owner of
the shared ES client + index alias/blue-green state) and
`ElasticsearchVectorStore` (the VectorStore role, a thin delegate onto
an `ElasticsearchStore` instance). Two classes, not one, because
`VectorStore.search(query_vector, ...)` and `KeywordStore.search(query,
...)` are two abstract methods with the SAME name and incompatible
signatures — Python has no method overloading, so one class can't
satisfy both interfaces at once. See stores/factory.py for how both
roles end up sharing one instance in practice, so callers still get two
objects (matching what Retriever/HybridIndexer already expect) that
happen to read/write the same underlying index.

Collection versioning (the VectorStore role): `index_name` (passed to
ElasticsearchStore.__init__) is an ES *alias*, not a physical index —
the stable, logical name every read goes through. create_collection()
creates a new, uniquely-named physical index without touching the alias
at all; upsert() populates it; publish() atomically repoints the alias
at it (and removes the previous physical index) via one
`indices.update_aliases()` call — ES guarantees the alias never has zero
or two targets in between. This is what makes a blue-green re-embed
possible with no downtime and a real rollback if the new version turns
out broken — see docs/technical-decisions.md §7 for the fuller history
behind this pattern. Once ES also stores vectors, it inherits the exact
failure mode
(embedding-model changes making stored vectors incompatible with new
queries) that this exists to solve; a keyword-only index never had that
problem, which is why create_index() below still has its own,
paired-with-the-vector-role story rather than blue-green of its own.

Writes merge, they don't replace: both roles write into the SAME
Elasticsearch document per chunk_id (the vector role's fields alongside
the keyword role's), via bulk `update` + `doc_as_upsert` rather than
plain `index` — plain `index` replaces a document's `_source` wholesale,
which would silently wipe out whichever role's fields weren't part of
that particular write (confirmed live: a doc_as_upsert `update` writing
only `{"vector": [...]}` followed by a second one writing only
`{"text": "..."}` ends up with BOTH fields present, not just the last
one written).

BM25 is Elasticsearch's default ranking function for `match` queries —
no special configuration needed beyond choosing a sensible analyzer to
get IDF-weighted, length-normalized lexical scoring.

Analyzer choice: the `standard` analyzer (lowercasing + whitespace/
punctuation tokenization, no stemming) is set explicitly, even though
it's ES's own default — stemming is deliberately left off, since it
risks mangling exact codes/identifiers ("CVE-2024-12345") in exchange
for generalizing prose forms ("test"/"tests"/"testing") we don't
strictly need for this corpus.

Known limitation, not solved here: the standard tokenizer splits on
hyphens, so an identifier like "internal-llama-70b" gets indexed as
three separate tokens ("internal", "llama", "70b"), not one. That's
usually fine (it still matches on any of those terms) but blurs exact-
phrase precision. A proper fix would add a second, un-analyzed
`keyword`-typed field for exact matching — a natural next step, not
built now.

Partial-failure handling on writes: index_chunks()/upsert() don't hand
their whole action list to elasticsearch.helpers.bulk() — that call
defaults to raising as soon as the first chunk in a batch is rejected,
which can leave later, perfectly fine chunks in the same call never
even attempted. Both write paths instead go through
_bulk_with_partial_failure(), built on the lower-level
elasticsearch.helpers.streaming_bulk(), which reports a real
True/False per chunk instead of one verdict for the whole request.
Only the chunks still outstanding get resent on each retry round (the
whole call is never unwound just because one chunk in it failed);
whatever is still failing once retries run out is reported explicitly
via UpsertBatchError, carrying which chunk_ids made it in and which
didn't.
"""

import time
import uuid
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk, streaming_bulk

from ..chunking.schema import Chunk, ChunkElement, ChunkMetadata
from ..metadata import DocumentMetadata
from ..providers.schema import EmbeddingVector, assert_single_model
from ..retry import retry_with_backoff
from .base import KeywordStore, VectorStore
from .filters import SearchFilter
from .schema import SearchResult

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF_SECONDS = 1.0
_ELASTICSEARCH_REQUEST_TIMEOUT_SECONDS = 300

# ES's own similarity names for dense_vector fields -- kept behind this
# map so create_collection()'s public contract (distance="cosine"/
# "euclidean"/"dot") stays vendor-neutral, not whatever ES happens to
# call its own similarity options.
_ES_SIMILARITY_MAP = {
    "cosine": "cosine",
    "euclidean": "l2_norm",
    "dot": "dot_product",
}

# elasticsearch.helpers.bulk() already splits a large `actions` list into
# multiple HTTP requests on its own (chunk_size=500 actions, or
# max_chunk_bytes=100MB, whichever comes first) -- no hand-rolled
# batching-by-size needed here.


def _build_query(query: str, search_filter: SearchFilter | None) -> dict:
    # Constraints go in `filter`, not `must` -- a document either
    # satisfies a filter or it doesn't; putting it there keeps it out of
    # BM25 scoring (so it can't make one match outrank another by
    # matching the filter "better") and makes it eligible for
    # Elasticsearch's filter cache, which matters once a filter (e.g. a
    # future mandatory security clause) runs on every single query.
    return {
        "bool": {
            "must": {"match": {"text": query}},
            "filter": _filter_clauses(search_filter),
            # A parent chunk (from parent-child chunking) is meant to be
            # reached only by resolving up from one of its children, never
            # matched directly -- excluded here natively rather than
            # relying on every caller to remember to filter it out
            # afterward.
            "must_not": {"term": {"is_parent": True}},
        }
    }


def _filter_clauses(search_filter: SearchFilter | None) -> list[dict]:
    """The SearchFilter -> ES query-DSL translation, factored out so
    BOTH _build_query (BM25) and the knn query's own `filter` (vector
    search) compose the identical clauses -- one translation, not two
    that could drift apart."""
    if search_filter is None:
        return []
    clauses: list[dict] = [
        {"terms": {field: values}} for field, values in search_filter.any_of.items()
    ]
    clauses.extend(
        {"range": {field: {"gte": from_date, "lte": to_date}}}
        for field, (from_date, to_date) in search_filter.date_range.items()
    )
    return clauses


def _index_mapping(dimension: int, distance: str, m: int, ef_construction: int) -> dict:
    if distance not in _ES_SIMILARITY_MAP:
        raise ValueError(f"Unknown distance {distance!r}; choose from {sorted(_ES_SIMILARITY_MAP)}")
    return {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "text": {"type": "text"},
            "source": {"type": "keyword"},
            "doc_id": {"type": "keyword"},
            "element_types": {"type": "keyword"},
            # Stored and returned verbatim, never analyzed or indexed --
            # nothing ever needs to full-text-search inside a base64
            # thumbnail or an element's own text (that's what the
            # top-level `text` field is for).
            "elements": {"type": "object", "enabled": False},
            "pages": {"type": "integer"},
            "slides": {"type": "integer"},
            "parent_id": {"type": "keyword"},
            "is_parent": {"type": "boolean"},
            # The vector role's own fields -- model_id backs
            # ModelMismatchError below, checked against incoming query
            # vectors at read time (see _stored_model_id() below).
            "model_id": {"type": "keyword"},
            "vector": {
                "type": "dense_vector",
                "dims": dimension,
                "index": True,
                "similarity": _ES_SIMILARITY_MAP[distance],
                "index_options": {"type": "hnsw", "m": m, "ef_construction": ef_construction},
            },
            # Document-level tags (see DocumentMetadata) -- keyword-typed
            # (exact match, filterable), not analyzed text, since none of
            # these are meant to be searched by relevance.
            "classification": {"type": "keyword"},
            "private": {"type": "boolean"},
            "owner": {"type": "keyword"},
            "author": {"type": "keyword"},
            # A real `date` field, not keyword -- this is what the
            # date-range side of the user-facing "Filters" panel narrows
            # on (SearchFilter.date_range), and a range query needs a
            # range-comparable type. A strict `date` mapping rejects a
            # non-ISO-parseable value at INDEX time instead of just
            # storing it -- that's why DocumentMetadata.doc_date now
            # validates the format itself, at the API boundary, well
            # before it would ever reach here.
            "doc_date": {"type": "date", "format": "yyyy-MM-dd"},
            "data_type": {"type": "keyword"},
            "tags": {"type": "keyword"},
            # Derived from private/owner, not a DocumentMetadata field
            # itself -- see DocumentMetadata.to_payload(). "*" for a
            # shared document, [owner] for a private one, [] for a
            # private document with no owner recorded (visible to
            # nobody, deliberately).
            "acl_allow": {"type": "keyword"},
            # Lifecycle (see DocumentMetadata). effective_* stay keyword
            # for the same reason doc_date does -- nothing range-filters
            # them yet.
            "status": {"type": "keyword"},
            "doc_family_id": {"type": "keyword"},
            "version": {"type": "integer"},
            "effective_from": {"type": "keyword"},
            "effective_to": {"type": "keyword"},
        }
    }


def _chunk_document(
    chunk: Chunk, vector: EmbeddingVector | None, doc_metadata: DocumentMetadata | None
) -> dict[str, Any]:
    """The full document body for one chunk -- shared by both roles'
    writes (upsert() supplies `vector`, index_chunks() doesn't), always
    written via `update`+`doc_as_upsert` (see module docstring) so
    whichever role writes second never wipes out what the other role
    already wrote for this same chunk_id."""
    doc: dict[str, Any] = {
        "chunk_id": chunk.id,
        "text": chunk.text,
        "source": chunk.metadata.source_file,
        # NOT chunk.metadata.source_file (see ChunkMetadata.doc_id) --
        # that's the human-readable filename; this is the STABLE id the
        # API hands out.
        "doc_id": chunk.metadata.doc_id,
        "element_types": chunk.metadata.element_types,
        "elements": [e.model_dump() for e in chunk.metadata.elements],
        "pages": chunk.metadata.pages,
        "slides": chunk.metadata.slides,
        "parent_id": chunk.parent_id,
        "is_parent": chunk.is_parent,
    }
    if vector is not None:
        doc["vector"] = vector.vector
        doc["model_id"] = vector.model_id
    if doc_metadata is not None:
        # Document-level tags (classification, author, tags, ...) --
        # merged in here rather than living on ChunkMetadata itself,
        # since they're a property of the DOCUMENT (mutable later via
        # set_document_metadata, no re-embed needed), not of how this
        # particular chunk was cut.
        doc.update(doc_metadata.to_payload())
    return doc


def _result_from_hit(hit: dict, vector: list[float] | None = None) -> SearchResult:
    source = hit["_source"]
    return SearchResult(
        chunk_id=source["chunk_id"],
        score=hit["_score"],
        text=source["text"],
        source=source["source"],
        doc_id=source["doc_id"],
        element_types=source["element_types"],
        elements=[ChunkElement(**e) for e in source.get("elements", [])],
        pages=source.get("pages", []),
        slides=source.get("slides", []),
        parent_id=source.get("parent_id"),
        model_id=source.get("model_id"),
        version=source.get("version", 1),
        doc_family_id=source.get("doc_family_id"),
        effective_from=source.get("effective_from"),
        status=source.get("status", "current"),
        tags=source.get("tags", []),
        author=source.get("author"),
        doc_date=source.get("doc_date"),
        classification=source.get("classification", "public"),
        private=source.get("private", False),
        vector=vector,
    )


class ModelMismatchError(RuntimeError):
    """Raised when a query vector's model_id doesn't match the model_id
    of the vectors actually stored in the index being searched -- the
    same invariant assert_single_model() enforces on write, now enforced
    on read too. Dimension matching alone isn't enough: two different
    models can share a dimension count while encoding meaning in
    completely incompatible spaces, and a mismatched search would return
    confident-looking, meaningless results with no error at all
    otherwise.
    """


class UpsertBatchError(RuntimeError):
    """Raised when one or more chunks still fail to write even after
    retrying just the failures a few times (see
    _bulk_with_partial_failure below). Carries the chunk_ids that DID
    make it in and the ones that didn't, so a caller can retry just the
    failures instead of redoing the whole call -- the already-succeeded
    chunks are already durably stored, there's nothing to redo for
    them.
    """

    def __init__(
        self, message: str, succeeded_chunk_ids: list[str], failed_chunk_ids: list[str]
    ) -> None:
        super().__init__(message)
        self.succeeded_chunk_ids = succeeded_chunk_ids
        self.failed_chunk_ids = failed_chunk_ids


def _bulk_with_partial_failure(
    client: Elasticsearch,
    actions: list[dict[str, Any]],
    max_retries: int,
    backoff_seconds: float,
) -> None:
    """Shared by index_chunks() and upsert() -- see this module's
    docstring section on partial-failure handling for why this exists
    instead of a plain bulk() call.

    `pending` starts as every action and only ever shrinks, one chunk_id
    at a time, the moment that chunk_id comes back confirmed. Each round
    resends whatever is STILL in `pending` (nothing already confirmed is
    ever resent -- harmless either way since writes are upserts, but
    wasteful for a large batch with one straggler) via streaming_bulk(),
    told not to raise on a rejected item so every chunk in the round
    gets a real attempt regardless of what happened to any other chunk
    in it. A connection-level failure mid-round (the whole request
    breaking, not just one document) is caught the same way
    retry_with_backoff treats any failed attempt -- whatever wasn't
    already confirmed this round simply stays pending for the next one.
    """
    pending = {action["_id"]: action for action in actions}
    succeeded: list[str] = []

    for attempt in range(max_retries):
        try:
            for ok, info in streaming_bulk(
                client, list(pending.values()), raise_on_error=False, raise_on_exception=False
            ):
                # info is e.g. {"update": {"_id": ..., "status": ..., ...}}
                # -- one key, named after the action's _op_type.
                item = next(iter(info.values()))
                if ok:
                    chunk_id = item["_id"]
                    succeeded.append(chunk_id)
                    pending.pop(chunk_id, None)
        except Exception:
            pass  # whatever's still in `pending` gets retried below

        if not pending:
            return
        if attempt < max_retries - 1:
            time.sleep(backoff_seconds * (2**attempt))

    raise UpsertBatchError(
        f"{len(pending)} of {len(actions)} chunks failed to upsert after "
        f"{max_retries} attempts; {len(succeeded)} succeeded.",
        succeeded_chunk_ids=succeeded,
        failed_chunk_ids=list(pending),
    )


class ElasticsearchStore(KeywordStore):
    def __init__(
        self,
        url: str,
        index_name: str,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self._client = Elasticsearch(
            url, request_timeout=_ELASTICSEARCH_REQUEST_TIMEOUT_SECONDS
        )
        # `index_name` is an ALIAS, not a physical index -- see this
        # module's docstring. Kept as `_alias` (not `_index_name`)
        # internally so every read/write site here has to spell out
        # which target it means.
        self._alias = index_name
        self._pending_index: str | None = None
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    # ---------------------------------------------------- shared (both roles)

    def create_collection(
        self,
        dimension: int,
        distance: str = "cosine",
        m: int = 16,
        ef_construct: int = 100,
        indexing_threshold: int = 20000,
    ) -> None:
        """The VectorStore role's entry point (see ElasticsearchVectorStore
        below, which delegates here) -- lives on this class because the
        alias/pending state it manages is shared with the keyword role.
        `indexing_threshold` has no ES equivalent (an optimizer setting
        some vector databases use to delay HNSW graph construction
        until enough points accumulate; ES builds it incrementally
        always) -- accepted for interface compatibility, unused."""
        if self._pending_index is not None:
            # An earlier create_collection() was never published -- its
            # physical index is otherwise orphaned. Not the live version
            # (never published), so nothing is lost by dropping it.
            self._client.indices.delete(index=self._pending_index, ignore_unavailable=True)

        physical_name = f"{self._alias}__{uuid.uuid4().hex[:8]}"
        self._client.indices.create(
            index=physical_name,
            settings={"analysis": {"analyzer": {"default": {"type": "standard"}}}},
            mappings=_index_mapping(dimension, distance, m, ef_construct),
        )
        self._pending_index = physical_name

    def publish(self) -> None:
        """Atomically repoints the alias at the pending index -- see
        this module's docstring."""
        if self._pending_index is None:
            raise RuntimeError(
                "No pending index to publish — call create_collection() (and "
                "upsert()/index_chunks() to populate it) first."
            )

        previous = self._current_alias_target()
        actions: list[dict[str, Any]] = []
        if previous is not None:
            actions.append({"remove": {"index": previous, "alias": self._alias}})
        actions.append({"add": {"index": self._pending_index, "alias": self._alias}})
        # Single call: the alias never has zero or two targets in
        # between -- search() sees either the old version or the new
        # one, atomically.
        self._client.indices.update_aliases(actions=actions)

        if previous is not None and previous != self._pending_index:
            self._client.indices.delete(index=previous, ignore_unavailable=True)

        self._pending_index = None

    def _current_alias_target(self) -> str | None:
        try:
            response = self._client.indices.get_alias(name=self._alias)
        except NotFoundError:
            return None
        return next(iter(response), None)

    def _write_target(self) -> str:
        """Resolves to the pending index if one's being
        built, otherwise the live alias -- so a write always lands
        wherever a concurrent ingest's chunks actually belong, never
        silently landing in the WRONG index version during a blue-green
        rebuild."""
        return self._pending_index or self._alias

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        actions = [
            {"_op_type": "delete", "_index": self._alias, "_id": chunk_id}
            for chunk_id in chunk_ids
        ]

        def call() -> None:
            # 404 (already absent) is a valid outcome here, not a
            # failure -- delete_chunks() is explicitly a no-op for
            # missing IDs. Both roles call this identically -- deleting
            # the same chunk_id twice is itself a no-op the second time.
            bulk(self._client, actions, ignore_status=404)
            self._client.indices.refresh(index=self._alias)

        retry_with_backoff(call, self._max_retries, self._retry_backoff_seconds)

    def set_document_metadata(self, doc_id: str, fields: dict[str, Any]) -> None:
        if not fields:
            return
        # _update_by_query has no plain "merge this partial doc" mode
        # the way a single-document `update` API call does -- a script
        # is the standard way to patch many documents' fields at once.
        # Built from `fields`' own keys rather than hand-listing every
        # possible metadata field name, so this can never drift out of
        # sync with DocumentMetadata -- and the keys always come from
        # that fixed pydantic schema, never from arbitrary caller input,
        # so there's no injection risk in interpolating them into the
        # script source (only the VALUES travel through `params`). Both
        # roles call this identically -- re-applying the same merge is
        # idempotent, not a clobber risk the way a plain `index` write
        # would be.
        script_source = "; ".join(f"ctx._source.{key} = params.{key}" for key in fields) + ";"
        self._client.update_by_query(
            index=self._alias,
            query={"term": {"doc_id": doc_id}},
            script={"source": script_source, "params": fields},
            refresh=True,
        )

    def list_chunk_ids(self) -> list[str]:
        if not self._client.indices.exists(index=self._alias):
            return []

        chunk_ids: list[str] = []
        search_after = None
        while True:
            response = self._client.search(
                index=self._alias,
                query={"match_all": {}},
                size=256,
                sort=[{"chunk_id": "asc"}],
                source_includes=["chunk_id"],
                search_after=search_after,
            )
            hits = response["hits"]["hits"]
            if not hits:
                break
            chunk_ids.extend(hit["_source"]["chunk_id"] for hit in hits)
            search_after = hits[-1]["sort"]
        return chunk_ids

    # -------------------------------------------------------- keyword role

    def create_index(self) -> None:
        """A no-op once create_collection() (the paired VectorStore
        role) has already created the shared index with its full
        mapping -- every real call site in this codebase already calls
        create_collection() first (see generation/demo.py,
        evaluation/run_eval.py, the *_compare_demo.py scripts: always
        `vector_store.create_collection(...)` immediately before
        `keyword_store.create_index()`). Raises if there is truly no
        pending or live index at all -- the keyword role alone has no
        embedding dimension to build a vector mapping with."""
        if self._pending_index is not None or self._current_alias_target() is not None:
            return
        raise RuntimeError(
            "create_index() requires create_collection(dimension=...) (the "
            "paired VectorStore role) to run first -- the keyword role alone "
            "has no dimension to build the shared index's vector mapping with."
        )

    def ensure_ready(self) -> None:
        """Same story as create_index() -- idempotently no-ops once the
        paired VectorStore role's ensure_ready(dimension) has already
        created the shared index (see api/main.py's lifespan, which
        always calls the vector role's ensure_ready() first for exactly
        this reason). Raises rather than silently creating a
        vector-less index if called first."""
        if self._current_alias_target() is not None:
            return
        raise RuntimeError(
            "ensure_ready() requires the paired VectorStore role's "
            "ensure_ready(dimension) to run first — see this method's "
            "docstring."
        )

    def index_chunks(
        self, chunks: list[Chunk], doc_metadata: DocumentMetadata | None = None
    ) -> None:
        target = self._write_target()
        actions = [
            {
                "_op_type": "update",
                "_index": target,
                "_id": chunk.id,
                "doc": _chunk_document(chunk, vector=None, doc_metadata=doc_metadata),
                "doc_as_upsert": True,
            }
            for chunk in chunks
        ]

        try:
            _bulk_with_partial_failure(
                self._client, actions, self._max_retries, self._retry_backoff_seconds
            )
        finally:
            # Refresh regardless of outcome -- whatever DID succeed
            # should be searchable immediately, even if UpsertBatchError
            # is about to be raised for the rest. ES refreshes on its
            # own roughly every 1s; force it so a search right after
            # indexing (demos, tests, the HybridIndexer) sees it too.
            self._client.indices.refresh(index=target)

    def search(
        self, query: str, top_k: int = 5, search_filter: SearchFilter | None = None
    ) -> list[SearchResult]:
        response = self._client.search(
            index=self._alias,
            query=_build_query(query, search_filter),
            size=top_k,
        )
        return [_result_from_hit(hit) for hit in response["hits"]["hits"]]


class ElasticsearchVectorStore(VectorStore):
    """The VectorStore role -- a thin delegate onto a shared
    `ElasticsearchStore` instance. See this module's docstring for why
    this is a separate class rather than one class implementing both
    interfaces."""

    def __init__(self, store: ElasticsearchStore) -> None:
        self._store = store

    @property
    def _client(self) -> Elasticsearch:
        """Passthrough onto the shared backend's client -- some callers
        (api/routers/ingest.py's embedder-override compatibility check)
        duck-type against a store's `_client`/`_current_alias_target`
        directly; this keeps that duck-typing working against the
        wrapper instead of every such caller needing to know it should
        reach through `._store` instead."""
        return self._store._client

    def _current_alias_target(self) -> str | None:
        return self._store._current_alias_target()

    def create_collection(
        self,
        dimension: int,
        distance: str = "cosine",
        m: int = 16,
        ef_construct: int = 100,
        indexing_threshold: int = 20000,
    ) -> None:
        self._store.create_collection(dimension, distance, m, ef_construct, indexing_threshold)

    def publish(self) -> None:
        self._store.publish()

    def ping(self) -> bool:
        return self._store.ping()

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        self._store.delete_chunks(chunk_ids)

    def set_document_metadata(self, doc_id: str, fields: dict[str, Any]) -> None:
        self._store.set_document_metadata(doc_id, fields)

    def list_chunk_ids(self) -> list[str]:
        return self._store.list_chunk_ids()

    def ensure_ready(self, dimension: int) -> None:
        if self._store._current_alias_target() is not None:
            # A shared index is already live. It may have been created
            # before `dimension`'s current value or before some mapping
            # field existed -- ES mappings can't be altered in place for
            # an existing dense_vector field (changing dims/similarity
            # needs a real reindex, i.e. a fresh
            # create_collection()+publish() blue-green cycle, not
            # attempted automatically here). Treated as a no-op -- an
            # existing deployment isn't silently rebuilt just because
            # the service restarted.
            return
        self.create_collection(dimension=dimension)
        self.publish()

    def upsert(
        self,
        chunks: list[Chunk],
        vectors: list[EmbeddingVector],
        doc_metadata: DocumentMetadata | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks and vectors must be the same length: {len(chunks)} != {len(vectors)}"
            )
        assert_single_model(vectors)

        target = self._store._write_target()
        actions = [
            {
                "_op_type": "update",
                "_index": target,
                "_id": chunk.id,
                "doc": _chunk_document(chunk, vector=vector, doc_metadata=doc_metadata),
                "doc_as_upsert": True,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        try:
            _bulk_with_partial_failure(
                self._store._client,
                actions,
                self._store._max_retries,
                self._store._retry_backoff_seconds,
            )
        finally:
            self._store._client.indices.refresh(index=target)

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
                f"index's stored vectors were produced by {stored_model_id!r}. "
                "Dimension matching alone doesn't guarantee compatibility."
            )

        knn: dict[str, Any] = {
            "field": "vector",
            "query_vector": query_vector.vector,
            "k": top_k,
            # num_candidates is ES's recall/latency knob for kNN,
            # wired from this search()'s own ef_search parameter -- a
            # wider candidate list examined before returning the top k.
            "num_candidates": max(ef_search or top_k * 4, top_k),
            "filter": {
                "bool": {
                    "filter": _filter_clauses(search_filter),
                    "must_not": {"term": {"is_parent": True}},
                }
            },
        }
        body: dict[str, Any] = {"knn": knn, "size": top_k, "_source": True}
        if with_vectors:
            # The `vector` field never appears in `_source` (confirmed
            # live against ES 9.4.1 -- a dense_vector field is stored
            # but not source-reconstructable by default), so it has to
            # be requested explicitly via `fields`; it comes back as a
            # plain list, not wrapped in an extra list the way most
            # multi-valued `fields` entries are.
            body["fields"] = ["vector"]

        response = self._store._client.search(index=self._store._alias, **body)
        results = []
        for hit in response["hits"]["hits"]:
            vector = None
            if with_vectors:
                vector = hit.get("fields", {}).get("vector")
            results.append(_result_from_hit(hit, vector=vector))
        return results

    def _stored_model_id(self) -> str | None:
        """Peek at one already-stored document to find which model
        produced the live index's vectors. None if the index doesn't
        exist yet or has no documents -- nothing to compare against, so
        nothing to reject."""
        try:
            response = self._store._client.search(
                index=self._store._alias,
                query={"exists": {"field": "model_id"}},
                size=1,
                source_includes=["model_id"],
            )
        except Exception:
            return None
        hits = response["hits"]["hits"]
        if not hits:
            return None
        model_id = hits[0]["_source"].get("model_id")
        return model_id if isinstance(model_id, str) else None

    def get_by_chunk_id(self, chunk_id: str) -> Chunk | None:
        try:
            document = self._store._client.get(index=self._store._alias, id=chunk_id)
        except NotFoundError:
            return None
        except Exception:
            # Index doesn't exist (or another retrieve-time issue) --
            # nothing to resolve, same as "not found".
            return None
        source = document["_source"]
        return Chunk(
            id=source["chunk_id"],
            text=source["text"],
            parent_id=source.get("parent_id"),
            is_parent=source.get("is_parent", False),
            metadata=ChunkMetadata(
                source_file=source["source"],
                doc_id=source.get("doc_id", ""),
                element_types=source["element_types"],
                elements=[ChunkElement(**e) for e in source.get("elements", [])],
                pages=source.get("pages", []),
                slides=source.get("slides", []),
            ),
        )
