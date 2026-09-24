"""Elasticsearch-backed KeywordStore implementation.

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
"""

from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ..chunking.schema import Chunk, ChunkElement
from ..metadata import DocumentMetadata
from ..retry import retry_with_backoff
from .base import KeywordStore
from .filters import SearchFilter
from .schema import SearchResult

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF_SECONDS = 1.0


def _build_query(query: str, search_filter: SearchFilter | None) -> dict:
    # Constraints go in `filter`, not `must` -- a document either
    # satisfies a filter or it doesn't; putting it there keeps it out of
    # BM25 scoring (so it can't make one match outrank another by
    # matching the filter "better") and makes it eligible for
    # Elasticsearch's filter cache, which matters once a filter (e.g. a
    # future mandatory security clause) runs on every single query.
    filter_clauses: list[dict] = []
    if search_filter is not None:
        filter_clauses = [
            {"terms": {field: values}} for field, values in search_filter.any_of.items()
        ]
        filter_clauses.extend(
            {"range": {field: {"gte": from_date, "lte": to_date}}}
            for field, (from_date, to_date) in search_filter.date_range.items()
        )
    return {
        "bool": {
            "must": {"match": {"text": query}},
            "filter": filter_clauses,
            # A parent chunk (from parent-child chunking) is meant to be
            # reached only by resolving up from one of its children, never
            # matched directly -- excluded here natively rather than
            # relying on every caller to remember to filter it out
            # afterward.
            "must_not": {"term": {"is_parent": True}},
        }
    }


class ElasticsearchStore(KeywordStore):
    def __init__(
        self,
        url: str,
        index_name: str,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
    ) -> None:
        self._client = Elasticsearch(url)
        self._index_name = index_name
        self._max_retries = max_retries
        self._retry_backoff_seconds = retry_backoff_seconds

    def create_index(self) -> None:
        self._client.indices.delete(index=self._index_name, ignore_unavailable=True)
        self._client.indices.create(
            index=self._index_name,
            settings={"analysis": {"analyzer": {"default": {"type": "standard"}}}},
            mappings={
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "text": {"type": "text"},
                    "source": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "element_types": {"type": "keyword"},
                    # Stored and returned verbatim, never analyzed or
                    # indexed -- nothing ever needs to full-text-search
                    # inside a base64 thumbnail or an element's own text
                    # (that's what the top-level `text` field is for).
                    "elements": {"type": "object", "enabled": False},
                    "pages": {"type": "integer"},
                    "slides": {"type": "integer"},
                    "parent_id": {"type": "keyword"},
                    "is_parent": {"type": "boolean"},
                    # Document-level tags (see DocumentMetadata) --
                    # keyword-typed (exact match, filterable), not
                    # analyzed text, since none of these are meant to be
                    # searched by relevance.
                    "classification": {"type": "keyword"},
                    "private": {"type": "boolean"},
                    "owner": {"type": "keyword"},
                    "author": {"type": "keyword"},
                    # A real `date` field, not keyword -- this is what the
                    # date-range side of the user-facing "Filters" panel
                    # narrows on (SearchFilter.date_range), and a range
                    # query needs a range-comparable type. A strict `date`
                    # mapping rejects a non-ISO-parseable value at INDEX
                    # time instead of just storing it -- that's why
                    # DocumentMetadata.doc_date now validates the format
                    # itself, at the API boundary, well before it would
                    # ever reach here.
                    "doc_date": {"type": "date", "format": "yyyy-MM-dd"},
                    "data_type": {"type": "keyword"},
                    "tags": {"type": "keyword"},
                    # Derived from private/owner, not a DocumentMetadata
                    # field itself -- see DocumentMetadata.to_payload().
                    # "*" for a shared document, [owner] for a private
                    # one, [] for a private document with no owner
                    # recorded (visible to nobody, deliberately).
                    "acl_allow": {"type": "keyword"},
                    # Lifecycle (see DocumentMetadata). effective_* stay
                    # keyword for the same reason doc_date does -- nothing
                    # range-filters them yet.
                    "status": {"type": "keyword"},
                    "doc_family_id": {"type": "keyword"},
                    "version": {"type": "integer"},
                    "effective_from": {"type": "keyword"},
                    "effective_to": {"type": "keyword"},
                }
            },
        )

    def ensure_ready(self) -> None:
        if self._client.indices.exists(index=self._index_name):
            return
        self.create_index()

    def ping(self) -> bool:
        try:
            return bool(self._client.ping())
        except Exception:
            return False

    def index_chunks(
        self, chunks: list[Chunk], doc_metadata: DocumentMetadata | None = None
    ) -> None:
        extra_source = doc_metadata.to_payload() if doc_metadata is not None else {}
        actions = [
            {
                "_index": self._index_name,
                "_id": chunk.id,
                "_source": {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.metadata.source_file,
                    # See qdrant_store._to_point's identical comment --
                    # this used to be chunk.metadata.source_file (the
                    # filename), which is the doc_id identity bug fixed
                    # here on the ES side too.
                    "doc_id": chunk.metadata.doc_id,
                    "element_types": chunk.metadata.element_types,
                    "elements": [e.model_dump() for e in chunk.metadata.elements],
                    "pages": chunk.metadata.pages,
                    "slides": chunk.metadata.slides,
                    "parent_id": chunk.parent_id,
                    "is_parent": chunk.is_parent,
                    # Document-level tags -- see qdrant_store._to_point's
                    # identical merge.
                    **extra_source,
                },
            }
            for chunk in chunks
        ]

        def call() -> None:
            bulk(self._client, actions)
            # ES refreshes on its own roughly every 1s; force it so a
            # search immediately after indexing (demos, tests, the
            # HybridIndexer) sees the new documents.
            self._client.indices.refresh(index=self._index_name)

        retry_with_backoff(call, self._max_retries, self._retry_backoff_seconds)

    def set_document_metadata(self, doc_id: str, fields: dict[str, Any]) -> None:
        if not fields:
            return
        # _update_by_query has no plain "merge this partial doc" mode the
        # way a single-document `update` API call does -- a script is the
        # standard way to patch many documents' fields at once. Built
        # from `fields`' own keys rather than hand-listing every possible
        # metadata field name, so this can never drift out of sync with
        # DocumentMetadata -- and the keys always come from that fixed
        # pydantic schema, never from arbitrary caller input, so there's
        # no injection risk in interpolating them into the script source
        # (only the VALUES travel through `params`).
        script_source = "; ".join(f"ctx._source.{key} = params.{key}" for key in fields) + ";"
        self._client.update_by_query(
            index=self._index_name,
            query={"term": {"doc_id": doc_id}},
            script={"source": script_source, "params": fields},
            refresh=True,
        )

    def search(
        self, query: str, top_k: int = 5, search_filter: SearchFilter | None = None
    ) -> list[SearchResult]:
        response = self._client.search(
            index=self._index_name,
            query=_build_query(query, search_filter),
            size=top_k,
        )
        return [
            SearchResult(
                chunk_id=hit["_source"]["chunk_id"],
                score=hit["_score"],
                text=hit["_source"]["text"],
                source=hit["_source"]["source"],
                doc_id=hit["_source"]["doc_id"],
                element_types=hit["_source"]["element_types"],
                elements=[ChunkElement(**e) for e in hit["_source"].get("elements", [])],
                pages=hit["_source"].get("pages", []),
                slides=hit["_source"].get("slides", []),
                parent_id=hit["_source"].get("parent_id"),
                model_id=None,
                version=hit["_source"].get("version", 1),
                doc_family_id=hit["_source"].get("doc_family_id"),
                effective_from=hit["_source"].get("effective_from"),
                status=hit["_source"].get("status", "current"),
            )
            for hit in response["hits"]["hits"]
        ]

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        if not chunk_ids:
            return
        actions = [
            {"_op_type": "delete", "_index": self._index_name, "_id": chunk_id}
            for chunk_id in chunk_ids
        ]

        def call() -> None:
            # 404 (already absent) is a valid outcome here, not a failure
            # -- delete_chunks() is explicitly a no-op for missing IDs.
            bulk(self._client, actions, ignore_status=404)
            self._client.indices.refresh(index=self._index_name)

        retry_with_backoff(call, self._max_retries, self._retry_backoff_seconds)

    def list_chunk_ids(self) -> list[str]:
        if not self._client.indices.exists(index=self._index_name):
            return []

        chunk_ids: list[str] = []
        search_after = None
        while True:
            response = self._client.search(
                index=self._index_name,
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
