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

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from ..chunking.schema import Chunk
from ..retry import retry_with_backoff
from .base import KeywordStore
from .schema import SearchResult

_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BACKOFF_SECONDS = 1.0


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
                    "pages": {"type": "integer"},
                    "slides": {"type": "integer"},
                    "parent_id": {"type": "keyword"},
                    "is_parent": {"type": "boolean"},
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

    def index_chunks(self, chunks: list[Chunk]) -> None:
        actions = [
            {
                "_index": self._index_name,
                "_id": chunk.id,
                "_source": {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.metadata.source_file,
                    "doc_id": chunk.metadata.source_file,
                    "element_types": chunk.metadata.element_types,
                    "pages": chunk.metadata.pages,
                    "slides": chunk.metadata.slides,
                    "parent_id": chunk.parent_id,
                    "is_parent": chunk.is_parent,
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

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # A parent chunk (from parent-child chunking) is meant to be
        # reached only by resolving up from one of its children, never
        # matched directly -- excluded here natively rather than relying
        # on every caller to remember to filter it out afterward.
        response = self._client.search(
            index=self._index_name,
            query={
                "bool": {
                    "must": {"match": {"text": query}},
                    "must_not": {"term": {"is_parent": True}},
                }
            },
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
                pages=hit["_source"].get("pages", []),
                slides=hit["_source"].get("slides", []),
                parent_id=hit["_source"].get("parent_id"),
                model_id=None,
            )
            for hit in response["hits"]["hits"]
        ]

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
