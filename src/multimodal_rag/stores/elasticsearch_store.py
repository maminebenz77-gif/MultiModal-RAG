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
from .base import KeywordStore
from .schema import SearchResult


class ElasticsearchStore(KeywordStore):
    def __init__(self, url: str, index_name: str) -> None:
        self._client = Elasticsearch(url)
        self._index_name = index_name

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
                }
            },
        )

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
                },
            }
            for chunk in chunks
        ]
        bulk(self._client, actions)
        # ES refreshes on its own roughly every 1s; force it so a search
        # immediately after indexing (demos, tests) sees the new documents.
        self._client.indices.refresh(index=self._index_name)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        response = self._client.search(
            index=self._index_name, query={"match": {"text": query}}, size=top_k
        )
        return [
            SearchResult(
                chunk_id=hit["_source"]["chunk_id"],
                score=hit["_score"],
                text=hit["_source"]["text"],
                source=hit["_source"]["source"],
                doc_id=hit["_source"]["doc_id"],
                element_types=hit["_source"]["element_types"],
                model_id=None,
            )
            for hit in response["hits"]["hits"]
        ]
