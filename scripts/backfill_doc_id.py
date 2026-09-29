"""One-time backfill for the doc_id payload identity bug (Phase 1 of the
metadata/ACL/versioning plan): chunks upserted before ChunkMetadata.doc_id
existed have "doc_id" == the human-readable filename in their payload
(same value as "source"), not the stable sha256(filename) id the API
hands out everywhere else.

Costs ZERO re-embeddings: chunk_id() (chunking/ids.py) already hashes
doc_id in as the id's prefix ("<doc_id>::<strategy>::<index>::<hash>"),
so the correct value is recoverable from data already stored -- this is
an Elasticsearch bulk update, not a re-ingest.

Idempotent: re-running it just re-derives and re-writes the same value.
Safe to run against a live index (patching this one field never touches
the stored text or vector).

Run: `uv run python scripts/backfill_doc_id.py [--dry-run]`
"""

import argparse
from typing import cast

from elasticsearch.helpers import bulk

from multimodal_rag.config import get_settings
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore
from multimodal_rag.stores.factory import get_keyword_store


def _derive_doc_id(chunk_id: str) -> str:
    # chunk_id() (chunking/ids.py) is "<doc_id>::<strategy>::<index>::<hash>"
    # -- doc_id is everything before the first "::". split(..., 1) so a
    # doc_id that itself happened to contain "::" (it can't today, sha256
    # hex digests never do, but this is cheap insurance) isn't truncated.
    return chunk_id.split("::", 1)[0]


def _backfill_elasticsearch(dry_run: bool) -> int:
    settings = get_settings()
    store = cast(ElasticsearchStore, get_keyword_store(settings, index_name="api_corpus"))

    chunk_ids = store.list_chunk_ids()
    if dry_run:
        print(f"would patch {len(chunk_ids)} documents")
        return len(chunk_ids)

    actions = [
        {
            "_op_type": "update",
            "_index": store._alias,
            "_id": chunk_id,
            "doc": {"doc_id": _derive_doc_id(chunk_id)},
        }
        for chunk_id in chunk_ids
    ]
    bulk(store._client, actions)
    store._client.indices.refresh(index=store._alias)
    print(f"patched {len(actions)} documents")
    return len(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change; write nothing."
    )
    args = parser.parse_args()

    _backfill_elasticsearch(args.dry_run)


if __name__ == "__main__":
    main()
