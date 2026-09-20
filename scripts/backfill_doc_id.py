"""One-time backfill for the doc_id payload identity bug (Phase 1 of the
metadata/ACL/versioning plan): chunks upserted before ChunkMetadata.doc_id
existed have "doc_id" == the human-readable filename in their payload
(same value as "source"), not the stable sha256(filename) id the API
hands out everywhere else.

Costs ZERO re-embeddings: chunk_id() (chunking/ids.py) already hashes
doc_id in as the id's prefix ("<doc_id>::<strategy>::<index>::<hash>"),
so the correct value is recoverable from data already stored -- this is
a payload patch (Qdrant set_payload + an Elasticsearch bulk update), not
a re-ingest.

Idempotent: re-running it just re-derives and re-writes the same value.
Safe to run against a live collection/index (both backends support
patching a payload field without touching the vector/text).

Run: `uv run python scripts/backfill_doc_id.py [--dry-run]`
"""

import argparse
import uuid
from typing import cast

from elasticsearch.helpers import bulk

from multimodal_rag.config import get_settings
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore
from multimodal_rag.stores.factory import get_keyword_store, get_vector_store
from multimodal_rag.stores.qdrant_store import QdrantStore


def _derive_doc_id(chunk_id: str) -> str:
    # chunk_id() (chunking/ids.py) is "<doc_id>::<strategy>::<index>::<hash>"
    # -- doc_id is everything before the first "::". split(..., 1) so a
    # doc_id that itself happened to contain "::" (it can't today, sha256
    # hex digests never do, but this is cheap insurance) isn't truncated.
    return chunk_id.split("::", 1)[0]


def _point_id(chunk_id: str) -> str:
    # Same derivation as qdrant_store._point_id() -- Qdrant point ids must
    # be an integer or UUID, not our human-readable chunk_id strings.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _backfill_qdrant(dry_run: bool) -> int:
    settings = get_settings()
    # get_vector_store() is typed to return the VectorStore ABC (no
    # set_payload -- that's a Qdrant-specific op, not part of the
    # cross-backend interface), but going through the factory rather than
    # constructing QdrantStore directly still runs the privacy guard on
    # the configured URL. This script is backend-specific by nature (it's
    # patching a Qdrant-only payload field), so the cast is honest about
    # that, not a workaround for a real type mismatch.
    store = cast(QdrantStore, get_vector_store(settings, collection_name="api_corpus"))

    updates: dict[str, list[str]] = {}
    for chunk_id in store.list_chunk_ids():
        updates.setdefault(_derive_doc_id(chunk_id), []).append(chunk_id)

    if dry_run:
        print(f"[qdrant] would patch {sum(len(v) for v in updates.values())} points "
              f"across {len(updates)} documents")
        return sum(len(v) for v in updates.values())

    patched = 0
    for doc_id, chunk_ids in updates.items():
        # One set_payload call per doc_id (batched over that document's
        # point ids), not one call per chunk -- same value written for
        # every chunk of a document, so there's no reason to pay N round
        # trips for it.
        store._client.set_payload(
            collection_name=store._alias,
            payload={"doc_id": doc_id},
            points=[_point_id(cid) for cid in chunk_ids],
            wait=True,
        )
        patched += len(chunk_ids)
    print(f"[qdrant] patched {patched} points across {len(updates)} documents")
    return patched


def _backfill_elasticsearch(dry_run: bool) -> int:
    settings = get_settings()
    store = cast(ElasticsearchStore, get_keyword_store(settings, index_name="api_corpus"))

    chunk_ids = store.list_chunk_ids()
    if dry_run:
        print(f"[elasticsearch] would patch {len(chunk_ids)} documents")
        return len(chunk_ids)

    actions = [
        {
            "_op_type": "update",
            "_index": store._index_name,
            "_id": chunk_id,
            "doc": {"doc_id": _derive_doc_id(chunk_id)},
        }
        for chunk_id in chunk_ids
    ]
    bulk(store._client, actions)
    store._client.indices.refresh(index=store._index_name)
    print(f"[elasticsearch] patched {len(actions)} documents")
    return len(actions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would change; write nothing."
    )
    args = parser.parse_args()

    _backfill_qdrant(args.dry_run)
    _backfill_elasticsearch(args.dry_run)


if __name__ == "__main__":
    main()
