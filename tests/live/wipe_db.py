"""Wipes all locally-ingested RAG data back to a clean slate: the
api_corpus collection in Qdrant, the api_corpus index in Elasticsearch,
and the sqlite tracking file (documents/queries/feedback). Does NOT
touch the running docker containers themselves -- see stop_stores.py
for that.

Reuses the same collection name / db path api/main.py defaults to, so
this always matches whatever the app actually wrote to, even if those
defaults change later.

Not a pytest test -- a manual convenience script. One-way: there's no
undo once this runs.

Run: uv run python tests/live/wipe_db.py
"""

from multimodal_rag.api.main import _COLLECTION_NAME, _DEFAULT_DB_PATH
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore
from multimodal_rag.stores.factory import get_keyword_store, get_vector_store
from multimodal_rag.stores.qdrant_store import QdrantStore


def main() -> None:
    # get_vector_store()/get_keyword_store() are typed to return the
    # abstract VectorStore/KeywordStore -- this script deliberately
    # reaches past that abstraction into store-specific internals
    # (deleting the whole physical collection/index), which isn't part
    # of the general interface. The casts are just to satisfy mypy about
    # that deliberate choice; get_vector_store()/get_keyword_store()
    # only ever construct these two concrete classes today.
    vector_store: QdrantStore = get_vector_store(collection_name=_COLLECTION_NAME)  # type: ignore[assignment]
    physical = vector_store._current_alias_target()
    if physical is not None:
        vector_store._client.delete_collection(physical)
        print(f"Deleted Qdrant collection {physical!r}.")
    else:
        print("No live Qdrant collection to delete.")

    keyword_store: ElasticsearchStore = get_keyword_store(  # type: ignore[assignment]
        index_name=_COLLECTION_NAME
    )
    existed = keyword_store._client.indices.exists(index=_COLLECTION_NAME)
    keyword_store._client.indices.delete(index=_COLLECTION_NAME, ignore_unavailable=True)
    print(f"Elasticsearch index {_COLLECTION_NAME!r} existed: {bool(existed)} -> deleted.")

    if _DEFAULT_DB_PATH.exists():
        _DEFAULT_DB_PATH.unlink()
        print(f"Deleted {_DEFAULT_DB_PATH}.")
    else:
        print("No sqlite tracking file to delete.")

    print("Clean slate.")


if __name__ == "__main__":
    main()
