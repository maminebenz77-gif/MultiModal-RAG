"""Wipes all locally-ingested RAG data back to a clean slate: the
api_corpus collection in Qdrant, the api_corpus index in Elasticsearch,
and the sqlite tracking file (documents/queries/feedback).

If Docker is unavailable, this script starts local services from
.local-services first so the wipe can still run on Windows-only setups.

Reuses the same collection name / db path api/main.py defaults to, so
this always matches whatever the app actually wrote to, even if those
defaults change later.

Not a pytest test -- a manual convenience script. One-way: there's no
undo once this runs.

Run: uv run python tests/live/wipe_db.py
"""

import shutil
import subprocess
from pathlib import Path

from multimodal_rag.api.main import _COLLECTION_NAME, _DEFAULT_DB_PATH
from multimodal_rag.stores.elasticsearch_store import ElasticsearchStore
from multimodal_rag.stores.factory import get_keyword_store, get_vector_store
from multimodal_rag.stores.qdrant_store import QdrantStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_START_SCRIPT = PROJECT_ROOT / ".local-services" / "scripts" / "start-stores.ps1"


def _ensure_stores_for_local_mode() -> None:
    if shutil.which("docker") is not None:
        return

    if not LOCAL_START_SCRIPT.exists():
        raise FileNotFoundError(
            "Docker is not available and local start script is missing: "
            f"{LOCAL_START_SCRIPT}"
        )

    print("Docker not found; starting local Qdrant + Elasticsearch first...")
    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LOCAL_START_SCRIPT),
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    _ensure_stores_for_local_mode()

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
