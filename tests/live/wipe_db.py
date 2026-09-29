"""Wipes all locally-ingested RAG data back to a clean slate: the
api_corpus index in Elasticsearch (which serves both the vector and
keyword search roles -- see stores/elasticsearch_store.py) and the
sqlite tracking file (documents/queries/feedback).

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
from multimodal_rag.stores.factory import get_keyword_store

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

    print("Docker not found; starting local Elasticsearch first...")
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

    # get_keyword_store() is typed to return the abstract KeywordStore --
    # this script deliberately reaches past that abstraction into
    # store-specific internals (deleting the whole physical index behind
    # the alias), which isn't part of the general interface. The cast is
    # just to satisfy mypy about that deliberate choice; get_keyword_store()
    # only ever constructs this one concrete class today.
    store: ElasticsearchStore = get_keyword_store(index_name=_COLLECTION_NAME)  # type: ignore[assignment]
    # _COLLECTION_NAME is an ALIAS, not a real index (see
    # elasticsearch_store.py) -- deleting by alias name wouldn't remove
    # the physical index behind it, so resolve to the real name first.
    physical = store._current_alias_target()
    if physical is not None:
        store._client.indices.delete(index=physical, ignore_unavailable=True)
        print(f"Deleted Elasticsearch index {physical!r} (alias {_COLLECTION_NAME!r}).")
    else:
        print(f"No live index behind alias {_COLLECTION_NAME!r} to delete.")

    if _DEFAULT_DB_PATH.exists():
        _DEFAULT_DB_PATH.unlink()
        print(f"Deleted {_DEFAULT_DB_PATH}.")
    else:
        print("No sqlite tracking file to delete.")

    print("Clean slate.")


if __name__ == "__main__":
    main()
