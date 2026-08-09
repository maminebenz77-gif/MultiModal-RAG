"""Direct tests of the Database class itself -- most of its behavior is
already exercised through the API-level tests (test_ingest.py,
test_feedback.py, ...), but the self-healing schema property below is
specific enough to Database's own internals that it deserves its own
test rather than being an incidental side effect of some API test.
"""

from pathlib import Path

from multimodal_rag.api.db import Database


def test_schema_recreates_itself_if_the_file_is_deleted_while_the_process_is_alive(
    tmp_path: Path,
) -> None:
    """Regression test: schema used to be created once, in __init__.
    If the underlying file was deleted out from under an already-running
    process (e.g. tests/live/wipe_db.py, which is explicitly meant to be
    safe to run against a live server) and sqlite3.connect() silently
    created a fresh, empty file in its place, every later query hit "no
    such table" until the process restarted -- this is exactly what
    happened testing this feature."""
    db_path = tmp_path / "state.db"
    db = Database(db_path)
    db.upsert_document("doc-1", "a.md", "hash-1", 1, 1)
    assert db.get_document("doc-1") is not None

    db_path.unlink()  # simulate wipe_db.py running against a live process

    # The same Database instance, no re-construction -- this must not
    # raise "no such table: documents".
    assert db.get_document("doc-1") is None
    db.upsert_document("doc-2", "b.md", "hash-2", 1, 1)
    assert db.get_document("doc-2") is not None


def test_get_document_by_content_hash_finds_it_regardless_of_doc_id(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document("doc-a", "a.md", "shared-hash", 1, 1)

    found = db.get_document_by_content_hash("shared-hash")

    assert found is not None
    assert found.doc_id == "doc-a"


def test_get_document_by_content_hash_returns_none_when_not_found(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    assert db.get_document_by_content_hash("nonexistent-hash") is None


def test_wipe_documents_deletes_all_rows_and_returns_the_count(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document("doc-a", "a.md", "hash-a", 1, 1)
    db.upsert_document("doc-b", "b.md", "hash-b", 2, 2)

    deleted = db.wipe_documents()

    assert deleted == 2
    assert db.list_documents() == []


def test_wipe_documents_does_not_touch_queries_or_feedback(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document("doc-a", "a.md", "hash-a", 1, 1)
    db.record_query("q-1", "a question", "an answer", False, "hybrid_rrf")
    db.record_feedback("q-1", "up", None)

    db.wipe_documents()

    assert db.query_exists("q-1")
    assert db.metrics().feedback_up == 1
