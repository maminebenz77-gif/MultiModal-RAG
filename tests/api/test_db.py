"""Direct tests of the Database class itself -- most of its behavior is
already exercised through the API-level tests (test_ingest.py,
test_feedback.py, ...), but the self-healing schema property below is
specific enough to Database's own internals that it deserves its own
test rather than being an incidental side effect of some API test.
"""

import sqlite3
from pathlib import Path

from multimodal_rag.api.db import Database
from multimodal_rag.chunking.schema import ChunkElement
from multimodal_rag.generation.schema import Citation


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


def test_delete_document_removes_only_that_row(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document("doc-a", "a.md", "hash-a", 1, 1)
    db.upsert_document("doc-b", "b.md", "hash-b", 2, 2)

    db.delete_document("doc-a")

    assert db.get_document("doc-a") is None
    assert db.get_document("doc-b") is not None


def test_wipe_documents_does_not_touch_queries_or_feedback(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document("doc-a", "a.md", "hash-a", 1, 1)
    db.record_query("q-1", "a question", "an answer", False, "hybrid_rrf")
    db.record_feedback("q-1", "up", None)

    db.wipe_documents()

    assert db.query_exists("q-1")
    assert db.metrics().feedback_up == 1


def test_create_conversation_returns_a_fresh_id_each_time(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")

    first = db.create_conversation()
    second = db.create_conversation()

    assert first != second
    assert db.conversation_exists(first)
    assert db.conversation_exists(second)


def test_conversation_exists_is_false_for_an_unknown_id(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    assert db.conversation_exists("nonexistent") is False


def test_get_recent_turns_returns_oldest_first_and_respects_limit(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation()
    for i in range(3):
        db.record_query(
            f"q-{i}", f"question {i}", f"answer {i}", False, "hybrid_rrf",
            conversation_id=conversation_id,
        )

    turns = db.get_recent_turns(conversation_id, limit=2)

    assert turns == [("question 1", "answer 1"), ("question 2", "answer 2")]


def test_get_recent_turns_only_includes_this_conversation(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_a = db.create_conversation()
    conversation_b = db.create_conversation()
    db.record_query(
        "q-a", "question a", "answer a", False, "hybrid_rrf", conversation_id=conversation_a
    )
    db.record_query(
        "q-b", "question b", "answer b", False, "hybrid_rrf", conversation_id=conversation_b
    )

    assert db.get_recent_turns(conversation_a, limit=10) == [("question a", "answer a")]


def test_list_conversations_excludes_conversations_with_no_turns(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    empty_conversation = db.create_conversation()
    active_conversation = db.create_conversation()
    db.record_query(
        "q-1", "a question", "an answer", False, "hybrid_rrf",
        conversation_id=active_conversation,
    )

    summaries = db.list_conversations()

    ids = [s.conversation_id for s in summaries]
    assert active_conversation in ids
    assert empty_conversation not in ids


def test_list_conversations_orders_by_most_recently_active_and_carries_preview(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    older = db.create_conversation()
    db.record_query(
        "q-older", "older question", "older answer", False, "hybrid_rrf",
        conversation_id=older,
    )
    newer = db.create_conversation()
    db.record_query(
        "q-newer", "newer question", "newer answer", False, "hybrid_rrf",
        conversation_id=newer,
    )

    summaries = db.list_conversations()

    assert [s.conversation_id for s in summaries] == [newer, older]
    assert summaries[0].preview == "newer question"
    assert summaries[0].message_count == 1


def test_set_conversation_title_is_preferred_over_the_first_question(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation()
    db.record_query(
        "q-1", "what was the raw first question", "an answer", False, "hybrid_rrf",
        conversation_id=conversation_id,
    )

    db.set_conversation_title(conversation_id, "A Generated Title")

    summaries = db.list_conversations()
    assert summaries[0].preview == "A Generated Title"


def test_delete_conversation_removes_the_conversation_and_its_queries(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation()
    db.record_query(
        "q-1", "a question", "an answer", False, "hybrid_rrf", conversation_id=conversation_id
    )

    db.delete_conversation(conversation_id)

    assert db.conversation_exists(conversation_id) is False
    assert db.query_exists("q-1") is False
    assert db.get_conversation_messages(conversation_id) == []


def test_delete_conversation_removes_citations_and_feedback(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation()
    db.record_query(
        "q-1", "a question", "an answer ⟦1⟧", False, "hybrid_rrf",
        conversation_id=conversation_id,
        citations=[Citation(marker=1, chunk_id="chunk-a", source="doc.md", pages=[], slides=[])],
    )
    db.record_feedback("q-1", "up", None)

    db.delete_conversation(conversation_id)

    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.row_factory = sqlite3.Row
        assert conn.execute("SELECT 1 FROM citations WHERE query_id = 'q-1'").fetchone() is None
        assert conn.execute("SELECT 1 FROM feedback WHERE query_id = 'q-1'").fetchone() is None


def test_delete_conversation_leaves_other_conversations_alone(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    keep = db.create_conversation()
    delete_me = db.create_conversation()
    db.record_query("q-keep", "keep this", "answer", False, "hybrid_rrf", conversation_id=keep)
    db.record_query(
        "q-delete", "delete this", "answer", False, "hybrid_rrf", conversation_id=delete_me
    )

    db.delete_conversation(delete_me)

    assert db.conversation_exists(keep) is True
    assert db.query_exists("q-keep") is True


def test_record_query_persists_citations_and_get_conversation_messages_reads_them_back(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation()
    citations = [
        Citation(marker=1, chunk_id="chunk-a", source="doc.md", pages=[2], slides=[]),
        Citation(marker=2, chunk_id="chunk-b", source="doc.md", pages=[], slides=[3]),
    ]

    db.record_query(
        "q-1", "a question", "an answer ⟦1⟧⟦2⟧", False, "hybrid_rrf",
        conversation_id=conversation_id, needs_clarification=False, citations=citations,
    )

    messages = db.get_conversation_messages(conversation_id)

    assert len(messages) == 1
    message = messages[0]
    assert message.question == "a question"
    assert message.answer == "an answer ⟦1⟧⟦2⟧"
    assert message.needs_clarification is False
    assert [(c.marker, c.chunk_id, c.pages, c.slides) for c in message.citations] == [
        (1, "chunk-a", [2], []),
        (2, "chunk-b", [], [3]),
    ]


def test_get_conversation_messages_is_ordered_oldest_first(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation()
    db.record_query(
        "q-1", "first", "first answer", False, "hybrid_rrf", conversation_id=conversation_id
    )
    db.record_query(
        "q-2", "second", "second answer", False, "hybrid_rrf", conversation_id=conversation_id
    )

    messages = db.get_conversation_messages(conversation_id)

    assert [m.question for m in messages] == ["first", "second"]


def test_needs_clarification_and_conversation_id_self_heal_onto_an_existing_queries_table(
    tmp_path: Path,
) -> None:
    """Regression test for the B2 migration: a queries table created
    BEFORE conversation_id/needs_clarification existed must still work
    once the new Database code runs against it -- see _ensure_column and
    its use in _create_tables."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE queries (
            query_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            refused INTEGER NOT NULL,
            retrieval_method TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    db = Database(db_path)
    conversation_id = db.create_conversation()

    # Must not raise "no such column" -- this is the whole point of the test.
    db.record_query(
        "q-1", "a question", "an answer", False, "hybrid_rrf",
        conversation_id=conversation_id, needs_clarification=True,
    )

    messages = db.get_conversation_messages(conversation_id)
    assert messages[0].needs_clarification is True


def test_citation_text_and_elements_round_trip(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation()
    elements = [
        ChunkElement(type="table", text="| A | B |\n| --- | --- |\n| 1 | 2 |"),
        ChunkElement(type="image", image_base64="aGVsbG8=", description="A photo."),
    ]
    citations = [
        Citation(
            marker=1, chunk_id="chunk-a", source="doc.md", pages=[2], slides=[],
            text="the full chunk text", elements=elements,
        ),
    ]

    db.record_query(
        "q-1", "a question", "an answer ⟦1⟧", False, "hybrid_rrf",
        conversation_id=conversation_id, citations=citations,
    )

    messages = db.get_conversation_messages(conversation_id)

    citation = messages[0].citations[0]
    assert citation.text == "the full chunk text"
    assert [(e.type, e.text, e.image_base64, e.description) for e in citation.elements] == [
        ("table", "| A | B |\n| --- | --- |\n| 1 | 2 |", None, None),
        ("image", None, "aGVsbG8=", "A photo."),
    ]


def test_citation_text_and_elements_self_heal_onto_an_existing_citations_table(
    tmp_path: Path,
) -> None:
    """Regression test: a citations table created before text/elements
    existed must still work once the new Database code runs against it."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE queries (
            query_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            refused INTEGER NOT NULL,
            retrieval_method TEXT NOT NULL,
            created_at TEXT NOT NULL,
            conversation_id TEXT,
            needs_clarification INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE citations (
            query_id TEXT NOT NULL,
            marker INTEGER NOT NULL,
            chunk_id TEXT NOT NULL,
            source TEXT NOT NULL,
            pages TEXT NOT NULL,
            slides TEXT NOT NULL,
            PRIMARY KEY (query_id, marker)
        )
        """
    )
    conn.commit()
    conn.close()

    db = Database(db_path)
    conversation_id = db.create_conversation()
    citations = [
        Citation(marker=1, chunk_id="chunk-a", source="doc.md", text="some text"),
    ]

    # Must not raise "no such column" -- this is the whole point of the test.
    db.record_query(
        "q-1", "a question", "an answer ⟦1⟧", False, "hybrid_rrf",
        conversation_id=conversation_id, citations=citations,
    )

    messages = db.get_conversation_messages(conversation_id)
    assert messages[0].citations[0].text == "some text"


def test_metrics_averages_latency_across_recorded_queries(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.record_query("q-1", "a", "answer a", False, "hybrid_rrf", latency_ms=100.0)
    db.record_query("q-2", "b", "answer b", False, "hybrid_rrf", latency_ms=300.0)

    assert db.metrics().avg_latency_ms == 200.0


def test_metrics_latency_is_zero_when_no_query_has_a_recorded_latency(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.record_query("q-1", "a", "answer a", False, "hybrid_rrf")

    assert db.metrics().avg_latency_ms == 0.0


def test_metrics_feedback_rate_reflects_the_fraction_of_queries_with_any_feedback(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    db.record_query("q-1", "a", "answer a", False, "hybrid_rrf")
    db.record_query("q-2", "b", "answer b", False, "hybrid_rrf")
    db.record_feedback("q-1", "up", None)

    assert db.metrics().feedback_rate == 0.5


def test_latency_ms_self_heals_onto_an_existing_queries_table(tmp_path: Path) -> None:
    """Regression test: a queries table created before latency_ms existed
    must still work once the new Database code runs against it, and old
    rows (NULL latency_ms) must not drag avg_latency_ms toward 0."""
    db_path = tmp_path / "state.db"
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE queries (
            query_id TEXT PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            refused INTEGER NOT NULL,
            retrieval_method TEXT NOT NULL,
            created_at TEXT NOT NULL,
            conversation_id TEXT,
            needs_clarification INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO queries VALUES "
        "('q-old', 'old q', 'old a', 0, 'hybrid_rrf', '2020-01-01T00:00:00', NULL, 0)"
    )
    conn.commit()
    conn.close()

    db = Database(db_path)
    # Must not raise "no such column" -- this is the whole point of the test.
    db.record_query("q-new", "new q", "new a", False, "hybrid_rrf", latency_ms=150.0)

    assert db.metrics().avg_latency_ms == 150.0
    assert db.metrics().total_queries == 2
