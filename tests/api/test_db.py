"""Direct tests of the Database class itself -- most of its behavior is
already exercised through the API-level tests (test_ingest.py,
test_feedback.py, ...), but the self-healing schema property below is
specific enough to Database's own internals that it deserves its own
test rather than being an incidental side effect of some API test.
"""

import sqlite3
from pathlib import Path

import pytest

from multimodal_rag.api.db import AccessDeniedError, Database, QueryNotFoundError
from multimodal_rag.chunking.schema import ChunkElement
from multimodal_rag.generation.schema import Citation
from multimodal_rag.identity import Principal
from multimodal_rag.metadata import DocumentMetadata

_MINIMAL_METADATA = DocumentMetadata(classification="public")

# Most tests here are about Database's own storage behavior, not about who
# is asking -- an admin sees and may do everything, so they stay focused on
# what they were written for. Scoping itself is covered at the bottom.
_ADMIN = Principal.unrestricted()


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
    db.upsert_document(_ADMIN, "doc-1", "a.md", "hash-1", 1, 1, _MINIMAL_METADATA)
    assert db.get_document(_ADMIN, "doc-1") is not None

    db_path.unlink()  # simulate wipe_db.py running against a live process

    # The same Database instance, no re-construction -- this must not
    # raise "no such table: documents".
    assert db.get_document(_ADMIN, "doc-1") is None
    db.upsert_document(_ADMIN, "doc-2", "b.md", "hash-2", 1, 1, _MINIMAL_METADATA)
    assert db.get_document(_ADMIN, "doc-2") is not None


def test_get_document_by_content_hash_finds_it_regardless_of_doc_id(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document(_ADMIN, "doc-a", "a.md", "shared-hash", 1, 1, _MINIMAL_METADATA)

    found = db.get_document_by_content_hash(_ADMIN, "shared-hash")

    assert found is not None
    assert found.doc_id == "doc-a"


def test_get_document_by_content_hash_returns_none_when_not_found(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    assert db.get_document_by_content_hash(_ADMIN, "nonexistent-hash") is None


def test_wipe_documents_deletes_all_rows_and_returns_the_count(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document(_ADMIN, "doc-a", "a.md", "hash-a", 1, 1, _MINIMAL_METADATA)
    db.upsert_document(_ADMIN, "doc-b", "b.md", "hash-b", 2, 2, _MINIMAL_METADATA)

    deleted = db.wipe_documents(_ADMIN)

    assert deleted == 2
    assert db.list_documents(_ADMIN) == []


def test_delete_document_removes_only_that_row(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document(_ADMIN, "doc-a", "a.md", "hash-a", 1, 1, _MINIMAL_METADATA)
    db.upsert_document(_ADMIN, "doc-b", "b.md", "hash-b", 2, 2, _MINIMAL_METADATA)

    db.delete_document(_ADMIN, "doc-a")

    assert db.get_document(_ADMIN, "doc-a") is None
    assert db.get_document(_ADMIN, "doc-b") is not None


def test_wipe_documents_does_not_touch_queries_or_feedback(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document(_ADMIN, "doc-a", "a.md", "hash-a", 1, 1, _MINIMAL_METADATA)
    db.record_query(_ADMIN, "q-1", "a question", "an answer", False, "hybrid_rrf")
    db.record_feedback(_ADMIN, "q-1", "up", None)

    db.wipe_documents(_ADMIN)

    assert db.query_exists(_ADMIN, "q-1")
    assert db.metrics(_ADMIN).feedback_up == 1


def test_update_document_metadata_changes_only_the_given_fields(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document(_ADMIN, "doc-a", "a.md", "hash-a", 1, 1, _MINIMAL_METADATA)

    updated = db.update_document_metadata(
        _ADMIN, "doc-a", {"classification": "c2", "tags": ["urgent"]}
    )

    assert updated is not None
    assert updated.metadata.classification == "c2"
    assert updated.metadata.tags == ["urgent"]
    # author was never in the patch -- must stay at its prior value.
    assert updated.metadata.author is None


def test_update_document_metadata_returns_none_for_an_unknown_doc_id(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    assert db.update_document_metadata(_ADMIN, "nonexistent", {"private": True}) is None


def test_update_document_metadata_with_an_empty_patch_returns_the_current_row(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    db.upsert_document(_ADMIN, "doc-a", "a.md", "hash-a", 1, 1, _MINIMAL_METADATA)

    updated = db.update_document_metadata(_ADMIN, "doc-a", {})

    assert updated is not None
    assert updated.metadata.classification == "public"


def test_create_conversation_returns_a_fresh_id_each_time(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")

    first = db.create_conversation(_ADMIN)
    second = db.create_conversation(_ADMIN)

    assert first != second
    assert db.conversation_exists(_ADMIN, first)
    assert db.conversation_exists(_ADMIN, second)


def test_conversation_exists_is_false_for_an_unknown_id(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    assert db.conversation_exists(_ADMIN, "nonexistent") is False


def test_get_recent_turns_returns_oldest_first_and_respects_limit(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation(_ADMIN)
    for i in range(3):
        db.record_query(
            _ADMIN,
            f"q-{i}", f"question {i}", f"answer {i}", False, "hybrid_rrf",
            conversation_id=conversation_id,
        )

    turns = db.get_recent_turns(_ADMIN, conversation_id, limit=2)

    assert turns == [("question 1", "answer 1"), ("question 2", "answer 2")]


def test_get_recent_turns_only_includes_this_conversation(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_a = db.create_conversation(_ADMIN)
    conversation_b = db.create_conversation(_ADMIN)
    db.record_query(
        _ADMIN,
        "q-a", "question a", "answer a", False, "hybrid_rrf", conversation_id=conversation_a
    )
    db.record_query(
        _ADMIN,
        "q-b", "question b", "answer b", False, "hybrid_rrf", conversation_id=conversation_b
    )

    assert db.get_recent_turns(_ADMIN, conversation_a, limit=10) == [("question a", "answer a")]


def test_list_conversations_excludes_conversations_with_no_turns(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    empty_conversation = db.create_conversation(_ADMIN)
    active_conversation = db.create_conversation(_ADMIN)
    db.record_query(
        _ADMIN,
        "q-1", "a question", "an answer", False, "hybrid_rrf",
        conversation_id=active_conversation,
    )

    summaries = db.list_conversations(_ADMIN)

    ids = [s.conversation_id for s in summaries]
    assert active_conversation in ids
    assert empty_conversation not in ids


def test_list_conversations_orders_by_most_recently_active_and_carries_preview(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    older = db.create_conversation(_ADMIN)
    db.record_query(
        _ADMIN,
        "q-older", "older question", "older answer", False, "hybrid_rrf",
        conversation_id=older,
    )
    newer = db.create_conversation(_ADMIN)
    db.record_query(
        _ADMIN,
        "q-newer", "newer question", "newer answer", False, "hybrid_rrf",
        conversation_id=newer,
    )

    summaries = db.list_conversations(_ADMIN)

    assert [s.conversation_id for s in summaries] == [newer, older]
    assert summaries[0].preview == "newer question"
    assert summaries[0].message_count == 1


def test_set_conversation_title_is_preferred_over_the_first_question(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation(_ADMIN)
    db.record_query(
        _ADMIN,
        "q-1", "what was the raw first question", "an answer", False, "hybrid_rrf",
        conversation_id=conversation_id,
    )

    db.set_conversation_title(_ADMIN, conversation_id, "A Generated Title")

    summaries = db.list_conversations(_ADMIN)
    assert summaries[0].preview == "A Generated Title"


def test_delete_conversation_removes_the_conversation_and_its_queries(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation(_ADMIN)
    db.record_query(
        _ADMIN,
        "q-1", "a question", "an answer", False, "hybrid_rrf", conversation_id=conversation_id
    )

    db.delete_conversation(_ADMIN, conversation_id)

    assert db.conversation_exists(_ADMIN, conversation_id) is False
    assert db.query_exists(_ADMIN, "q-1") is False
    assert db.get_conversation_messages(_ADMIN, conversation_id) == []


def test_delete_conversation_removes_citations_and_feedback(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation(_ADMIN)
    db.record_query(
        _ADMIN,
        "q-1", "a question", "an answer ⟦1⟧", False, "hybrid_rrf",
        conversation_id=conversation_id,
        citations=[Citation(marker=1, chunk_id="chunk-a", source="doc.md", pages=[], slides=[])],
    )
    db.record_feedback(_ADMIN, "q-1", "up", None)

    db.delete_conversation(_ADMIN, conversation_id)

    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.row_factory = sqlite3.Row
        assert conn.execute("SELECT 1 FROM citations WHERE query_id = 'q-1'").fetchone() is None
        assert conn.execute("SELECT 1 FROM feedback WHERE query_id = 'q-1'").fetchone() is None


def test_delete_conversation_leaves_other_conversations_alone(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    keep = db.create_conversation(_ADMIN)
    delete_me = db.create_conversation(_ADMIN)
    db.record_query(
        _ADMIN, "q-keep", "keep this", "answer", False, "hybrid_rrf", conversation_id=keep
    )
    db.record_query(
        _ADMIN,
        "q-delete", "delete this", "answer", False, "hybrid_rrf", conversation_id=delete_me
    )

    db.delete_conversation(_ADMIN, delete_me)

    assert db.conversation_exists(_ADMIN, keep) is True
    assert db.query_exists(_ADMIN, "q-keep") is True


def test_record_query_persists_citations_and_get_conversation_messages_reads_them_back(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation(_ADMIN)
    citations = [
        Citation(marker=1, chunk_id="chunk-a", source="doc.md", pages=[2], slides=[]),
        Citation(marker=2, chunk_id="chunk-b", source="doc.md", pages=[], slides=[3]),
    ]

    db.record_query(
        _ADMIN,
        "q-1", "a question", "an answer ⟦1⟧⟦2⟧", False, "hybrid_rrf",
        conversation_id=conversation_id, needs_clarification=False, citations=citations,
    )

    messages = db.get_conversation_messages(_ADMIN, conversation_id)

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
    conversation_id = db.create_conversation(_ADMIN)
    db.record_query(
        _ADMIN,
        "q-1", "first", "first answer", False, "hybrid_rrf", conversation_id=conversation_id
    )
    db.record_query(
        _ADMIN,
        "q-2", "second", "second answer", False, "hybrid_rrf", conversation_id=conversation_id
    )

    messages = db.get_conversation_messages(_ADMIN, conversation_id)

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
    conversation_id = db.create_conversation(_ADMIN)

    # Must not raise "no such column" -- this is the whole point of the test.
    db.record_query(
        _ADMIN,
        "q-1", "a question", "an answer", False, "hybrid_rrf",
        conversation_id=conversation_id, needs_clarification=True,
    )

    messages = db.get_conversation_messages(_ADMIN, conversation_id)
    assert messages[0].needs_clarification is True


def test_citation_text_and_elements_round_trip(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation_id = db.create_conversation(_ADMIN)
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
        _ADMIN,
        "q-1", "a question", "an answer ⟦1⟧", False, "hybrid_rrf",
        conversation_id=conversation_id, citations=citations,
    )

    messages = db.get_conversation_messages(_ADMIN, conversation_id)

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
    conversation_id = db.create_conversation(_ADMIN)
    citations = [
        Citation(marker=1, chunk_id="chunk-a", source="doc.md", text="some text"),
    ]

    # Must not raise "no such column" -- this is the whole point of the test.
    db.record_query(
        _ADMIN,
        "q-1", "a question", "an answer ⟦1⟧", False, "hybrid_rrf",
        conversation_id=conversation_id, citations=citations,
    )

    messages = db.get_conversation_messages(_ADMIN, conversation_id)
    assert messages[0].citations[0].text == "some text"


def test_metrics_averages_latency_across_recorded_queries(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.record_query(_ADMIN, "q-1", "a", "answer a", False, "hybrid_rrf", latency_ms=100.0)
    db.record_query(_ADMIN, "q-2", "b", "answer b", False, "hybrid_rrf", latency_ms=300.0)

    assert db.metrics(_ADMIN).avg_latency_ms == 200.0


def test_metrics_latency_is_zero_when_no_query_has_a_recorded_latency(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    db.record_query(_ADMIN, "q-1", "a", "answer a", False, "hybrid_rrf")

    assert db.metrics(_ADMIN).avg_latency_ms == 0.0


def test_metrics_feedback_rate_reflects_the_fraction_of_queries_with_any_feedback(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    db.record_query(_ADMIN, "q-1", "a", "answer a", False, "hybrid_rrf")
    db.record_query(_ADMIN, "q-2", "b", "answer b", False, "hybrid_rrf")
    db.record_feedback(_ADMIN, "q-1", "up", None)

    assert db.metrics(_ADMIN).feedback_rate == 0.5


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
    db.record_query(_ADMIN, "q-new", "new q", "new a", False, "hybrid_rrf", latency_ms=150.0)

    assert db.metrics(_ADMIN).avg_latency_ms == 150.0
    assert db.metrics(_ADMIN).total_queries == 2


# ---------------------------------------------------------------------------
# Scoping: who is asking changes what comes back. The "not found" answers are
# deliberately identical for "doesn't exist" and "exists but isn't yours".
# ---------------------------------------------------------------------------

_ALICE = Principal(principal_id="user:alice", clearance="c2")
_BOB = Principal(principal_id="user:bob", clearance="c2")


def _doc(db: Database, principal: Principal, doc_id: str, **metadata: object) -> None:
    owned = DocumentMetadata(
        classification="public", owner=principal.principal_id, **metadata  # type: ignore[arg-type]
    )
    db.upsert_document(principal, doc_id, f"{doc_id}.md", f"hash-{doc_id}", 1, 1, owned)


def test_a_private_document_is_invisible_to_others_everywhere_it_could_show_up(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    _doc(db, _ALICE, "doc-a", private=True)

    assert db.get_document(_BOB, "doc-a") is None
    assert db.list_documents(_BOB) == []
    assert db.get_documents_by_ids(_BOB, {"doc-a"}) == []
    assert db.get_document_by_content_hash(_BOB, "hash-doc-a") is None
    assert db.get_document(_ALICE, "doc-a") is not None


def test_content_hash_lookup_ignores_documents_the_caller_cannot_see(tmp_path: Path) -> None:
    """The duplicate-upload oracle: matching against EVERYONE's documents
    would tell a caller that someone else holds the same file."""
    db = Database(tmp_path / "state.db")
    _doc(db, _ALICE, "doc-a", private=True)

    assert db.get_document_by_content_hash(_BOB, "hash-doc-a") is None


def test_a_shared_document_is_visible_but_only_its_owner_can_change_it(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    _doc(db, _ALICE, "doc-a")  # shared: private defaults to False

    assert db.get_document(_BOB, "doc-a") is not None
    with pytest.raises(AccessDeniedError):
        db.update_document_metadata(_BOB, "doc-a", {"tags": ["mine now"]})
    with pytest.raises(AccessDeniedError):
        db.delete_document(_BOB, "doc-a")
    assert db.get_document(_ALICE, "doc-a") is not None


def test_a_non_admin_cannot_reassign_ownership_even_of_their_own_document(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    _doc(db, _ALICE, "doc-a")

    with pytest.raises(AccessDeniedError):
        db.update_document_metadata(_ALICE, "doc-a", {"owner": "user:bob"})
    assert db.update_document_metadata(_ADMIN, "doc-a", {"owner": "user:bob"}) is not None


def test_updating_or_deleting_a_document_you_cannot_see_behaves_as_if_it_isnt_there(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    _doc(db, _ALICE, "doc-a", private=True)

    assert db.update_document_metadata(_BOB, "doc-a", {"tags": ["x"]}) is None
    db.delete_document(_BOB, "doc-a")  # silently nothing -- no error to probe with
    assert db.get_document(_ALICE, "doc-a") is not None


def test_upsert_refuses_to_overwrite_a_document_owned_by_someone_else(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    _doc(db, _ALICE, "doc-a")

    with pytest.raises(AccessDeniedError):
        _doc(db, _BOB, "doc-a")  # same doc_id, different caller


def test_only_an_admin_can_wipe_the_whole_corpus(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    _doc(db, _ALICE, "doc-a")

    with pytest.raises(AccessDeniedError):
        db.wipe_documents(_ALICE)
    assert db.wipe_documents(_ADMIN) == 1


def test_conversations_belong_to_whoever_started_them(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    conversation = db.create_conversation(_ALICE)
    db.record_query(
        _ALICE, "q-1", "alice's question", "an answer", False, "hybrid_rrf",
        conversation_id=conversation,
    )

    assert db.conversation_exists(_ALICE, conversation) is True
    assert db.conversation_exists(_BOB, conversation) is False
    assert db.get_conversation_messages(_BOB, conversation) == []
    assert db.get_recent_turns(_BOB, conversation, limit=10) == []
    assert [c.conversation_id for c in db.list_conversations(_BOB)] == []
    assert [c.conversation_id for c in db.list_conversations(_ALICE)] == [conversation]
    assert [c.conversation_id for c in db.list_conversations(_ADMIN)] == [conversation]


def test_someone_else_cannot_delete_retitle_or_append_to_your_conversation(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    conversation = db.create_conversation(_ALICE)
    db.record_query(
        _ALICE, "q-1", "alice's question", "an answer", False, "hybrid_rrf",
        conversation_id=conversation,
    )

    db.delete_conversation(_BOB, conversation)
    db.set_conversation_title(_BOB, conversation, "hijacked")
    with pytest.raises(AccessDeniedError):
        db.record_query(
            _BOB, "q-2", "injected", "answer", False, "hybrid_rrf", conversation_id=conversation
        )

    assert db.conversation_exists(_ALICE, conversation) is True
    assert [c.preview for c in db.list_conversations(_ALICE)] == ["alice's question"]


def test_feedback_on_someone_elses_query_is_indistinguishable_from_a_missing_query(
    tmp_path: Path,
) -> None:
    db = Database(tmp_path / "state.db")
    conversation = db.create_conversation(_ALICE)
    db.record_query(
        _ALICE, "q-1", "question", "answer", False, "hybrid_rrf", conversation_id=conversation
    )

    with pytest.raises(QueryNotFoundError):
        db.record_feedback(_BOB, "q-1", "up", None)
    with pytest.raises(QueryNotFoundError):
        db.record_feedback(_BOB, "no-such-query", "up", None)
    assert db.record_feedback(_ALICE, "q-1", "up", None)


def test_metrics_count_only_what_the_caller_may_see(tmp_path: Path) -> None:
    db = Database(tmp_path / "state.db")
    _doc(db, _ALICE, "doc-a", private=True)
    _doc(db, _BOB, "doc-b")
    conversation = db.create_conversation(_ALICE)
    db.record_query(
        _ALICE, "q-1", "question", "answer", False, "hybrid_rrf", conversation_id=conversation
    )
    db.record_feedback(_ALICE, "q-1", "up", None)

    bob = db.metrics(_BOB)
    assert (bob.total_documents, bob.total_queries, bob.feedback_up) == (1, 0, 0)
    alice = db.metrics(_ALICE)
    assert (alice.total_documents, alice.total_queries, alice.feedback_up) == (2, 1, 1)
    admin = db.metrics(_ADMIN)
    assert (admin.total_documents, admin.total_queries, admin.feedback_up) == (2, 1, 1)


def test_a_conversation_with_no_owner_is_visible_only_to_an_admin(tmp_path: Path) -> None:
    """Fail closed: a row from before conversations had owners belongs to
    nobody, not to everybody."""
    db = Database(tmp_path / "state.db")
    conversation = db.create_conversation(_ALICE)
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute("UPDATE conversations SET owner = NULL")

    assert db.conversation_exists(_ALICE, conversation) is False
    assert db.conversation_exists(_ADMIN, conversation) is True
