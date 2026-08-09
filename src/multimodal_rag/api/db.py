"""SQLite persistence for API-level state that doesn't fit in Qdrant/
Elasticsearch, which only know about chunks: which documents have been
ingested, a log of past queries (so /feedback has something to
reference and /metrics has something to summarize), and feedback on
those queries.

A short-lived connection per call, rather than one held open for the
app's lifetime -- this isn't the hot path (retrieval/generation are),
so the simplicity of "always open a fresh connection" outweighs any
pooling benefit, and it sidesteps sqlite3's cross-thread
connection-sharing restrictions entirely: every call, even ones run in
a threadpool worker thread, gets its own connection.

Plain sqlite3, not an ORM -- this is the first persistence layer in the
project, and raw SQL keeps exactly what's stored and how fully visible
rather than adding a new abstraction layer on top of everything else.
"""

import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from .schemas import DocumentSummary, IngestResponse, MetricsResponse


class QueryNotFoundError(LookupError):
    """Raised by record_feedback() when query_id doesn't match any
    logged query -- POST /feedback turns this into a 404."""


class Database:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        # Re-applied on every connection, not just once at construction.
        # CREATE TABLE IF NOT EXISTS is cheap and idempotent, and this
        # makes the schema self-healing if the underlying file is ever
        # deleted or replaced out from under an already-running process
        # -- exactly what tests/live/wipe_db.py does, and it's
        # explicitly meant to be safe to run against a live server.
        # Schema-only-at-__init__ isn't: the file getting wiped while a
        # server holds a Database instance left every later query
        # hitting "no such table" until that process restarted.
        self._create_tables(conn)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _create_tables(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                num_parent_chunks INTEGER NOT NULL,
                num_child_chunks INTEGER NOT NULL,
                ingested_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                refused INTEGER NOT NULL,
                retrieval_method TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                query_id TEXT NOT NULL REFERENCES queries(query_id),
                rating TEXT NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            )
            """
        )

    def get_document(self, doc_id: str) -> DocumentSummary | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return self._row_to_document(row) if row is not None else None

    def get_document_by_content_hash(self, content_hash: str) -> DocumentSummary | None:
        """Finds a document by its CONTENT, regardless of filename/doc_id
        -- used to catch "this exact file was already ingested under a
        different-looking name" (e.g. the same file uploaded once via
        the single-file picker and once via the folder picker, which
        reports a path-prefixed name for the same bytes). If more than
        one doc_id happens to share this content_hash, the most
        recently ingested one wins -- an arbitrary but stable tiebreak,
        not a case expected to come up often."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? ORDER BY ingested_at DESC LIMIT 1",
                (content_hash,),
            ).fetchone()
        return self._row_to_document(row) if row is not None else None

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentSummary:
        return DocumentSummary(
            doc_id=row["doc_id"],
            filename=row["filename"],
            content_hash=row["content_hash"],
            num_parent_chunks=row["num_parent_chunks"],
            num_child_chunks=row["num_child_chunks"],
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
        )

    def upsert_document(
        self,
        doc_id: str,
        filename: str,
        content_hash: str,
        num_parent_chunks: int,
        num_child_chunks: int,
        status: Literal["ingested", "already_ingested"] = "ingested",
    ) -> IngestResponse:
        ingested_at = datetime.now(UTC)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, filename, content_hash, num_parent_chunks, num_child_chunks,
                     ingested_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename = excluded.filename,
                    content_hash = excluded.content_hash,
                    num_parent_chunks = excluded.num_parent_chunks,
                    num_child_chunks = excluded.num_child_chunks,
                    ingested_at = excluded.ingested_at
                """,
                (
                    doc_id,
                    filename,
                    content_hash,
                    num_parent_chunks,
                    num_child_chunks,
                    ingested_at.isoformat(),
                ),
            )
        return IngestResponse(
            doc_id=doc_id,
            filename=filename,
            status=status,
            num_parent_chunks=num_parent_chunks,
            num_child_chunks=num_child_chunks,
            ingested_at=ingested_at,
        )

    def list_documents(self) -> list[DocumentSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [self._row_to_document(row) for row in rows]

    def wipe_documents(self) -> int:
        """Deletes every row from `documents` -- the sqlite side of a
        full corpus reset (see HybridIndexer.delete_all() for the store
        side). Leaves queries/feedback history alone; those are a log of
        past activity, not corpus state, and wiping the corpus doesn't
        make past questions or feedback about it meaningless."""
        with self._connect() as conn:
            deleted = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            conn.execute("DELETE FROM documents")
        return int(deleted)

    def record_query(
        self, query_id: str, question: str, answer: str, refused: bool, retrieval_method: str
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO queries
                    (query_id, question, answer, refused, retrieval_method, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    question,
                    answer,
                    int(refused),
                    retrieval_method,
                    datetime.now(UTC).isoformat(),
                ),
            )

    def query_exists(self, query_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM queries WHERE query_id = ?", (query_id,)
            ).fetchone()
        return row is not None

    def record_feedback(self, query_id: str, rating: str, comment: str | None) -> str:
        if not self.query_exists(query_id):
            raise QueryNotFoundError(f"No query logged with query_id={query_id!r}")
        feedback_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO feedback (feedback_id, query_id, rating, comment, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (feedback_id, query_id, rating, comment, datetime.now(UTC).isoformat()),
            )
        return feedback_id

    def metrics(self) -> MetricsResponse:
        with self._connect() as conn:
            total_documents, total_chunks = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(num_parent_chunks + num_child_chunks), 0) "
                "FROM documents"
            ).fetchone()
            total_queries, total_refused = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(refused), 0) FROM queries"
            ).fetchone()
            feedback_up = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE rating = 'up'"
            ).fetchone()[0]
            feedback_down = conn.execute(
                "SELECT COUNT(*) FROM feedback WHERE rating = 'down'"
            ).fetchone()[0]

        refusal_rate = (total_refused / total_queries) if total_queries else 0.0
        return MetricsResponse(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_queries=total_queries,
            refusal_rate=refusal_rate,
            feedback_up=feedback_up,
            feedback_down=feedback_down,
        )
