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

from .schemas import DocumentSummary, IngestResponse, MetricsResponse


class QueryNotFoundError(LookupError):
    """Raised by record_feedback() when query_id doesn't match any
    logged query -- POST /feedback turns this into a 404."""


class Database:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
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

    def upsert_document(
        self, doc_id: str, filename: str, num_parent_chunks: int, num_child_chunks: int
    ) -> IngestResponse:
        ingested_at = datetime.now(UTC)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, filename, num_parent_chunks, num_child_chunks, ingested_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename = excluded.filename,
                    num_parent_chunks = excluded.num_parent_chunks,
                    num_child_chunks = excluded.num_child_chunks,
                    ingested_at = excluded.ingested_at
                """,
                (doc_id, filename, num_parent_chunks, num_child_chunks, ingested_at.isoformat()),
            )
        return IngestResponse(
            doc_id=doc_id,
            filename=filename,
            status="ingested",
            num_parent_chunks=num_parent_chunks,
            num_child_chunks=num_child_chunks,
            ingested_at=ingested_at,
        )

    def list_documents(self) -> list[DocumentSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [
            DocumentSummary(
                doc_id=row["doc_id"],
                filename=row["filename"],
                num_parent_chunks=row["num_parent_chunks"],
                num_child_chunks=row["num_child_chunks"],
                ingested_at=datetime.fromisoformat(row["ingested_at"]),
            )
            for row in rows
        ]

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
