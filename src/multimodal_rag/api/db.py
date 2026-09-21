"""SQLite persistence for API-level state that doesn't fit in Qdrant/
Elasticsearch, which only know about chunks: which documents have been
ingested, a log of past queries (so /feedback has something to
reference and /metrics has something to summarize) grouped into
conversations (so /query can load history server-side instead of the
client re-sending it every call), the citations each query produced,
and feedback on those queries.

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

import json
import logging
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..generation.schema import Citation
from ..identity import Principal
from ..metadata import DocumentMetadata
from .schemas import (
    ChunkElementOut,
    CitationOut,
    ConversationMessageOut,
    ConversationSummaryOut,
    DocumentMetadataOut,
    DocumentSummary,
    IngestResponse,
    MetricsResponse,
)

_logger = logging.getLogger(__name__)


class AccessDeniedError(PermissionError):
    """The caller can SEE the thing they tried to change, but isn't
    allowed to change it (a 403). Deliberately NOT raised for "can't see
    it at all" -- that's indistinguishable from "doesn't exist" (a 404),
    on purpose: a 403 there would confirm the thing exists, which is
    exactly what someone probing for other people's data wants to know."""


class DocumentNotFoundError(LookupError):
    """A document the caller named doesn't exist -- or exists but they
    cannot see it, which is deliberately the same thing (see
    AccessDeniedError). POST /ingest turns this into a 404 when
    supersedes_doc_id points at one."""


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
                ingested_at TEXT NOT NULL,
                classification TEXT NOT NULL DEFAULT '',
                private INTEGER NOT NULL DEFAULT 0,
                owner TEXT,
                author TEXT,
                doc_date TEXT,
                data_type TEXT,
                tags TEXT NOT NULL DEFAULT '[]',
                effective_from TEXT,
                status TEXT NOT NULL DEFAULT 'current',
                doc_family_id TEXT,
                version INTEGER NOT NULL DEFAULT 1,
                effective_to TEXT
            )
            """
        )
        # Self-healed for the same reason as citations' columns below --
        # an already-existing documents table (from before DocumentMetadata
        # existed) needs these added directly. classification's DEFAULT ''
        # here is a SQLite necessity (ADD COLUMN needs a constant), not a
        # real value -- the app layer never writes '' deliberately; POST
        # /ingest requires a real classification on every new document (see
        # metadata.DocumentMetadata), so '' only ever marks a row that
        # predates this migration.
        Database._ensure_column(conn, "documents", "classification", "TEXT NOT NULL DEFAULT ''")
        Database._ensure_column(conn, "documents", "private", "INTEGER NOT NULL DEFAULT 0")
        Database._ensure_column(conn, "documents", "owner", "TEXT")
        Database._ensure_column(conn, "documents", "author", "TEXT")
        Database._ensure_column(conn, "documents", "doc_date", "TEXT")
        Database._ensure_column(conn, "documents", "data_type", "TEXT")
        Database._ensure_column(conn, "documents", "tags", "TEXT NOT NULL DEFAULT '[]'")
        # Lifecycle (see metadata.Status). A row from before these existed
        # is, correctly, a current first version.
        Database._ensure_column(conn, "documents", "effective_from", "TEXT")
        Database._ensure_column(conn, "documents", "status", "TEXT NOT NULL DEFAULT 'current'")
        Database._ensure_column(conn, "documents", "doc_family_id", "TEXT")
        Database._ensure_column(conn, "documents", "version", "INTEGER NOT NULL DEFAULT 1")
        Database._ensure_column(conn, "documents", "effective_to", "TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                title TEXT,
                owner TEXT
            )
            """
        )
        Database._ensure_column(conn, "conversations", "title", "TEXT")
        # Who the conversation belongs to (a Principal.principal_id).
        # Nullable on purpose: a row that predates this column has no
        # owner, and a NULL owner matches nobody -- only an admin can
        # see it. Fail closed, not "belongs to everyone."
        Database._ensure_column(conn, "conversations", "owner", "TEXT")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                refused INTEGER NOT NULL,
                retrieval_method TEXT NOT NULL,
                created_at TEXT NOT NULL,
                conversation_id TEXT REFERENCES conversations(conversation_id),
                needs_clarification INTEGER NOT NULL DEFAULT 0,
                latency_ms REAL
            )
            """
        )
        # Self-healed rather than only declared above -- an already-existing
        # queries table (from before this migration) needs these added onto
        # it directly; CREATE TABLE IF NOT EXISTS only helps a fresh database.
        Database._ensure_column(
            conn, "queries", "conversation_id", "TEXT REFERENCES conversations(conversation_id)"
        )
        Database._ensure_column(
            conn, "queries", "needs_clarification", "INTEGER NOT NULL DEFAULT 0"
        )
        # No NOT NULL/DEFAULT here on purpose -- a query recorded before
        # this column existed has genuinely unknown latency, and AVG()
        # in metrics() already skips NULLs. Defaulting old rows to 0
        # would silently drag avg_latency_ms down instead of just being
        # honest that we don't know.
        Database._ensure_column(conn, "queries", "latency_ms", "REAL")
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS citations (
                query_id TEXT NOT NULL REFERENCES queries(query_id),
                marker INTEGER NOT NULL,
                chunk_id TEXT NOT NULL,
                source TEXT NOT NULL,
                doc_id TEXT NOT NULL DEFAULT '',
                pages TEXT NOT NULL,
                slides TEXT NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                elements TEXT NOT NULL DEFAULT '[]',
                PRIMARY KEY (query_id, marker)
            )
            """
        )
        # Self-healed for the same reason as queries.conversation_id above:
        # an already-existing citations table needs these added directly.
        Database._ensure_column(conn, "citations", "text", "TEXT NOT NULL DEFAULT ''")
        Database._ensure_column(conn, "citations", "elements", "TEXT NOT NULL DEFAULT '[]'")
        Database._ensure_column(conn, "citations", "doc_id", "TEXT NOT NULL DEFAULT ''")

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_type: str) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_type}")

    def _fetch_document(self, doc_id: str) -> DocumentSummary | None:
        """Raw, UNSCOPED row fetch -- private on purpose. Every public
        read below applies the caller's visibility on top of this; only
        code that has just written the row itself (and so already knows
        the caller may see it) uses this directly."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        if row is None:
            return None
        readable = self._readable([row])
        return readable[0] if readable else None  # unreadable reads as not found

    @staticmethod
    def _visible(principal: Principal, doc: DocumentSummary) -> bool:
        return principal.can_see(
            doc.metadata.classification, doc.metadata.private, doc.metadata.owner
        )

    def get_document(self, principal: Principal, doc_id: str) -> DocumentSummary | None:
        """None both when the document doesn't exist AND when it exists
        but the caller can't see it -- the two are deliberately
        indistinguishable (see AccessDeniedError)."""
        doc = self._fetch_document(doc_id)
        return doc if doc is not None and self._visible(principal, doc) else None

    def get_own_document(self, principal: Principal, doc_id: str) -> DocumentSummary | None:
        """A document the caller OWNS, whether or not they can currently
        SEE it. The two differ: an owner can raise their own document's
        classification above their own clearance, after which
        get_document() returns None for them. Ingest needs this one --
        it decides "is this a re-upload of a document I already have?",
        and a caller who can't see their own document must still get
        the edit path (diff the chunks, delete the stale ones), not the
        brand-new-document path that leaves the old version's chunks
        behind in the stores. Safe because ownership is still checked:
        this never returns someone else's document."""
        doc = self._fetch_document(doc_id)
        return doc if doc is not None and principal.can_modify(doc.metadata.owner) else None

    def get_documents_by_ids(
        self, principal: Principal, doc_ids: set[str]
    ) -> list[DocumentSummary]:
        """Fetch multiple documents in one round trip, keeping only the
        ones this caller may see -- built for ScopedRetriever's
        post-check (retrieval/scoped.py): re-verifying a handful of
        just-returned search results against the CURRENT, authoritative
        row (not the possibly-stale denormalized copy on each chunk's
        payload) is only cheap because it's one indexed lookup over a
        small `IN (...)` set, not one query per result. An id that
        doesn't exist and one the caller can't see both simply don't
        come back -- the caller treats them the same."""
        if not doc_ids:
            return []
        placeholders = ",".join("?" for _ in doc_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM documents WHERE doc_id IN ({placeholders})",
                tuple(doc_ids),
            ).fetchall()
        return [d for d in self._readable(rows) if self._visible(principal, d)]

    def get_document_by_content_hash(
        self, principal: Principal, content_hash: str
    ) -> DocumentSummary | None:
        """Finds a document by its CONTENT, regardless of filename/doc_id
        -- used to catch "this exact file was already ingested under a
        different-looking name" (e.g. the same file uploaded once via
        the single-file picker and once via the folder picker, which
        reports a path-prefixed name for the same bytes).

        Only documents the CALLER can see count. That's a security
        property, not a convenience: matching against everyone's
        documents would let anyone upload a file they already have and
        learn -- from the duplicate_of filename -- that someone else has
        it too. If more than one visible doc_id shares this content, the
        most recently ingested one wins."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? ORDER BY ingested_at DESC",
                (content_hash,),
            ).fetchall()
        for doc in self._readable(rows):
            if self._visible(principal, doc):
                return doc
        return None

    @classmethod
    def _readable(cls, rows: list[sqlite3.Row]) -> list[DocumentSummary]:
        """The rows that can be read, skipping (loudly) any that can't.

        A row from before classification existed has classification '' and
        _row_to_metadata refuses to read it. Raising there is right for a
        single row -- inventing a classification would be a guess about
        access -- but letting it propagate out of a LIST turned one old
        row into a 500 on every endpoint that lists, counts or looks up
        documents. So a list skips it. It is not hidden quietly: each one
        is logged at ERROR with what to do about it, and it is never
        assigned a classification, so it stays invisible (and unsearchable)
        until someone deliberately re-ingests or backfills it."""
        readable: list[DocumentSummary] = []
        for row in rows:
            try:
                readable.append(cls._row_to_document(row))
            except ValueError as exc:
                _logger.error(
                    "Skipping unreadable document row %r: %s -- re-ingest it (wipe, or delete and "
                    "upload again) or backfill its metadata.",
                    row["doc_id"],
                    exc,
                )
        return readable

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentSummary:
        return DocumentSummary(
            doc_id=row["doc_id"],
            filename=row["filename"],
            content_hash=row["content_hash"],
            num_parent_chunks=row["num_parent_chunks"],
            num_child_chunks=row["num_child_chunks"],
            ingested_at=datetime.fromisoformat(row["ingested_at"]),
            metadata=Database._row_to_metadata(row),
        )

    @staticmethod
    def _row_to_metadata(row: sqlite3.Row) -> DocumentMetadataOut:
        # classification == "" only for a row written before this column
        # existed -- POST /ingest has required a real classification on
        # every write since (see metadata.DocumentMetadata), so this can
        # only mean a database that predates the migration, not a
        # legitimate "no classification" state. Raising here (rather than
        # inventing a default) is deliberate: a document with no real
        # classification is one nobody can safely reason about access
        # for, so it should be visibly broken, not silently shown as
        # some guessed level.
        classification = row["classification"]
        if classification not in ("public", "c1", "c2", "c3"):
            raise ValueError(
                f"document {row['doc_id']!r} has no valid classification "
                f"({classification!r}) -- it predates DocumentMetadata and needs "
                "a manual backfill before it can be read."
            )
        return DocumentMetadataOut(
            classification=classification,
            private=bool(row["private"]),
            owner=row["owner"],
            author=row["author"],
            doc_date=row["doc_date"],
            data_type=row["data_type"],
            tags=json.loads(row["tags"]),
            effective_from=row["effective_from"],
            status=row["status"],
            doc_family_id=row["doc_family_id"],
            version=row["version"],
            effective_to=row["effective_to"],
        )

    def upsert_document(
        self,
        principal: Principal,
        doc_id: str,
        filename: str,
        content_hash: str,
        num_parent_chunks: int,
        num_child_chunks: int,
        metadata: DocumentMetadata,
        status: Literal["ingested", "already_ingested"] = "ingested",
    ) -> IngestResponse:
        ingested_at = datetime.now(UTC)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT owner FROM documents WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if existing is not None and not principal.can_modify(existing["owner"]):
                # Defense in depth: doc_id is derived from the uploader
                # AND the filename (routers/ingest.py), so two different
                # users can't normally land on the same doc_id -- but if
                # anything ever makes them, refusing to overwrite someone
                # else's row is the difference between a rejected upload
                # and a document takeover.
                raise AccessDeniedError(f"Not allowed to overwrite document {doc_id!r}")
            conn.execute(
                """
                INSERT INTO documents
                    (doc_id, filename, content_hash, num_parent_chunks, num_child_chunks,
                     ingested_at, classification, private, owner, author, doc_date,
                     data_type, tags, effective_from, status, doc_family_id, version,
                     effective_to)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename = excluded.filename,
                    content_hash = excluded.content_hash,
                    num_parent_chunks = excluded.num_parent_chunks,
                    num_child_chunks = excluded.num_child_chunks,
                    ingested_at = excluded.ingested_at,
                    classification = excluded.classification,
                    private = excluded.private,
                    owner = excluded.owner,
                    author = excluded.author,
                    doc_date = excluded.doc_date,
                    data_type = excluded.data_type,
                    tags = excluded.tags,
                    effective_from = excluded.effective_from,
                    status = excluded.status,
                    doc_family_id = excluded.doc_family_id,
                    version = excluded.version,
                    effective_to = excluded.effective_to
                """,
                (
                    doc_id,
                    filename,
                    content_hash,
                    num_parent_chunks,
                    num_child_chunks,
                    ingested_at.isoformat(),
                    metadata.classification,
                    int(metadata.private),
                    metadata.owner,
                    metadata.author,
                    metadata.doc_date,
                    metadata.data_type,
                    json.dumps(metadata.tags),
                    metadata.effective_from,
                    metadata.status,
                    metadata.doc_family_id,
                    metadata.version,
                    metadata.effective_to,
                ),
            )
        return IngestResponse(
            doc_id=doc_id,
            filename=filename,
            status=status,
            num_parent_chunks=num_parent_chunks,
            num_child_chunks=num_child_chunks,
            ingested_at=ingested_at,
            metadata=DocumentMetadataOut(**metadata.model_dump()),
        )

    def update_document_metadata(
        self, principal: Principal, doc_id: str, patch: dict[str, Any]
    ) -> DocumentSummary | None:
        """Merge `patch` (from DocumentMetadataUpdate.model_dump(
        exclude_unset=True) -- only the fields a caller actually sent)
        into the stored row. None if the caller can't see doc_id (same
        as it not existing); AccessDeniedError if they can see it but
        don't own it, or if a non-admin tries to change `owner` --
        reassigning ownership is an admin action. An empty patch is a
        no-op that still returns the current row."""
        doc = self.get_document(principal, doc_id)
        if doc is None:
            return None
        if not principal.can_modify(doc.metadata.owner):
            raise AccessDeniedError(f"Not allowed to modify document {doc_id!r}")
        if "owner" in patch and not principal.is_admin:
            raise AccessDeniedError("Only an admin may change a document's owner")
        if not patch:
            return doc

        allowed = {
            "classification",
            "private",
            "owner",
            "author",
            "doc_date",
            "data_type",
            "tags",
            "effective_from",
            "status",
        }
        unknown = set(patch) - allowed
        if unknown:
            # Defensive, not expected in practice -- the router validates
            # against DocumentMetadataUpdate's fixed field set before this
            # is ever called. Guards against a field name ending up in a
            # dynamically-built SQL statement by any other path.
            raise ValueError(f"Unknown metadata field(s): {sorted(unknown)}")

        if "status" in patch:
            # effective_to follows status automatically -- it is not
            # something a caller sets. Retiring stamps it; undoing the
            # retirement clears it, so a current document never carries an
            # end date.
            patch = {
                **patch,
                "effective_to": (
                    datetime.now(UTC).isoformat() if patch["status"] == "superseded" else None
                ),
            }

        set_clauses = []
        values: list[Any] = []
        for key, value in patch.items():
            set_clauses.append(f"{key} = ?")
            if key == "private":
                values.append(int(value))
            elif key == "tags":
                values.append(json.dumps(value))
            else:
                values.append(value)
        values.append(doc_id)

        with self._connect() as conn:
            conn.execute(
                f"UPDATE documents SET {', '.join(set_clauses)} WHERE doc_id = ?",
                values,
            )
        # Raw fetch, not get_document(principal, ...): the owner may have
        # just changed something (e.g. raised the classification past
        # their own clearance) that makes the row invisible to THEM
        # afterwards -- they still just made this change and are owed
        # its result.
        return self._fetch_document(doc_id)

    def list_documents(self, principal: Principal) -> list[DocumentSummary]:
        # Filtered in Python through Principal.can_see (not an equivalent
        # SQL WHERE) so the visibility rule exists in exactly one place.
        # Fine while this endpoint returns every visible document
        # anyway; pushing it into SQL is the optimization if that ever
        # stops being true.
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY ingested_at DESC"
            ).fetchall()
        return [d for d in self._readable(rows) if self._visible(principal, d)]

    def wipe_documents(self, principal: Principal) -> int:
        """Deletes every row from `documents` -- the sqlite side of a
        full corpus reset (see HybridIndexer.delete_all() for the store
        side). Admin only: it removes EVERYONE's documents. Leaves
        queries/feedback history alone; those are a log of past
        activity, not corpus state, and wiping the corpus doesn't make
        past questions or feedback about it meaningless."""
        if not principal.is_admin:
            raise AccessDeniedError("Only an admin may wipe the whole corpus")
        with self._connect() as conn:
            deleted = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            conn.execute("DELETE FROM documents")
        return int(deleted)

    def delete_document(self, principal: Principal, doc_id: str) -> None:
        """Deletes one row from `documents` -- the sqlite side of a
        single-document delete (see HybridIndexer.delete_document() for
        the store side). Same "leave query/feedback history alone"
        reasoning as wipe_documents(), just scoped to one document.
        A no-op if the caller can't see it (it may as well not exist);
        AccessDeniedError if they can see it but don't own it."""
        doc = self.get_document(principal, doc_id)
        if doc is None:
            return
        if not principal.can_modify(doc.metadata.owner):
            raise AccessDeniedError(f"Not allowed to delete document {doc_id!r}")
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    @staticmethod
    def _owns_conversation(
        conn: sqlite3.Connection, principal: Principal, conversation_id: str
    ) -> bool:
        """Owner-or-admin -- which is exactly Principal.can_modify. A
        conversation has no "shared" state, so seeing and owning are the
        same thing here, unlike documents. False both for "doesn't
        exist" and "someone else's": callers must not be able to tell
        the two apart."""
        row = conn.execute(
            "SELECT owner FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return row is not None and principal.can_modify(row["owner"])

    def record_query(
        self,
        principal: Principal,
        query_id: str,
        question: str,
        answer: str,
        refused: bool,
        retrieval_method: str,
        *,
        conversation_id: str | None = None,
        needs_clarification: bool = False,
        citations: list[Citation] | None = None,
        latency_ms: float | None = None,
    ) -> None:
        with self._connect() as conn:
            # A query is only ever recorded into a conversation the
            # caller owns. A query with NO conversation belongs to
            # nobody, so only an admin may write one.
            if conversation_id is None:
                if not principal.is_admin:
                    raise AccessDeniedError("Only an admin may record a query with no conversation")
            elif not self._owns_conversation(conn, principal, conversation_id):
                raise AccessDeniedError(f"Not allowed to write to conversation {conversation_id!r}")
            conn.execute(
                """
                INSERT INTO queries
                    (query_id, question, answer, refused, retrieval_method, created_at,
                     conversation_id, needs_clarification, latency_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    question,
                    answer,
                    int(refused),
                    retrieval_method,
                    datetime.now(UTC).isoformat(),
                    conversation_id,
                    int(needs_clarification),
                    latency_ms,
                ),
            )
            for citation in citations or []:
                conn.execute(
                    """
                    INSERT INTO citations
                        (query_id, marker, chunk_id, source, doc_id, pages, slides, text, elements)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        query_id,
                        citation.marker,
                        citation.chunk_id,
                        citation.source,
                        citation.doc_id,
                        json.dumps(citation.pages),
                        json.dumps(citation.slides),
                        citation.text,
                        json.dumps([e.model_dump() for e in citation.elements]),
                    ),
                )

    def create_conversation(self, principal: Principal) -> str:
        conversation_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (conversation_id, created_at, owner) VALUES (?, ?, ?)",
                (conversation_id, datetime.now(UTC).isoformat(), principal.principal_id),
            )
        return conversation_id

    def conversation_exists(self, principal: Principal, conversation_id: str) -> bool:
        """True only if it exists AND the caller owns it -- so this can
        be used directly as a 404 gate without becoming an oracle for
        other people's conversation ids."""
        with self._connect() as conn:
            return self._owns_conversation(conn, principal, conversation_id)

    def delete_conversation(self, principal: Principal, conversation_id: str) -> None:
        """Deletes a conversation and everything hanging off it. No
        ON DELETE CASCADE in this schema, so the order matters:
        feedback and citations both reference queries, which reference
        the conversation -- deleting outside-in avoids ever leaving an
        orphaned row that would dangle after a partial failure. A no-op if
        the caller doesn't own it."""
        with self._connect() as conn:
            if not self._owns_conversation(conn, principal, conversation_id):
                return
            conn.execute(
                """
                DELETE FROM feedback
                WHERE query_id IN (SELECT query_id FROM queries WHERE conversation_id = ?)
                """,
                (conversation_id,),
            )
            conn.execute(
                """
                DELETE FROM citations
                WHERE query_id IN (SELECT query_id FROM queries WHERE conversation_id = ?)
                """,
                (conversation_id,),
            )
            conn.execute("DELETE FROM queries WHERE conversation_id = ?", (conversation_id,))
            conn.execute(
                "DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,)
            )

    def get_recent_turns(
        self, principal: Principal, conversation_id: str, limit: int
    ) -> list[tuple[str, str]]:
        """Last `limit` (question, answer) pairs, oldest-first -- fed
        straight into AgentChain.answer()'s `history` parameter. Windowed
        here rather than at the API schema layer (the old client-owned
        `history` field capped itself at max_length=10) since the server
        now owns the full conversation. Empty for a conversation the
        caller doesn't own."""
        with self._connect() as conn:
            if not self._owns_conversation(conn, principal, conversation_id):
                return []
            rows = conn.execute(
                """
                SELECT question, answer FROM queries
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [(row["question"], row["answer"]) for row in reversed(rows)]

    def set_conversation_title(
        self, principal: Principal, conversation_id: str, title: str
    ) -> None:
        with self._connect() as conn:
            if not self._owns_conversation(conn, principal, conversation_id):
                return
            conn.execute(
                "UPDATE conversations SET title = ? WHERE conversation_id = ?",
                (title, conversation_id),
            )

    def list_conversations(
        self, principal: Principal, limit: int = 20
    ) -> list[ConversationSummaryOut]:
        """Conversations that have at least one recorded turn, most
        recently active first -- a conversation row created right before
        a /query call that then failed has nothing worth resuming, so
        it's excluded rather than showing up as an empty entry."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    c.conversation_id AS conversation_id,
                    COALESCE(
                        c.title,
                        (SELECT question FROM queries WHERE conversation_id = c.conversation_id
                            ORDER BY created_at LIMIT 1)
                    ) AS preview,
                    (SELECT COUNT(*) FROM queries WHERE conversation_id = c.conversation_id)
                        AS message_count,
                    (SELECT MAX(created_at) FROM queries WHERE conversation_id = c.conversation_id)
                        AS updated_at
                FROM conversations c
                WHERE EXISTS (SELECT 1 FROM queries WHERE conversation_id = c.conversation_id)
                  AND (? OR c.owner = ?)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (int(principal.is_admin), principal.principal_id, limit),
            ).fetchall()
        return [
            ConversationSummaryOut(
                conversation_id=row["conversation_id"],
                preview=row["preview"],
                message_count=row["message_count"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    def get_conversation_messages(
        self, principal: Principal, conversation_id: str
    ) -> list[ConversationMessageOut]:
        with self._connect() as conn:
            if not self._owns_conversation(conn, principal, conversation_id):
                return []
            rows = conn.execute(
                "SELECT * FROM queries WHERE conversation_id = ? ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
            messages = []
            for row in rows:
                citation_rows = conn.execute(
                    "SELECT * FROM citations WHERE query_id = ? ORDER BY marker",
                    (row["query_id"],),
                ).fetchall()
                messages.append(
                    ConversationMessageOut(
                        query_id=row["query_id"],
                        question=row["question"],
                        answer=row["answer"],
                        citations=[
                            CitationOut(
                                marker=c["marker"],
                                chunk_id=c["chunk_id"],
                                source=c["source"],
                                doc_id=c["doc_id"],
                                pages=json.loads(c["pages"]),
                                slides=json.loads(c["slides"]),
                                text=c["text"],
                                elements=[
                                    ChunkElementOut(**e) for e in json.loads(c["elements"])
                                ],
                            )
                            for c in citation_rows
                        ],
                        refused=bool(row["refused"]),
                        needs_clarification=bool(row["needs_clarification"]),
                        retrieval_method=row["retrieval_method"],
                        created_at=datetime.fromisoformat(row["created_at"]),
                    )
                )
        return messages

    def query_exists(self, principal: Principal, query_id: str) -> bool:
        """True only if the query exists AND sits in a conversation the
        caller owns -- so /feedback's 404 can't be used to probe for
        other people's query ids."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT conversation_id FROM queries WHERE query_id = ?", (query_id,)
            ).fetchone()
            if row is None:
                return False
            if row["conversation_id"] is None:
                return principal.is_admin
            return self._owns_conversation(conn, principal, row["conversation_id"])

    def record_feedback(
        self, principal: Principal, query_id: str, rating: str, comment: str | None
    ) -> str:
        if not self.query_exists(principal, query_id):
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

    def metrics(self, principal: Principal) -> MetricsResponse:
        """Summary numbers over what THIS caller may see: an admin gets
        the whole system; everyone else gets their own -- documents
        they can see, and queries/feedback from conversations they own.
        Global counts would leak how much other people are using (or
        storing in) the system, which on a small deployment is
        identifying on its own."""
        with self._connect() as conn:
            doc_rows = conn.execute(
                "SELECT * FROM documents"
            ).fetchall()
            visible_docs = [d for d in self._readable(doc_rows) if self._visible(principal, d)]
            total_documents = len(visible_docs)
            total_chunks = sum(d.num_parent_chunks + d.num_child_chunks for d in visible_docs)

            # Fixed SQL fragments only (never caller input) -- the
            # caller's identity travels as a bound parameter.
            params: tuple[str, ...]
            if principal.is_admin:
                q_joins, q_where, params = "", "1 = 1", ()
            else:
                q_joins = "JOIN conversations c ON c.conversation_id = q.conversation_id"
                q_where, params = "c.owner = ?", (principal.principal_id,)

            total_queries, total_refused = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(q.refused), 0) "
                f"FROM queries q {q_joins} WHERE {q_where}",
                params,
            ).fetchone()
            # AVG() over a column that's NULL for every pre-migration row
            # (see latency_ms's self-heal above) ignores those NULLs on its
            # own -- no COALESCE needed to exclude them, only to turn "no
            # rows with a known latency at all" into 0.0 instead of None.
            avg_latency_ms = conn.execute(
                f"SELECT COALESCE(AVG(q.latency_ms), 0) FROM queries q {q_joins} WHERE {q_where}",
                params,
            ).fetchone()[0]
            feedback_up = conn.execute(
                "SELECT COUNT(*) FROM feedback f "
                f"JOIN queries q ON q.query_id = f.query_id {q_joins} "
                f"WHERE {q_where} AND f.rating = 'up'",
                params,
            ).fetchone()[0]
            feedback_down = conn.execute(
                "SELECT COUNT(*) FROM feedback f "
                f"JOIN queries q ON q.query_id = f.query_id {q_joins} "
                f"WHERE {q_where} AND f.rating = 'down'",
                params,
            ).fetchone()[0]

        refusal_rate = (total_refused / total_queries) if total_queries else 0.0
        feedback_rate = (
            (feedback_up + feedback_down) / total_queries if total_queries else 0.0
        )
        return MetricsResponse(
            total_documents=total_documents,
            total_chunks=total_chunks,
            total_queries=total_queries,
            refusal_rate=refusal_rate,
            feedback_up=feedback_up,
            feedback_down=feedback_down,
            avg_latency_ms=avg_latency_ms,
            feedback_rate=feedback_rate,
        )
