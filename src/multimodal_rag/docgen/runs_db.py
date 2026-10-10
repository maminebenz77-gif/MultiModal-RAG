"""SQLite persistence for discoverability of docgen runs -- a thin,
separate table, never merged into api/db.py's own schema (docgen stays
a module that doesn't modify existing ones).

This exists only for what the LangGraph checkpointer itself can't give
us: enumerating a caller's own thread_ids, a human-readable title, and
an error message if a background run thread dies outright. Live
run/paused/done status is NEVER tracked here -- that's always
re-derived fresh from graph.get_state() (see api/routers/docgen.py),
so there is exactly one source of truth for it, not two that could
drift apart.

Same house style as api/db.py: plain sqlite3, a fresh short-lived
connection per call, CREATE TABLE IF NOT EXISTS.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..config import PROJECT_ROOT
from ..identity import Principal

DEFAULT_RUNS_DB_PATH = PROJECT_ROOT / "data" / "docgen_runs.db"


@dataclass(frozen=True)
class RunRow:
    thread_id: str
    title: str
    owner: str
    created_at: datetime
    last_error: str | None


class DocgenRunsDB:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS docgen_runs (
                thread_id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_error TEXT
            )
            """
        )
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _row_to_run(row: sqlite3.Row) -> RunRow:
        return RunRow(
            thread_id=row["thread_id"],
            title=row["title"],
            owner=row["owner"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_error=row["last_error"],
        )

    def create_run(self, principal: Principal, thread_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO docgen_runs (thread_id, owner, title, created_at, last_error) "
                "VALUES (?, ?, ?, ?, NULL)",
                (thread_id, principal.principal_id, title, datetime.now(UTC).isoformat()),
            )

    def list_runs(self, principal: Principal) -> list[RunRow]:
        with self._connect() as conn:
            if principal.is_admin:
                rows = conn.execute("SELECT * FROM docgen_runs ORDER BY created_at DESC").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM docgen_runs WHERE owner = ? ORDER BY created_at DESC",
                    (principal.principal_id,),
                ).fetchall()
        return [self._row_to_run(row) for row in rows]

    def get_run(self, principal: Principal, thread_id: str) -> RunRow | None:
        """None for "doesn't exist" AND for "exists but isn't yours" --
        deliberately the same thing, the same reasoning as api/db.py's
        DocumentNotFoundError docstring: the router 404s either way,
        never 403s, so probing someone else's thread_id can't even
        confirm it exists."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM docgen_runs WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        if row is None:
            return None
        run = self._row_to_run(row)
        if not principal.is_admin and run.owner != principal.principal_id:
            return None
        return run

    def record_error(self, thread_id: str, message: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE docgen_runs SET last_error = ? WHERE thread_id = ?", (message, thread_id)
            )

    def clear_error(self, thread_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE docgen_runs SET last_error = NULL WHERE thread_id = ?", (thread_id,)
            )
