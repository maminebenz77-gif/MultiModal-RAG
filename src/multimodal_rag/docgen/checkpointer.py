"""SQLite-backed checkpointer for the docgen graph -- what lets a run
survive a process restart and resume an ask_human interrupt exactly
where it left off, rather than starting the question over.

Reaching an interrupt doesn't actually need a checkpointer (LangGraph
only requires one to process a Command(resume=...)); this exists so a
REAL run can resume at all, across a killed-and-restarted process, not
just across calls within one already-running Python process the way
an in-memory checkpointer would.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

from ..config import PROJECT_ROOT

DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "data" / "docgen_checkpoints.sqlite"
"""Separate from api/main.py's DEFAULT_DB_PATH on purpose -- that file
is the live app's document catalogue; this one is graph execution
state (which node is next, what's pending on an interrupt), a
different kind of data with a different lifecycle."""

_ALLOWED_MSGPACK_MODULES = [
    ("multimodal_rag.docgen.sources", "SourceSpec"),
    ("multimodal_rag.docgen.nodes.retrieval", "RetrievedChunk"),
    ("multimodal_rag.stores.schema", "SearchResult"),
    ("multimodal_rag.chunking.schema", "ChunkElement"),
]
"""Every custom (non-builtin) type that can end up inside DocGenState
and therefore gets written into a checkpoint -- SourceSpec (state.sources),
RetrievedChunk and the SearchResult/ChunkElement it wraps (current.chunks,
Attempt.chunks). langgraph-checkpoint's default is to deserialize ANY
type it finds with just a loud warning (real run output: "Deserializing
unregistered type ... This will be blocked in a future version") --
that default exists for backward compatibility, not because it's safe:
unrestricted deserialization of a checkpoint file is exactly the kind
of thing an attacker who can write to that file would exploit. Listing
the actual types we put there, explicitly, is the documented fix
(langgraph_checkpoint_sqlite's own README calls this out as the
security-relevant setting), not a cosmetic one to silence a warning."""


@contextmanager
def build_checkpointer(path: Path = DEFAULT_CHECKPOINT_PATH) -> Iterator[SqliteSaver]:
    path.parent.mkdir(parents=True, exist_ok=True)
    serde = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES)
    # Replicates SqliteSaver.from_conn_string()'s own connection setup --
    # that classmethod doesn't expose a way to pass a custom `serde`,
    # so this is the only way to get our allowlist (rather than the
    # unrestricted default) actually used.
    with closing(sqlite3.connect(str(path), check_same_thread=False)) as conn:
        yield SqliteSaver(conn, serde=serde)
