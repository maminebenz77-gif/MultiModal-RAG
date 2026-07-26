"""Shared chunk-ID construction, used by every chunking strategy.

IDs are content-addressed, not purely positional: they include a hash of
the chunk's own text, not just its document/strategy/index. Purely
positional IDs (doc::strategy::index) look identical across re-ingestion
runs even when the underlying text changed — which means there's no
cheap way to tell "this chunk actually changed" from "this chunk is
unchanged but we re-ran the pipeline" without diffing full text. A
content hash makes that distinction part of the ID itself: an unchanged
chunk re-embeds to the exact same ID; a changed one gets a new ID.

Trade-off worth naming, not hiding: when a chunk's text *does* change,
its old ID doesn't automatically disappear from the vector store — the
new content lands under a new ID, and the stale one is now an orphan
unless something explicitly deletes it (e.g. by doc_id) during
re-ingestion. That cleanup doesn't exist yet; this only gives the
pipeline the *information* needed to do it later.
"""

import hashlib


def chunk_id(doc_id: str, strategy: str, index: int, text: str) -> str:
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:10]
    return f"{doc_id}::{strategy}::{index}::{text_hash}"
