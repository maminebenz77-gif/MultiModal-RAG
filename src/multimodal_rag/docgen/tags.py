"""Tag helpers for docgen.

Introduces no new metadata field or store schema -- a docgen source is
identified by the same free-form `tags` field every document already
carries (metadata.DocumentMetadata.tags), filterable today via
SearchFilter.any_of({"tags": [...]}) with no changes needed anywhere in
stores/ or retrieval/.

The "docgen:<role>:" prefix built by build_docgen_tag() is a naming
convention, not a mechanism: it exists purely so a later run's
reuse-suggestions can tell "a batch this workflow itself created" apart
from any other tag already in the corpus, by matching a string prefix.
"""

from __future__ import annotations

from typing import Literal

from ..api.db import Database
from ..identity import Principal

Role = Literal["task", "ref"]

_PREFIX = "docgen"


def build_docgen_tag(role: Role, label: str) -> str:
    """A new tag for a docgen-ingested batch, e.g. build_docgen_tag("task",
    "q3-audit") -> "docgen:task:q3-audit". `label` is the part the user
    actually chose and recognizes; the prefix is added, never typed."""
    label = label.strip()
    if not label:
        raise ValueError("label must not be empty")
    if ":" in label:
        raise ValueError(f"label must not contain ':' (the tag separator): {label!r}")
    return f"{_PREFIX}:{role}:{label}"


def list_known_tags(
    db: Database, *, prefix: str | None = None, principal: Principal | None = None
) -> list[str]:
    """All distinct tags currently in use, optionally narrowed to ones
    starting with `prefix` (e.g. "docgen:task:" to see only this
    workflow's own past task-doc batches). Read-only reuse of
    Database.list_documents -- the same data GET /documents returns, and
    the same way the frontend's own "previously used tags" filter is
    built (frontend/app.py's _distinct_tags), just called directly
    instead of over HTTP, since docgen has no API layer of its own."""
    principal = principal or Principal.unrestricted()
    tags: set[str] = set()
    for doc in db.list_documents(principal):
        tags.update(doc.metadata.tags)
    if prefix is not None:
        tags = {tag for tag in tags if tag.startswith(prefix)}
    return sorted(tags)
