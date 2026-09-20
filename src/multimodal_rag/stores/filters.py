"""Backend-neutral search filter: the shape a caller expresses a
constraint in, translated separately by each store into whatever real
query language it actually speaks (Qdrant's Filter/FieldCondition,
Elasticsearch's bool.filter) -- see qdrant_store._build_query_filter and
elasticsearch_store._build_query.

Deliberately minimal: just what document-scoped retrieval needs.
Range conditions and exclusions still aren't built here -- adding them
ahead of a real caller would mean guessing at a contract instead of
letting one fall out of an actual second use. merge() (below) exists
because that real second use has now arrived: a mandatory security
filter (retrieval/scoped.py) composing with a caller-supplied one
(Retriever.retrieve()'s doc_ids).
"""

from pydantic import BaseModel, ConfigDict


class SearchFilter(BaseModel):
    model_config = ConfigDict(frozen=True)

    any_of: dict[str, list[str]] = {}
    """field -> the values it may take. Multiple entries AND together
    (every field's constraint must hold); the values for one field OR
    together (any one satisfies that field) -- e.g.
    {"doc_id": ["a", "b"]} means "doc_id is a OR b". An empty value list
    for a field is a deliberate "matches nothing" clause (not "no
    constraint" -- that's what leaving the field out entirely means),
    matching what the post-retrieval filter this replaces did for
    doc_ids=[]: exclude everything rather than nothing."""


def merge(*filters: "SearchFilter | None") -> "SearchFilter | None":
    """AND every given filter together. None entries are ignored (no
    constraint to add); an all-None input returns None (no filter at
    all).

    For a field present in MORE than one filter, the merged constraint
    is the INTERSECTION of their value lists, not the union. This is
    the whole security property that makes a mandatory filter mandatory:
    a caller-supplied constraint on a field the security filter already
    constrains can only ever NARROW it -- asking for a value the
    security filter doesn't already allow intersects down to nothing,
    never adds it. There is no way to compose two filters here that
    ends up WIDER than either one alone."""
    present = [f for f in filters if f is not None]
    if not present:
        return None

    merged: dict[str, list[str]] = {}
    for f in present:
        for field, values in f.any_of.items():
            if field in merged:
                merged[field] = [v for v in merged[field] if v in values]
            else:
                merged[field] = list(values)
    return SearchFilter(any_of=merged)
