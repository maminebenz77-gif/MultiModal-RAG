"""Backend-neutral search filter: the shape a caller expresses a
constraint in, translated separately by each store into whatever real
query language it actually speaks (Qdrant's Filter/FieldCondition,
Elasticsearch's bool.filter) -- see qdrant_store._build_query_filter and
elasticsearch_store._build_query.

Deliberately minimal: just what document-scoped retrieval needs.
merge() (below) exists because a real second use arrived early on: a
mandatory security filter (retrieval/scoped.py) composing with a
caller-supplied one (Retriever.retrieve()'s doc_ids). date_range is the
next real use to arrive -- a user-facing date filter (frontend "Filters"
panel) -- rather than something guessed at ahead of a caller.
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

    date_range: dict[str, tuple[str, str]] = {}
    """field -> (from_date, to_date), both inclusive ISO date strings
    ("YYYY-MM-DD"). A field absent here has no date constraint. Only one
    range per field is expressible here -- see merge() for what happens
    when two filters both constrain the same field. Lexicographic
    comparison on "YYYY-MM-DD" strings already matches chronological
    order, so callers never need to parse these into real dates just to
    narrow one."""


def merge(*filters: "SearchFilter | None") -> "SearchFilter | None":
    """AND every given filter together. None entries are ignored (no
    constraint to add); an all-None input returns None (no filter at
    all).

    For an any_of field present in MORE than one filter, the merged
    constraint is the INTERSECTION of their value lists, not the union.
    This is the whole security property that makes a mandatory filter
    mandatory: a caller-supplied constraint on a field the security
    filter already constrains can only ever NARROW it -- asking for a
    value the security filter doesn't already allow intersects down to
    nothing, never adds it. There is no way to compose two filters here
    that ends up WIDER than either one alone.

    A date_range field present in more than one filter narrows the same
    way: the merged range is the OVERLAP (the later of the two
    from-dates, the earlier of the two to-dates), never a range wider
    than either input alone."""
    present = [f for f in filters if f is not None]
    if not present:
        return None

    merged_any_of: dict[str, list[str]] = {}
    merged_date_range: dict[str, tuple[str, str]] = {}
    for f in present:
        for field, values in f.any_of.items():
            if field in merged_any_of:
                merged_any_of[field] = [v for v in merged_any_of[field] if v in values]
            else:
                merged_any_of[field] = list(values)
        for field, (from_date, to_date) in f.date_range.items():
            if field in merged_date_range:
                existing_from, existing_to = merged_date_range[field]
                merged_date_range[field] = (
                    max(existing_from, from_date),
                    min(existing_to, to_date),
                )
            else:
                merged_date_range[field] = (from_date, to_date)
    return SearchFilter(any_of=merged_any_of, date_range=merged_date_range)
