"""SearchFilter's merge() -- the mechanism that lets a mandatory
security filter compose with a caller-supplied one (see
retrieval/scoped.py). The property that actually matters: a merge can
only NARROW, never widen or replace, what either filter alone allowed.
"""

from multimodal_rag.stores.filters import SearchFilter, merge


def test_merge_of_nothing_is_none() -> None:
    assert merge() is None
    assert merge(None, None) is None


def test_merge_with_one_real_filter_returns_it_unchanged() -> None:
    f = SearchFilter(any_of={"doc_id": ["a", "b"]})
    assert merge(f, None) == f
    assert merge(None, f) == f


def test_merge_unions_fields_that_appear_in_only_one_filter() -> None:
    security = SearchFilter(any_of={"classification": ["public", "c1"]})
    caller = SearchFilter(any_of={"doc_id": ["a"]})

    result = merge(security, caller)

    assert result is not None
    assert result.any_of == {"classification": ["public", "c1"], "doc_id": ["a"]}


def test_merge_intersects_a_field_present_in_both_filters() -> None:
    # The load-bearing case: a mandatory filter (e.g. a security clause)
    # and a caller-supplied one both constrain the SAME field. The
    # result must be the intersection, not the union -- otherwise a
    # caller could WIDEN what the mandatory filter allows just by
    # supplying their own values for that field.
    security = SearchFilter(any_of={"acl_allow": ["user:alice", "*"]})
    caller = SearchFilter(any_of={"acl_allow": ["*", "user:mallory"]})

    result = merge(security, caller)

    assert result is not None
    assert result.any_of == {"acl_allow": ["*"]}


def test_merge_cannot_be_used_to_escape_the_security_filter() -> None:
    """A caller (or an LLM agent picking its own filter arguments) asking
    for a value the security filter doesn't already allow must intersect
    down to nothing, not somehow gain access to it."""
    security = SearchFilter(any_of={"classification": ["public"]})
    caller = SearchFilter(any_of={"classification": ["c3"]})

    result = merge(security, caller)

    assert result is not None
    assert result.any_of == {"classification": []}


def test_merge_of_three_filters_intersects_pairwise() -> None:
    a = SearchFilter(any_of={"doc_id": ["x", "y", "z"]})
    b = SearchFilter(any_of={"doc_id": ["x", "y"]})
    c = SearchFilter(any_of={"doc_id": ["y"]})

    result = merge(a, b, c)

    assert result is not None
    assert result.any_of == {"doc_id": ["y"]}


def test_merge_with_one_real_date_range_filter_returns_it_unchanged() -> None:
    f = SearchFilter(date_range={"doc_date": ("2024-01-01", "2024-12-31")})
    assert merge(f, None) == f


def test_merge_narrows_a_date_range_present_in_both_filters_to_the_overlap() -> None:
    # Same security property as any_of's intersection case: a caller's
    # own date range can only shrink what a mandatory range already
    # allows, never extend past it in either direction.
    wide = SearchFilter(date_range={"doc_date": ("2024-01-01", "2025-12-31")})
    narrower = SearchFilter(date_range={"doc_date": ("2025-01-01", "2025-06-30")})

    result = merge(wide, narrower)

    assert result is not None
    assert result.date_range == {"doc_date": ("2025-01-01", "2025-06-30")}


def test_merge_a_date_range_that_does_not_overlap_produces_an_empty_range() -> None:
    # No special-casing needed: an inverted (from > to) range is simply
    # never satisfied by any real date, the same "matches nothing"
    # outcome any_of's empty-list case produces deliberately.
    a = SearchFilter(date_range={"doc_date": ("2024-01-01", "2024-06-30")})
    b = SearchFilter(date_range={"doc_date": ("2025-01-01", "2025-06-30")})

    result = merge(a, b)

    assert result is not None
    assert result.date_range == {"doc_date": ("2025-01-01", "2024-06-30")}


def test_merge_unions_date_range_fields_present_in_only_one_filter() -> None:
    a = SearchFilter(any_of={"classification": ["public"]})
    b = SearchFilter(date_range={"doc_date": ("2024-01-01", "2024-12-31")})

    result = merge(a, b)

    assert result is not None
    assert result.any_of == {"classification": ["public"]}
    assert result.date_range == {"doc_date": ("2024-01-01", "2024-12-31")}
