"""MetadataFilterRequest.to_search_filter() -- the one place the
frontend's "Filters" panel request shape becomes a real SearchFilter.
Everything else in schemas.py is a plain pydantic model with no logic
of its own worth testing in isolation."""

from multimodal_rag.api.schemas import MetadataFilterRequest


def test_an_entirely_empty_filter_becomes_no_constraint_at_all() -> None:
    assert MetadataFilterRequest().to_search_filter() is None


def test_tags_and_author_become_any_of_clauses() -> None:
    result = MetadataFilterRequest(tags=["runbook"], author=["Alice"]).to_search_filter()
    assert result is not None
    assert result.any_of == {"tags": ["runbook"], "author": ["Alice"]}
    assert result.date_range == {}


def test_a_full_date_range_is_used_as_given() -> None:
    result = MetadataFilterRequest(
        date_from="2024-01-01", date_to="2024-12-31"
    ).to_search_filter()
    assert result is not None
    assert result.date_range == {"doc_date": ("2024-01-01", "2024-12-31")}


def test_an_open_ended_from_date_widens_to_the_full_possible_range_on_that_side() -> None:
    # The frontend need not already know the corpus's actual earliest
    # doc_date just to set only the "to" side of the range.
    result = MetadataFilterRequest(date_to="2024-12-31").to_search_filter()
    assert result is not None
    assert result.date_range == {"doc_date": ("0000-01-01", "2024-12-31")}


def test_an_open_ended_to_date_widens_to_the_full_possible_range_on_that_side() -> None:
    result = MetadataFilterRequest(date_from="2024-01-01").to_search_filter()
    assert result is not None
    assert result.date_range == {"doc_date": ("2024-01-01", "9999-12-31")}
