"""DocumentMetadata itself -- mainly the derived acl_allow field and the
classification ladder, since everything else is a plain pydantic model
already exercised through the store/API tests."""

import pytest
from pydantic import ValidationError

from multimodal_rag.metadata import DocumentMetadata, clearance_or_below


def test_classification_has_no_default() -> None:
    try:
        DocumentMetadata()  # type: ignore[call-arg]
        raised = False
    except Exception:
        raised = True
    assert raised, "DocumentMetadata() without classification must fail, not default"


def test_doc_date_accepts_a_real_iso_date() -> None:
    metadata = DocumentMetadata(classification="public", doc_date="2024-05-01")
    assert metadata.doc_date == "2024-05-01"


def test_doc_date_of_none_is_fine() -> None:
    metadata = DocumentMetadata(classification="public", doc_date=None)
    assert metadata.doc_date is None


def test_doc_date_rejects_a_malformed_value() -> None:
    # Elasticsearch now maps doc_date as a real `date` field (for the
    # date-range filter panel), which would otherwise turn this into a
    # bulk-indexing failure deep inside HybridIndexer.index() instead of
    # a 422 at the API boundary -- the earliest point a caller's typo
    # can be caught.
    with pytest.raises(ValidationError, match="doc_date must be an ISO date"):
        DocumentMetadata(classification="public", doc_date="not-a-date")


def test_to_payload_marks_a_non_private_document_shared_with_everyone() -> None:
    metadata = DocumentMetadata(classification="public")
    assert metadata.to_payload()["acl_allow"] == ["*"]


def test_to_payload_marks_a_private_document_visible_only_to_its_owner() -> None:
    metadata = DocumentMetadata(classification="c2", private=True, owner="alice@example.com")
    assert metadata.to_payload()["acl_allow"] == ["alice@example.com"]


def test_to_payload_marks_an_unclaimed_private_document_visible_to_nobody() -> None:
    """Fail closed: private with no owner is broken, not "shared with
    everyone by accident.\""""
    metadata = DocumentMetadata(classification="c1", private=True)
    assert metadata.to_payload()["acl_allow"] == []


def test_clearance_or_below_is_cumulative() -> None:
    assert clearance_or_below("public") == ["public"]
    assert clearance_or_below("c1") == ["public", "c1"]
    assert clearance_or_below("c2") == ["public", "c1", "c2"]
    assert clearance_or_below("c3") == ["public", "c1", "c2", "c3"]


def test_a_new_document_is_a_current_first_version_with_no_end_date() -> None:
    payload = DocumentMetadata(classification="public").to_payload()

    assert payload["status"] == "current"
    assert payload["version"] == 1
    assert payload["effective_to"] is None
    assert payload["doc_family_id"] is None


def test_lifecycle_fields_travel_in_the_store_payload() -> None:
    """Every write path sends the whole to_payload(); if a lifecycle field
    were missing from it, the stores' copy would silently go stale."""
    metadata = DocumentMetadata(
        classification="c1",
        status="superseded",
        doc_family_id="fam",
        version=3,
        effective_from="2026-01-01",
        effective_to="2026-06-01T00:00:00+00:00",
    )

    payload = metadata.to_payload()

    assert payload["status"] == "superseded"
    assert payload["doc_family_id"] == "fam"
    assert payload["version"] == 3
    assert payload["effective_from"] == "2026-01-01"
    assert payload["effective_to"] == "2026-06-01T00:00:00+00:00"
