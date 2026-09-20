"""DocumentMetadata itself -- mainly the derived acl_allow field and the
classification ladder, since everything else is a plain pydantic model
already exercised through the store/API tests."""

from multimodal_rag.metadata import DocumentMetadata, clearance_or_below


def test_classification_has_no_default() -> None:
    try:
        DocumentMetadata()  # type: ignore[call-arg]
        raised = False
    except Exception:
        raised = True
    assert raised, "DocumentMetadata() without classification must fail, not default"


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
