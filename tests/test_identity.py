"""Principal's visibility and modification rules -- the ONE place they're
written down; every endpoint defers to these (see identity.py)."""

from multimodal_rag.identity import Principal

_ALICE = Principal(principal_id="user:alice", clearance="c2")
_ADMIN = Principal.unrestricted()


def test_a_shared_document_within_clearance_is_visible() -> None:
    assert _ALICE.can_see("c1", private=False, owner="user:bob") is True


def test_a_document_above_clearance_is_not_visible() -> None:
    assert _ALICE.can_see("c3", private=False, owner="user:bob") is False


def test_a_private_document_is_visible_only_to_its_owner() -> None:
    assert _ALICE.can_see("public", private=True, owner="user:alice") is True
    assert _ALICE.can_see("public", private=True, owner="user:bob") is False


def test_a_private_document_with_no_owner_is_visible_to_nobody_but_admin() -> None:
    assert _ALICE.can_see("public", private=True, owner=None) is False
    assert _ADMIN.can_see("public", private=True, owner=None) is True


def test_admin_sees_everything() -> None:
    assert _ADMIN.can_see("c3", private=True, owner="user:bob") is True


def test_only_the_owner_or_an_admin_can_modify() -> None:
    assert _ALICE.can_modify("user:alice") is True
    assert _ALICE.can_modify("user:bob") is False
    assert _ALICE.can_modify(None) is False
    assert _ADMIN.can_modify("user:bob") is True
    assert _ADMIN.can_modify(None) is True


def test_seeing_a_shared_document_does_not_grant_the_right_to_modify_it() -> None:
    assert _ALICE.can_see("public", private=False, owner="user:bob") is True
    assert _ALICE.can_modify("user:bob") is False
