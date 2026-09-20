"""Principal: who is calling, and what they're cleared to see.

Neutral, top-level module -- not under api/ -- for the same reason
metadata.py is: both api/identity.py (validates a real OIDC token into
one of these) and retrieval/scoped.py (uses one to build a security
filter) need it, and retrieval must never depend on api/ (see
stores/base.py's ports/adapters boundary, which applies here too --
api/ is the outer layer that imports from the rest of this package,
never the reverse).
"""

from dataclasses import dataclass

from .metadata import Classification, clearance_or_below


@dataclass(frozen=True)
class Principal:
    principal_id: str
    """Stable identifier for the caller, e.g. "user:alice@example.com" --
    what a document's acl_allow is compared against (see
    retrieval/scoped.py)."""

    clearance: Classification
    """The highest classification this caller may see. Compared against
    a document's own classification via metadata.clearance_or_below()."""

    is_admin: bool = False
    """True bypasses every ACL/classification check entirely (see
    retrieval/scoped.py's _security_filter_for) -- a real, audited
    superuser property, not something ordinary OIDC claims should set by
    default. Currently only Principal.unrestricted() sets it."""

    def can_see(self, classification: Classification, private: bool, owner: str | None) -> bool:
        """THE visibility rule -- the one place it's written down. Search
        (retrieval/scoped.py encodes the same rule as a store filter),
        the document list, and every other read path defer to this, so
        "who may see a document" can't quietly mean different things in
        different endpoints. A private document with no owner is visible
        to nobody, admins aside (fail closed)."""
        if self.is_admin:
            return True
        if private and owner != self.principal_id:
            return False
        return classification in clearance_or_below(self.clearance)

    def can_modify(self, owner: str | None) -> bool:
        """Editing or deleting is stricter than seeing: only the owner
        (or an admin). Being able to SEE a shared document says nothing
        about being allowed to change it."""
        return self.is_admin or (owner is not None and owner == self.principal_id)

    @classmethod
    def unrestricted(cls) -> "Principal":
        """The auth_mode=disabled principal. principal_id="*" is
        cosmetic here (is_admin=True means acl_allow is never even
        consulted -- see retrieval/scoped.py), not itself a magic value
        anything is matched against."""
        return cls(principal_id="*", clearance="c3", is_admin=True)
