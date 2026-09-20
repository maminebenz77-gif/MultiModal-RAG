"""Document-level metadata: the tags a WHOLE document carries
(classification, author, dates, free tags...) as opposed to
chunking.schema.ChunkMetadata (how one chunk was cut FROM that
document).

Lives in this neutral, top-level module -- not under api/ or stores/ --
because both sides need it: api/db.py (the sqlite `documents` table, the
source of truth) and stores/* (the denormalized copy written onto every
chunk's payload, so a search can filter on it without a join) both
import it, and stores must never depend on api/ (see stores/base.py's
own ports/adapters boundary).
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

Classification = Literal["public", "c1", "c2", "c3"]
"""Confidentiality level, cumulative from least to most sensitive --
someone cleared for c2 can see public/c1/c2, not c3. Required on every
document (DocumentMetadata.classification has no default): a document
with no classification is one nobody can reason about access for, and
defaulting it silently to something "safe-looking" is exactly the kind
of swallowed requirement that turns into a leak or a lockout nobody can
diagnose from the outside."""

_CLEARANCE_LADDER: list[Classification] = ["public", "c1", "c2", "c3"]


def clearance_or_below(level: Classification) -> list[Classification]:
    """Every level a caller cleared at `level` may see -- e.g. "c2" ->
    ["public", "c1", "c2"]. Used to build the classification side of a
    security filter (see retrieval/scoped.py): a principal's clearance
    is one value, but the filter needs the whole allowed RANGE."""
    return _CLEARANCE_LADDER[: _CLEARANCE_LADDER.index(level) + 1]


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    classification: Classification
    """Required, no default -- see the Classification docstring above.
    Rejected at the API boundary (POST /ingest) if missing, not silently
    assumed."""

    private: bool = False
    """"Only the owner should see this." Enforced everywhere a document
    can be seen -- search (retrieval/scoped.py), the document list, and
    every lookup (Principal.can_see, identity.py). A private document
    with no `owner` is visible to nobody but an admin."""

    owner: str | None = None
    """Who owns this, compared against Principal.principal_id
    (identity.py): it gates `private` visibility AND who may edit or
    delete the document. At upload it is set from the authenticated
    caller for everyone except an admin, who may name any owner; only an
    admin can reassign it afterwards."""

    author: str | None = None
    """The document's actual author, as opposed to `owner` (who
    uploaded it into this system) -- often the same person, not always
    (someone uploading a colleague's report, a shared policy doc)."""

    doc_date: str | None = None
    """When the document's CONTENT was written or published -- an ISO
    date string ("2024-05-01"), caller-supplied. Deliberately distinct
    from `documents.ingested_at` (when this system first saw it, see
    api/db.py) -- conflating "written" with "ingested" is a common way
    to pick the wrong document when two versions disagree."""

    data_type: str | None = None
    """What kind of document this is (policy, runbook, spec, contract,
    report, ...). Free text for now, not yet a closed vocabulary --
    fine for storing and displaying, not yet reliable as a filter (two
    uploaders spelling the same type two different ways would silently
    split what should be one facet)."""

    tags: list[str] = []
    """Free-form labels -- the general escape hatch for whatever
    doesn't fit a dedicated field above. Validated against nothing yet."""

    def to_payload(self) -> dict[str, Any]:
        """The flat dict merged into every chunk's stored payload
        (Qdrant + Elasticsearch) -- the one place these field names are
        spelled for that purpose, so the two backends can't drift apart.
        See stores.qdrant_store._to_point and
        stores.elasticsearch_store.index_chunks.

        Includes `acl_allow` -- a field with NO corresponding attribute
        on this model, computed fresh from private/owner every time this
        is called rather than stored as its own field. That's
        deliberate: acl_allow only exists to be fast to FILTER on (a
        single array-contains check, see retrieval/scoped.py). Deriving
        it (rather than storing it as a separate field on this model)
        means it can't disagree with private/owner INSIDE this model --
        but the copy in Qdrant/Elasticsearch can still go stale if a
        writer sends a partial dict and skips this method (exactly the
        PATCH /documents/{doc_id} bug this design once had). Every write
        path must therefore send the whole to_payload(), never a
        hand-picked subset."""
        payload = self.model_dump()
        payload["acl_allow"] = self._acl_allow()
        return payload

    def _acl_allow(self) -> list[str]:
        if not self.private:
            return ["*"]
        # A private document with no owner recorded is visible to NOBODY
        # -- not even an accident away from being effectively public.
        # Fail closed: an unclaimed private document should read as
        # "broken, fix the owner", not "oops, shared with everyone."
        return [self.owner] if self.owner else []
