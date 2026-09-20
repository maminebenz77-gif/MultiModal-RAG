"""ScopedRetriever: wraps a Retriever and forces a server-derived
security SearchFilter into every retrieve() call.

Lives BELOW the agent, not inside it: AgentChain lets the model choose
its own tool arguments (generation/agent.py), so a prompt-injected
document could otherwise steer what gets searched -- the security
filter has to sit somewhere the model cannot reach or influence at all.
ScopedRetriever satisfies the same shape RetrieverLike
(generation/chain.py) already describes, so RagChain/AgentChain can be
handed one instead of a plain Retriever with no code change on their
side; they never need to know scoping exists.

Two layers, not one:
  1. A store-level SearchFilter (acl_allow / classification), so a
     search that can only see 1% of the corpus still gets real,
     store-filtered results -- not five results post-filtered down to
     zero (see stores/filters.py and Retriever.retrieve()'s
     search_filter parameter).
  2. A post-check against sqlite (the catalogue, the authoritative
     source -- see api/db.py), re-verifying every result AFTER the
     search. Cheap: one indexed lookup over the handful of doc_ids in a
     top_k result set, not the whole corpus. This is what makes a
     permission change take effect immediately even if the denormalized
     payload copy hasn't been repatched yet, and what catches a missing
     payload index silently letting an unfiltered result through.
"""

import logging
from typing import TYPE_CHECKING

from ..identity import Principal
from ..metadata import clearance_or_below
from ..stores.filters import SearchFilter
from .retriever import Retriever
from .schema import RetrievalMethod

if TYPE_CHECKING:
    from ..api.db import Database
    from ..stores.schema import SearchResult

_logger = logging.getLogger(__name__)


def _security_filter_for(principal: Principal) -> SearchFilter | None:
    """None means "no constraint" -- deliberately, not a special case:
    is_admin bypasses ACL/classification entirely (a real, audited
    superuser property), so there's nothing to encode into a filter
    rather than some magic wildcard value every document would need to
    carry. merge() (stores/filters.py) already treats None as "nothing
    to add", so this composes correctly with doc_ids/search_filter
    without ScopedRetriever needing to special-case it either."""
    if principal.is_admin:
        return None
    return SearchFilter(
        any_of={
            "acl_allow": [principal.principal_id, "*"],
            "classification": list(clearance_or_below(principal.clearance)),
        }
    )


class ScopedRetriever:
    def __init__(self, inner: Retriever, principal: Principal, db: "Database") -> None:
        self._inner = inner
        self._principal = principal
        self._db = db

    def retrieve(
        self,
        query: str,
        method: RetrievalMethod,
        top_k: int,
        *,
        rerank: bool = False,
        resolve_parent_context: bool = False,
        doc_ids: list[str] | None = None,
    ) -> "list[SearchResult]":
        results = self._inner.retrieve(
            query,
            method=method,
            top_k=top_k,
            rerank=rerank,
            resolve_parent_context=resolve_parent_context,
            doc_ids=doc_ids,
            search_filter=_security_filter_for(self._principal),
        )
        return self._post_check(results)

    def _post_check(self, results: "list[SearchResult]") -> "list[SearchResult]":
        if not results or self._principal.is_admin:
            return results

        # The database applies Principal.can_see itself (see
        # Database.get_documents_by_ids), so "not returned" covers both
        # "exists but not visible to this caller" and "not in the
        # catalogue at all" -- neither can be confirmed safe to show,
        # so neither is. Fail closed, same direction as every other gap
        # in this check.
        visible = {
            doc.doc_id
            for doc in self._db.get_documents_by_ids(
                self._principal, {r.doc_id for r in results}
            )
        }
        allowed = [r for r in results if r.doc_id in visible]
        dropped = [r.chunk_id for r in results if r.doc_id not in visible]

        if dropped:
            # This should be rare -- it means the store-level filter let
            # through something sqlite (the authority) says this
            # principal can't see: a missing/stale payload index, a
            # permission change that hasn't been repatched yet, or a
            # chunk indexed before classification/acl_allow existed.
            # ERROR, not a silent drop, because each occurrence is worth
            # investigating even though this check is doing exactly its
            # job by catching it.
            _logger.error(
                "ScopedRetriever post-check dropped %d result(s) for principal %r that "
                "the store-level filter should not have returned: chunk_ids=%s",
                len(dropped),
                self._principal.principal_id,
                dropped,
            )
        return allowed
