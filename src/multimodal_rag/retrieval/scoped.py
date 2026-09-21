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
from ..stores.filters import SearchFilter, merge
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


_CURRENT_ONLY = SearchFilter(any_of={"status": ["current"]})
"""The lifecycle default: a document someone has replaced is not an
answer, it is history. Deliberately NOT part of _security_filter_for --
this is not about who may see a document, and an admin gets it too
(an admin asking a question also wants the current policy)."""


class ScopedRetriever:
    def __init__(
        self,
        inner: Retriever,
        principal: Principal,
        db: "Database",
        include_superseded: bool = False,
    ) -> None:
        self._inner = inner
        self._principal = principal
        self._db = db
        self._include_superseded = include_superseded

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
            # merge() intersects, so the lifecycle default can only narrow
            # what the security filter allows -- and asking for history
            # (include_superseded) removes ONLY the lifecycle clause, never
            # the security one: someone else's private old version stays
            # invisible.
            search_filter=merge(
                _security_filter_for(self._principal),
                None if self._include_superseded else _CURRENT_ONLY,
            ),
        )
        return self._post_check(results)

    def _post_check(self, results: "list[SearchResult]") -> "list[SearchResult]":
        if not results:
            return results
        if self._principal.is_admin and self._include_superseded:
            return results  # admin asked for everything: no rule left to verify

        # The database applies Principal.can_see itself (see
        # Database.get_documents_by_ids), so "not returned" covers both
        # "exists but not visible to this caller" and "not in the
        # catalogue at all" -- neither can be confirmed safe to show,
        # so neither is. Fail closed, same direction as every other gap
        # in this check. Status is verified here too, against the
        # catalogue, so a store copy that hasn't caught up with a
        # supersession can't put a retired version back in an answer.
        documents = {
            doc.doc_id: doc
            for doc in self._db.get_documents_by_ids(
                self._principal, {r.doc_id for r in results}
            )
        }

        def acceptable(result: "SearchResult") -> bool:
            doc = documents.get(result.doc_id)
            if doc is None:
                return False
            return self._include_superseded or doc.metadata.status == "current"

        allowed = [r for r in results if acceptable(r)]
        dropped = [r.chunk_id for r in results if not acceptable(r)]

        if dropped:
            # This should be rare -- it means the store-level filter let
            # through something sqlite (the authority) says this
            # principal can't see or shouldn't get: a missing/stale
            # payload index, a permission or supersession change that
            # hasn't been repatched yet, or a chunk indexed before
            # classification/acl_allow/status existed.
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
