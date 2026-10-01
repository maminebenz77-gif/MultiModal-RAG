"""The retrieve step of docgen's per-question loop: for one question,
looks up chunks from each of its required sources and labels each
chunk with which source it came from.

A pure function, not a graph node yet -- LangGraph's typed state
doesn't exist as a type until docgen/state.py (a later phase), and
keeping this decoupled from it means it's testable with a real
retriever and a couple of SourceSpecs, no graph machinery involved.
The eventual graph node (a later phase) will just narrow
state.sources down to one question's `sources_required` and call this.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...retrieval.retriever import Retriever
from ...retrieval.schema import RetrievalMethod
from ...stores.filters import SearchFilter
from ...stores.schema import SearchResult
from ..sources import SourceRole, SourceSpec


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: SearchResult
    source_role: SourceRole
    """Which docgen source this came from (task_docs/reference_kb) --
    deliberately a separate field, not SearchResult.source, which
    already means something else (the chunk's source FILENAME)."""


def retrieve_for_question(
    query: str,
    sources: list[SourceSpec],
    retriever: Retriever,
    *,
    method: RetrievalMethod = RetrievalMethod.HYBRID_RRF,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """Calls Retriever.retrieve() once per source, scoped to that
    source's tag, and merges the results labeled by which source they
    came from. A source matching nothing just contributes no chunks to
    the merged list -- judging whether that's enough to answer is
    validate_answer's job, not this function's."""
    labeled: list[RetrievedChunk] = []
    for source in sources:
        results = retriever.retrieve(
            query,
            method=method,
            top_k=top_k,
            search_filter=SearchFilter(any_of={"tags": [source.tag]}),
        )
        labeled.extend(
            RetrievedChunk(chunk=result, source_role=source.role) for result in results
        )
    return labeled
