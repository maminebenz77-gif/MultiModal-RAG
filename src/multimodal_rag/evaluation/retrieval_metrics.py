"""Layer 1 (retrieval) evaluation metrics: recall@k, MRR, nDCG@k.

Pure functions over a ranked list and a golden item's expected_sources --
no I/O, no LLM calls, deterministic. Relevance is FILENAME-level (the
`.source` attribute), matching the golden set's ground truth (see
data/golden_set.json) -- chunk_id is content-hashed and would silently
break on any re-chunking, so it can't be the relevance key.
"""

import math
from collections.abc import Sequence
from typing import Protocol


class HasSource(Protocol):
    """Structural, not `SearchResult` itself -- these functions only ever
    read `.source`. `SearchResult` already satisfies this; widened so a
    lighter-weight stand-in works too (see evaluation/langfuse_experiment.py,
    which reconstructs retrieval results from a Langfuse experiment task's
    JSON-serializable output rather than passing real SearchResult objects
    through, deliberately -- keeps SearchResult.elements' possible base64
    image data out of a trace payload)."""

    source: str


def recall_at_k(retrieved: Sequence[HasSource], expected_sources: list[str]) -> float:
    """Fraction of the expected sources that appear anywhere in `retrieved`.

    Not binary "was anything relevant found" -- a golden item can name more
    than one acceptable source (near-identical facts stated in two
    documents), and finding one out of two is real partial credit, not a
    pass/fail.
    """
    if not expected_sources:
        raise ValueError("recall_at_k is undefined for a golden item with no expected sources")
    found = {r.source for r in retrieved} & set(expected_sources)
    return len(found) / len(expected_sources)


def mrr(retrieved: Sequence[HasSource], expected_sources: list[str]) -> float:
    """1 / (rank of the first relevant result), 1-indexed; 0.0 if none of
    `retrieved` is relevant. Unlike recall_at_k, this is sensitive to
    *where* the first relevant result lands -- rank 1 and rank k score
    identically under recall_at_k but very differently here, which matters
    because context assembly truncates by token budget (see
    generation/chain.py), not by "was it anywhere in the top-k."
    """
    if not expected_sources:
        raise ValueError("mrr is undefined for a golden item with no expected sources")
    expected = set(expected_sources)
    for rank, result in enumerate(retrieved, start=1):
        if result.source in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[HasSource], expected_sources: list[str]) -> float:
    """Normalized Discounted Cumulative Gain: like recall_at_k, but a
    relevant result ranked higher counts for more than the same result
    ranked lower (via a log2 position discount), and -- unlike mrr --
    every relevant result contributes, not just the first.

    Relevance is binary per retrieved position (its source is/isn't in
    expected_sources); the ideal ranking used for normalization assumes
    one relevant result per expected source, in the best possible
    positions (1..len(expected_sources)).

    A document contributes at most ONE hit to the score, no matter how
    many of its chunks land in `retrieved` -- relevance here is
    document-level (one expected source = one relevant unit), matching
    what IDCG assumes below. Without this, a document with several
    retrieved child chunks (routine with ParentChildChunker) would score
    multiple hits against IDCG's single ideal slot for it, pushing nDCG
    above 1.0 -- caught by actually running this against the live stack,
    not by the unit tests, which only ever used one chunk per source.
    """
    if not expected_sources:
        raise ValueError("ndcg_at_k is undefined for a golden item with no expected sources")
    expected = set(expected_sources)

    dcg = 0.0
    already_credited: set[str] = set()
    for rank, result in enumerate(retrieved, start=1):
        if result.source in expected and result.source not in already_credited:
            dcg += 1.0 / math.log2(rank + 1)
            already_credited.add(result.source)
    ideal_hits = min(len(expected_sources), len(retrieved))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
