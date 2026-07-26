"""Retrieval method selection.

Unlike providers/stores, this isn't chosen per deployment environment
and swapped through config — it's chosen per query, by the caller
(eventually a UI dropdown). That's why Retriever is one class with a
method parameter, not a ports/adapters ABC with one implementation per
method.
"""

from enum import StrEnum


class RetrievalMethod(StrEnum):
    COSINE = "cosine"
    MMR = "mmr"
    BM25 = "bm25"
    HYBRID_RRF = "hybrid_rrf"
