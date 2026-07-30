"""Assembles retrieved chunks into a token-budgeted, numbered context
block for the RAG prompt.
"""

import tiktoken

from ..stores.schema import SearchResult

# cl100k_base is accurate for OpenAI-family models and a reasonable
# approximation for others -- the "same chain on both backends" design
# means this count is never exact for whatever model runs on the server
# profile, only close enough to bound cost/latency, not to bill against.
_ENCODING_NAME = "cl100k_base"


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return len(encoding.encode(text))


def assemble_context(results: list[SearchResult], token_budget: int) -> list[SearchResult]:
    """Greedily include ranked results (best first) until the budget
    would be exceeded. Drops the TAIL of the ranking — lower-relevance
    chunks — rather than truncating any individual chunk's text; a
    partial chunk read out of context is worse than one fewer whole
    chunk. Always includes at least the top result, even if it alone
    exceeds the budget — an empty context defeats the point."""
    included: list[SearchResult] = []
    used = 0
    for result in results:
        cost = count_tokens(result.text)
        if included and used + cost > token_budget:
            break
        included.append(result)
        used += cost
    return included


def format_context_block(index: int, result: SearchResult) -> str:
    return f"[{index}] (source: {result.source}{_format_location(result)})\n{result.text}"


def _format_location(result: SearchResult) -> str:
    if result.pages:
        return f", page {', '.join(str(p) for p in result.pages)}"
    if result.slides:
        return f", slide {', '.join(str(s) for s in result.slides)}"
    return ""
