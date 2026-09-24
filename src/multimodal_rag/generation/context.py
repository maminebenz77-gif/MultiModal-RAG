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

_TABLE_ELEMENT_TYPE = "table"
"""Mirrors ingestion.schema.ElementType.TABLE's value as a plain string --
generation deliberately has no import dependency on the ingestion layer
(same reasoning as ChunkElement.type itself being a plain str, not the
enum -- see chunking/schema.py)."""


def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding(_ENCODING_NAME)
    return len(encoding.encode(text))


def _generation_text(result: SearchResult) -> str:
    """What the generator actually reads for this chunk -- usually just
    result.text (also what was embedded), but for a table-derived chunk
    result.text is the LLM summary used to make it matchable by search
    (see chunking/text.py's element_text), not the real numbers. The raw
    table survives on the chunk's element snapshot regardless of what got
    embedded in its place, so reconstruct from there when a table is
    present. Harmless for a chunk with no table: rebuilding from
    `elements` reproduces the same text result.text already has, since
    neither substitutes anything for non-table element types.

    Known narrow limitation: `elements` is the whole PARENT section's
    elements even on a raw child chunk (see ParentChildChunker), so if a
    caller ever generates directly from an un-resolved child
    (resolve_parent_context=False -- not what any real call site uses
    today) and that child's section contains a table, this returns the
    whole section's reconstructed text rather than just the child's
    narrow slice. Not worth extra machinery for a path nothing currently
    exercises for generation.
    """
    if not result.elements:
        return result.text
    has_table = any(el.type == _TABLE_ELEMENT_TYPE for el in result.elements)
    if not has_table:
        return result.text
    parts = [text for el in result.elements if (text := (el.description or el.text))]
    return "\n\n".join(parts) if parts else result.text


def assemble_context(results: list[SearchResult], token_budget: int) -> list[SearchResult]:
    """Greedily include ranked results (best first) until the budget
    would be exceeded. Drops the TAIL of the ranking — lower-relevance
    chunks — rather than truncating any individual chunk's text; a
    partial chunk read out of context is worse than one fewer whole
    chunk. Always includes at least the top result, even if it alone
    exceeds the budget — an empty context defeats the point. Budgets
    against _generation_text, not result.text -- a table-derived chunk's
    real cost is the raw table sent to the generator, not the short
    summary that happened to get embedded."""
    included: list[SearchResult] = []
    used = 0
    for result in results:
        cost = count_tokens(_generation_text(result))
        if included and used + cost > token_budget:
            break
        included.append(result)
        used += cost
    return included


def format_context_block(index: int, result: SearchResult) -> str:
    location = _format_location(result)
    provenance = _format_provenance(result)
    return f"⟦{index}⟧ (source: {result.source}{location}{provenance})\n{_generation_text(result)}"


def _format_location(result: SearchResult) -> str:
    if result.pages:
        return f", page {', '.join(str(p) for p in result.pages)}"
    if result.slides:
        return f", slide {', '.join(str(s) for s in result.slides)}"
    return ""


def _format_provenance(result: SearchResult) -> str:
    """Version lineage (metadata.py), shown to the model so it can act on
    the CONFLICT_RESOLUTION_RULE (prompt.py) instead of silently picking
    one of two disagreeing sources. Silent for a document with none of
    this tagged -- version 1, current, no effective date -- so an
    ordinary citation looks exactly as it did before this existed."""
    parts = []
    if result.version > 1:
        parts.append(f"v{result.version}")
    if result.status != "current":
        parts.append(result.status)
    if result.effective_from:
        parts.append(f"effective {result.effective_from}")
    return f", {', '.join(parts)}" if parts else ""
