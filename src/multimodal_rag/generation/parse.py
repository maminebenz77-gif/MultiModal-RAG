"""Parses a raw LLM answer into a RagAnswer: extracts which citation
markers were actually cited and maps them back to real chunk metadata
WE already have — never trusting the model's own recall of filenames or
page numbers, only which numbered block it referenced.

Markers use double-angled brackets (e.g. "⟦1⟧") instead of plain
"[1]" — ordinary square-bracketed numbers show up all the time in real
technical documents (citations, footnotes, array/step indices) and in
the model's own enumerated lists, and either would be misread as a
citation by a plain "\\[(\\d+)\\]" regex. The double-angled form is
distinctive enough that it only appears when WE put it there.

Being told to use ⟦N⟧ doesn't guarantee a model will (observed with
gpt-4o-mini: it sometimes reverts to its own trained-in browsing-citation
format, 【N†source】, despite the system prompt explicitly forbidding it —
prompting shapes behavior, it doesn't enforce it). Rather than silently
losing the citation when that happens, the one variant actually observed
is normalized back to ⟦N⟧ before parsing. This is deliberately narrow: it
targets exactly the pattern seen in practice, not a broad guess at every
format a model might someday use.
"""

import re

from ..stores.schema import SearchResult
from .prompt import REFUSAL_TEXT
from .schema import Citation, RagAnswer

_CITATION_RE = re.compile(r"⟦(\d+)⟧")
_ALTERNATE_CITATION_RE = re.compile(r"【(\d+)†[^】]*】")


def _normalize_citation_markers(raw_answer: str) -> str:
    return _ALTERNATE_CITATION_RE.sub(lambda m: f"⟦{m.group(1)}⟧", raw_answer)


def parse_answer(raw_answer: str, context_results: list[SearchResult]) -> RagAnswer:
    raw_answer = _normalize_citation_markers(raw_answer)
    cited_numbers = sorted({int(match) for match in _CITATION_RE.findall(raw_answer)})
    citations = [
        Citation(
            marker=number,
            chunk_id=context_results[number - 1].chunk_id,
            source=context_results[number - 1].source,
            doc_id=context_results[number - 1].doc_id,
            pages=context_results[number - 1].pages,
            slides=context_results[number - 1].slides,
            text=context_results[number - 1].text,
            elements=context_results[number - 1].elements,
        )
        for number in cited_numbers
        if 1 <= number <= len(context_results)
    ]
    refused = REFUSAL_TEXT.lower() in raw_answer.strip().lower()
    return RagAnswer(
        answer=raw_answer,
        citations=citations,
        refused=refused,
        retrieved_chunks=context_results,
    )
