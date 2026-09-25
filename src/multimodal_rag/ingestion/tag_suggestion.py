"""Suggests tags for a document being ingested, from an excerpt of its
own parsed text -- optional, LLM-derived, never authoritative. The
frontend pre-fills the Tags field with these before the ingest form is
ever submitted, so a human still has to look at and confirm them (edit,
clear, or leave as-is) before anything is actually written -- the same
trust boundary this project already draws around every other LLM-
derived, human-reviewed field (a chart's description, a table's
one-sentence summary, see excel_charts.py/tables.py), applied here to a
field that also drives access-adjacent filtering (SearchFilter.any_of),
which is exactly why an LLM guess is never allowed to become the tag on
its own.
"""

import json
import re

from ..providers.factory import get_llm

_MAX_TAGS = 5
_MAX_EXCERPT_CHARS = 2000
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

_PROMPT = f"""Suggest up to {_MAX_TAGS} short, lowercase tags for the following document \
excerpt -- single words or short hyphenated phrases (e.g. "runbook", "q3-2026", \
"incident-postmortem"), describing what KIND of document this is or what topic/period it \
covers. Do not invent facts the excerpt doesn't contain.

Respond with EXACTLY one JSON array of strings, nothing else -- no markdown code fences, no \
prose before or after it, e.g. ["runbook", "incident-postmortem"]

## DOCUMENT EXCERPT
{{excerpt}}"""


def build_excerpt(texts: list[str]) -> str:
    """The first _MAX_EXCERPT_CHARS characters of a document's own text
    (title, headings, opening paragraphs -- whatever its parser produced
    first), joined and truncated -- enough for a model to judge what
    KIND of document this is without sending a whole document through
    an extra LLM call just to suggest a few tags."""
    return "\n".join(t for t in texts if t)[:_MAX_EXCERPT_CHARS]


def suggest_tags(excerpt: str) -> list[str]:
    """Returns an empty list if no LLM provider is configured, the call
    fails, or the response can't be parsed as a JSON array of strings --
    suggesting tags is a nice-to-have, never a reason to fail ingest (or
    even this standalone endpoint): an empty list just means the Tags
    field starts blank, same as before this existed.
    """
    if not excerpt.strip():
        return []
    try:
        llm = get_llm()
    except Exception:
        return []
    try:
        raw = llm.generate([{"role": "user", "content": _PROMPT.format(excerpt=excerpt)}])
    except Exception:
        return []
    return _parse_tags(raw)


def _parse_tags(raw: str) -> list[str]:
    match = _JSON_ARRAY_RE.search(raw)
    if match is None:
        return []
    try:
        parsed = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []
    tags = [stripped for t in parsed if isinstance(t, str) and (stripped := t.strip())]
    return tags[:_MAX_TAGS]
