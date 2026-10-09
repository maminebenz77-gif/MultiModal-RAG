"""Harmonization: after every question is answered, one LLM call reads
every question/answer pair together and rewrites each answer's WORDING
for consistent tone and to remove redundancy between answers -- it has
to see all of them at once, since the entire point is how they read
together, not any single one in isolation.

Deliberately fails soft: if the call fails, or its output can't be
parsed, the run keeps the original, unharmonized answers rather than
stopping. A polish pass that didn't work leaves a slightly less
polished document, not a broken run -- same "nice-to-have, never
blocks the real thing" precedent generation/title.py's generate_title()
and tracing.py's cost-lookup already follow in this codebase.
"""

from __future__ import annotations

import json
import re

from ...providers.base import LLMProvider
from ...providers.factory import get_llm
from ...tracing import get_rendered_prompt, traced_span, update_span_output
from ..state import Answer, Question

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_PROMPT_NAME = "docgen_harmonize_prompt"

# {{double-braced}} -- Langfuse's own templating syntax (see
# tracing.get_rendered_prompt), not Python's str.format().
_HARMONIZE_PROMPT = """## ROLE
You are an editor. You harmonize a set of already-answered questions so the full set reads as \
one consistent document -- same tone, no redundant repetition between answers, consistent \
terminology.

## TASK
Rewrite each ANSWER below for tone, consistency, and to remove redundancy with the OTHER \
answers. Do NOT change what any answer actually says -- every factual claim must stay exactly \
as true and exactly as supported as it already was. You are editing prose, not re-answering \
the questions or adding anything a question's own answer didn't already say.

## OUTPUT FORMAT
Respond with EXACTLY one JSON object, nothing else -- no markdown code fences, no prose before \
or after it. Map each question's id to its rewritten answer text, one entry per question below:
{"q1": "<rewritten answer>", "q2": "<rewritten answer>", ...}

## QUESTIONS AND ANSWERS
{{qa_pairs}}"""


class HarmonizeParseError(ValueError):
    """The model responded, but its output couldn't be parsed into a
    valid id -> text mapping -- same split validate_answer's
    ValidationParseError draws, for the same reason: distinct from a
    provider/network failure, which propagates normally."""


def _format_qa_pairs(questions: list[Question], answers: dict[str, Answer]) -> str:
    lines = []
    for question in questions:
        answer = answers.get(question["id"])
        if answer is None:
            continue  # skipped, or otherwise never accepted -- nothing to harmonize
        lines.append(f"- [{question['id']}] Question: {question['text']}")
        lines.append(f"  Answer: {answer['text']}")
    return "\n".join(lines)


def _parse_harmonized(raw: str) -> dict[str, str]:
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise HarmonizeParseError(f"No JSON object found in model output: {raw!r}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise HarmonizeParseError(f"Model output wasn't valid JSON: {raw!r}") from exc
    if not isinstance(parsed, dict) or not all(isinstance(v, str) for v in parsed.values()):
        raise HarmonizeParseError(f"Model output wasn't a str -> str mapping: {raw!r}")
    return parsed


def harmonize_answers_text(
    questions: list[Question], answers: dict[str, Answer], llm: LLMProvider | None = None
) -> dict[str, str]:
    """Raises HarmonizeParseError if the model's output can't be
    parsed; propagates any provider/network error from get_llm() as-is
    -- same contract as validate_answer/interpret_request_text. A
    question id missing from the result (the model skipped it, or
    parsing partially failed upstream of here) is the caller's to
    handle -- this function only returns what it actually got back."""
    llm = llm or get_llm()
    prompt = get_rendered_prompt(
        _PROMPT_NAME, _HARMONIZE_PROMPT, qa_pairs=_format_qa_pairs(questions, answers)
    )
    with traced_span("docgen_harmonize_answers", as_type="generation", input=prompt) as span:
        raw = llm.generate([{"role": "user", "content": prompt}])
        update_span_output(span, raw)
    return _parse_harmonized(raw)
