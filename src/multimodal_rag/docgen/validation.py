"""Grounding check for docgen's per-question loop: is the generated
answer actually supported by the retrieved chunks, and does it address
the question that was asked?

A standalone, swappable function (question, answer, chunks) ->
ValidationResult -- easy to unit test on its own, and easy to later
replace or wrap with a third-party guardrails library without
restructuring the graph (see the docgen build plan's guardrails
section). Modeled on evaluation/judge.py's LLM-as-judge pattern (strict
grading-assistant framing, defensive JSON parsing) but as ONE combined
call covering both groundedness and relevance, returning a direct
valid/invalid verdict plus the reason -- not two separate float scores
-- since that's the shape the retry loop actually needs (the reason
text feeds back into the next attempt).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..providers.base import LLMProvider
from ..providers.factory import get_llm
from ..tracing import get_rendered_prompt, traced_span, update_span_output, update_span_score
from .nodes.retrieval import RetrievedChunk, format_chunks

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_PROMPT_NAME = "docgen_grounding_prompt"

# {{double-braced}} -- Langfuse's own templating syntax (see
# tracing.get_rendered_prompt), not Python's str.format(). The single
# braces in the JSON example below are deliberately NOT escaped the
# way a str.format() template would need -- Langfuse only ever
# substitutes a {{...}} pair, so a lone "{" needs no escaping at all.
_GROUNDING_PROMPT = """## ROLE
You are a strict grading assistant. You judge whether a generated ANSWER is properly grounded \
in the provided CONTEXT, and whether it actually addresses the QUESTION that was asked.

## TASK
The ANSWER is VALID only if both are true:
- Every factual claim it makes is supported by the CONTEXT -- no claims the CONTEXT doesn't \
back, no contradictions.
- It actually addresses the QUESTION asked, not something else.

If either fails, the ANSWER is INVALID. Explain why in one sentence, specific enough that \
someone retrying with a different search could avoid the same mistake.

## OUTPUT FORMAT
Respond with EXACTLY one JSON object, nothing else -- no markdown code fences, no prose before \
or after it:
{"valid": <true or false>, "reason": "<one sentence>"}

## QUESTION
{{question}}

## CONTEXT
{{context}}

## ANSWER
{{answer}}"""


class ValidationParseError(ValueError):
    """The judge LLM responded, but its output couldn't be parsed into a
    verdict -- distinct from a provider/network failure (which
    propagates normally), same split evaluation/judge.py's
    JudgeParseError draws for the same reason."""


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: str
    llm_called: bool = True
    """False only for the zero-chunks short-circuit below -- lets a
    caller (docgen/graph.py's usage counter) know whether this call
    actually spent a real LLM request, without needing to re-check
    `chunks` itself or duplicate this function's own short-circuit
    condition at the call site."""


def _parse_verdict(raw: str) -> ValidationResult:
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise ValidationParseError(f"No JSON object found in judge output: {raw!r}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValidationParseError(f"Judge output wasn't valid JSON: {raw!r}") from exc
    valid = parsed.get("valid")
    reason = parsed.get("reason")
    if not isinstance(valid, bool) or not isinstance(reason, str):
        raise ValidationParseError(f"Judge output missing valid/reason fields: {raw!r}")
    return ValidationResult(valid=valid, reason=reason)


def validate_answer(
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    llm: LLMProvider | None = None,
) -> ValidationResult:
    """Raises ValidationParseError if the judge's output can't be
    parsed; propagates any provider/network error from get_llm() as-is
    -- same contract as evaluation/judge.py's score_* functions.

    Zero retrieved chunks short-circuits to invalid with no LLM call at
    all -- there is no possible way an answer is grounded in evidence
    that doesn't exist, so there's nothing for a judge to usefully
    decide."""
    if not chunks:
        return ValidationResult(
            valid=False,
            reason="No chunks were retrieved to support this answer.",
            llm_called=False,
        )

    llm = llm or get_llm()
    prompt = get_rendered_prompt(
        _PROMPT_NAME,
        _GROUNDING_PROMPT,
        question=question,
        context=format_chunks(chunks),
        answer=answer,
    )
    with traced_span("docgen_validate_answer", as_type="generation", input=prompt) as span:
        raw = llm.generate([{"role": "user", "content": prompt}])
        result = _parse_verdict(raw)
        update_span_output(span, raw)
        update_span_score(
            span, "docgen_grounding", 1.0 if result.valid else 0.0, comment=result.reason
        )
    return result
