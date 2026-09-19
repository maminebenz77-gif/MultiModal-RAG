"""Layer 2 (generation) evaluation via a hand-rolled LLM-as-judge.

Reuses get_llm() -- the same provider used everywhere else in this
codebase, not a separate RAGAS-style model wrapper -- to score a generated
answer along two independent axes:

  - Faithfulness: is every claim in the answer actually supported by the
    context it was generated from? Deliberately takes only (answer,
    context) -- NOT the question -- matching the real definition (an
    answer's groundedness in its context doesn't depend on what was
    asked; a faithfulness judge that saw the question could be tempted to
    reward "sounds like a good answer" instead of "is actually
    supported," which is a different thing entirely).
  - Relevance: does the answer actually address the question that was
    asked, independent of whether it's grounded? Takes (question, answer)
    -- NOT the context -- for the same reason in reverse: relevance is
    about the answer/question relationship, not about the evidence.
  - Correctness: does the answer agree, in substance, with a human
    expert's reference answer? Unlike the two above, this ONE needs a
    ground truth (see evaluation/run_expert_eval.py) -- it's not a
    property of the answer and its own context/question alone.

All three return a single JSON object, parsed defensively: a judge that
returns malformed output raises JudgeParseError rather than crashing
whatever's scoring a whole batch of items -- see run_eval.py, which
catches it per-item so one bad judge call doesn't take down the run.
"""

import json
import re

from ..providers.factory import get_llm

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_OUTPUT_FORMAT_INSTRUCTIONS = """## OUTPUT FORMAT
Respond with EXACTLY one JSON object, nothing else -- no markdown code fences, no prose before \
or after it:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<one sentence>"}}"""

_FAITHFULNESS_PROMPT = f"""## ROLE
You are a strict grading assistant. You judge whether a generated ANSWER is fully supported by \
a given CONTEXT -- nothing more.

## TASK
Score how well every factual claim in the ANSWER is supported by the CONTEXT, from 0.0 to 1.0:
- 1.0: every claim in the ANSWER is directly supported by the CONTEXT.
- 0.0: the ANSWER makes claims the CONTEXT doesn't support at all, or contradicts it.
- In between: some claims are supported and some aren't, or a claim is a plausible but \
unsupported extrapolation beyond what the CONTEXT actually says.

Do NOT judge whether the ANSWER is a good or complete answer to any question -- only whether \
what it says is actually backed by the CONTEXT. An answer that says less than it could, but \
everything it does say is grounded, still scores 1.0.

{_OUTPUT_FORMAT_INSTRUCTIONS}

## CONTEXT
{{context}}

## ANSWER
{{answer}}"""

_RELEVANCE_PROMPT = f"""## ROLE
You are a strict grading assistant. You judge whether a generated ANSWER actually addresses the \
QUESTION that was asked -- nothing else.

## TASK
Score how directly the ANSWER addresses the QUESTION, from 0.0 to 1.0:
- 1.0: the ANSWER directly and completely addresses what was asked.
- 0.0: the ANSWER doesn't address the QUESTION at all (answers something else, or is a generic \
non-answer).
- In between: the ANSWER is partially on-topic, or addresses only part of a multi-part QUESTION.

Do NOT judge whether the ANSWER is factually correct or well-supported by any evidence -- only \
whether it's actually trying to answer what was asked. A confidently wrong answer that directly \
addresses the question still scores high here; a correct-but-evasive answer scores low.

{_OUTPUT_FORMAT_INSTRUCTIONS}

## QUESTION
{{question}}

## ANSWER
{{answer}}"""

_CORRECTNESS_PROMPT = f"""## ROLE
You are a strict grading assistant. You judge whether a generated ANSWER agrees, in substance, \
with a REFERENCE ANSWER written by a human expert -- nothing more.

## TASK
Score how well the ANSWER agrees with the REFERENCE ANSWER, from 0.0 to 1.0:
- 1.0: the ANSWER conveys the same key facts and conclusions as the REFERENCE ANSWER. Different \
wording, different length, and extra correct detail are all fine -- what matters is substance, \
not phrasing.
- 0.0: the ANSWER contradicts the REFERENCE ANSWER, or is missing its substance entirely.
- In between: the ANSWER gets some of the REFERENCE ANSWER's substance right but misses or \
gets wrong a meaningful part of it.

Do NOT judge whether the ANSWER directly addresses the QUESTION or is well-written -- only \
whether it agrees with the REFERENCE ANSWER. The QUESTION is given only so you can tell which \
parts of the REFERENCE ANSWER are the ones actually being asked about.

{_OUTPUT_FORMAT_INSTRUCTIONS}

## QUESTION
{{question}}

## REFERENCE ANSWER
{{reference_answer}}

## ANSWER
{{answer}}"""


class JudgeParseError(ValueError):
    """The judge LLM responded, but its output couldn't be parsed into a
    score -- distinct from a provider/network failure (which propagates
    normally), and deliberately catchable on its own so a caller scoring
    many items can skip just this one instead of losing the whole batch.
    """


def _extract_score(raw: str) -> float:
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise JudgeParseError(f"No JSON object found in judge output: {raw!r}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"Judge output wasn't valid JSON: {raw!r}") from exc
    score = parsed.get("score")
    if not isinstance(score, int | float):
        raise JudgeParseError(f"Judge output had no numeric 'score' field: {raw!r}")
    return max(0.0, min(1.0, float(score)))


def score_faithfulness(answer: str, context: str) -> float:
    """Raises JudgeParseError if the judge's output can't be parsed;
    propagates any provider/network error from get_llm() as-is.
    """
    raw = get_llm().generate(
        [{"role": "user", "content": _FAITHFULNESS_PROMPT.format(context=context, answer=answer)}]
    )
    return _extract_score(raw)


def score_relevance(question: str, answer: str) -> float:
    """Same contract as score_faithfulness."""
    raw = get_llm().generate(
        [{"role": "user", "content": _RELEVANCE_PROMPT.format(question=question, answer=answer)}]
    )
    return _extract_score(raw)


def score_answer_correctness(question: str, answer: str, reference_answer: str) -> float:
    """The one judge in this module that needs a ground truth -- unlike
    faithfulness/relevance, which are properties of an answer and its own
    context/question alone, this compares against a human expert's
    reference_answer (see evaluation/run_expert_eval.py). Same contract as
    score_faithfulness otherwise.
    """
    raw = get_llm().generate(
        [
            {
                "role": "user",
                "content": _CORRECTNESS_PROMPT.format(
                    question=question, reference_answer=reference_answer, answer=answer
                ),
            }
        ]
    )
    return _extract_score(raw)
