"""The configuration loop: turns a person's free-text description of
what they want answered into the real Question entries the rest of
the graph (already built, and tested without knowing this phase would
ever exist) already knows how to work with.

Runs once, before anything else -- route_at_start (docgen/graph.py)
only enters this loop at all when state["configuration"]["confirmed"]
is still False, so a caller that already has a fully-formed, confirmed
question list (every test built in earlier phases does exactly this)
skips it entirely and goes straight to select_next_question, unchanged.

interpret_request always regenerates the FULL question list from the
original request text plus every correction so far, rather than
patching a previous partial list -- the same "start fresh from
accumulated context" approach formulate_query already uses for query
reformulation, and for the same reason: patching a list incrementally
needs its own diffing logic, where regenerating from the whole history
needs none.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from langgraph.types import interrupt

from ...providers.base import LLMProvider
from ...providers.factory import get_llm
from ...tracing import get_rendered_prompt, traced_span, update_span_output
from ..sources import SourceRole
from ..state import Configuration, DocGenState, Question

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_PROMPT_NAME = "docgen_interpret_request_prompt"

# {{double-braced}} -- Langfuse's own templating syntax (see
# tracing.get_rendered_prompt), not Python's str.format(). The single
# braces in the JSON example below need no escaping -- Langfuse only
# ever substitutes a matched {{...}} pair.
_INTERPRET_PROMPT = """## ROLE
You turn a person's free-text description of what they want answered into a structured list \
of questions, plus the output format they asked for.

## TASK
Read the REQUEST below (and any CORRECTIONS, which must all still be honored -- a correction \
adds to what was already asked, it never silently drops an earlier one). Produce:
- a list of standalone, clearly-phrased questions
- for each question, which of the AVAILABLE SOURCES it needs (by name) -- most questions only \
need "task_docs"; only use "reference_kb" for a question that explicitly compares against a \
reference or standard, and only if "reference_kb" is actually listed as available
- the output format: "pptx" or "docx"
- a short template name if the person mentioned one, otherwise an empty string

## AVAILABLE SOURCES
{{available_sources}}

## OUTPUT FORMAT
Respond with EXACTLY one JSON object, nothing else -- no markdown code fences, no prose before \
or after it:
{"questions": [{"text": "<question text>", "sources_required": ["<source name>", ...]}], \
"format": "pptx or docx", "template": "<template name, or an empty string>"}

## REQUEST
{{request_text}}
{{corrections}}"""

_CORRECTIONS_SECTION = """
## CORRECTIONS (given after seeing an earlier interpretation -- all of these still apply)
{items}
"""


class InterpretParseError(ValueError):
    """The model responded, but its output couldn't be parsed into a
    valid question list -- same split validate_answer's
    ValidationParseError draws, for the same reason: distinct from a
    provider/network failure, which propagates normally."""


@dataclass(frozen=True)
class InterpretedQuestion:
    text: str
    sources_required: list[str]


@dataclass(frozen=True)
class InterpretedRequest:
    questions: list[InterpretedQuestion]
    format: Literal["pptx", "docx"]
    template: str


def _format_corrections(corrections: list[str]) -> str:
    return "\n".join(f"- {correction}" for correction in corrections)


def _parse_interpretation(raw: str) -> InterpretedRequest:
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise InterpretParseError(f"No JSON object found in model output: {raw!r}")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise InterpretParseError(f"Model output wasn't valid JSON: {raw!r}") from exc

    questions_raw = parsed.get("questions")
    output_format = parsed.get("format")
    template = parsed.get("template")
    if (
        not isinstance(questions_raw, list)
        or not questions_raw
        or output_format not in ("pptx", "docx")
        or not isinstance(template, str)
    ):
        raise InterpretParseError(f"Model output missing required fields: {raw!r}")

    questions: list[InterpretedQuestion] = []
    for item in questions_raw:
        if not isinstance(item, dict):
            raise InterpretParseError(f"Model output had a non-object question: {raw!r}")
        text = item.get("text")
        sources_required = item.get("sources_required")
        if not isinstance(text, str) or not text or not isinstance(sources_required, list):
            raise InterpretParseError(f"Model output had a malformed question: {raw!r}")
        questions.append(InterpretedQuestion(text=text, sources_required=list(sources_required)))

    return InterpretedRequest(questions=questions, format=output_format, template=template)


def interpret_request_text(
    request_text: str,
    corrections: list[str],
    available_sources: Sequence[str],
    llm: LLMProvider | None = None,
) -> InterpretedRequest:
    """Raises InterpretParseError if the model's output can't be
    parsed; propagates any provider/network error from get_llm() as-is
    -- same contract as validate_answer/formulate_query."""
    llm = llm or get_llm()
    corrections_section = (
        _CORRECTIONS_SECTION.format(items=_format_corrections(corrections)) if corrections else ""
    )
    prompt = get_rendered_prompt(
        _PROMPT_NAME,
        _INTERPRET_PROMPT,
        available_sources=", ".join(available_sources) if available_sources else "(none)",
        request_text=request_text,
        corrections=corrections_section,
    )
    with traced_span("docgen_interpret_request", as_type="generation", input=prompt) as span:
        raw = llm.generate([{"role": "user", "content": prompt}])
        update_span_output(span, raw)
    return _parse_interpretation(raw)


class ConfirmationResponse(TypedDict):
    action: Literal["confirm", "revise"]
    text: str
    """Only used when action == "revise" -- the correction added to
    the request's history."""


def _format_confirmation_summary(questions: list[Question], configuration: Configuration) -> str:
    if not questions:
        return (
            'I wasn\'t able to turn that into any questions. Reply with "revise" and a '
            "rephrased description to try again."
        )
    lines = [f"I'll answer {len(questions)} question(s):"]
    for i, question in enumerate(questions, start=1):
        sources = ", ".join(question["sources_required"]) or "no sources"
        lines.append(f"{i}. {question['text']} (using: {sources})")
    template = configuration["template"] or "the default template"
    lines.append(f"Output: a {configuration['format']} file, using {template}.")
    return "\n".join(lines)


def reformulate_for_confirmation(state: DocGenState) -> dict[str, Any]:
    summary = _format_confirmation_summary(state["questions"], state["configuration"])
    response = interrupt(
        {"kind": "confirm_configuration", "summary": summary}, response_schema=ConfirmationResponse
    )
    if response["action"] == "confirm":
        return {"configuration": {**state["configuration"], "confirmed": True}}
    request = state["request"]
    return {"request": {**request, "corrections": [*request["corrections"], response["text"]]}}


def route_after_confirmation(state: DocGenState) -> Literal["frozen", "revise"]:
    return "frozen" if state["configuration"]["confirmed"] else "revise"


def freeze_configuration(state: DocGenState) -> dict[str, Any]:
    """Drops any source role an interpreted question references that
    isn't actually in state["sources"] -- a hallucinated or stale
    source name can never be satisfied anyway. This is a programmatic
    sanity check on the model's own output, not a human decision, so
    it's corrected silently here rather than escalated or bounced back
    to reformulate_for_confirmation."""
    available_roles = {source.role for source in state["sources"]}
    questions = [
        {
            **question,
            "sources_required": [
                role for role in question["sources_required"] if role in available_roles
            ],
        }
        for question in state["questions"]
    ]
    return {"questions": cast(list[Question], questions)}


def build_questions(interpreted: InterpretedRequest) -> list[Question]:
    """Turns the model's output into real Question entries, with
    code-assigned sequential ids -- never trusting a model to invent
    unique, well-formed ids, the same reasoning doc_id/chunk_id are
    computed by code elsewhere in this project. sources_required is
    NOT yet validated against state["sources"] here (cast() says
    "trust this for now," not "this is already correct") -- that
    happens in freeze_configuration, once a human has confirmed this
    draft, not before."""
    return [
        {
            "id": f"q{i}",
            "text": question.text,
            "sources_required": cast(list[SourceRole], question.sources_required),
            "status": "pending",
        }
        for i, question in enumerate(interpreted.questions, start=1)
    ]
