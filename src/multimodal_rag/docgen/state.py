"""Typed state schema for the docgen LangGraph workflow.

Defined before any graph wiring (the next phase) -- earlier phases'
pure functions (resolve_source, retrieve_for_question, validate_answer)
were built and tested without depending on this at all, per the docgen
build plan. A plain TypedDict with full-replace semantics on every
field, LangGraph's default when a node returns a partial update --
rather than speculating about which fields might need a custom merge
reducer before any real node exists to show that it actually does.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from .nodes.retrieval import RetrievedChunk
from .sources import SourceRole, SourceSpec


class Attempt(TypedDict):
    """One REJECTED try at answering a question -- carried into the next
    retry so the model doesn't repeat the same mistake (see
    validation.validate_answer's `reason`)."""

    query: str
    chunks: list[RetrievedChunk]
    answer: str
    reason: str


class Question(TypedDict):
    id: str
    text: str
    sources_required: list[SourceRole]
    status: Literal["pending", "answered", "escalated", "skipped"]


class Answer(TypedDict):
    text: str
    attempts: list[Attempt]
    """Every rejected attempt, oldest first -- not the accepted answer,
    which is `text` above. len(attempts) IS the retry count; there is
    no separate counter that could drift out of sync with it."""
    accepted_by: Literal["validation", "human"]


class Configuration(TypedDict):
    template: str
    format: Literal["pptx", "docx"]
    confirmed: bool
    max_retries: int
    """How many automated attempt_answer -> validate_answer cycles a
    question gets before escalating to a human -- per-run, not a fixed
    constant, so a person launching a run can trade off effort against
    document complexity (nodes.answering.DEFAULT_MAX_RETRIES is only
    the fallback used if nothing else sets this)."""


class Review(TypedDict):
    decision: Literal["pending", "approved", "edit_requested"]
    flagged_question_ids: list[str]


class RequestContext(TypedDict):
    text: str
    """The person's original free-text description of what they want
    answered -- never edited in place."""

    corrections: list[str]
    """Every correction given after an earlier interpret_request draft
    was shown back and rejected, oldest first -- ADDED to, never
    replaced: a later correction must not silently drop an earlier
    one (the docgen build plan's own wording for this loop)."""


class Usage(TypedDict):
    llm_calls: int
    """Every real LLM call made so far across the whole run (formulate_query,
    generate_answer_text, validate_answer -- not validate_answer's own
    zero-chunks short-circuit, which never calls out at all). Incremented at
    the graph-node level (docgen/graph.py), not inside those functions
    themselves, for the same reason error-handling lives there: it's a
    wiring/observability concern, not what each function itself does."""


class InProgress(TypedDict):
    """Scratch space for the ONE question currently moving through
    select_next_question -> retrieve -> generate_answer -> validate_answer.
    Bundled as one object, not five separate top-level fields, because
    all five only mean anything together, as a description of the same
    in-flight question -- see the docgen build plan's phase 6 notes."""

    question_id: str
    query: str
    chunks: list[RetrievedChunk]
    answer: str
    attempts: list[Attempt]
    """Rejected tries so far for THIS question -- becomes Answer.attempts
    once accepted. len(attempts) is the retry count, same reasoning as
    Answer.attempts."""

    valid: bool | None
    """Set by the validate_answer node right before the router reads it;
    None only ever appears briefly, between select_next_question
    creating this record and validate_answer's first run."""

    human_guidance: str | None
    """Set only when a human, escalated to via ask_human, chooses to
    reformulate rather than answer directly -- steering context for the
    NEXT formulate_query/generate_answer_text call (the original
    Question.text is deliberately never overwritten). Persists across
    whatever retries follow, cleared only when a fresh InProgress is
    opened for a different question."""


class DocGenState(TypedDict):
    sources: list[SourceSpec]
    questions: list[Question]
    answers: dict[str, Answer]
    """Keyed by Question.id -- only questions accepted so far (via
    validation or a human) have an entry."""
    configuration: Configuration
    review: Review
    current: InProgress | None
    """None when no question is being actively worked on (before the
    first one starts, and after the last one finishes)."""
    usage: Usage
    request: RequestContext
