"""Optional Langfuse tracing -- entirely opt-in, no code path requires it.

Never sends data to the public Langfuse Cloud, full stop -- not gated on
allow_external, since "this profile may call our own company's external
AI gateway" and "this profile may hand a third party's cloud our real
query/answer content" are different questions with different answers.
Tracing only activates once LANGFUSE_HOST is set explicitly to something
that isn't cloud.langfuse.com (its regional variants included) --
intended to be a self-hosted Langfuse instance you or your company run.

Deliberately NOT wired through litellm's own `success_callback =
["langfuse"]` integration. litellm==1.85.0 (pinned deliberately, see
config.py) only declares compatibility with the OLD v2 langfuse client
(`langfuse==2.59.7`), which itself pins `wrapt<2.0` -- directly
incompatible with `unstructured[pdf]`'s `wrapt>=2.1.1` requirement,
already a hard dependency for document ingestion. Confirmed by direct
inspection, not assumption: litellm's LangfuseLogger does
`langfuse.version.__version__`, a module that doesn't even exist in the
modern (4.x, OpenTelemetry-based) client this project actually installs.
So this module instruments `litellm.completion()` calls manually instead,
using the v4 client directly.

Nesting: langfuse v4's `start_as_current_observation()` nests
automatically via OpenTelemetry's own context propagation (contextvars
under the hood), which is also what lets it survive the
`run_in_threadpool` boundary FastAPI uses to call the (synchronous) agent
code -- no manual trace_id threading through LLMProvider's interface is
needed, and no existing test's fake LLMProvider needs to change at all.

Every actual call into the Langfuse SDK in this file is wrapped
defensively: "not configured" (get_langfuse_client() returns None) was
always a no-op, but "configured, then failing mid-request" (host
unreachable, service down, a network blip) was not fully covered before
-- a Langfuse hiccup could otherwise have broken a real, already-
successful query. Every function here is safe to call whether or not
Langfuse is reachable, full stop.
"""

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Protocol
from urllib.parse import urlparse

import litellm
from langfuse import Langfuse, propagate_attributes

from .config import get_settings
from .privacy_guard import ExternalCallBlockedError, enforce_privacy_guard

_TIMEOUT_SECONDS = 10
"""Explicit, not the SDK default -- this project has already been burned
more than once this session by a call with no bounded timeout hanging for
hours; better to fail visibly than silently stall a request."""

_LANGFUSE_CLOUD_HOSTS = {"cloud.langfuse.com", "us.cloud.langfuse.com", "eu.cloud.langfuse.com"}
"""Blocked UNCONDITIONALLY, regardless of allow_external. allow_external
is this project's "is this profile permitted to call ANY external
service" switch -- true on a normal company laptop, since it needs to
reach the company's own external LLM gateway. That is a completely
different question from "may this specific data (real user questions/
answers/retrieved content) be sent to a third-party's cloud," which must
stay no even when the first answer is yes. If tracing is ever wanted
against the public Langfuse Cloud, that has to be a deliberate, separate
decision -- not a side effect of a profile flag set for an unrelated
reason.
"""

_logger = logging.getLogger(__name__)


class _SearchResultLike(Protocol):
    source: str
    score: float
    chunk_id: str


def _safe_enter(context_manager: Any) -> bool:
    """Best-effort `__enter__` -- returns whether it actually succeeded,
    so a caller can skip the matching `__exit__` if not. Never raises:
    a context manager that fails to open (e.g. the SDK doing something
    network-bound on entry) must not be able to take a real request
    down with it.
    """
    try:
        context_manager.__enter__()
        return True
    except Exception:
        _logger.warning("Langfuse tracing failed to start; continuing without it.", exc_info=True)
        return False


def _safe_exit(context_manager: Any) -> None:
    """Best-effort `__exit__`, for a context manager `_safe_enter` already
    confirmed was opened. Never raises -- a failure closing a span must
    never surface after the real work it wrapped already completed
    (successfully or not).
    """
    try:
        context_manager.__exit__(None, None, None)
    except Exception:
        _logger.warning("Langfuse tracing failed to close cleanly.", exc_info=True)


@lru_cache(maxsize=1)
def get_langfuse_client() -> Langfuse | None:
    """None when tracing isn't configured (the default) -- every other
    function in this module treats that as "do nothing," never an error.
    Cached so the client (and its background export thread) is
    constructed once per process, not once per query.
    """
    settings = get_settings()
    if settings.langfuse_public_key is None or settings.langfuse_secret_key is None:
        return None

    # No silent default to the public cloud -- a host must be given
    # explicitly, in full, by whoever configures this. The SDK itself
    # defaults an unset host to "https://cloud.langfuse.com"; that
    # default is refused here, not inherited, so tracing simply stays off
    # until someone deliberately points it at a real (ideally
    # self-hosted, internal) address.
    host = settings.langfuse_host
    if not host:
        _logger.warning(
            "Langfuse tracing disabled: LANGFUSE_PUBLIC_KEY/SECRET_KEY are set but "
            "LANGFUSE_HOST is not. Refusing to default to the public Langfuse Cloud -- "
            "set LANGFUSE_HOST explicitly (a self-hosted instance) to enable tracing."
        )
        return None

    hostname = urlparse(host).hostname
    if hostname is not None and hostname.lower() in _LANGFUSE_CLOUD_HOSTS:
        _logger.warning(
            "Langfuse tracing disabled: LANGFUSE_HOST=%r points at the public Langfuse "
            "Cloud, which this project refuses to send data to regardless of "
            "allow_external -- real query/answer/retrieved content is not something to "
            "hand to a third party by default. Point LANGFUSE_HOST at a self-hosted "
            "instance instead.",
            host,
        )
        return None

    try:
        enforce_privacy_guard(host, settings.allow_external)
    except ExternalCallBlockedError:
        # Tracing is a nice-to-have, never a reason to break every query
        # -- but silently disabling it would leave a developer wondering
        # why traces never show up. Logged loudly, disabled quietly.
        _logger.warning(
            "Langfuse tracing disabled: allow_external=False and %r is not a "
            "local/internal host. Point LANGFUSE_HOST at a self-hosted instance "
            "on this network, or leave LANGFUSE_PUBLIC_KEY/SECRET_KEY unset.",
            host,
        )
        return None

    try:
        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=host,
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception:
        # Caught, not just fail-soft-by-convention: this is @lru_cache'd,
        # and lru_cache does NOT cache a raised exception -- an unguarded
        # failure here would retry (and fail) on every single request
        # forever, once Langfuse is configured but unreachable. Returning
        # None instead is what actually gets cached, so this constructor
        # runs at most once per process even when it fails.
        _logger.warning(
            "Langfuse tracing disabled: failed to construct a client for host %r.",
            host,
            exc_info=True,
        )
        return None


@contextmanager
def traced_query(conversation_id: str, query_id: str) -> Iterator[None]:
    """Wraps one /query call as its own trace, grouped into a session per
    conversation -- every turn of a conversation becomes its own trace,
    all visible together under one session in the Langfuse UI. A no-op if
    tracing isn't configured OR if opening the trace fails for any
    reason -- either way, the wrapped query still runs normally.
    """
    client = get_langfuse_client()
    if client is None:
        yield
        return

    attrs_cm = propagate_attributes(session_id=conversation_id, trace_name="query")
    attrs_open = _safe_enter(attrs_cm)
    span_cm = client.start_as_current_observation(
        name="query", as_type="span", metadata={"query_id": query_id}
    )
    span_open = attrs_open and _safe_enter(span_cm)
    try:
        yield
    finally:
        if span_open:
            _safe_exit(span_cm)
        if attrs_open:
            _safe_exit(attrs_cm)


def log_search_event(query: str, results: Sequence[_SearchResultLike]) -> None:
    """Records one search_knowledge_base round as a point-in-time event --
    not a span, since AgentChain's on_tool_call hook (see generation/
    agent.py) fires AFTER the search already completed, with no start
    time available to measure a real duration against. A no-op if tracing
    isn't configured. Nests under whatever span/trace is currently
    active, same as everything else here.
    """
    client = get_langfuse_client()
    if client is None:
        return
    try:
        client.create_event(
            name="search_knowledge_base",
            input=query,
            output=[
                {"source": r.source, "score": r.score, "chunk_id": r.chunk_id} for r in results
            ],
        )
    except Exception:
        _logger.warning("Langfuse search-event logging failed; continuing.", exc_info=True)


@contextmanager
def traced_generation(name: str, model: str, messages: list[dict[str, Any]]) -> Iterator[Any]:
    """Wraps one litellm.completion() call as a generation observation --
    real start/end timing, unlike log_search_event, plus the actual
    prompt as `input` (visible in the Langfuse UI). Yields None (a no-op)
    if tracing isn't configured OR if opening the observation fails for
    any reason -- callers don't need to branch on whether tracing is on;
    they just need to treat None as "nothing to attach a result to" (see
    record_generation_result).
    """
    client = get_langfuse_client()
    if client is None:
        yield None
        return

    span_cm = client.start_as_current_observation(
        name=name, as_type="generation", model=model, input=messages
    )
    try:
        generation = span_cm.__enter__()
    except Exception:
        _logger.warning("Langfuse tracing failed to start; continuing without it.", exc_info=True)
        yield None
        return
    try:
        yield generation
    finally:
        _safe_exit(span_cm)


def record_generation_result(generation: Any, response: Any, output: Any) -> None:
    """Attaches output/usage/cost to an in-flight generation observation,
    once the completion response is available. A no-op if `generation` is
    None (see traced_generation). Cost lookup is fail-soft on its own: an
    unrecognized/custom model just means no cost attached, not a broken
    trace -- same "nice-to-have, never blocks the real thing" precedent
    as generation/title.py's generate_title(). The `.update()` call
    itself is wrapped too -- this runs right after a real, successful LLM
    response comes back, and a Langfuse hiccup here must not be able to
    take that already-obtained answer down with it.
    """
    if generation is None:
        return
    usage = getattr(response, "usage", None)
    usage_details = (
        {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        if usage is not None
        else None
    )
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None
    try:
        generation.update(
            output=output,
            usage_details=usage_details,
            cost_details={"total": cost} if cost is not None else None,
        )
    except Exception:
        _logger.warning("Langfuse generation-result logging failed; continuing.", exc_info=True)
