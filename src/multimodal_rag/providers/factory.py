"""Factories: the ONE place allowed to import concrete provider classes.

Everything else in the app should call get_llm() / get_embedder() /
get_vision() / get_reranker() and depend only on the abstract interfaces in
base.py — never import a concrete provider directly. That single rule is
what lets the backend change per environment (Mac vs. air-gapped server)
through config alone, and it's also what makes the privacy guard below
actually effective: if code could reach around the factory and construct
`LiteLLMProvider(base_url="https://api.openai.com")` directly, the guard
would never run.

The privacy guard: when a profile sets allow_external=False (the
air-gapped server), it must be structurally impossible to construct a
provider that talks to a non-local host. This is enforced here, not
trusted to config alone.
"""

import ipaddress
import os
import socket
from urllib.parse import urlparse

from ..config import Settings, get_settings
from ..device import resolve_device
from .base import EmbeddingProvider, LLMProvider, Reranker, VisionProvider
from .embeddings import APIEmbeddingProvider, SentenceTransformerEmbeddingProvider
from .llm import InternalServerLLM, LiteLLMProvider


class ExternalCallBlockedError(RuntimeError):
    """Raised when the active profile forbids external calls but a
    provider would require one."""


def _is_internal_host(host: str) -> bool:
    """True if `host` is loopback or a private-network address.

    Handles both literal IPs and hostnames (resolved via DNS, which is
    exactly what should succeed for an internal company hostname on the
    server's own network, and fail closed otherwise).
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_private
    except ValueError:
        pass
    try:
        resolved = socket.gethostbyname(host)
    except socket.gaierror:
        return False
    return ipaddress.ip_address(resolved).is_private


def _enforce_privacy_guard(base_url: str | None, allow_external: bool) -> None:
    if allow_external:
        return

    # Belt-and-suspenders: even if a provider correctly checks base_url,
    # also force offline mode so a library (e.g. huggingface_hub) can't
    # silently try to download an uncached model from the internet.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if base_url is None:
        return  # no network endpoint at all -> nothing to check

    host = urlparse(base_url).hostname
    if host is None or not _is_internal_host(host):
        raise ExternalCallBlockedError(
            f"Refusing to build a provider pointing at {base_url!r}: "
            f"allow_external=False and {host!r} is not a local/internal host."
        )


def get_llm(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    _enforce_privacy_guard(settings.llm_base_url, settings.allow_external)

    if settings.llm_provider == "litellm":
        return LiteLLMProvider(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    if settings.llm_provider == "internal_server":
        return InternalServerLLM(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")


def get_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    _enforce_privacy_guard(settings.embed_base_url, settings.allow_external)

    if settings.embed_provider == "sentence_transformers":
        device = resolve_device(settings.device)
        return SentenceTransformerEmbeddingProvider(model_name=settings.embed_model, device=device)
    if settings.embed_provider == "api":
        return APIEmbeddingProvider(
            model=settings.embed_model,
            base_url=settings.embed_base_url,
            api_key=settings.embed_api_key,
        )
    raise ValueError(f"Unknown embed_provider: {settings.embed_provider!r}")


def get_vision(settings: Settings | None = None) -> VisionProvider:
    raise NotImplementedError("No concrete VisionProvider implemented yet — a future step.")


def get_reranker(settings: Settings | None = None) -> Reranker:
    raise NotImplementedError("No concrete Reranker implemented yet — a future step.")
