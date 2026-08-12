"""Factories: the ONE place allowed to import concrete provider classes.

Everything else in the app should call get_llm() / get_embedder() /
get_vision() / get_reranker() and depend only on the abstract interfaces in
base.py — never import a concrete provider directly. That single rule is
what lets the backend change per environment (Mac vs. air-gapped server)
through config alone, and it's also what makes the privacy guard
(multimodal_rag.privacy_guard) actually effective: if code could reach
around the factory and construct `LiteLLMProvider(base_url="https://api.openai.com")`
directly, the guard would never run.
"""

from ..config import Settings, get_settings
from ..device import resolve_device
from ..privacy_guard import enforce_privacy_guard
from .base import EmbeddingProvider, LLMProvider, Reranker, VisionProvider
from .embeddings import LiteLLMEmbeddingProvider, SentenceTransformerEmbeddingProvider
from .llm import InternalServerLLM, LiteLLMProvider
from .rerank import CrossEncoderReranker
from .vision import InternalServerVisionProvider, LiteLLMVisionProvider


def get_llm(settings: Settings | None = None) -> LLMProvider:
    settings = settings or get_settings()
    enforce_privacy_guard(settings.llm_base_url, settings.allow_external)

    if settings.llm_provider == "litellm":
        return LiteLLMProvider(
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    if settings.llm_provider == "internal_server":
        return InternalServerLLM(base_url=settings.llm_base_url, api_key=settings.llm_api_key)
    raise ValueError(f"Unknown llm_provider: {settings.llm_provider!r}")


def llm_from_override(
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    *,
    allow_external: bool,
) -> LLMProvider:
    enforce_privacy_guard(base_url, allow_external)
    if provider == "litellm":
        return LiteLLMProvider(model=model, base_url=base_url, api_key=api_key)
    if provider == "internal_server":
        return InternalServerLLM(base_url=base_url, api_key=api_key)
    raise ValueError(f"Unknown llm_provider: {provider!r}")


def get_embedder(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    enforce_privacy_guard(settings.embed_base_url, settings.allow_external)

    if settings.embed_provider == "sentence_transformers":
        device = resolve_device(settings.device)
        return SentenceTransformerEmbeddingProvider(model_name=settings.embed_model, device=device)
    if settings.embed_provider == "litellm":
        return LiteLLMEmbeddingProvider(
            model=settings.embed_model,
            base_url=settings.embed_base_url,
            api_key=settings.embed_api_key,
        )
    raise ValueError(f"Unknown embed_provider: {settings.embed_provider!r}")


def embedder_from_override(
    provider: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    *,
    allow_external: bool,
    device: str,
) -> EmbeddingProvider:
    enforce_privacy_guard(base_url, allow_external)
    if provider == "sentence_transformers":
        return SentenceTransformerEmbeddingProvider(model_name=model, device=device)
    if provider == "litellm":
        return LiteLLMEmbeddingProvider(model=model, base_url=base_url, api_key=api_key)
    raise ValueError(f"Unknown embed_provider: {provider!r}")


def get_vision(settings: Settings | None = None) -> VisionProvider:
    settings = settings or get_settings()

    if settings.vision_provider is None:
        raise NotImplementedError(
            "Vision is not configured for this profile — set VISION_PROVIDER in your .env file."
        )

    enforce_privacy_guard(settings.vision_base_url, settings.allow_external)

    if settings.vision_provider == "litellm":
        if settings.vision_model is None:
            raise ValueError("VISION_MODEL must be set when VISION_PROVIDER=litellm")
        return LiteLLMVisionProvider(
            model=settings.vision_model,
            base_url=settings.vision_base_url,
            api_key=settings.vision_api_key,
        )
    if settings.vision_provider == "internal_server":
        return InternalServerVisionProvider(
            base_url=settings.vision_base_url, api_key=settings.vision_api_key
        )
    raise ValueError(f"Unknown vision_provider: {settings.vision_provider!r}")


def get_reranker(settings: Settings | None = None) -> Reranker:
    settings = settings or get_settings()

    if settings.reranker_provider is None:
        raise NotImplementedError(
            "Reranker is not configured for this profile — set RERANKER_PROVIDER "
            "in your .env file."
        )

    # Even the fully-local cross_encoder path downloads its model from HF
    # Hub the first time — same belt-and-suspenders as get_embedder().
    enforce_privacy_guard(settings.reranker_base_url, settings.allow_external)

    if settings.reranker_provider == "cross_encoder":
        if settings.reranker_model is None:
            raise ValueError("RERANKER_MODEL must be set when RERANKER_PROVIDER=cross_encoder")
        device = resolve_device(settings.device)
        return CrossEncoderReranker(model_name=settings.reranker_model, device=device)
    raise ValueError(f"Unknown reranker_provider: {settings.reranker_provider!r}")
