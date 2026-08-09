"""Environment-portability config: which profile (local|server) is active,
and every setting that differs between developing on a laptop and running
on the air-gapped company GPU server.

Profile selection happens via the RAG_ENV *OS* environment variable, read
before pydantic ever loads a .env file — RAG_ENV has to be known first,
since it decides which file (.env.local or .env.server) supplies everything
else.
"""

import logging
import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# litellm otherwise fetches its model-cost map from GitHub on first
# import -- pointless on any network that can't reach it (air-gapped
# server, corporate proxy) and pure wasted latency everywhere else,
# since we don't use its cost-tracking features. Must run before litellm
# is imported anywhere; config.py is imported before that happens
# everywhere in this project, and doesn't import litellm itself.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
logging.getLogger("LiteLLM").setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RagEnv(StrEnum):
    LOCAL = "local"
    SERVER = "server"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    rag_env: RagEnv

    # LLM
    llm_provider: str
    llm_model: str
    llm_base_url: str | None = None
    llm_api_key: str | None = None

    # Embeddings
    embed_provider: str
    embed_model: str
    embed_base_url: str | None = None
    embed_api_key: str | None = None

    # Vision (interface exists already; no concrete provider yet, so these
    # are optional until a future step wires one up).
    vision_provider: str | None = None
    vision_model: str | None = None
    vision_base_url: str | None = None
    vision_api_key: str | None = None

    # Reranker — cross-encoder, applied to an already-retrieved candidate
    # set, not a first-class provider selected per environment the way
    # LLM/embed/vision are. Optional until configured.
    reranker_provider: str | None = None
    reranker_model: str | None = None
    reranker_base_url: str | None = None
    reranker_api_key: str | None = None

    # Vector / search stores
    qdrant_url: str
    elastic_url: str

    # Runtime
    device: str = "auto"
    allow_external: bool = True

    # Certificates
    trust_system_certs: bool = False

    @field_validator(
        "llm_base_url",
        "llm_api_key",
        "embed_base_url",
        "embed_api_key",
        "vision_provider",
        "vision_model",
        "vision_base_url",
        "vision_api_key",
        "reranker_provider",
        "reranker_model",
        "reranker_base_url",
        "reranker_api_key",
        mode="before",
    )
    @classmethod
    def _blank_env_value_means_unset(cls, value: str | None) -> str | None:
        # A .env file with `KEY=` (blank) loads as "", not unset — normalize
        # to None so downstream `is None` checks (e.g. the privacy guard)
        # behave the same whether a var is blank or absent entirely.
        return value or None


def _env_file_for_profile(profile: RagEnv) -> Path:
    return PROJECT_ROOT / f".env.{profile.value}"


def load_settings() -> Settings:
    """Build Settings for whichever profile RAG_ENV names (default: local)."""
    raw_profile = os.environ.get("RAG_ENV", "local")
    try:
        profile = RagEnv(raw_profile)
    except ValueError as exc:
        raise ValueError(
            f"RAG_ENV={raw_profile!r} is not valid; must be 'local' or 'server'"
        ) from exc

    env_file = _env_file_for_profile(profile)
    settings = Settings(_env_file=env_file, rag_env=profile)

    if settings.trust_system_certs:
        # Corporate networks (e.g. the work laptop) sit behind a
        # TLS-inspecting proxy whose root CA is trusted by the OS but not
        # by certifi's bundled CA list, which httpx/litellm use by default.
        # truststore patches ssl to defer to the OS trust store instead.
        # Only enabled where needed (see .env.server / work-machine env
        # files) — inert everywhere else, so this never changes behavior
        # on Linux or in CI.
        import truststore
        truststore.inject_into_ssl()

    return settings


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. The rest of the app should call this, never Settings() directly."""
    return load_settings()
