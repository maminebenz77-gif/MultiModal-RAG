"""Environment-portability config: which profile (local|server) is active,
and every setting that differs between developing on a laptop and running
on the air-gapped company GPU server.

Profile selection happens via the RAG_ENV *OS* environment variable, read
before pydantic ever loads a .env file — RAG_ENV has to be known first,
since it decides which file (.env.local or .env.server) supplies everything
else.
"""

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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

    # Vector / search stores
    qdrant_url: str
    elastic_url: str

    # Runtime
    device: str = "auto"
    allow_external: bool = True

    @field_validator(
        "llm_base_url",
        "llm_api_key",
        "embed_base_url",
        "embed_api_key",
        "vision_provider",
        "vision_model",
        "vision_base_url",
        "vision_api_key",
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
    return Settings(_env_file=env_file, rag_env=profile)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton. The rest of the app should call this, never Settings() directly."""
    return load_settings()
