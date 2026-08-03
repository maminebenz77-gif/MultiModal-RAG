"""Frontend-only settings: just enough to find the API.

Deliberately separate from the backend's Settings (multimodal_rag.config)
-- the frontend never touches an LLM/embedding/store directly, only the
API over HTTP, so it has no business requiring backend secrets
(LLM keys, store URLs) just to start. It DOES reuse the same "RAG_ENV
picks .env.local vs .env.server" mechanism the rest of the project uses
for local-vs-server portability, just scoped down to the one setting
the frontend actually needs.
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file_encoding="utf-8", extra="ignore")

    api_base_url: str = "http://127.0.0.1:8000"


def _env_file_for_profile(profile: str) -> Path:
    return PROJECT_ROOT / f".env.{profile}"


@lru_cache
def get_frontend_settings() -> FrontendSettings:
    profile = os.environ.get("RAG_ENV", "local")
    env_file = _env_file_for_profile(profile)
    return FrontendSettings(_env_file=env_file)
