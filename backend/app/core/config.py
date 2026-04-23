"""Application settings loaded from environment (never commit secrets)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_ENV_FILE = _BACKEND_ROOT / ".env"


class Settings(BaseSettings):
    """Runtime configuration via `.env` and process environment."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Env: LLM_PROVIDER, OPENAI_API_KEY, GROQ_API_KEY, GEMINI_API_KEY, *_MODEL, GROQ_BASE_URL
    llm_provider: Literal["openai", "groq", "gemini"] = "openai"
    openai_api_key: str | None = None
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    openai_model: str = "gpt-4o-mini"
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_model: str = "gemini-2.0-flash"
    groq_base_url: str = "https://api.groq.com/openai/v1"

    # RAG / ops (env: CHROMA_PERSIST_DIRECTORY, LOG_DIR)
    chroma_persist_directory: str | None = None
    log_directory: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LOG_DIR", "LOG_DIRECTORY", "log_directory"),
    )

    @field_validator("chroma_persist_directory", "log_directory", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: object) -> str | None:
        if v is None:
            return None
        if isinstance(v, str) and not v.strip():
            return None
        return str(v).strip() if isinstance(v, str) else v  # type: ignore[return-value]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()
