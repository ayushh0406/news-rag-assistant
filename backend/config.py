"""
Configuration Module
====================
Centralised application configuration using Pydantic Settings.
All environment variables are loaded here and exposed via a singleton
`settings` object that every other module imports.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Base directory for the project (the repo root)
# ---------------------------------------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    app_name: str = Field(default="AI News Research Assistant")
    app_version: str = Field(default="1.0.0")
    debug: bool = Field(default=False)

    # ------------------------------------------------------------------
    # Google / Gemini
    # ------------------------------------------------------------------
    google_api_key: str = Field(default="", description="Google Generative AI API key")
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_output_tokens: int = Field(default=8192)

    # ------------------------------------------------------------------
    # ChromaDB
    # ------------------------------------------------------------------
    chroma_persist_dir: str = Field(default="./chroma_db")
    chroma_collection_name: str = Field(default="news_articles")

    # ------------------------------------------------------------------
    # RAG / Chunking
    # ------------------------------------------------------------------
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    max_urls: int = Field(default=10, gt=0, le=20)
    top_k_results: int = Field(default=5, gt=0)

    # ------------------------------------------------------------------
    # FastAPI
    # ------------------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=True)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default="logs/app.log")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------
    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info) -> int:  # noqa: ANN001
        chunk_size = info.data.get("chunk_size", 1000)
        if v >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({v}) must be less than chunk_size ({chunk_size})"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    @property
    def chroma_persist_path(self) -> Path:
        """Resolved absolute path to the ChromaDB persistence directory."""
        return (BASE_DIR / self.chroma_persist_dir).resolve()

    @property
    def log_file_path(self) -> Optional[Path]:
        """Resolved absolute path to the log file, or None if not set."""
        if self.log_file:
            return (BASE_DIR / self.log_file).resolve()
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Module-level singleton for convenient importing:
#   from backend.config import settings
settings: Settings = get_settings()
