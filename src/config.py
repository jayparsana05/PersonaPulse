"""
PersonaPulse – Configuration & Environment Loading
===================================================
Central module for loading, validating, and exposing all
environment variables as a strongly-typed Settings dataclass.

Usage:
    from src.config import settings
    print(settings.GEMINI_API_KEY)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env file from the project root (two levels up from src/)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _require(key: str) -> str:
    """Return env var value or exit with a clear error message."""
    val = os.getenv(key, "").strip()
    if not val:
        print(f"[Config] ❌  Required environment variable '{key}' is missing or empty.")
        sys.exit(1)
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Settings:
    # ── LLM / Embeddings ──────────────────────────────────────────────────
    GEMINI_API_KEY: str

    # ── Search ────────────────────────────────────────────────────────────
    TAVILY_API_KEY: str

    # ── Supabase ──────────────────────────────────────────────────────────
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str

    # ── Telegram ──────────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str

    # ── LinkedIn ──────────────────────────────────────────────────────────
    LINKEDIN_ACCESS_TOKEN: str
    LINKEDIN_AUTHOR_URN: str
    LINKEDIN_TOKEN_EXPIRY_DATE: str          # raw string, e.g. "2025-12-31"

    # ── X (Twitter) ───────────────────────────────────────────────────────
    X_API_KEY: str = ""
    X_API_SECRET: str = ""
    X_ACCESS_TOKEN: str = ""
    X_ACCESS_SECRET: str = ""

    # ── Derived / Optional ────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    EMBEDDING_DIMENSIONS: int = 768
    LLM_MODEL: str = "gemini-3.6-flash"
    LLM_FALLBACK_MODEL: str = "gemini-3.5-flash-lite"
    DUPLICATE_THRESHOLD: float = 0.85
    TOKEN_WARN_DAYS: int = 5                 # alert if LinkedIn token expires within N days

    @property
    def fallback_models(self) -> list[str]:
        """Ordered list of fallback models to try if the primary model fails or is overloaded."""
        candidates = [
            self.LLM_FALLBACK_MODEL,
            "gemini-3.5-flash-lite",
            "gemini-3.6-flash",
        ]
        seen = {self.LLM_MODEL}
        fallbacks: list[str] = []
        for model in candidates:
            if model and model not in seen:
                seen.add(model)
                fallbacks.append(model)
        return fallbacks

    @property
    def linkedin_token_expiry(self) -> date:
        """Parse LINKEDIN_TOKEN_EXPIRY_DATE as a Python date object."""
        try:
            return datetime.strptime(self.LINKEDIN_TOKEN_EXPIRY_DATE, "%Y-%m-%d").date()
        except ValueError as exc:
            print(
                f"[Config] ❌  LINKEDIN_TOKEN_EXPIRY_DATE must be YYYY-MM-DD, "
                f"got '{self.LINKEDIN_TOKEN_EXPIRY_DATE}': {exc}"
            )
            sys.exit(1)

    @property
    def linkedin_token_days_remaining(self) -> int:
        """Number of calendar days until the LinkedIn token expires."""
        return (self.linkedin_token_expiry - date.today()).days

    @property
    def telegram_api_base(self) -> str:
        return f"https://api.telegram.org/bot{self.TELEGRAM_BOT_TOKEN}"


# ---------------------------------------------------------------------------
# Singleton – constructed once at import time
# ---------------------------------------------------------------------------

settings = Settings(
    # LLM
    GEMINI_API_KEY=_require("GEMINI_API_KEY"),
    # Search
    TAVILY_API_KEY=_require("TAVILY_API_KEY"),
    # Supabase
    SUPABASE_URL=_require("SUPABASE_URL"),
    SUPABASE_SERVICE_ROLE_KEY=_require("SUPABASE_SERVICE_ROLE_KEY"),
    # Telegram
    TELEGRAM_BOT_TOKEN=_require("TELEGRAM_BOT_TOKEN"),
    TELEGRAM_CHAT_ID=_require("TELEGRAM_CHAT_ID"),
    # LinkedIn
    LINKEDIN_ACCESS_TOKEN=_require("LINKEDIN_ACCESS_TOKEN"),
    LINKEDIN_AUTHOR_URN=_require("LINKEDIN_AUTHOR_URN"),
    LINKEDIN_TOKEN_EXPIRY_DATE=_require("LINKEDIN_TOKEN_EXPIRY_DATE"),
    # X (Twitter) - optional for local testing
    X_API_KEY=_optional("X_API_KEY"),
    X_API_SECRET=_optional("X_API_SECRET"),
    X_ACCESS_TOKEN=_optional("X_ACCESS_TOKEN"),
    X_ACCESS_SECRET=_optional("X_ACCESS_SECRET"),
    # Optional model overrides
    LLM_MODEL=_optional("LLM_MODEL", "gemini-3.6-flash"),
    LLM_FALLBACK_MODEL=_optional("LLM_FALLBACK_MODEL", "gemini-3.5-flash-lite"),
)

__all__ = ["settings"]
