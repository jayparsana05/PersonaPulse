"""
Unit tests for src.memory.store_draft row construction.

Covers:
- article_url and image_url are stored as separate columns
- the embedding is serialized as a pgvector literal
- status defaults to PENDING
- optional URL columns default to None

Uses a fake Supabase client, so no network or API keys are needed.
Dummy env vars are installed before importing src.memory so config
validation passes without a populated .env file.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

# Dummy env vars (see tests/test_ingestion_discovery.py) so src.config
# imports cleanly in CI / without a populated .env.
_REQUIRED_ENV = {
    "GEMINI_API_KEY": "test-gemini",
    "TAVILY_API_KEY": "test-tavily",
    "SUPABASE_URL": "https://test.supabase.co",
    "SUPABASE_SERVICE_ROLE_KEY": "test-sb-key",
    "TELEGRAM_BOT_TOKEN": "test-bot",
    "TELEGRAM_CHAT_ID": "12345",
    "LINKEDIN_ACCESS_TOKEN": "test-li-token",
    "LINKEDIN_AUTHOR_URN": "urn:li:person:TEST",
    "LINKEDIN_TOKEN_EXPIRY_DATE": "2099-12-31",
}
for _key, _value in _REQUIRED_ENV.items():
    os.environ.setdefault(_key, _value)

from src.memory import store_draft  # noqa: E402


class FakeResult:
    data = [{"id": "00000000-0000-0000-0000-000000000000"}]


class FakeSupabase:
    """Minimal stand-in for the Supabase client that records the insert."""

    def __init__(self):
        self.table_name = None
        self.row = None

    def table(self, name):
        self.table_name = name
        return self

    def insert(self, row):
        self.row = row
        return self

    def execute(self):
        return FakeResult()


class StoreDraftTest(unittest.TestCase):
    def setUp(self):
        self.fake_client = FakeSupabase()

    def _store(self, **kwargs):
        defaults = {
            "platform": "both",
            "topic": "The Rise of Agentic AI",
            "content": "LINKEDIN:\n...\n\nX:\n...",
            "embedding": [0.1, 0.2, 0.3],
        }
        defaults.update(kwargs)
        with patch("src.memory._get_supabase", return_value=self.fake_client):
            return store_draft(**defaults)

    def test_returns_post_id(self):
        post_id = self._store()
        self.assertEqual(post_id, FakeResult.data[0]["id"])

    def test_inserts_into_posts_table(self):
        self._store()
        self.assertEqual(self.fake_client.table_name, "posts")

    def test_article_and_image_urls_are_separate_columns(self):
        self._store(
            article_url="https://example.com/agentic-ai",
            image_url="https://img.example.com/hero.jpg",
        )
        self.assertEqual(
            self.fake_client.row["article_url"], "https://example.com/agentic-ai"
        )
        self.assertEqual(
            self.fake_client.row["image_url"], "https://img.example.com/hero.jpg"
        )

    def test_optional_urls_default_to_none(self):
        self._store()
        self.assertIsNone(self.fake_client.row["article_url"])
        self.assertIsNone(self.fake_client.row["image_url"])

    def test_embedding_is_a_pgvector_literal(self):
        self._store(embedding=[0.1, 0.2, 0.3])
        self.assertEqual(self.fake_client.row["embedding"], "[0.10000000,0.20000000,0.30000000]")

    def test_status_is_pending(self):
        self._store()
        self.assertEqual(self.fake_client.row["status"], "PENDING")


if __name__ == "__main__":
    unittest.main()