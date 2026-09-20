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

from src.memory import store_draft, store_research_question, store_research_sources  # noqa: E402
from src.models import ResearchQuestion, ResearchSource  # noqa: E402


class FakeResult:
    data = [{"id": "00000000-0000-0000-0000-000000000000"}]


class FakeSupabase:
    """Minimal stand-in for the Supabase client that records the insert."""

    def __init__(self):
        self.table_name = None
        self.row = None
        self.rows = []

    def table(self, name):
        self.table_name = name
        return self

    def insert(self, row):
        self.row = row
        self.rows.append(row)
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


class StoreResearchQuestionTest(unittest.TestCase):
    """store_research_question inserts a traceable row with topic + question."""

    def setUp(self):
        self.fake_client = FakeSupabase()

    def _question(self, **kwargs):
        defaults = {
            "topic": "Agentic orchestration",
            "question": "Which orchestration framework scales best?",
            "aspects": ["reliability", "cost"],
            "status": ResearchQuestion.STATUS_PROPOSED,
            "priority": ResearchQuestion.PRIORITY_NORMAL,
        }
        defaults.update(kwargs)
        return ResearchQuestion(**defaults)

    def _store(self, **kwargs):
        with patch("src.memory._get_supabase", return_value=self.fake_client):
            return store_research_question(self._question(**kwargs))

    def test_returns_question_id(self):
        self.assertEqual(self._store(), FakeResult.data[0]["id"])

    def test_inserts_into_research_questions_table(self):
        self._store()
        self.assertEqual(self.fake_client.table_name, "research_questions")

    def test_row_records_topic_and_question(self):
        self._store()
        self.assertEqual(self.fake_client.row["topic"], "Agentic orchestration")
        self.assertEqual(
            self.fake_client.row["question"], "Which orchestration framework scales best?"
        )

    def test_aspects_are_serialized_as_json(self):
        self._store()
        self.assertEqual(
            self.fake_client.row["aspects"], '["reliability", "cost"]'
        )

    def test_status_and_priority_are_recorded(self):
        self._store(status="proposed", priority="normal")
        self.assertEqual(self.fake_client.row["status"], "proposed")
        self.assertEqual(self.fake_client.row["priority"], "normal")

    def test_current_status_and_priority_round_trip(self):
        self._store(status="researching", priority="high")
        self.assertEqual(self.fake_client.row["status"], "researching")
        self.assertEqual(self.fake_client.row["priority"], "high")


class StoreResearchSourcesTest(unittest.TestCase):
    """store_research_sources inserts each normalized source into research_sources."""

    def setUp(self):
        self.fake_client = FakeSupabase()

    def _sources(self, n=2):
        return [
            ResearchSource(
                url=f"https://ex.com/{i}",
                title=f"Source {i}",
                body="A body snippet.",
                published="2026-09-19",
                source="ex.com",
                score=0.9 - i / 10,
                source_type=ResearchSource.SOURCE_TYPE_SECONDARY,
            )
            for i in range(n)
        ]

    def _store(self, sources=None, question_id="qid-1"):
        with patch("src.memory._get_supabase", return_value=self.fake_client):
            return store_research_sources(question_id, sources if sources is not None else self._sources())

    def test_inserts_into_research_sources_table(self):
        self._store()
        self.assertEqual(self.fake_client.table_name, "research_sources")

    def test_persists_each_normalized_source(self):
        self._store(self._sources(3))
        self.assertEqual(len(self.fake_client.rows), 3)

    def test_row_contains_source_metadata(self):
        self._store(self._sources(1))
        row = self.fake_client.rows[0]
        self.assertEqual(row["url"], "https://ex.com/0")
        self.assertEqual(row["title"], "Source 0")
        self.assertEqual(row["body"], "A body snippet.")
        self.assertEqual(row["source"], "ex.com")
        self.assertEqual(row["published"], "2026-09-19")
        self.assertEqual(row["score"], 0.9)
        self.assertEqual(row["source_type"], ResearchSource.SOURCE_TYPE_SECONDARY)
        self.assertIsNotNone(row["accessed_at"])

    def test_links_each_row_to_the_question_id(self):
        self._store(self._sources(2), question_id="qid-7")
        for row in self.fake_client.rows:
            self.assertEqual(row["research_question_id"], "qid-7")

    def test_question_id_none_when_unknown(self):
        self._store(question_id=None)
        self.assertIsNone(self.fake_client.rows[0]["research_question_id"])

    def test_returns_one_id_per_source(self):
        ids = self._store(self._sources(2))
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(i == FakeResult.data[0]["id"] for i in ids))


if __name__ == "__main__":
    unittest.main()