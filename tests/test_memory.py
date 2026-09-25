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

from src.memory import (  # noqa: E402
    check_topic_researched,
    delete_research_question,
    filter_repeated_sources,
    get_known_research_questions,
    get_known_source_urls,
    get_research_sources_for_question,
    link_research_sources,
    store_draft,
    store_research_question,
    store_research_sources,
)
from src.models import ResearchQuestion, ResearchSource  # noqa: E402


class FakeResult:
    data = [{"id": "00000000-0000-0000-0000-000000000000"}]


class FakeSupabase:
    """Minimal stand-in for the Supabase client that records the insert."""

    def __init__(self):
        self.table_name = None
        self.row = None
        self.rows = []
        self.deleted_table = None
        self.deleted_filter = None

    def table(self, name):
        self.table_name = name
        return self

    def insert(self, row):
        self.row = row
        self.rows.append(row)
        return self

    def delete(self):
        self.deleted_table = self.table_name
        return self

    def eq(self, col, val):
        self.deleted_filter = (col, val)
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


class LinkResearchSourcesTest(unittest.TestCase):
    """link_research_sources writes lightweight per-session link rows (url +
    title only, no duplicated full content) for already-known sources."""

    def setUp(self):
        self.fake_client = FakeSupabase()

    def _source(self):
        return ResearchSource(
            url="https://ex.com/known",
            title="Known source",
            body="A body snippet.",
            source="ex.com",
            published="2026-09-19",
            score=0.9,
        )

    def _link(self, sources=None, question_id="qid-1"):
        with patch("src.memory._get_supabase", return_value=self.fake_client):
            return link_research_sources(
                question_id, sources if sources is not None else [self._source()]
            )

    def test_inserts_into_research_sources_table(self):
        self._link()
        self.assertEqual(self.fake_client.table_name, "research_sources")

    def test_link_rows_record_url_and_title_only(self):
        self._link()
        row = self.fake_client.rows[0]
        self.assertEqual(row["url"], "https://ex.com/known")
        self.assertEqual(row["title"], "Known source")
        # No duplicated full content: body / source / published are omitted.
        self.assertNotIn("body", row)
        self.assertNotIn("source", row)
        self.assertNotIn("published", row)
        self.assertIsNotNone(row["accessed_at"])

    def test_link_rows_are_scoped_to_the_session(self):
        self._link(question_id="qid-7")
        self.assertEqual(self.fake_client.rows[0]["research_question_id"], "qid-7")

    def test_returns_one_id_per_link_row(self):
        ids = self._link([self._source(), self._source()])
        self.assertEqual(len(self.fake_client.rows), 2)
        self.assertEqual(len(ids), 2)
        self.assertTrue(all(i == FakeResult.data[0]["id"] for i in ids))

    def test_question_id_none_when_unknown(self):
        self._link(question_id=None)
        self.assertIsNone(self.fake_client.rows[0]["research_question_id"])


class FakeResultWithData:
    def __init__(self, data):
        self.data = data


class FakeSelectableSupabase:
    """Chainable fake supporting select/order/limit/eq/execute for reads."""

    def __init__(self, result_rows):
        self._result_rows = result_rows
        self.table_name = None
        self.calls = []

    def table(self, name):
        self.table_name = name
        return self

    def select(self, *cols):
        self.calls.append(("select", cols))
        return self

    def order(self, col, desc=False):
        self.calls.append(("order", col, desc))
        return self

    def limit(self, n):
        self.calls.append(("limit", n))
        return self

    def eq(self, col, val):
        self.calls.append(("eq", col, val))
        return self

    def in_(self, col, values):
        self.calls.append(("in_", col, values))
        return self

    def execute(self):
        return FakeResultWithData(self._result_rows)


KNOWN_SESSIONS = [
    {
        "id": "qid-1",
        "topic": "Agentic orchestration",
        "question": "Which orchestration framework scales best?",
        "status": "answered",
        "embedding": [1.0, 0.0],
    },
    {
        "id": "qid-2",
        "topic": "Memory for agents",
        "question": "How should agent memory be managed?",
        "status": "proposed",
        "embedding": [0.0, 1.0],
    },
]


class CheckTopicResearchedTest(unittest.TestCase):
    """Research-session memory: exact + semantic duplicate detection."""

    def _check(self, topic, question, **kwargs):
        with patch("src.memory.get_known_research_questions", return_value=KNOWN_SESSIONS), \
             patch("src.memory.get_normalized_embedding") as embed:
            result = check_topic_researched(topic, question, **kwargs)
        return result, embed

    def test_exact_duplicate_topic_and_question_is_matched(self):
        result, _ = self._check(
            "Agentic  orchestration",
            "Which orchestration framework scales best?",
            use_embedding=False,
        )
        self.assertTrue(result["matched"])
        self.assertEqual(result["reason"], "exact")
        self.assertEqual(result["question_id"], "qid-1")

    def test_duplicate_question_is_matched_regardless_of_topic_wording(self):
        result, _ = self._check(
            "Scaling agentic AI (reworded topic)",
            "which orchestration framework scales best?",
            use_embedding=False,
        )
        self.assertTrue(result["matched"])
        self.assertEqual(result["reason"], "exact")
        self.assertEqual(result["question_id"], "qid-1")

    def test_related_but_different_question_is_not_a_duplicate(self):
        """Same topic family, meaningfully different question → follow-up allowed."""
        result, _ = self._check(
            "Agentic orchestration",
            "What are the cost implications of agent orchestration?",
            use_embedding=False,
        )
        self.assertFalse(result["matched"])
        self.assertIsNone(result["reason"])
        self.assertIsNone(result["question_id"])

    def test_unrelated_topic_and_question_is_not_a_duplicate(self):
        result, _ = self._check(
            "RAG evaluation",
            "What is the best evaluation harness for retrieval pipelines?",
            use_embedding=False,
        )
        self.assertFalse(result["matched"])

    def test_empty_known_sessions_never_match(self):
        with patch("src.memory.get_known_research_questions", return_value=[]):
            result = check_topic_researched(
                "A", "B", use_embedding=False,
            )
        self.assertFalse(result["matched"])

    def test_semantic_match_above_threshold(self):
        with patch("src.memory.get_known_research_questions", return_value=KNOWN_SESSIONS), \
             patch("src.memory.get_normalized_embedding", return_value=[1.0, 0.0]):
            result = check_topic_researched(
                "Scaling production agents",
                "What is the most scalable orchestration framework?",
            )
        self.assertTrue(result["matched"])
        self.assertEqual(result["reason"], "semantic")
        self.assertEqual(result["question_id"], "qid-1")
        self.assertGreaterEqual(result["similarity"], 0.88)

    def test_semantic_similarity_below_threshold_allows_follow_up(self):
        with patch("src.memory.get_known_research_questions", return_value=KNOWN_SESSIONS), \
             patch("src.memory.get_normalized_embedding", return_value=[-1.0, 0.0]):
            result = check_topic_researched(
                "Scaling production agents",
                "What is the most scalable orchestration framework?",
            )
        self.assertFalse(result["matched"])
        self.assertIsNone(result["reason"])
        self.assertIsNotNone(result["similarity"])

    def test_memory_store_failure_is_non_blocking(self):
        with patch("src.memory.get_known_research_questions", side_effect=RuntimeError("db down")):
            result = check_topic_researched("A", "B", use_embedding=False)
        self.assertFalse(result["matched"])

    def test_embedding_failure_falls_back_to_exact_only(self):
        with patch("src.memory.get_known_research_questions", return_value=KNOWN_SESSIONS), \
             patch("src.memory.get_normalized_embedding", side_effect=RuntimeError("gemini down")):
            result = check_topic_researched(
                "RAG evaluation", "What is the best evaluation harness?", use_embedding=True,
            )
        self.assertFalse(result["matched"])


class GetKnownResearchQuestionsTest(unittest.TestCase):
    def test_returns_rows_from_research_questions(self):
        rows = [KNOWN_SESSIONS[0]]
        fake = FakeSelectableSupabase(rows)
        with patch("src.memory._get_supabase", return_value=fake):
            known = get_known_research_questions(limit=20)
        self.assertEqual(known, rows)
        self.assertEqual(fake.table_name, "research_questions")

    def test_store_failure_returns_empty(self):
        fake = FakeSelectableSupabase([])
        with patch("src.memory._get_supabase", side_effect=RuntimeError("db down")):
            known = get_known_research_questions()
        self.assertEqual(known, [])


class StoreResearchQuestionMemoryTest(unittest.TestCase):
    def test_optional_embedding_is_stored_as_pgvector_literal(self):
        fake = FakeSupabase()
        question = ResearchQuestion(topic="T", question="Q?", aspects=[])
        with patch("src.memory._get_supabase", return_value=fake):
            store_research_question(question, embedding=[0.1, 0.2])
        self.assertEqual(fake.row["embedding"], "[0.10000000,0.20000000]")

    def test_without_embedding_no_vector_column_is_sent(self):
        fake = FakeSupabase()
        question = ResearchQuestion(topic="T", question="Q?", aspects=[])
        with patch("src.memory._get_supabase", return_value=fake):
            store_research_question(question)
        self.assertNotIn("embedding", fake.row)


class GetResearchSourcesForQuestionTest(unittest.TestCase):
    def test_hydrates_sources_and_ids_for_a_session(self):
        row = {
            "id": "src-1",
            "url": "https://ex.com/a",
            "title": "Source A",
            "body": "Body.",
            "source": "ex.com",
            "published": "2026-09-19",
            "score": 0.9,
            "source_type": "secondary",
            "accessed_at": "2026-09-19T10:00:00+00:00",
        }
        fake = FakeSelectableSupabase([row])
        with patch("src.memory._get_supabase", return_value=fake):
            sources, ids = get_research_sources_for_question("qid-1")
        self.assertEqual(ids, ["src-1"])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].url, "https://ex.com/a")

    def test_no_question_id_returns_empty(self):
        with patch("src.memory._get_supabase") as sb:
            sources, ids = get_research_sources_for_question("")
        sb.assert_not_called()
        self.assertEqual((sources, ids), ([], []))

    def test_store_failure_returns_empty(self):
        with patch("src.memory._get_supabase", side_effect=RuntimeError("db down")):
            sources, ids = get_research_sources_for_question("qid-1")
        self.assertEqual((sources, ids), ([], []))

    def test_full_rows_do_not_trigger_enrichment_lookup(self):
        row = {
            "id": "src-1",
            "url": "https://ex.com/a",
            "title": "Source A",
            "body": "Body.",
            "source": "ex.com",
            "published": "2026-09-19",
            "score": 0.9,
            "source_type": "secondary",
            "accessed_at": "2026-09-19T10:00:00+00:00",
        }
        fake = FakeSelectableSupabase([row])
        with patch("src.memory._get_supabase", return_value=fake):
            sources, ids = get_research_sources_for_question("qid-1")
        self.assertEqual(ids, ["src-1"])
        self.assertEqual(sources[0].body, "Body.")
        self.assertNotIn("in_", [c[0] for c in fake.calls])

    def test_link_rows_are_enriched_with_existing_full_records(self):
        """Per-session link rows (url + empty body) are hydrated with the full
        content already stored for the same URL in an earlier session."""
        link_row = {
            "id": "lnk-1",
            "url": "https://ex.com/known",
            "title": "Known",
            "body": "",
            "source": "",
            "published": "",
            "score": 0.0,
            "source_type": "secondary",
            "accessed_at": "2026-09-20T10:00:00+00:00",
        }
        canonical = {
            "id": "src-0",
            "url": "https://ex.com/known",
            "title": "Known source",
            "body": "A snippet about orchestration.",
            "source": "ex.com",
            "published": "2026-09-19",
            "score": 0.9,
            "source_type": "secondary",
            "accessed_at": "2026-09-19T10:00:00+00:00",
        }

        class TwoStage(FakeSelectableSupabase):
            def __init__(self):
                super().__init__([link_row])
                self._reads = 0

            def execute(self):
                self._reads += 1
                return FakeResultWithData([canonical] if self._reads > 1 else self._result_rows)

        fake = TwoStage()
        with patch("src.memory._get_supabase", return_value=fake):
            sources, ids = get_research_sources_for_question("qid-1")
        # The session's link-row id is kept; content comes from the existing record.
        self.assertEqual(ids, ["lnk-1"])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].url, "https://ex.com/known")
        self.assertEqual(sources[0].title, "Known source")
        self.assertEqual(sources[0].body, "A snippet about orchestration.")
        self.assertEqual(sources[0].score, 0.9)

    def test_newer_link_row_does_not_override_full_source_content(self):
        """A lightweight link row written AFTER the original full source (same
        URL) is never chosen as the canonical record: hydration must return the
        full source content, not the younger, content-less link row."""
        session_link = {
            "id": "lnk-2",
            "url": "https://ex.com/known",
            "title": "Known",
            "body": "",
            "source": "",
            "published": "",
            "score": 0.0,
            "source_type": "secondary",
            "accessed_at": "2026-09-21T10:00:00+00:00",
        }
        full_source = {
            "id": "src-0",
            "url": "https://ex.com/known",
            "title": "Known source",
            "body": "A snippet about orchestration.",
            "source": "ex.com",
            "published": "2026-09-19",
            "score": 0.9,
            "source_type": "secondary",
            "accessed_at": "2026-09-19T10:00:00+00:00",
        }
        # The cross-session lookup returns rows newest-first: the session's own
        # link row (latest accessed_at) BEFORE the older full record.
        candidates = [session_link, full_source]

        class TwoStage(FakeSelectableSupabase):
            def __init__(self):
                super().__init__([session_link])
                self._reads = 0

            def execute(self):
                self._reads += 1
                return FakeResultWithData(candidates if self._reads > 1 else self._result_rows)

        fake = TwoStage()
        with patch("src.memory._get_supabase", return_value=fake):
            sources, ids = get_research_sources_for_question("qid-1")
        self.assertEqual(ids, ["lnk-2"])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].url, "https://ex.com/known")
        self.assertEqual(sources[0].title, "Known source")
        self.assertEqual(sources[0].body, "A snippet about orchestration.")
        self.assertEqual(sources[0].source, "ex.com")
        self.assertEqual(sources[0].published, "2026-09-19")
        self.assertEqual(sources[0].score, 0.9)


class GetKnownSourceUrlsTest(unittest.TestCase):
    def test_returns_normalized_known_urls(self):
        fake = FakeSelectableSupabase([
            {"url": "https://www.Example.com/a/?utm_source=x#frag"},
            {"url": "https://ex.com/b"},
            {"url": ""},
        ])
        with patch("src.memory._get_supabase", return_value=fake):
            known = get_known_source_urls()
        self.assertIn("https://example.com/a", known)
        self.assertIn("https://ex.com/b", known)

    def test_store_failure_returns_empty_set(self):
        with patch("src.memory._get_supabase", side_effect=RuntimeError("db down")):
            known = get_known_source_urls()
        self.assertEqual(known, set())


class FilterRepeatedSourcesTest(unittest.TestCase):
    def _source(self, url):
        return ResearchSource(url=url, title="T", body="B")

    def test_fresh_sources_all_pass(self):
        sources = [self._source("https://ex.com/1"), self._source("https://ex.com/2")]
        to_store, repeated = filter_repeated_sources(sources, known_urls=set())
        self.assertEqual(len(to_store), 2)
        self.assertEqual(repeated, [])

    def test_known_source_is_flagged_repeated(self):
        sources = [self._source("https://ex.com/1"), self._source("https://ex.com/2")]
        to_store, repeated = filter_repeated_sources(sources, known_urls={"https://ex.com/1"})
        self.assertEqual([s.url for s in to_store], ["https://ex.com/2"])
        self.assertEqual([s.url for s in repeated], ["https://ex.com/1"])

    def test_repeat_within_batch_keeps_first_occurrence(self):
        sources = [
            self._source("https://ex.com/a"),
            self._source("https://ex.com/a"),
            self._source("https://ex.com/b"),
        ]
        to_store, repeated = filter_repeated_sources(sources, known_urls=set())
        self.assertEqual([s.url for s in to_store], ["https://ex.com/a", "https://ex.com/b"])
        self.assertEqual([s.url for s in repeated], ["https://ex.com/a"])

    def test_known_normalized_equivalent_url_is_repeated(self):
        sources = [self._source("https://www.example.com/a/?utm_source=rss#top")]
        to_store, repeated = filter_repeated_sources(sources, known_urls={"https://example.com/a"})
        self.assertEqual(to_store, [])
        self.assertEqual(len(repeated), 1)


class DeleteResearchQuestionTest(unittest.TestCase):
    """delete_research_question removes the session row (sources cascade)."""

    def test_deletes_from_research_questions(self):
        fake = FakeSupabase()
        with patch("src.memory._get_supabase", return_value=fake):
            delete_research_question("qid-1")
        self.assertEqual(fake.deleted_table, "research_questions")

    def test_filters_on_row_id(self):
        fake = FakeSupabase()
        with patch("src.memory._get_supabase", return_value=fake):
            delete_research_question("qid-1")
        self.assertEqual(fake.deleted_filter, ("id", "qid-1"))

    def test_no_id_does_not_touch_the_store(self):
        with patch("src.memory._get_supabase") as sb:
            delete_research_question("")
            delete_research_question(None)
        sb.assert_not_called()

    def test_store_failure_is_non_blocking(self):
        with patch("src.memory._get_supabase", side_effect=RuntimeError("db down")):
            delete_research_question("qid-1")


if __name__ == "__main__":
    unittest.main()