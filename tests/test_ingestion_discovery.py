"""
Unit tests for Phase-1 topic discovery (src/ingestion.py).

Covers:
- multiple candidates
- duplicate candidates (URL-based and source+title based)
- missing metadata
- empty discovery result

Uses a mocked Tavily client, so no network or API keys are needed.
A set of dummy env vars is installed before importing src.ingestion
so config validation passes even without a real .env file.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

# Dummy env vars so src.config (which _require()s several keys) imports
# cleanly in CI / without a populated .env.
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

from src.ingestion import (  # noqa: E402
    _normalize_url,
    discover_topic_candidates,
    fetch_trending_tech_news,
)
from src.models import TopicCandidate  # noqa: E402


class FakeTavily:
    """Minimal stand-in for TavilyClient that records search kwargs."""

    def __init__(self, results=None):
        self.results = results or []
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return {"results": self.results}


def raw_result(
    url="https://example.com/agentic-ai",
    title="The Rise of Agentic AI",
    content="A short snippet about agentic AI.",
    published="2026-09-19",
    score=0.91,
    keywords=None,
):
    return {
        "url": url,
        "title": title,
        "content": content,
        "published_date": published,
        "score": score,
        "keywords": keywords,
    }


class DiscoverTopicCandidatesTest(unittest.TestCase):
    def setUp(self):
        self.fake = FakeTavily()

    def _search_args(self):
        return self.fake.calls[-1]

    def test_multiple_candidates(self):
        self.fake.results = [
            raw_result(url="https://ex.com/a", title="A", content="One", published="2026-09-01", score=0.9),
            raw_result(url="https://ex.com/b", title="B", content="Two", published="2026-09-02", score=0.8),
            raw_result(url="https://ex.com/c", title="C", content="Three", published="2026-09-03", score=0.7),
        ]
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            candidates = discover_topic_candidates(query="agentic AI", limit=3)

        self.assertIsInstance(candidates, list)
        self.assertEqual(len(candidates), 3)
        self.assertTrue(all(isinstance(c, TopicCandidate) for c in candidates))

        first = candidates[0]
        self.assertEqual(first.url, "https://ex.com/a")
        self.assertEqual(first.title, "A")
        self.assertEqual(first.source, "ex.com")
        self.assertEqual(first.published, "2026-09-01")
        self.assertEqual(first.description, "One")
        self.assertEqual(first.search_score, 0.9)
        self.assertIsNotNone(first.discovered_at)

        # Results come back sorted by search score descending.
        self.assertEqual([c.search_score for c in candidates], [0.9, 0.8, 0.7])

    def test_candidates_are_sorted_after_dedup(self):
        self.fake.results = [
            raw_result(url="https://ex.com/a", title="A", content="One", published="", score=0.4),
            raw_result(url="https://ex.com/b", title="B", content="Two", published="", score=0.95),
        ]
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            candidates = discover_topic_candidates(query="agentic AI", limit=2)
        self.assertEqual([c.title for c in candidates], ["B", "A"])

    def test_search_uses_limit_and_candidate_params(self):
        self.fake.results = [raw_result()]
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            discover_topic_candidates(query="agentic AI", limit=7)

        kwargs = self._search_args()
        self.assertEqual(kwargs["max_results"], 7)
        self.assertIs(kwargs["include_raw_content"], False)
        self.assertEqual(kwargs["topic"], "news")
        self.assertEqual(kwargs["days"], 7)
        self.assertIn("facebook.com", kwargs["exclude_domains"])

    def test_no_query_uses_rotated_query(self):
        self.fake.results = [raw_result()]
        with patch("src.ingestion._get_tavily", return_value=self.fake), \
             patch("src.ingestion._rotate_search_query", return_value="patched-query"):
            discover_topic_candidates(limit=1)

        self.assertEqual(self._search_args()["query"], "patched-query")

    def test_limit_of_zero_returns_empty_without_search(self):
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            candidates = discover_topic_candidates(query="agentic AI", limit=0)
        self.assertEqual(candidates, [])
        self.assertEqual(self.fake.calls, [])

    def test_duplicate_urls_are_deduped(self):
        self.fake.results = [
            raw_result(url="https://www.Example.com/news/agents?utm_source=rss#top", title="Dup A"),
            raw_result(url="https://example.com/news/agents", title="Dup B"),
            raw_result(url="https://other.com/unique", title="Unique"),
        ]
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            candidates = discover_topic_candidates(query="agentic AI", limit=3)

        self.assertEqual(len(candidates), 2)
        urls = {_normalize_url(c.url) for c in candidates}
        self.assertEqual(urls, {"https://other.com/unique", "https://example.com/news/agents"})

    def test_duplicate_source_and_title_are_deduped(self):
        self.fake.results = [
            raw_result(url="https://a.example.com/x", title="Agent Framework Released"),
            raw_result(url="https://a.example.com/y", title="  Agent   Framework   Released "),
            raw_result(url="https://b.example.com/z", title="Agent Framework Released"),
        ]
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            candidates = discover_topic_candidates(query="agentic AI", limit=3)

        # a.example.com duplicate is collapsed; b.example.com is a distinct source.
        self.assertEqual(len(candidates), 2)

    def test_keywords_are_preserved_when_provided(self):
        self.fake.results = [
            raw_result(keywords=["agentic", "orchestration"]),
            raw_result(url="https://ex.com/no-kw", title="No Keywords", keywords=[]),
            raw_result(url="https://ex.com/bad-kw", title="Bad Keywords", keywords=["", 42, "agents"]),
        ]
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            candidates = discover_topic_candidates(query="agentic AI", limit=3)

        by_url = {c.url: c for c in candidates}
        self.assertEqual(by_url["https://example.com/agentic-ai"].keywords, ["agentic", "orchestration"])
        self.assertEqual(by_url["https://ex.com/no-kw"].keywords, [])
        self.assertEqual(by_url["https://ex.com/bad-kw"].keywords, ["agents"])

    def test_missing_metadata_is_tolerated(self):
        self.fake.results = [
            raw_result(url="https://ex.com/only-url", title="", content="", published="", score=None),  # no metadata
            raw_result(url="https://ex.com/title-only", title="Title Only"),  # partial
            {"title": "No URL at all"},                                      # skipped
        ]
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            candidates = discover_topic_candidates(query="agentic AI", limit=3)

        self.assertEqual(len(candidates), 2)
        by_url = {c.url: c for c in candidates}
        only_url = by_url["https://ex.com/only-url"]
        self.assertEqual(only_url.title, "")
        self.assertEqual(only_url.published, "")
        self.assertEqual(only_url.description, "")
        self.assertEqual(only_url.search_score, 0.0)

        self.assertEqual(by_url["https://ex.com/title-only"].title, "Title Only")

    def test_empty_discovery_result(self):
        self.fake.results = []
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            candidates = discover_topic_candidates(query="agentic AI", limit=5)

        self.assertEqual(candidates, [])

    def test_discovered_at_is_fresh(self):
        self.fake.results = [raw_result()]
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            candidates = discover_topic_candidates(query="agentic AI", limit=1)
        self.assertEqual(len(candidates), 1)
        self.assertIsNotNone(candidates[0].discovered_at)


class NormalizeUrlTest(unittest.TestCase):
    def test_normalizes_host_case_and_www(self):
        self.assertEqual(
            _normalize_url("https://WWW.Example.COM/news"),
            "https://example.com/news",
        )

    def test_drops_fragment_and_utm_params(self):
        self.assertEqual(
            _normalize_url("https://example.com/news?utm_source=rss&id=7#top"),
            "https://example.com/news?id=7",
        )

    def test_trailing_slash_is_normalized(self):
        self.assertEqual(
            _normalize_url("https://example.com/news/"),
            "https://example.com/news",
        )

    def test_empty_url(self):
        self.assertEqual(_normalize_url(""), "")
        self.assertEqual(_normalize_url(None), "")


class LegacyFetchStillWorksTest(unittest.TestCase):
    """Guard: the existing single-article flow is unchanged by the refactor."""

    def setUp(self):
        self.fake = FakeTavily()

    def test_returns_article_dict(self):
        self.fake.results = [
            {
                "url": "https://ex.com/agentic",
                "title": "  The Rise of Agentic AI  ",
                "raw_content": "Line one.\n\n\n\nLine two.   ",
                "published_date": "2026-09-19",
            }
        ]
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            article = fetch_trending_tech_news(query="agentic AI")

        self.assertEqual(article["url"], "https://ex.com/agentic")
        self.assertEqual(article["title"], "The Rise of Agentic AI")
        self.assertEqual(article["source"], "ex.com")
        self.assertEqual(article["published"], "2026-09-19")
        self.assertEqual(article["body"], "Line one.\n\nLine two.")

    def test_empty_raises_runtime_error(self):
        self.fake.results = []
        with patch("src.ingestion._get_tavily", return_value=self.fake):
            with self.assertRaises(RuntimeError):
                fetch_trending_tech_news(query="agentic AI")


if __name__ == "__main__":
    unittest.main()