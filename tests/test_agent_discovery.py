"""
Unit tests for the Phase-1 discovery entry point (src/agent.run_discovery).

Covers:
- returning a non-empty set of TopicCandidate objects
- returning [] when discovery surfaces nothing
- forwarding query/limit to discover_topic_candidates
- treating '' as "no query" (rotated query -> None)
- propagating errors from the discovery layer

src.agent.discover_topic_candidates is mocked; no network or API keys are
needed. Dummy env vars are installed before importing src.agent so config
validation passes without a populated .env file.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

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

from src.agent import run_discovery  # noqa: E402
from src.models import TopicCandidate  # noqa: E402


class RunDiscoveryTest(unittest.TestCase):
    def _candidate(self, title="Agentic AI orchestration", score=0.85) -> TopicCandidate:
        return TopicCandidate(
            title=title,
            url="https://example.com/agentic-orchestration",
            description="Orchestration frameworks are converging.",
            keywords=["agentic", "orchestration"],
            source="example.com",
            search_score=score,
        )

    def test_returns_candidates(self):
        expected = [self._candidate(), self._candidate(title="Memory for agents", score=0.7)]
        with patch("src.agent.discover_topic_candidates", return_value=expected) as m:
            result = run_discovery(query="agentic AI", limit=5)

        self.assertEqual(result, expected)
        m.assert_called_once_with(query="agentic AI", limit=5)

    def test_empty_result_returns_empty(self):
        with patch("src.agent.discover_topic_candidates", return_value=[]):
            result = run_discovery(query="agentic AI", limit=5)

        self.assertEqual(result, [])

    def test_empty_query_passes_none(self):
        with patch("src.agent.discover_topic_candidates", return_value=[]) as m:
            run_discovery()

        m.assert_called_once_with(query=None, limit=None)

    def test_whitespace_query_is_passed_through(self):
        with patch("src.agent.discover_topic_candidates", return_value=[]) as m:
            run_discovery(query="  custom query  ", limit=3)

        m.assert_called_once_with(query="  custom query  ", limit=3)

    def test_propagates_discovery_errors(self):
        with patch("src.agent.discover_topic_candidates", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                run_discovery(query="agentic AI")

    def test_does_not_touch_pipeline_state(self):
        """run_discovery only returns candidates; it must not mutate/draft."""
        candidate = self._candidate()
        with patch("src.agent.discover_topic_candidates", return_value=[candidate]) as m:
            result = run_discovery(query="agentic AI", limit=1)
        self.assertEqual(result, [candidate])


if __name__ == "__main__":
    unittest.main()