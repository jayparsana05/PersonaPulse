"""
Unit tests for the phase-2 selection entry point (src.agent.run_selection).

Covers:
- using injected candidates (skips discovery)
- discovering candidates when none are provided
- returning selection + research question
- returning a None question when nothing was selected

Discovery and the selection/question functions are mocked; no network or
API keys are needed. Dummy env vars are installed before importing src.agent.
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

# Dummy env vars (see tests/test_agent_discovery.py) so src.config imports
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

from src.agent import run_selection  # noqa: E402
from src.models import (  # noqa: E402
    ResearchQuestion,
    TopicCandidate,
    TopicSelection,
)


def candidate(title="Agentic orchestration") -> TopicCandidate:
    return TopicCandidate(
        title=title,
        url=f"https://example.com/{title.lower().replace(' ', '-')}",
        description=f"A description about {title}.",
        source="example.com",
        published="2026-09-19",
        search_score=0.85,
    )


def selection_for(topic, mode=TopicSelection.MODE_LLM) -> TopicSelection:
    return TopicSelection(
        selected=topic,
        reasoning="Strongest engineering relevance.",
        criteria=["Engineering relevance"],
        mode=mode,
    )


def question_for(topic) -> ResearchQuestion:
    return ResearchQuestion(
        topic=topic.title,
        question="Which framework scales?",
        aspects=["reliability"],
    )


class RunSelectionTest(unittest.TestCase):
    def test_injected_candidates_skip_discovery(self):
        topics = [candidate(), candidate(title="Memory for agents")]
        selection = selection_for(topics[0])
        question = question_for(topics[0])
        with patch("src.agent.discover_topic_candidates") as discover, \
             patch("src.agent.select_topic", return_value=selection) as select, \
             patch("src.agent.frame_question", return_value=question) as frame, \
             patch("src.agent.store_research_question", return_value="qid-1") as store:
            result = run_selection(query="agentic AI", limit=5, candidates=topics)

        discover.assert_not_called()
        select.assert_called_once_with(topics, query="agentic AI")
        frame.assert_called_once_with(topics[0], query="agentic AI")
        self.assertEqual(result["topic_candidates"], topics)
        self.assertEqual(result["selection"], selection)
        self.assertEqual(result["research_question"], question)
        self.assertEqual(result["research_question_id"], "qid-1")
        store.assert_called_once_with(question)

    def test_discovers_candidates_when_none_provided(self):
        topics = [candidate()]
        selection = selection_for(topics[0])
        with patch("src.agent.discover_topic_candidates", return_value=topics) as discover, \
             patch("src.agent.select_topic", return_value=selection), \
             patch("src.agent.frame_question", return_value=question_for(topics[0])), \
             patch("src.agent.store_research_question", return_value="qid-2") as store:
            result = run_selection(query="agentic AI", limit=5)

        discover.assert_called_once_with(query="agentic AI", limit=5)
        self.assertEqual(result["topic_candidates"], topics)
        self.assertEqual(result["research_question_id"], "qid-2")
        store.assert_called_once()

    def test_not_selected_returns_none_question(self):
        """When selection picks nothing, no research question is framed or stored."""
        empty_selection = TopicSelection(
            reasoning="Nothing fit.",
            mode=TopicSelection.MODE_NONE_FIT,
        )
        with patch("src.agent.discover_topic_candidates", return_value=[]), \
             patch("src.agent.select_topic", return_value=empty_selection), \
             patch("src.agent.frame_question", return_value=None) as frame, \
             patch("src.agent.store_research_question") as store:
            result = run_selection(query="agentic AI", limit=5)

        self.assertIsNone(result["selection"].selected)
        self.assertIsNone(result["research_question"])
        self.assertIsNone(result["research_question_id"])
        frame.assert_called_once_with(None, query="agentic AI")
        store.assert_not_called()

    def test_empty_query_rotates_via_discovery(self):
        with patch("src.agent.discover_topic_candidates", return_value=[]) as discover, \
             patch("src.agent.select_topic", return_value=TopicSelection(mode=TopicSelection.MODE_EMPTY)), \
             patch("src.agent.frame_question", return_value=None), \
             patch("src.agent.store_research_question"):
            run_selection()

        discover.assert_called_once_with(query=None, limit=None)

    def test_persistence_failure_does_not_block_selection(self):
        """A Supabase failure while storing must not break the selection result."""
        topics = [candidate()]
        selection = selection_for(topics[0])
        question = question_for(topics[0])
        with patch("src.agent.discover_topic_candidates", return_value=topics), \
             patch("src.agent.select_topic", return_value=selection), \
             patch("src.agent.frame_question", return_value=question), \
             patch("src.agent.store_research_question", side_effect=RuntimeError("db down")) as store:
            result = run_selection(query="agentic AI", limit=5, candidates=topics)

        self.assertEqual(result["selection"], selection)
        self.assertEqual(result["research_question"], question)
        self.assertIsNone(result["research_question_id"])
        store.assert_called_once()


class RunSelectionResearchMemoryTest(unittest.TestCase):
    """run_selection(check_researched=True) remembers previously researched
    topics/questions and never duplicates a research session."""

    def _setup(self, dup_response):
        topics = [candidate()]
        selection = selection_for(topics[0])
        question = question_for(topics[0])
        return topics, selection, question

    def _run(self, dup_response, store_id="qid-2"):
        topics, selection, question = self._setup(dup_response)
        with patch("src.agent.discover_topic_candidates", return_value=topics), \
             patch("src.agent.select_topic", return_value=selection), \
             patch("src.agent.frame_question", return_value=question), \
             patch("src.agent.check_topic_researched", return_value=dup_response) as check, \
             patch("src.agent.store_research_question", return_value=store_id) as store:
            result = run_selection(query="agentic AI", limit=5, candidates=topics, check_researched=True)
        return result, check, store

    def test_exact_duplicate_reuses_existing_session(self):
        dup = {
            "matched": True, "reason": "exact", "question_id": "qid-existing",
            "matched_question": {"id": "qid-existing"}, "similarity": None,
        }
        result, check, store = self._run(dup)
        self.assertTrue(result["already_researched"])
        self.assertEqual(result["research_question_id"], "qid-existing")
        store.assert_not_called()

    def test_new_topic_stores_and_marks_not_researched(self):
        dup = {
            "matched": False, "reason": None, "question_id": None,
            "matched_question": None, "similarity": None,
        }
        result, check, store = self._run(dup)
        self.assertFalse(result["already_researched"])
        self.assertEqual(result["research_question_id"], "qid-2")
        store.assert_called_once()

    def test_follow_up_different_question_is_allowed(self):
        dup = {
            "matched": False, "reason": None, "question_id": None,
            "matched_question": None, "similarity": 0.4,
        }
        result, _, store = self._run(dup)
        self.assertFalse(result["already_researched"])
        store.assert_called_once()


if __name__ == "__main__":
    unittest.main()