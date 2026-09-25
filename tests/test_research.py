"""
Unit tests for the Prompt-4 research stage (src/research.py) and its
entry point (src.agent.run_research).

Covers:
- research query generation (LLM + deterministic fallback)
- multi-source search (multi-query, partial failure, empty results)
- provider-result → ResearchSource normalization
- URL normalization + deduplication
- configurable source limits + deterministic ordering
- persistence (store_research_sources)
- end-to-end run_research()

LLM (complete_text) and the provider search (search_news) are mocked, so no
network or API keys are needed. Dummy env vars are installed before importing
src.research so config validation passes without a populated .env file.
"""

from __future__ import annotations

import json
import os
import unittest
from contextlib import contextmanager
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

from src.research import generate_research_queries, research_question  # noqa: E402
from src.agent import persist_research_question, run_research, run_research_or_reuse  # noqa: E402
from src.models import ResearchQuestion, ResearchSource  # noqa: E402


def research_question_fixture() -> ResearchQuestion:
    return ResearchQuestion(
        topic="Agentic orchestration",
        question="Which orchestration framework scales best for production agents?",
        aspects=["reliability", "cost"],
        status=ResearchQuestion.STATUS_PROPOSED,
        priority=ResearchQuestion.PRIORITY_NORMAL,
    )


def raw_result(
    url="https://ex.com/a",
    title="First source",
    content="A snippet about orchestration.",
    published="2026-09-19",
    score=0.9,
):
    return {
        "url": url,
        "title": title,
        "content": content,
        "published_date": published,
        "score": score,
    }


def _dump(payload) -> str:
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

class GenerateResearchQueriesTest(unittest.TestCase):
    def _generate(self, payload, **kwargs):
        with patch("src.research.complete_text", return_value=_dump(payload)) as m:
            result = generate_research_queries(research_question_fixture(), **kwargs)
        return result, m

    def test_valid_research_question_generates_queries(self):
        queries, _ = self._generate({"queries": ["q1", "q2"]})
        self.assertEqual(queries, ["q1", "q2"])

    def test_question_is_included_in_query_generation(self):
        _, m = self._generate({"queries": ["q1"]})
        self.assertIn("Which orchestration framework scales best", m.call_args.args[1])

    def test_aspects_influence_generated_queries(self):
        _, m = self._generate({"queries": ["q1"]})
        prompt = m.call_args.args[1]
        self.assertIn("reliability", prompt)
        self.assertIn("cost", prompt)

    def test_duplicate_queries_are_removed(self):
        queries, _ = self._generate({"queries": ["agentic", "agentic", "  agentic  ",
                                                 " agentic " , "multi-agent"]})
        self.assertEqual(queries, ["agentic", "multi-agent"])

    def test_casefold_duplicates_are_removed(self):
        queries, _ = self._generate({"queries": ["Agentic AI", "agentic ai"]})
        self.assertEqual(queries, ["Agentic AI"])

    def test_whitespace_is_normalized(self):
        queries, _ = self._generate({"queries": ["  multi   source   search  "]})
        self.assertEqual(queries, ["multi source search"])

    def test_maximum_query_count_is_enforced(self):
        queries, _ = self._generate({"queries": ["q1", "q2", "q3", "q4"]}, max_queries=2)
        self.assertEqual(queries, ["q1", "q2"])

    def test_non_string_entries_are_dropped(self):
        queries, _ = self._generate({"queries": ["ok", 5, None, "", "  "]})
        self.assertEqual(queries, ["ok"])

    def test_malformed_llm_response_triggers_fallback(self):
        queries, _ = self._generate("not json at all")
        self.assertEqual(queries[0], research_question_fixture().question)
        self.assertTrue(all(q.strip() for q in queries))

    def test_wrong_shaped_llm_response_triggers_fallback(self):
        queries, _ = self._generate({"queries": "not a list"})
        self.assertEqual(queries[0], research_question_fixture().question)

    def test_llm_failure_triggers_fallback(self):
        with patch("src.research.complete_text", side_effect=RuntimeError("LLM down")):
            queries = generate_research_queries(research_question_fixture())
        self.assertEqual(queries[0], research_question_fixture().question)

    def test_fallback_includes_question_and_aspects(self):
        queries = generate_research_queries(research_question_fixture(), use_llm=False)
        self.assertLessEqual(len(queries), 3)
        self.assertEqual(queries[0], research_question_fixture().question)
        self.assertTrue(any("reliability" in q for q in queries))
        self.assertTrue(any("cost" in q for q in queries))
        self.assertTrue(all(q.strip() for q in queries))

    def test_fallback_respects_max_queries(self):
        rq = ResearchQuestion(
            topic="T",
            question="Q?",
            aspects=["a1", "a2", "a3", "a4", "a5"],
        )
        queries = generate_research_queries(rq, use_llm=False, max_queries=2)
        self.assertEqual(len(queries), 2)

    def test_empty_research_question_returns_empty(self):
        rq = ResearchQuestion(topic="", question="", aspects=[])
        with patch("src.research.complete_text", side_effect=RuntimeError("down")):
            queries = generate_research_queries(rq, use_llm=False)
        self.assertEqual(queries, [])

    def test_none_research_question_returns_empty(self):
        with patch("src.research.complete_text") as m:
            queries = generate_research_queries(None)
        m.assert_not_called()
        self.assertEqual(queries, [])


# ---------------------------------------------------------------------------
# Multi-source search
# ---------------------------------------------------------------------------

class ResearchQuestionStageTest(unittest.TestCase):
    """research_question(): multi-query, resilient provider search."""

    def _patch(self, queries, side_effect):
        """Apply both patches in a single context manager that yields the
        search_news mock so tests can inspect call ordering/counts."""

        @contextmanager
        def _manager():
            with patch("src.research.generate_research_queries", return_value=queries), \
                 patch("src.research.search_news", side_effect=side_effect) as search:
                yield (None, search)

        return _manager()

    def test_multiple_queries_are_executed(self):
        with self._patch(["q1", "q2"], side_effect=[[], []]) as (_, search):
            research_question(research_question_fixture())
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args_list[0].kwargs["query"], "q1")
        self.assertEqual(search.call_args_list[1].kwargs["query"], "q2")

    def test_results_from_multiple_queries_are_collected(self):
        with self._patch(["q1", "q2"], side_effect=[[raw_result(url="https://ex.com/1")],
                                                    [raw_result(url="https://ex.com/2")]]):
            sources = research_question(research_question_fixture())
        self.assertEqual(len(sources), 2)

    def test_empty_query_result_does_not_stop_research(self):
        with self._patch(["q1", "q2"], side_effect=[[], [raw_result(url="https://ex.com/2")]]):
            sources = research_question(research_question_fixture())
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].url, "https://ex.com/2")

    def test_one_query_failure_does_not_stop_remaining(self):
        with self._patch(["q1", "q2"], side_effect=[RuntimeError("provider down"),
                                                    [raw_result(url="https://ex.com/2")]]):
            sources = research_question(research_question_fixture())
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].url, "https://ex.com/2")

    def test_all_failures_return_empty_safely(self):
        with self._patch(["q1", "q2"], side_effect=[RuntimeError("a"), RuntimeError("b")]):
            sources = research_question(research_question_fixture())
        self.assertEqual(sources, [])

    def test_all_empty_results_return_empty(self):
        with self._patch(["q1", "q2"], side_effect=[[], []]):
            sources = research_question(research_question_fixture())
        self.assertEqual(sources, [])

    def test_per_query_max_results_is_respected(self):
        with self._patch(["q1"], side_effect=[[]]) as (_, search):
            research_question(research_question_fixture(), max_sources_per_query=3)
        self.assertEqual(search.call_args.kwargs["max_results"], 3)

    def test_no_research_question_returns_empty(self):
        with patch("src.research.generate_research_queries") as m:
            self.assertEqual(research_question(None), [])
        m.assert_not_called()


# ---------------------------------------------------------------------------
# Source normalization
# ---------------------------------------------------------------------------

class SourceNormalizationTest(unittest.TestCase):
    def _single(self, results):
        with patch("src.research.generate_research_queries", return_value=["q"]), \
             patch("src.research.search_news", return_value=results):
            return research_question(research_question_fixture())

    def test_maps_provider_result_to_research_source(self):
        sources = self._single([raw_result()])
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertIsInstance(source, ResearchSource)
        self.assertEqual(source.url, "https://ex.com/a")
        self.assertEqual(source.title, "First source")
        self.assertEqual(source.body, "A snippet about orchestration.")
        self.assertEqual(source.source, "ex.com")
        self.assertEqual(source.published, "2026-09-19")
        self.assertEqual(source.score, 0.9)
        self.assertEqual(source.source_type, ResearchSource.SOURCE_TYPE_SECONDARY)
        self.assertIsNotNone(source.accessed_at)

    def test_missing_optional_metadata_is_handled(self):
        sources = self._single([raw_result(title="", content="", published="", score=None)])
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertEqual(source.title, "")
        self.assertEqual(source.body, "")
        self.assertEqual(source.published, "")
        self.assertEqual(source.score, 0.0)

    def test_publication_date_is_preserved_when_available(self):
        sources = self._single([raw_result(published="2026-09-19")])
        self.assertEqual(sources[0].published, "2026-09-19")

    def test_search_score_is_preserved_when_available(self):
        sources = self._single([raw_result(score=0.87)])
        self.assertEqual(sources[0].score, 0.87)

    def test_result_without_url_is_skipped(self):
        sources = self._single([raw_result(url="")])
        self.assertEqual(sources, [])

    def test_relevance_floor_filters_low_score_results(self):
        results = [
            raw_result(url="https://ex.com/high", score=0.9),
            raw_result(url="https://ex.com/low", score=0.02),
            {"url": "https://ex.com/noscore", "title": "No score"},
        ]
        with patch("src.research.generate_research_queries", return_value=["q"]), \
             patch("src.research.search_news", return_value=results):
            sources = research_question(research_question_fixture(), min_score=0.1)
        urls = [s.url for s in sources]
        self.assertIn("https://ex.com/high", urls)
        self.assertIn("https://ex.com/noscore", urls)
        self.assertNotIn("https://ex.com/low", urls)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class DeduplicationTest(unittest.TestCase):
    def _run(self, results, **kwargs):
        with patch("src.research.generate_research_queries", return_value=["q"]), \
             patch("src.research.search_news", return_value=results):
            return research_question(research_question_fixture(), **kwargs)

    def test_identical_urls_are_deduplicated(self):
        sources = self._run([
            raw_result(url="https://ex.com/a", title="A1", score=0.9),
            raw_result(url="https://ex.com/a", title="A2", score=0.8),
        ])
        self.assertEqual(len(sources), 1)

    def test_normalized_equivalent_urls_are_deduplicated(self):
        sources = self._run([
            raw_result(url="https://www.Example.com/news/agents?utm_source=rss#top", score=0.7),
            raw_result(url="https://example.com/news/agents/", score=0.9),
        ])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].score, 0.9)

    def test_duplicate_keeps_strongest_metadata(self):
        sources = self._run([
            raw_result(url="https://ex.com/a", title="Short", content="", score=0.8),
            raw_result(url="https://ex.com/a", title="A much longer and more descriptive title", content="", score=0.8),
        ])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].title, "A much longer and more descriptive title")

    def test_duplicate_prefers_entry_with_body(self):
        sources = self._run([
            raw_result(url="https://ex.com/a", title="T", content="", score=0.8),
            raw_result(url="https://ex.com/a", title="T", content="Has a body snippet.", score=0.8),
        ])
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].body, "Has a body snippet.")

    def test_duplicate_across_queries_produces_one_source(self):
        with patch("src.research.generate_research_queries", return_value=["q1", "q2"]), \
             patch("src.research.search_news", side_effect=[
                 [raw_result(url="https://ex.com/same", score=0.5)],
                 [raw_result(url="https://ex.com/same", score=0.95)],
             ]):
            sources = research_question(research_question_fixture())
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].score, 0.95)

    def test_final_ordering_is_deterministic_by_score(self):
        results = [
            raw_result(url="https://ex.com/low", score=0.2),
            raw_result(url="https://ex.com/high", score=0.99),
            raw_result(url="https://ex.com/mid", score=0.5),
        ]
        first = self._run(results)
        second = self._run(results)
        self.assertEqual([s.url for s in first], [s.url for s in second])
        self.assertEqual([s.url for s in first],
                         ["https://ex.com/high", "https://ex.com/mid", "https://ex.com/low"])

    def test_equal_score_and_title_length_order_urls_ascending(self):
        results = [
            raw_result(url="https://ex.com/zebra", title="Same Title Length", score=0.5),
            raw_result(url="https://ex.com/apple", title="Same Title Length", score=0.5),
            raw_result(url="https://ex.com/mango", title="Same Title Length", score=0.5),
        ]
        sources = self._run(results)
        self.assertEqual([s.score for s in sources], [0.5, 0.5, 0.5])
        self.assertEqual(len({len(s.title) for s in sources}), 1)
        self.assertEqual([s.url for s in sources],
                         ["https://ex.com/apple", "https://ex.com/mango", "https://ex.com/zebra"])

    def test_equal_score_and_title_length_order_normalized_urls_ascending(self):
        results = [
            raw_result(url="https://www.Example.com/zebra/?x=1#frag", title="T", score=0.5),
            raw_result(url="https://EX.com/apple", title="T", score=0.5),
            raw_result(url="https://example.com/mango/", title="T", score=0.5),
        ]
        sources = self._run(results)
        self.assertEqual([s.url for s in sources],
                         ["https://EX.com/apple", "https://example.com/mango/", "https://www.Example.com/zebra/?x=1#frag"])


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

class LimitsTest(unittest.TestCase):
    def test_final_source_limit_is_respected(self):
        results = [raw_result(url=f"https://ex.com/{i}", score=0.9 - i / 10) for i in range(8)]
        with patch("src.research.generate_research_queries", return_value=["q"]), \
             patch("src.research.search_news", return_value=results):
            sources = research_question(research_question_fixture(), max_sources=3)
        self.assertEqual(len(sources), 3)

    def test_deduplication_happens_before_final_limit(self):
        results = [
            raw_result(url="https://ex.com/1", score=0.9),
            raw_result(url="https://ex.com/1", score=0.85),
            raw_result(url="https://ex.com/2", score=0.8),
            raw_result(url="https://ex.com/3", score=0.7),
        ]
        with patch("src.research.generate_research_queries", return_value=["q"]), \
             patch("src.research.search_news", return_value=results):
            sources = research_question(research_question_fixture(), max_sources=5)
        self.assertEqual(len(sources), 3)  # 3 unique even though 4 raw results
        self.assertEqual(len({s.url for s in sources}), 3)


# ---------------------------------------------------------------------------
# Entry point: run_research()
# ---------------------------------------------------------------------------

class RunResearchTest(unittest.TestCase):
    def test_end_to_end_produces_research_sources(self):
        queries = ["Which orchestration framework scales best for production agents?"]
        results = [
            raw_result(url="https://ex.com/1", score=0.9),
            raw_result(url="https://ex.com/2", score=0.8),
        ]
        with patch("src.research.complete_text", return_value=_dump({"queries": queries})), \
             patch("src.research.search_news", return_value=results), \
             patch("src.agent.store_research_sources", return_value=["id-1", "id-2"]) as store, \
             patch("src.agent.draft_post") as draft, \
             patch("src.agent.store_draft") as store_post:
            result = run_research(research_question_fixture(), research_question_id="qid-1")

        sources = result["research_sources"]
        self.assertEqual(len(sources), 2)
        self.assertEqual({s.url for s in sources}, {"https://ex.com/1", "https://ex.com/2"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["research_source_ids"], ["id-1", "id-2"])
        store.assert_called_once()
        self.assertEqual(store.call_args.args[0], "qid-1")
        draft.assert_not_called()
        store_post.assert_not_called()

    def test_boundary_stops_at_research_sources(self):
        with patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=[raw_result()]), \
             patch("src.agent.store_research_sources", return_value=["id-1"]):
            result = run_research(research_question_fixture())
        self.assertEqual(set(result.keys()),
                         {"research_question", "research_sources", "status", "research_source_ids"})
        self.assertTrue(all(isinstance(s, ResearchSource) for s in result["research_sources"]))
        self.assertNotIn("evidence", result)
        self.assertNotIn("report", result)

    def test_dedup_before_persistence(self):
        query = ["q"]
        results = [
            raw_result(url="https://ex.com/same", score=0.5),
            raw_result(url="https://ex.com/same", score=0.9),
        ]
        with patch("src.research.complete_text", return_value=_dump({"queries": query})), \
             patch("src.research.search_news", return_value=results), \
             patch("src.agent.store_research_sources", return_value=["id-1"]) as store:
            result = run_research(research_question_fixture(), research_question_id="qid-1")
        self.assertEqual(result["status"], "ok")
        # Only the deduplicated source reaches persistence.
        self.assertEqual(store.call_args.args[1], [result["research_sources"][0]])
        self.assertEqual(len(result["research_sources"]), 1)

    def test_persistence_failure_does_not_destroy_result(self):
        with patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=[raw_result()]), \
             patch("src.agent.store_research_sources", side_effect=RuntimeError("db down")):
            result = run_research(research_question_fixture(), research_question_id="qid-1")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["research_source_ids"], [])
        self.assertEqual(len(result["research_sources"]), 1)

    def test_no_research_question_id_skips_persistence_but_returns_sources(self):
        with patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=[raw_result()]), \
             patch("src.agent.store_research_sources") as store:
            result = run_research(research_question_fixture(), research_question_id=None)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["research_sources"]), 1)
        self.assertEqual(result["research_source_ids"], [])
        store.assert_not_called()

    def test_no_sources_returns_empty_status(self):
        with patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=[]), \
             patch("src.agent.store_research_sources") as store:
            result = run_research(research_question_fixture())
        self.assertEqual(result["research_sources"], [])
        self.assertEqual(result["status"], "empty")
        store.assert_not_called()


# ---------------------------------------------------------------------------
# Research question persistence (used to link stored sources to their session)
# ---------------------------------------------------------------------------

class PersistResearchQuestionTest(unittest.TestCase):
    def test_returns_uuid_when_store_succeeds(self):
        with patch("src.agent.store_research_question", return_value="qid-9") as store:
            question_id = persist_research_question(research_question_fixture())
        self.assertEqual(question_id, "qid-9")
        store.assert_called_once_with(research_question_fixture())

    def test_returns_none_when_store_is_unavailable(self):
        with patch("src.agent.store_research_question", side_effect=RuntimeError("db down")):
            question_id = persist_research_question(research_question_fixture())
        self.assertIsNone(question_id)

    def test_stored_question_id_links_to_persisted_sources(self):
        with patch("src.agent.store_research_question", return_value="qid-9") as store, \
             patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=[raw_result()]), \
             patch("src.agent.store_research_sources", return_value=["id-1"]) as sources_store:
            result = run_research(research_question_fixture(), research_question_id=store.return_value)
        sources_store.assert_called_once()
        self.assertEqual(sources_store.call_args.args[0], "qid-9")
        self.assertEqual(result["research_source_ids"], ["id-1"])


class RunResearchSkipKnownSourcesTest(unittest.TestCase):
    """run_research(skip_known_sources=True) keeps source-level dedup across
    sessions: already-known URLs still appear in the in-memory result but are
    not persisted again."""

    def test_known_source_is_filtered_from_persistence_only(self):
        results = [
            raw_result(url="https://ex.com/known", score=0.9),
            raw_result(url="https://ex.com/fresh", score=0.8),
        ]
        with patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=results), \
             patch("src.agent.get_known_source_urls", return_value={"https://ex.com/known"}), \
             patch("src.agent.store_research_sources", return_value=["id-2"]) as store:
            result = run_research(research_question_fixture(), research_question_id="qid-1", skip_known_sources=True)

        self.assertEqual(result["status"], "ok")
        # In-memory result keeps both sources (dedup is only about persistence).
        self.assertEqual(len(result["research_sources"]), 2)
        # Only the fresh source reaches persistence.
        self.assertEqual(len(store.call_args.args[1]), 1)
        self.assertEqual(store.call_args.args[1][0].url, "https://ex.com/fresh")

    def test_default_disables_cross_session_filter(self):
        results = [raw_result(url="https://ex.com/known", score=0.9)]
        with patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=results), \
             patch("src.agent.get_known_source_urls") as known, \
             patch("src.agent.store_research_sources", return_value=["id-1"]) as store:
            run_research(research_question_fixture(), research_question_id="qid-1")
        known.assert_not_called()
        self.assertEqual(len(store.call_args.args[1]), 1)

    def test_memory_unavailability_falls_back_to_persist_everything(self):
        results = [raw_result(url="https://ex.com/known", score=0.9)]
        with patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=results), \
             patch("src.agent.get_known_source_urls", side_effect=RuntimeError("db down")), \
             patch("src.agent.store_research_sources", return_value=["id-1"]) as store:
            result = run_research(research_question_fixture(), research_question_id="qid-1", skip_known_sources=True)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(store.call_args.args[1]), 1)


class MemorySim:
    """In-memory mirror of the research-session store (exact matching only).

    Mirrors the exact-match semantics of ``check_topic_researched`` plus the
    store / hydrate / delete functions, so ``run_research_or_reuse``'s reuse
    gate and discard-cleanup run end to end without Supabase. The ``search``
    context manager installs the standard patches and yields the ``search_news``
    mock for call assertions.
    """

    def __init__(self):
        self.sessions = []
        self._next = 1

    @staticmethod
    def _norm(text):
        return " ".join((text or "").split()).casefold()

    def session_ids(self):
        return [s["id"] for s in self.sessions]

    def check_topic_researched(self, topic, question, **kwargs):
        norm_topic = self._norm(topic)
        norm_question = self._norm(question)
        for session in self.sessions:
            same_topic = bool(norm_topic) and self._norm(session["topic"]) == norm_topic
            same_question = bool(norm_question) and self._norm(session["question"]) == norm_question
            if same_topic and same_question:
                return {"matched": True, "reason": "exact", "question_id": session["id"],
                        "matched_question": session["meta"], "similarity": None}
            if same_question:
                return {"matched": True, "reason": "exact", "question_id": session["id"],
                        "matched_question": session["meta"], "similarity": None}
        return {"matched": False, "reason": None, "question_id": None,
                "matched_question": None, "similarity": None}

    def get_research_sources_for_question(self, question_id):
        for session in self.sessions:
            if session["id"] == question_id:
                return list(session["sources"]), list(session["source_ids"])
        return [], []

    def store_research_question(self, rq):
        question_id = f"qid-{self._next}"
        self._next += 1
        meta = {"id": question_id, "topic": rq.topic, "question": rq.question,
                "status": getattr(rq, "status", None)}
        self.sessions.append({"id": question_id, "topic": rq.topic, "question": rq.question,
                              "sources": [], "source_ids": [], "meta": meta})
        return question_id

    def store_research_sources(self, research_question_id, sources):
        for session in self.sessions:
            if session["id"] == research_question_id:
                ids = [f"src-{len(session['source_ids']) + i}" for i in range(len(sources))]
                session["sources"] = list(sources)
                session["source_ids"] = ids
                return ids
        return []

    def delete_research_question(self, research_question_id):
        self.sessions[:] = [s for s in self.sessions if s["id"] != research_question_id]

    def seed(self, question_id="qid-old", with_sources=False):
        """Pre-seed a remembered session for the fixture topic/question."""
        rq = research_question_fixture()
        session = {"id": question_id, "topic": rq.topic, "question": rq.question,
                   "sources": [], "source_ids": [],
                   "meta": {"id": question_id, "topic": rq.topic,
                            "question": rq.question, "status": rq.status}}
        if with_sources:
            session["sources"] = [ResearchSource(url="https://ex.com/old", title="Old source")]
            session["source_ids"] = ["src-0"]
        self.sessions.append(session)

    @contextmanager
    def search(self, search_side_effect):
        with patch.multiple(
            "src.agent",
            check_topic_researched=self.check_topic_researched,
            get_research_sources_for_question=self.get_research_sources_for_question,
            get_known_source_urls=lambda: set(),
            store_research_question=self.store_research_question,
            store_research_sources=self.store_research_sources,
            delete_research_question=self.delete_research_question,
        ), patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
           patch("src.research.search_news", side_effect=search_side_effect) as search_mock:
            yield search_mock


class RunResearchOrReuseTest(unittest.TestCase):
    """run_research_or_reuse(): remember researched topics/questions, reuse
    exact/semantic duplicates, and never prevent follow-up research."""

    def test_exact_duplicate_reuses_session(self):
        rq = research_question_fixture()
        with patch("src.agent.check_topic_researched", return_value={
                "matched": True, "reason": "exact",
                "question_id": "qid-old", "matched_question": {}, "similarity": None,
            }) as check, \
             patch("src.agent.get_research_sources_for_question",
                   return_value=([ResearchSource(url="https://ex.com/a", title="A")], ["src-1"])), \
             patch("src.agent.research_stage") as stage, \
             patch("src.agent.store_research_sources") as store:
            result = run_research_or_reuse(rq)

        check.assert_called_once_with(rq.topic, rq.question, threshold=None)
        self.assertEqual(result["status"], "reused")
        self.assertEqual(result["duplicate_reason"], "exact")
        self.assertEqual(result["reused_question_id"], "qid-old")
        self.assertEqual(result["research_question_id"], "qid-old")
        self.assertEqual([s.url for s in result["research_sources"]], ["https://ex.com/a"])
        stage.assert_not_called()
        store.assert_not_called()

    def test_semantic_duplicate_reuses_session(self):
        with patch("src.agent.check_topic_researched", return_value={
                "matched": True, "reason": "semantic",
                "question_id": "qid-old", "matched_question": {}, "similarity": 0.94,
            }), \
             patch("src.agent.get_research_sources_for_question",
                   return_value=([ResearchSource(url="https://ex.com/b")], ["src-2"])):
            result = run_research_or_reuse(research_question_fixture())
        self.assertEqual(result["status"], "reused")
        self.assertEqual(result["duplicate_reason"], "semantic")

    def test_related_but_different_question_runs_fresh_research(self):
        """Follow-up research on a meaningfully different question proceeds."""
        with patch("src.agent.check_topic_researched", return_value={
                "matched": False, "reason": None, "question_id": None,
                "matched_question": None, "similarity": 0.4,
            }), \
             patch("src.agent.store_research_question", return_value="qid-new") as store_q, \
             patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=[raw_result(url="https://ex.com/1")]), \
             patch("src.agent.get_known_source_urls", return_value=set()), \
             patch("src.agent.store_research_sources", return_value=["id-1"]):
            result = run_research_or_reuse(research_question_fixture())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["duplicate_reason"], None)
        self.assertEqual(result["research_question_id"], "qid-new")
        self.assertEqual(len(result["research_sources"]), 1)
        store_q.assert_called_once()

    def test_memory_unavailable_does_not_prevent_research(self):
        with patch("src.agent.check_topic_researched", side_effect=RuntimeError("db down")), \
             patch("src.agent.store_research_question", return_value="qid-new"), \
             patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news", return_value=[raw_result(url="https://ex.com/1")]), \
             patch("src.agent.get_known_source_urls", return_value=set()), \
             patch("src.agent.store_research_sources", return_value=["id-1"]):
            result = run_research_or_reuse(research_question_fixture())
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["research_question_id"], "qid-new")

    def test_remembered_session_without_sources_runs_fresh_research(self):
        """A matched session with no usable stored sources is not reused empty."""
        with patch("src.agent.check_topic_researched", return_value={
                "matched": True, "reason": "exact",
                "question_id": "qid-old", "matched_question": {}, "similarity": None,
            }), \
             patch("src.agent.get_research_sources_for_question", return_value=([], [])), \
             patch("src.agent.store_research_question", return_value="qid-new") as store_q, \
             patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
             patch("src.research.search_news",
                   return_value=[raw_result(url="https://ex.com/1")]) as search, \
             patch("src.agent.get_known_source_urls", return_value=set()), \
             patch("src.agent.store_research_sources", return_value=["id-1"]):
            result = run_research_or_reuse(research_question_fixture())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["duplicate_reason"], None)
        self.assertEqual(result["research_question_id"], "qid-new")
        self.assertEqual(len(result["research_sources"]), 1)
        store_q.assert_called_once()
        search.assert_called_once()


def _hydration_failure(*args, **kwargs):
    raise RuntimeError("hydration down")


class RunResearchOrReuseMemoryTest(unittest.TestCase):
    """run_research_or_reuse against an in-memory session mirror:

    - failed / incomplete research is never remembered,
    - an empty or un-hydratable remembered session never yields an empty
      ``reused`` result (fresh research runs instead),
    - successful research is reused on a later run.
    """

    def setUp(self):
        self.sim = MemorySim()

    def _run(self, search_results, **kwargs):
        """Run the fixture question once; returns (result, search_news mock)."""
        with self.sim.search(search_results) as search:
            result = run_research_or_reuse(research_question_fixture(), **kwargs)
        return result, search

    def test_first_attempt_without_sources_then_next_run_researches_fresh(self):
        first, _ = self._run([[]])
        self.assertEqual(first["status"], "empty")
        self.assertIsNone(first["research_question_id"])
        self.assertEqual(self.sim.sessions, [])

        second, _ = self._run([[raw_result(url="https://ex.com/fresh")]])
        self.assertEqual(second["status"], "ok")
        self.assertEqual(len(self.sim.sessions), 1)
        self.assertNotEqual(second["research_question_id"], first["research_question_id"])

    def test_research_failure_then_next_run_researches_fresh(self):
        first, _ = self._run([[RuntimeError("provider down")]])
        self.assertEqual(first["status"], "empty")
        self.assertIsNone(first["research_question_id"])
        self.assertEqual(self.sim.sessions, [])

        second, _ = self._run([[raw_result(url="https://ex.com/ok")]])
        self.assertEqual(second["status"], "ok")
        self.assertEqual(len(self.sim.sessions), 1)

    def test_successful_research_is_reused_on_the_next_run(self):
        first, _ = self._run([[raw_result(url="https://ex.com/a")]])
        self.assertEqual(first["status"], "ok")
        qid = first["research_question_id"]
        self.assertEqual(self.sim.session_ids(), [qid])
        self.assertEqual(len(self.sim.sessions[0]["source_ids"]), 1)

        second, search = self._run([[raw_result(url="https://ex.com/ignored")]])
        self.assertEqual(second["status"], "reused")
        self.assertEqual(second["duplicate_reason"], "exact")
        self.assertEqual(second["reused_question_id"], qid)
        self.assertEqual(second["research_question_id"], qid)
        self.assertEqual([s.url for s in second["research_sources"]], ["https://ex.com/a"])
        search.assert_not_called()

    def test_failed_research_cleans_up_the_created_session(self):
        with patch.multiple(
            "src.agent",
            check_topic_researched=self.sim.check_topic_researched,
            get_research_sources_for_question=self.sim.get_research_sources_for_question,
            get_known_source_urls=lambda: set(),
            store_research_question=self.sim.store_research_question,
            store_research_sources=self.sim.store_research_sources,
            delete_research_question=self.sim.delete_research_question,
        ), patch("src.research.complete_text",
                            return_value=_dump({"queries": ["q"]})), \
            patch("src.research.search_news", return_value=[]):
            result = run_research_or_reuse(research_question_fixture())

        self.assertEqual(result["status"], "empty")
        self.assertIsNone(result["research_question_id"])
        self.assertEqual(self.sim.sessions, [])

    def test_remembered_session_without_stored_sources_runs_fresh(self):
        self.sim.seed("qid-old", with_sources=False)
        result, _ = self._run([[raw_result(url="https://ex.com/back-to-work")]])
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["duplicate_reason"], None)
        self.assertNotEqual(result["research_question_id"], "qid-old")
        self.assertEqual(len(self.sim.sessions), 2)
        self.assertEqual(self.sim.session_ids(), ["qid-old", result["research_question_id"]])

    def test_remembered_session_hydration_failure_runs_fresh(self):
        self.sim.seed("qid-old", with_sources=True)
        with patch.multiple(
            "src.agent",
            check_topic_researched=self.sim.check_topic_researched,
            get_research_sources_for_question=_hydration_failure,
            get_known_source_urls=lambda: set(),
            store_research_question=self.sim.store_research_question,
            store_research_sources=self.sim.store_research_sources,
            delete_research_question=self.sim.delete_research_question,
        ), patch("src.research.complete_text", return_value=_dump({"queries": ["q"]})), \
            patch("src.research.search_news",
                  return_value=[raw_result(url="https://ex.com/fallback")]):
            result = run_research_or_reuse(research_question_fixture())

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["duplicate_reason"], None)
        self.assertNotEqual(result["research_question_id"], "qid-old")
        self.assertEqual(result["reused_question_id"], None)

    def test_remembered_session_with_valid_sources_is_reused(self):
        self.sim.seed("qid-old", with_sources=True)
        result, search = self._run([[]])
        self.assertEqual(result["status"], "reused")
        self.assertEqual(result["duplicate_reason"], "exact")
        self.assertEqual(result["reused_question_id"], "qid-old")
        self.assertEqual(result["research_question_id"], "qid-old")
        self.assertEqual([s.url for s in result["research_sources"]], ["https://ex.com/old"])
        search.assert_not_called()


if __name__ == "__main__":
    unittest.main()