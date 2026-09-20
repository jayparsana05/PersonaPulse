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
from src.agent import run_research  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()