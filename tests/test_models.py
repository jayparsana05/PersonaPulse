"""
Tests for the PersonaPulse research domain models (src/models.py).

Runs with the stdlib test runner (no third-party framework required):

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import unittest
from datetime import datetime

from src.models import (
    Claim,
    Evidence,
    ResearchQuestion,
    ResearchReport,
    ResearchSource,
    TopicCandidate,
)


class TopicCandidateTest(unittest.TestCase):
    def test_defaults(self):
        candidate = TopicCandidate(title="Agentic AI orchestration")
        self.assertEqual(candidate.title, "Agentic AI orchestration")
        self.assertEqual(candidate.description, "")
        self.assertEqual(candidate.url, "")
        self.assertEqual(candidate.keywords, [])
        self.assertIsNone(candidate.source)
        self.assertEqual(candidate.published, "")
        self.assertEqual(candidate.search_score, 0.0)
        self.assertIsNone(candidate.discovered_at)

    def test_round_trip(self):
        candidate = TopicCandidate(
            title="Agentic AI orchestration",
            description="Frameworks for coordinating multiple agents.",
            url="https://example.com/agentic-orchestration",
            keywords=["agentic", "orchestration"],
            source="example.com",
            published="2026-09-19",
            search_score=0.87,
            discovered_at=datetime(2026, 9, 20, 10, 30, 0),
        )
        restored = TopicCandidate.from_dict(candidate.to_dict())
        self.assertEqual(restored, candidate)

    def test_to_dict_uses_search_score_key(self):
        candidate = TopicCandidate(title="T", search_score=0.42)
        data = candidate.to_dict()
        self.assertIn("search_score", data)
        self.assertNotIn("relevance_score", data)
        self.assertEqual(data["search_score"], 0.42)

    def test_from_dict_accepts_legacy_relevance_score_key(self):
        candidate = TopicCandidate.from_dict(
            {"title": "T", "relevance_score": 0.63}
        )
        self.assertEqual(candidate.search_score, 0.63)

    def test_round_trip_with_missing_optional_values(self):
        candidate = TopicCandidate(title="Minimal")
        restored = TopicCandidate.from_dict(candidate.to_dict())
        self.assertEqual(restored, candidate)

    def test_malformed_datetime_is_ignored(self):
        candidate = TopicCandidate.from_dict({"title": "T", "discovered_at": "not-a-date"})
        self.assertIsNone(candidate.discovered_at)


class ResearchQuestionTest(unittest.TestCase):
    def test_defaults_and_constants(self):
        question = ResearchQuestion(topic="Agentic orchestration", question="Which framework scales?")
        self.assertEqual(question.status, ResearchQuestion.STATUS_PROPOSED)
        self.assertEqual(question.priority, ResearchQuestion.PRIORITY_NORMAL)
        self.assertEqual(question.aspects, [])

    def test_status_constants(self):
        self.assertEqual(ResearchQuestion.STATUS_PROPOSED, "proposed")
        self.assertEqual(ResearchQuestion.STATUS_RESEARCHING, "researching")
        self.assertEqual(ResearchQuestion.STATUS_ANSWERED, "answered")
        self.assertEqual(ResearchQuestion.STATUS_DROPPED, "dropped")

    def test_round_trip(self):
        question = ResearchQuestion(
            topic="Agentic orchestration",
            question="Which framework scales?",
            aspects=["langgraph", "crewai", "autogen"],
            status=ResearchQuestion.STATUS_RESEARCHING,
            priority=ResearchQuestion.PRIORITY_HIGH,
            created_at=datetime(2026, 9, 20, 12, 0, 0),
        )
        restored = ResearchQuestion.from_dict(question.to_dict())
        self.assertEqual(restored, question)


class ResearchSourceTest(unittest.TestCase):
    # The exact dict shape produced by ingestion.fetch_trending_tech_news()
    LEGACY_ARTICLE = {
        "url": "https://example.com/agentic-ai",
        "title": "The Rise of Agentic AI",
        "body": "Full article body text.",
        "published": "2026-09-19",
        "source": "example.com",
    }

    def test_from_article_reuses_ingestion_shape(self):
        source = ResearchSource.from_article(self.LEGACY_ARTICLE)
        self.assertEqual(source.url, self.LEGACY_ARTICLE["url"])
        self.assertEqual(source.title, self.LEGACY_ARTICLE["title"])
        self.assertEqual(source.body, self.LEGACY_ARTICLE["body"])
        self.assertEqual(source.published, self.LEGACY_ARTICLE["published"])
        self.assertEqual(source.source, self.LEGACY_ARTICLE["source"])
        self.assertEqual(source.source_type, ResearchSource.SOURCE_TYPE_SECONDARY)

    def test_from_article_accepts_overrides(self):
        source = ResearchSource.from_article(
            self.LEGACY_ARTICLE,
            source_type=ResearchSource.SOURCE_TYPE_PRIMARY,
            score=0.92,
        )
        self.assertEqual(source.source_type, ResearchSource.SOURCE_TYPE_PRIMARY)
        self.assertEqual(source.score, 0.92)

    def test_to_dict_keeps_legacy_keys_first(self):
        source = ResearchSource.from_article(self.LEGACY_ARTICLE)
        data = source.to_dict()
        for key in ("url", "title", "body", "published", "source"):
            self.assertEqual(data[key], self.LEGACY_ARTICLE[key])
            self.assertIn(key, data)
        self.assertIn("score", data)

    def test_round_trip(self):
        source = ResearchSource.from_article(
            self.LEGACY_ARTICLE,
            accessed_at=datetime(2026, 9, 20, 14, 0, 0),
        )
        restored = ResearchSource.from_dict(source.to_dict())
        self.assertEqual(restored, source)


class EvidenceTest(unittest.TestCase):
    def test_defaults_and_constants(self):
        evidence = Evidence(claim_text="X scales to 1M agents")
        self.assertEqual(evidence.verification_status, Evidence.VERIFICATION_CLAIMED)
        self.assertEqual(evidence.confidence, 0.0)
        self.assertIsNone(evidence.source_url)
        self.assertEqual(evidence.supporting_quote, "")

    def test_verification_constants(self):
        self.assertEqual(Evidence.VERIFICATION_CLAIMED, "claimed")
        self.assertEqual(Evidence.VERIFICATION_CORROBORATED, "corroborated")
        self.assertEqual(Evidence.VERIFICATION_CONTRADICTED, "contradicted")

    def test_claim_is_an_alias_for_evidence(self):
        self.assertIs(Claim, Evidence)
        evidence = Claim(claim_text="Aliased")
        self.assertIsInstance(evidence, Evidence)

    def test_round_trip(self):
        evidence = Evidence(
            claim_text="X scales to 1M agents",
            source_url="https://example.com/report",
            supporting_quote="Benchmarks show 1M coordinated agents.",
            confidence=0.9,
            verification_status=Evidence.VERIFICATION_CORROBORATED,
        )
        restored = Evidence.from_dict(evidence.to_dict())
        self.assertEqual(restored, evidence)

    def test_context_and_direct_support_fields(self):
        evidence = Evidence(
            claim_text="X scales to 1M agents",
            source_url="https://example.com/report",
            supporting_quote="Benchmarks show 1M coordinated agents.",
            confidence=0.9,
            context="Benchmarking study across three frameworks.",
            directly_supports=False,
        )
        restored = Evidence.from_dict(evidence.to_dict())
        self.assertEqual(restored, evidence)
        self.assertEqual(restored.context, "Benchmarking study across three frameworks.")
        self.assertFalse(restored.directly_supports)

    def test_context_and_direct_support_defaults(self):
        evidence = Evidence(claim_text="Plain claim")
        self.assertEqual(evidence.context, "")
        self.assertTrue(evidence.directly_supports)
        restored = Evidence.from_dict({"claim_text": "Plain claim"})
        self.assertEqual(restored.context, "")
        self.assertTrue(restored.directly_supports)


class ResearchReportTest(unittest.TestCase):
    def _sample_report(self) -> ResearchReport:
        return ResearchReport(
            topic="Agentic AI orchestration",
            summary="Multi-agent systems are converging on orchestration standards.",
            conclusions=["Adopt a standards-first approach."],
            questions=[
                ResearchQuestion(
                    topic="Agentic AI orchestration",
                    question="Which framework scales?",
                    status=ResearchQuestion.STATUS_ANSWERED,
                )
            ],
            sources=[ResearchSource.from_article(ResearchSourceTest.LEGACY_ARTICLE)],
            evidence=[
                Evidence(
                    claim_text="Orchestration frameworks are converging.",
                    source_url="https://example.com/agentic-ai",
                )
            ],
            confidence_score=0.75,
            created_at=datetime(2026, 9, 20, 15, 0, 0),
        )

    def test_defaults(self):
        report = ResearchReport(topic="T")
        self.assertEqual(report.summary, "")
        self.assertEqual(report.conclusions, [])
        self.assertEqual(report.questions, [])
        self.assertEqual(report.sources, [])
        self.assertEqual(report.evidence, [])
        self.assertEqual(report.confidence_score, 0.0)

    def test_round_trip_with_nested_models(self):
        report = self._sample_report()
        restored = ResearchReport.from_dict(report.to_dict())
        self.assertEqual(restored, report)

    def test_from_dict_with_empty_sections(self):
        report = ResearchReport.from_dict({"topic": "T"})
        self.assertEqual(report.questions, [])
        self.assertEqual(report.sources, [])
        self.assertEqual(report.evidence, [])


if __name__ == "__main__":
    unittest.main()