"""
Unit tests for the Phase-3 critical analysis stage (src/analysis.py) and its
entry point (src.agent.run_critical_analysis).

Covers:
- supporting evidence is attached from the input (never invented)
- conflicting evidence / counterarguments are identified and grounded
- claims with missing evidence are recorded without fabricating support
- uncertainty / limitations / unresolved questions are captured
- classification into documented_fact / interpretation / opinion /
  unresolved_question
- source references are preserved verbatim for evidence-backed statements
- malformed / failed LLM responses degrade to an empty analysis
- run_critical_analysis() boundary (analysis only – no synthesis/drafting)

LLM (complete_text) is mocked, so no network or API keys are needed. Dummy
env vars are installed before importing src.* so config validation passes
without a populated .env file (see tests/test_evidence.py).
"""

from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

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

from src.agent import run_critical_analysis  # noqa: E402
from src.analysis import analyze_evidence  # noqa: E402
from src.models import (  # noqa: E402
    ClaimAnalysis,
    Counterargument,
    CriticalAnalysis,
    Evidence,
    ResearchQuestion,
)

URL_ONE = "https://ex.com/one"
URL_TWO = "https://ex.com/two"


def question_fixture() -> ResearchQuestion:
    return ResearchQuestion(
        topic="Agentic orchestration",
        question="Which orchestration framework scales best for production agents?",
        aspects=["reliability", "cost"],
    )


def evidence_fixture(
    claim_text: str = "LangGraph scaled to one million agents.",
    source_url: str = URL_ONE,
    quote: str = "LangGraph scaled to one million agents in production.",
    confidence: float = 0.8,
    verification_status: str = "unverified",
) -> Evidence:
    return Evidence(
        claim_text=claim_text,
        source_url=source_url,
        supporting_quote=quote,
        confidence=confidence,
        context="Benchmark report.",
        directly_supports=True,
        verification_status=verification_status,
    )


def analysis_json(**overrides) -> str:
    data = {
        "claims": [],
        "counterarguments": [],
        "limitations": [],
        "uncertainties": [],
        "unresolved_questions": [],
    }
    data.update(overrides)
    return json.dumps(data)


def claim_item(
    claim: str = "LangGraph is production-ready.",
    classification: str = "documented_fact",
    supporting=(),
    conflicting=(),
    confidence=0.8,
    reasoning: str = "Multiple sources agree.",
    limitations=(),
) -> dict:
    return {
        "claim": claim,
        "classification": classification,
        "supporting_source_urls": list(supporting),
        "conflicting_source_urls": list(conflicting),
        "confidence": confidence,
        "reasoning": reasoning,
        "limitations": list(limitations),
    }


class SupportingEvidenceTest(unittest.TestCase):
    def test_supporting_evidence_is_attached_from_input(self):
        item = evidence_fixture(claim_text="LangGraph scales.", source_url=URL_ONE)
        payload = analysis_json(claims=[claim_item(supporting=[URL_ONE])])

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(len(analysis.claims), 1)
        claim = analysis.claims[0]
        self.assertTrue(claim.has_evidence)
        self.assertEqual(len(claim.supporting_evidence), 1)
        # The ACTUAL input Evidence object is preserved (quote + attribution).
        self.assertIs(claim.supporting_evidence[0], item)
        self.assertEqual(claim.supporting_evidence[0].source_url, URL_ONE)
        self.assertEqual(claim.supporting_evidence[0].supporting_quote, item.supporting_quote)
        self.assertEqual(claim.source_urls, [URL_ONE])

    def test_claim_referencing_multiple_sources_keeps_all_references(self):
        one = evidence_fixture(claim_text="A", source_url=URL_ONE)
        two = evidence_fixture(claim_text="B", source_url=URL_TWO)
        payload = analysis_json(claims=[claim_item(supporting=[URL_ONE, URL_TWO])])

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [one, two])

        self.assertEqual(analysis.claims[0].source_urls, [URL_ONE, URL_TWO])

    def test_unknown_source_reference_is_ignored(self):
        item = evidence_fixture(source_url=URL_ONE)
        payload = analysis_json(claims=[claim_item(supporting=[URL_ONE, "https://evil.example/x"])])

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(analysis.claims[0].source_urls, [URL_ONE])

    def test_source_url_matching_tolerates_trailing_slash(self):
        item = evidence_fixture(source_url=URL_ONE)
        payload = analysis_json(claims=[claim_item(supporting=[URL_ONE + "/"])])

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertTrue(analysis.claims[0].has_evidence)


class ConflictingEvidenceTest(unittest.TestCase):
    def test_conflicting_evidence_is_attached(self):
        one = evidence_fixture(claim_text="Framework X scales.", source_url=URL_ONE)
        two = evidence_fixture(claim_text="Framework X does not scale.", source_url=URL_TWO)
        payload = analysis_json(
            claims=[claim_item(supporting=[URL_ONE], conflicting=[URL_TWO])]
        )

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [one, two])

        claim = analysis.claims[0]
        self.assertEqual([e.source_url for e in claim.supporting_evidence], [URL_ONE])
        self.assertEqual([e.source_url for e in claim.conflicting_evidence], [URL_TWO])

    def test_grounded_counterargument_is_kept(self):
        item = evidence_fixture(source_url=URL_TWO)
        payload = analysis_json(
            counterarguments=[
                {"argument": "Costs may outweigh gains.", "source_urls": [URL_TWO], "rebuttal": "Only at small scale."}
            ]
        )

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(len(analysis.counterarguments), 1)
        counter = analysis.counterarguments[0]
        self.assertEqual(counter.source_urls, [URL_TWO])
        self.assertEqual(counter.rebuttal, "Only at small scale.")

    def test_ungrounded_counterargument_is_dropped(self):
        item = evidence_fixture(source_url=URL_ONE)
        payload = analysis_json(
            counterarguments=[{"argument": "Some invented objection.", "source_urls": [], "rebuttal": ""}]
        )

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(analysis.counterarguments, [])

    def test_counterargument_with_unknown_source_is_dropped(self):
        item = evidence_fixture(source_url=URL_ONE)
        payload = analysis_json(
            counterarguments=[{"argument": "Objection.", "source_urls": ["https://unknown.example/x"]}]
        )

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(analysis.counterarguments, [])


class MissingEvidenceTest(unittest.TestCase):
    def test_ungrounded_claim_is_dropped(self):
        """documented_fact/interpretation/opinion claims need resolved evidence."""
        item = evidence_fixture(source_url=URL_ONE)
        payload = analysis_json(claims=[claim_item(claim="An open hypothesis.", supporting=[], conflicting=[])])

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(analysis.claims, [])

    def test_claim_with_only_unknown_urls_is_dropped(self):
        """An unmatched LLM-supplied URL is not grounds for keeping a claim."""
        item = evidence_fixture(source_url=URL_ONE)
        payload = analysis_json(
            claims=[claim_item(claim="Grounded nowhere.", supporting=["https://unknown.example/x"])]
        )

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(analysis.claims, [])

    def test_unresolved_question_without_evidence_is_retained(self):
        item = evidence_fixture(source_url=URL_ONE)
        payload = analysis_json(
            claims=[claim_item(claim="Unanswered by the material.", classification="unresolved_question")]
        )

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(len(analysis.claims), 1)
        claim = analysis.claims[0]
        self.assertEqual(claim.classification, ClaimAnalysis.CLASSIFICATION_UNRESOLVED)
        self.assertFalse(claim.has_evidence)

    def test_unresolved_question_with_evidence_is_retained(self):
        item = evidence_fixture(source_url=URL_ONE)
        payload = analysis_json(
            claims=[claim_item(claim="Touched on but open.", classification="unresolved_question", supporting=[URL_ONE])]
        )

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(len(analysis.claims), 1)
        self.assertTrue(analysis.claims[0].has_evidence)

    def test_no_evidence_input_returns_empty_analysis(self):
        with patch("src.analysis.complete_text") as mocked:
            analysis = analyze_evidence(question_fixture(), [])

        mocked.assert_not_called()
        self.assertEqual(analysis.status, CriticalAnalysis.STATUS_EMPTY)
        self.assertEqual(analysis.claims, [])
        self.assertEqual(analysis.counterarguments, [])

    def test_none_question_returns_empty_analysis(self):
        analysis = analyze_evidence(None, [evidence_fixture()])
        self.assertEqual(analysis.status, CriticalAnalysis.STATUS_EMPTY)

    def test_use_llm_false_returns_empty_without_calling_llm(self):
        with patch("src.analysis.complete_text") as mocked:
            analysis = analyze_evidence(question_fixture(), [evidence_fixture()], use_llm=False)

        mocked.assert_not_called()
        self.assertEqual(analysis.status, CriticalAnalysis.STATUS_EMPTY)

    def test_claim_without_text_is_dropped(self):
        item = evidence_fixture()
        payload = analysis_json(claims=[claim_item(claim="   "), claim_item(claim="Kept.", supporting=[URL_ONE])])

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual([c.claim for c in analysis.claims], ["Kept."])


class UncertaintyTest(unittest.TestCase):
    def test_limitations_uncertainties_and_open_questions_are_captured(self):
        item = evidence_fixture()
        payload = analysis_json(
            limitations=["Benchmarks come from vendor blogs."],
            uncertainties=["Production scale is not independently verified."],
            unresolved_questions=["How does it behave under partition?"],
        )

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(analysis.limitations, ["Benchmarks come from vendor blogs."])
        self.assertEqual(analysis.uncertainties, ["Production scale is not independently verified."])
        self.assertEqual(analysis.unresolved_questions, ["How does it behave under partition?"])
        self.assertEqual(analysis.status, CriticalAnalysis.STATUS_OK)

    def test_per_claim_limitations_are_kept(self):
        item = evidence_fixture()
        payload = analysis_json(claims=[claim_item(limitations=["Single source."], supporting=[URL_ONE])])

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(analysis.claims[0].limitations, ["Single source."])

    def test_blank_and_duplicate_strings_are_removed(self):
        item = evidence_fixture()
        payload = analysis_json(
            limitations=["Same point.", "  ", "same point."],
            uncertainties=[None, 42, "Real uncertainty."],
        )

        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])

        self.assertEqual(analysis.limitations, ["Same point."])
        self.assertEqual(analysis.uncertainties, ["Real uncertainty."])


class ClassificationTest(unittest.TestCase):
    def _classification(self, raw) -> str:
        item = evidence_fixture()
        payload = analysis_json(claims=[claim_item(classification=raw, supporting=[URL_ONE])])
        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])
        return analysis.claims[0].classification

    def test_documented_fact(self):
        self.assertEqual(self._classification("documented_fact"), ClaimAnalysis.CLASSIFICATION_FACT)
        self.assertEqual(self._classification("fact"), ClaimAnalysis.CLASSIFICATION_FACT)

    def test_interpretation(self):
        self.assertEqual(self._classification("interpretation"), ClaimAnalysis.CLASSIFICATION_INTERPRETATION)

    def test_opinion(self):
        self.assertEqual(self._classification("opinion"), ClaimAnalysis.CLASSIFICATION_OPINION)

    def test_unresolved_question(self):
        self.assertEqual(self._classification("open_question"), ClaimAnalysis.CLASSIFICATION_UNRESOLVED)
        self.assertEqual(self._classification("unresolved"), ClaimAnalysis.CLASSIFICATION_UNRESOLVED)

    def test_unknown_classification_defaults_to_interpretation(self):
        self.assertEqual(self._classification("wild_guess"), ClaimAnalysis.CLASSIFICATION_INTERPRETATION)
        self.assertEqual(self._classification(None), ClaimAnalysis.CLASSIFICATION_INTERPRETATION)

    def test_confidence_is_validated(self):
        item = evidence_fixture()
        payload = analysis_json(claims=[claim_item(confidence="0.9", supporting=[URL_ONE])])
        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])
        self.assertAlmostEqual(analysis.claims[0].confidence, 0.9)

        payload = analysis_json(claims=[claim_item(confidence=1.1, supporting=[URL_ONE])])
        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])
        self.assertEqual(analysis.claims[0].confidence, 0.5)


class MalformedResponseTest(unittest.TestCase):
    def test_invalid_json_returns_empty_analysis(self):
        with patch("src.analysis.complete_text", return_value="not json at all"):
            analysis = analyze_evidence(question_fixture(), [evidence_fixture()])
        self.assertEqual(analysis.status, CriticalAnalysis.STATUS_EMPTY)
        self.assertEqual(analysis.claims, [])

    def test_non_object_json_returns_empty_analysis(self):
        with patch("src.analysis.complete_text", return_value="[1, 2, 3]"):
            analysis = analyze_evidence(question_fixture(), [evidence_fixture()])
        self.assertEqual(analysis.status, CriticalAnalysis.STATUS_EMPTY)

    def test_llm_failure_returns_empty_analysis(self):
        with patch("src.analysis.complete_text", side_effect=RuntimeError("boom")):
            analysis = analyze_evidence(question_fixture(), [evidence_fixture()])
        self.assertEqual(analysis.status, CriticalAnalysis.STATUS_EMPTY)

    def test_wrong_typed_sections_are_ignored(self):
        item = evidence_fixture()
        payload = json.dumps({"claims": "nope", "limitations": {"a": 1}, "counterarguments": None})
        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])
        self.assertEqual(analysis.claims, [])
        self.assertEqual(analysis.limitations, [])
        self.assertEqual(analysis.counterarguments, [])
        self.assertEqual(analysis.status, CriticalAnalysis.STATUS_EMPTY)

    def test_non_dict_claim_entries_are_skipped(self):
        item = evidence_fixture()
        payload = json.dumps({"claims": ["nope", 5, claim_item(claim="Valid.", supporting=[URL_ONE])]})
        with patch("src.analysis.complete_text", return_value=payload):
            analysis = analyze_evidence(question_fixture(), [item])
        self.assertEqual([c.claim for c in analysis.claims], ["Valid."])


class PromptTest(unittest.TestCase):
    def test_prompt_contains_question_and_evidence(self):
        item = evidence_fixture(claim_text="Distinct claim text.", source_url=URL_ONE)
        with patch("src.analysis.complete_text", return_value=analysis_json()) as mocked:
            analyze_evidence(question_fixture(), [item])

        prompt = mocked.call_args.args[1]
        self.assertIn("Which orchestration framework scales best", prompt)
        self.assertIn("Distinct claim text.", prompt)
        self.assertIn(URL_ONE, prompt)


class RunCriticalAnalysisTest(unittest.TestCase):
    def test_returns_structured_analysis(self):
        rq = question_fixture()
        item = evidence_fixture()
        payload = analysis_json(claims=[claim_item(supporting=[URL_ONE])])

        with patch("src.analysis.complete_text", return_value=payload):
            result = run_critical_analysis(rq, [item])

        self.assertEqual(result["research_question"], rq)
        self.assertIsInstance(result["analysis"], CriticalAnalysis)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["evidence"], [item])

    def test_empty_evidence_reports_empty_status(self):
        result = run_critical_analysis(question_fixture(), [])
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["analysis"].claims, [])

    def test_does_not_draft_or_publish(self):
        rq = question_fixture()
        payload = analysis_json(claims=[claim_item(supporting=[URL_ONE])])
        with patch("src.analysis.complete_text", return_value=payload), \
                patch("src.agent.draft_post") as draft, \
                patch("src.agent.store_draft") as store:
            run_critical_analysis(rq, [evidence_fixture()])

        draft.assert_not_called()
        store.assert_not_called()


class CriticalAnalysisModelTest(unittest.TestCase):
    def test_round_trip_preserves_structure(self):
        original = CriticalAnalysis(
            topic="Agentic orchestration",
            research_question="Q?",
            claims=[
                ClaimAnalysis(
                    claim="C1",
                    classification=ClaimAnalysis.CLASSIFICATION_FACT,
                    supporting_evidence=[evidence_fixture()],
                    conflicting_evidence=[evidence_fixture(source_url=URL_TWO)],
                    confidence=0.7,
                    reasoning="because",
                    limitations=["small sample"],
                )
            ],
            counterarguments=[Counterargument(argument="A", evidence=[evidence_fixture(source_url=URL_TWO)], rebuttal="R")],
            limitations=["L"],
            uncertainties=["U"],
            unresolved_questions=["OQ"],
            status=CriticalAnalysis.STATUS_OK,
        )

        restored = CriticalAnalysis.from_dict(original.to_dict())

        self.assertEqual(restored.topic, original.topic)
        self.assertEqual(restored.status, original.status)
        self.assertEqual(len(restored.claims), 1)
        self.assertEqual(restored.claims[0].claim, "C1")
        self.assertEqual(restored.claims[0].classification, ClaimAnalysis.CLASSIFICATION_FACT)
        self.assertEqual(restored.claims[0].source_urls, [URL_ONE, URL_TWO])
        self.assertEqual(restored.claims[0].supporting_evidence[0].supporting_quote, evidence_fixture().supporting_quote)
        self.assertEqual(restored.counterarguments[0].argument, "A")
        self.assertEqual(restored.counterarguments[0].source_urls, [URL_TWO])
        self.assertEqual(restored.limitations, ["L"])
        self.assertEqual(restored.uncertainties, ["U"])
        self.assertEqual(restored.unresolved_questions, ["OQ"])

    def test_empty_analysis_round_trip(self):
        original = CriticalAnalysis(topic="T", research_question="Q?")
        restored = CriticalAnalysis.from_dict(original.to_dict())
        self.assertEqual(restored.claims, [])
        self.assertEqual(restored.counterarguments, [])
        self.assertEqual(restored.status, CriticalAnalysis.STATUS_OK)


if __name__ == "__main__":
    unittest.main()
