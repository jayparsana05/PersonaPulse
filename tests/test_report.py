"""
Unit tests for the Phase-3 report synthesis stage (src/report.py) and its
entry point (src.agent.run_report).

Covers:
- complete reports: every section (topic, question, findings, supporting /
  conflicting evidence, counterarguments, limitations, sources, references,
  synthesis, conclusions) is present
- traceability: findings reference the ACTUAL input Evidence (source references
  preserved verbatim)
- incomplete reports: missing analysis/evidence still build deterministically
  and never fabricate content
- grounded synthesis: the LLM receives only the collected material and the
  fallback restates only validated findings
- malformed / failed LLM synthesis degrades to the deterministic fallback
- the report is LinkedIn-agnostic (no drafting / publishing)

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

from src.agent import run_report  # noqa: E402
from src.models import (  # noqa: E402
    ClaimAnalysis,
    Counterargument,
    CriticalAnalysis,
    Evidence,
    ResearchQuestion,
    ResearchReport,
    ResearchSource,
)
from src.report import synthesize_report  # noqa: E402

URL_ONE = "https://ex.com/one"
URL_TWO = "https://ex.com/two"


def question_fixture() -> ResearchQuestion:
    return ResearchQuestion(
        topic="Agentic orchestration",
        question="Which orchestration framework scales best for production agents?",
        aspects=["reliability", "cost"],
    )


def source_fixture(url: str = URL_ONE, title: str = "Benchmarks") -> ResearchSource:
    return ResearchSource(
        url=url,
        title=title,
        body="Production evidence from the vendor report.",
        published="2026-09-19",
        source="ex.com",
        score=0.9,
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


def critical_analysis_fixture(claim) -> CriticalAnalysis:
    grounding = list(claim.supporting_evidence) or list(claim.conflicting_evidence)
    return CriticalAnalysis(
        topic="Agentic orchestration",
        research_question="Which orchestration framework scales best for production agents?",
        claims=[claim],
        counterarguments=(
            [
                Counterargument(
                    argument="Costs may outweigh gains.",
                    evidence=grounding,
                )
            ]
            if grounding
            else []
        ),
        limitations=["Benchmarks come from vendor blogs."],
        uncertainties=["Production scale is not independently verified."],
        unresolved_questions=["How does it behave under partition?"],
        status=CriticalAnalysis.STATUS_OK,
    )


def synthesis_json() -> str:
    """A structurally-valid default LLM payload (grounded only by its shape;
    whether the text survives depends on the finding claims in the test)."""
    return json.dumps({
        "synthesis": [{"text": "Synthesis narrative.", "finding_indices": [0]}],
        "conclusions": [],
    })


class CompleteReportTest(unittest.TestCase):
    def test_contains_all_sections(self):
        item = evidence_fixture()
        finding = ClaimAnalysis(
            claim="LangGraph handles large scale.",
            classification=ClaimAnalysis.CLASSIFICATION_FACT,
            supporting_evidence=[item],
        )
        analysis = critical_analysis_fixture(finding)
        segment_a = "LangGraph handles large scale."
        segment_b = "At large scale, LangGraph handles it."
        payload = json.dumps({
            "synthesis": [
                {"text": segment_a, "finding_indices": [0]},
                {"text": segment_b, "finding_indices": [0]},
            ],
            "conclusions": [{"text": segment_a, "finding_indices": [0]}],
        })

        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(
                question_fixture(), [source_fixture()], [item], analysis=analysis
            )

        self.assertEqual(report.topic, "Agentic orchestration")
        self.assertEqual(report.research_question, "Which orchestration framework scales best for production agents?")
        self.assertEqual(len(report.findings), 1)
        self.assertEqual(len(report.sources), 1)
        self.assertEqual(report.sources[0].url, URL_ONE)
        self.assertEqual(report.sources[0].title, "Benchmarks")
        self.assertEqual(report.evidence, [item])
        self.assertEqual(len(report.supporting_evidence), 1)
        self.assertEqual(len(report.conflicting_evidence), 0)
        self.assertEqual(len(report.counterarguments), 1)
        self.assertEqual(report.limitations, ["Benchmarks come from vendor blogs."])
        self.assertEqual(report.uncertainties, ["Production scale is not independently verified."])
        self.assertEqual(report.unresolved_questions, ["How does it behave under partition?"])
        synthesis = " ".join([segment_a, segment_b])
        self.assertEqual(report.synthesis, synthesis)
        self.assertEqual(report.synthesis_segments, [segment_a, segment_b])
        self.assertEqual(report.conclusions, [segment_a])
        self.assertEqual(report.summary, synthesis)
        self.assertIsNotNone(report.analysis)
        self.assertEqual(report.analysis.claims[0].claim, finding.claim)

    def test_finding_traceability_preserves_input_evidence(self):
        item = evidence_fixture()
        finding = ClaimAnalysis(claim="C", classification=ClaimAnalysis.CLASSIFICATION_FACT, supporting_evidence=[item])
        analysis = critical_analysis_fixture(finding)

        with patch("src.report.complete_text", return_value=synthesis_json()):
            report = synthesize_report(question_fixture(), [source_fixture()], [item], analysis=analysis)

        claim = report.findings[0]
        # The ACTUAL input Evidence object (quote + URL verbatim).
        self.assertIs(claim.supporting_evidence[0], item)
        self.assertEqual(report.findings[0].source_urls, [URL_ONE])

    def test_conflicting_evidence_section(self):
        one = evidence_fixture()
        two = evidence_fixture(claim_text="X does not scale.", source_url=URL_TWO)
        finding = ClaimAnalysis(
            claim="X scales.",
            classification=ClaimAnalysis.CLASSIFICATION_INTERPRETATION,
            supporting_evidence=[one],
            conflicting_evidence=[two],
        )
        analysis = critical_analysis_fixture(finding)

        with patch("src.report.complete_text", return_value=synthesis_json()):
            report = synthesize_report(question_fixture(), [], [one, two], analysis=analysis)

        self.assertEqual([e.source_url for e in report.supporting_evidence], [URL_ONE])
        self.assertEqual([e.source_url for e in report.conflicting_evidence], [URL_TWO])

    def test_duplicate_evidence_is_deduplicated(self):
        item = evidence_fixture()
        finding = ClaimAnalysis(
            claim="C", classification=ClaimAnalysis.CLASSIFICATION_FACT,
            supporting_evidence=[item] * 3,
        )
        analysis = critical_analysis_fixture(finding)

        with patch("src.report.complete_text", return_value=synthesis_json()):
            report = synthesize_report(question_fixture(), [], [item], analysis=analysis)

        self.assertEqual(len(report.supporting_evidence), 1)

    def test_confidence_score_is_mean_of_evidence(self):
        items = [evidence_fixture(claim_text="A", confidence=0.6), evidence_fixture(claim_text="B", confidence=0.8)]
        with patch("src.report.complete_text", return_value=synthesis_json()):
            report = synthesize_report(question_fixture(), [], items)

        self.assertAlmostEqual(report.confidence_score, 0.7)

    def test_llm_receives_only_collected_material(self):
        item = evidence_fixture(claim_text="A very specific claim.", source_url=URL_ONE)
        finding = ClaimAnalysis(claim="A very specific claim.", classification=ClaimAnalysis.CLASSIFICATION_FACT, supporting_evidence=[item])
        analysis = critical_analysis_fixture(finding)

        with patch("src.report.complete_text", return_value=synthesis_json()) as mocked:
            synthesize_report(question_fixture(), [source_fixture()], [item], analysis=analysis)

        prompt = mocked.call_args.args[1]
        self.assertIn("A very specific claim.", prompt)
        self.assertIn(URL_ONE, prompt)
        self.assertIn("Benchmarks", prompt)
        self.assertIn("Costs may outweigh gains.", prompt)
        self.assertNotIn("S&P 500", prompt)  # nothing beyond the material


class IncompleteReportTest(unittest.TestCase):
    def test_no_analysis_derives_findings_from_evidence(self):
        item = evidence_fixture(claim_text="Orchestrators converge.")
        report = synthesize_report(question_fixture(), [source_fixture()], [item])

        self.assertEqual(len(report.findings), 1)
        self.assertEqual(report.findings[0].claim, "Orchestrators converge.")
        self.assertEqual(report.findings[0].classification, ClaimAnalysis.CLASSIFICATION_INTERPRETATION)
        self.assertTrue(report.findings[0].has_evidence)
        self.assertEqual(report.findings[0].source_urls, [URL_ONE])

    def test_corroborated_evidence_falls_back_to_documented_fact(self):
        a = evidence_fixture(claim_text="Same claim.", source_url=URL_ONE)
        b = evidence_fixture(claim_text="Same claim.", source_url=URL_TWO)
        report = synthesize_report(question_fixture(), [], [a, b])

        self.assertEqual(report.findings[0].classification, ClaimAnalysis.CLASSIFICATION_FACT)
        self.assertEqual(report.findings[0].source_urls, [URL_ONE, URL_TWO])

    def test_empty_inputs_produce_minimal_non_fabricating_report(self):
        report = synthesize_report(None)

        self.assertEqual(report.topic, "")
        self.assertEqual(report.research_question, "")
        self.assertEqual(report.findings, [])
        self.assertEqual(report.sources, [])
        self.assertEqual(report.evidence, [])
        self.assertEqual(report.confidence_score, 0.0)
        self.assertEqual(report.conclusions, [])
        # Fallback restates that nothing was analysed; no invented facts.
        self.assertIn("No analysis was produced", report.synthesis)

    def test_missing_parts_do_not_fabricate_source_references(self):
        report = synthesize_report(question_fixture())  # no sources, no evidence

        self.assertEqual(report.findings, [])
        for finding in report.findings:
            self.assertEqual(finding.source_urls, [])

    def test_malformed_synthesis_falls_back(self):
        item = evidence_fixture()
        with patch("src.report.complete_text", return_value="not json"):
            report = synthesize_report(question_fixture(), [source_fixture()], [item])

        self.assertEqual(len(report.findings), 1)
        self.assertIn("Key findings:", report.synthesis)
        self.assertEqual(report.conclusions[0], "LangGraph scaled to one million agents.")

    def test_llm_failure_falls_back(self):
        item = evidence_fixture()
        with patch("src.report.complete_text", side_effect=RuntimeError("boom")):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(len(report.findings), 1)
        self.assertIn("Key findings:", report.synthesis)

    def test_use_llm_false_does_not_call_llm(self):
        item = evidence_fixture()
        with patch("src.report.complete_text") as mocked:
            report = synthesize_report(question_fixture(), [], [item], use_llm=False)

        mocked.assert_not_called()
        self.assertEqual(len(report.findings), 1)
        self.assertIn("Key findings:", report.synthesis)

    def test_partial_llm_output_keeps_conclusions_fallback(self):
        item = evidence_fixture()
        # Grounded segment (quotes the finding claim) with empty conclusions.
        payload = json.dumps({
            "synthesis": [
                {"text": "LangGraph scaled to one million agents.", "finding_indices": [0]}
            ],
            "conclusions": [],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(report.synthesis, "LangGraph scaled to one million agents.")
        self.assertTrue(report.conclusions)  # conclusions fell back to findings

    def test_blank_synthesis_uses_fallback(self):
        item = evidence_fixture()
        payload = json.dumps({
            "synthesis": [{"text": "   ", "finding_indices": [0]}],
            "conclusions": [],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertIn("Key findings:", report.synthesis)


class LinkedinAgnosticTest(unittest.TestCase):
    def test_report_has_no_social_drafting_fields(self):
        item = evidence_fixture()
        payload = synthesis_json()
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [source_fixture()], [item])

        for key in ("draft", "linkedin", "hashtags", "post"):
            self.assertNotIn(key, report.to_dict())
        self.assertFalse(hasattr(report, "linkedin_post"))


class SynthesisGroundingTest(unittest.TestCase):
    """Fix 1 – the LLM narrative must be grounded in the validated research.
    Structural contract: every segment/conclusion references findings and
    stays confined to their content."""

    def test_hallucinated_synthesis_and_conclusions_are_rejected(self):
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        payload = json.dumps({
            "synthesis": [
                {"text": "The market will reach one trillion dollars.", "finding_indices": [0]}
            ],
            "conclusions": [
                {"text": "Adopt a framework not mentioned anywhere in the research.", "finding_indices": [0]}
            ],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        # Unsupported narrative → deterministic grounded fallback.
        self.assertNotIn("one trillion dollars", report.synthesis)
        self.assertIn("Key findings:", report.synthesis)
        # Unsupported conclusion rejected; only the grounded finding remains.
        self.assertEqual(report.conclusions, ["Orchestrators converge."])
        # No new evidence is created by synthesis.
        self.assertEqual(report.evidence, [item])
        self.assertEqual(report.findings[0].source_urls, [URL_ONE])
        self.assertEqual(
            [e.source_url for f in report.findings for e in f.supporting_evidence], [URL_ONE]
        )

    def test_mixed_conclusions_keep_only_grounded_items(self):
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        payload = json.dumps({
            "synthesis": [
                {"text": "Orchestrators converge.", "finding_indices": [0]}
            ],
            "conclusions": [
                {"text": "Orchestrators converge.", "finding_indices": [0]},
                {"text": "A conclusion about something never researched.", "finding_indices": [0]},
            ],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(report.conclusions, ["Orchestrators converge."])
        self.assertEqual(report.synthesis, "Orchestrators converge.")

    def test_grounded_llm_synthesis_is_retained_without_fallback(self):
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        finding = ClaimAnalysis(
            claim="Orchestrators converge.",
            classification=ClaimAnalysis.CLASSIFICATION_FACT,
            supporting_evidence=[item],
        )
        analysis = critical_analysis_fixture(finding)
        payload = json.dumps({
            "synthesis": [
                {"text": "Orchestrators converge.", "finding_indices": [0]},
                {"text": "The orchestrators do converge.", "finding_indices": [0]},
            ],
            "conclusions": [{"text": "Orchestrators converge.", "finding_indices": [0]}],
        })

        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [source_fixture()], [item], analysis=analysis)

        self.assertEqual(
            report.synthesis, "Orchestrators converge. The orchestrators do converge."
        )
        self.assertEqual(report.conclusions, ["Orchestrators converge."])
        # Existing evidence is preserved, unchanged, untouched by synthesis.
        self.assertEqual(report.evidence, [item])
        self.assertIs(report.findings[0].supporting_evidence[0], item)

    def test_grounded_prefix_plus_hallucinated_assertion_is_rejected(self):
        """A grounded segment does not excuse an unsupported second segment."""
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        payload = json.dumps({
            "synthesis": [
                {"text": "Orchestrators converge.", "finding_indices": [0]},
                {"text": "Framework X dominates the enterprise market.", "finding_indices": [0]},
            ],
            "conclusions": [],
        })

        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        # Only the grounded segment survives; the hallucinated content never appears.
        self.assertEqual(report.synthesis, "Orchestrators converge.")
        self.assertNotIn("Framework X", report.synthesis)
        # No new evidence was created during synthesis.
        self.assertEqual(report.evidence, [item])
        self.assertEqual(report.findings[0].source_urls, [URL_ONE])

    def test_fully_unsupported_synthesis_is_rejected(self):
        """A narrative unrelated to all findings/evidence must not be accepted."""
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        payload = json.dumps({
            "synthesis": [
                {"text": "Blockchains are the future of retail payments.", "finding_indices": [0]}
            ],
            "conclusions": [],
        })

        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertIn("Key findings:", report.synthesis)
        self.assertNotIn("Blockchains", report.synthesis)


class ConclusionGroundingTest(unittest.TestCase):
    """Structural conclusion grounding: reject negation / added specifics."""

    def test_negated_conclusion_is_rejected(self):
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        payload = json.dumps({
            "synthesis": [{"text": "Orchestrators converge.", "finding_indices": [0]}],
            "conclusions": [{"text": "Orchestrators do not converge.", "finding_indices": [0]}],
        })

        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(report.conclusions, ["Orchestrators converge."])

    def test_quantitative_conclusion_is_rejected(self):
        item = evidence_fixture(
            claim_text="Teams increasingly adopt event-driven architectures.", source_url=URL_ONE
        )
        payload = json.dumps({
            "synthesis": [
                {"text": "Teams increasingly adopt event-driven architectures.", "finding_indices": [0]}
            ],
            "conclusions": [
                {"text": "Event-driven architecture adoption increased by 73% in 2026.", "finding_indices": [0]}
            ],
        })

        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(report.conclusions, ["Teams increasingly adopt event-driven architectures."])

    def test_valid_paraphrase_conclusion_is_retained(self):
        item = evidence_fixture(
            claim_text="Teams increasingly adopt event-driven architectures.", source_url=URL_ONE
        )
        payload = json.dumps({
            "synthesis": [
                {"text": "Teams increasingly adopt event-driven architectures.", "finding_indices": [0]}
            ],
            "conclusions": [
                {"text": "Event-driven architectures are increasingly adopted by teams.", "finding_indices": [0]}
            ],
        })

        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(
            report.conclusions,
            ["Event-driven architectures are increasingly adopted by teams."],
        )


class StructuralGroundingTest(unittest.TestCase):
    """Phase-4 structural synthesis grounding contract (finding_indices).

    A  grounded paraphrase retained
    B  grounded + unsupported addition rejected
    C  no references rejected
    D  invalid finding_indices rejected
    E  negation rejected
    F  unsupported quantitative rejected
    G  valid paraphrase retained (see ConclusionGroundingTest)
    H  multiple findings combined retained
    I  complete LLM failure → deterministic fallback, no fabrication
    """

    def _two_finding_analysis(self):
        item_a = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        item_b = evidence_fixture(
            claim_text="LangGraph scales to one million agents.", source_url=URL_TWO
        )
        f1 = ClaimAnalysis(
            claim="Orchestrators converge.",
            classification=ClaimAnalysis.CLASSIFICATION_FACT,
            supporting_evidence=[item_a],
        )
        f2 = ClaimAnalysis(
            claim="LangGraph scales to one million agents.",
            classification=ClaimAnalysis.CLASSIFICATION_FACT,
            supporting_evidence=[item_b],
        )
        return (
            [item_a, item_b],
            CriticalAnalysis(
                topic="Agentic orchestration",
                research_question="Which orchestration framework scales best for production agents?",
                claims=[f1, f2],
                counterarguments=[],
                limitations=["Benchmarks come from vendor blogs."],
                uncertainties=[],
                unresolved_questions=[],
                status=CriticalAnalysis.STATUS_OK,
            ),
        )

    def test_A_grounded_paraphrase_retained(self):
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        payload = json.dumps({
            "synthesis": [{"text": "The orchestrators converge.", "finding_indices": [0]}],
            "conclusions": [],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(report.synthesis, "The orchestrators converge.")

    def test_B_grounded_plus_unsupported_addition_rejected(self):
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        payload = json.dumps({
            "synthesis": [
                {"text": "Orchestrators converge.", "finding_indices": [0]},
                {"text": "Orchestrators converge, and Framework X dominates "
                         "the enterprise market.", "finding_indices": [0]},
            ],
            "conclusions": [],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(report.synthesis, "Orchestrators converge.")
        self.assertNotIn("Framework X", report.synthesis)

    def test_C_no_references_rejected(self):
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        payload = json.dumps({
            "synthesis": [{"text": "Orchestrators converge."}],  # no finding_indices
            "conclusions": [],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertIn("Key findings:", report.synthesis)

    def test_D_invalid_finding_indices_rejected(self):
        items, analysis = self._two_finding_analysis()
        payload = json.dumps({
            "synthesis": [
                {"text": "The orchestrators converge.", "finding_indices": [99]}
            ],
            "conclusions": [
                {"text": "LangGraph scales to one million agents.", "finding_indices": [2]}
            ],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], items, analysis=analysis)

        self.assertIn("Key findings:", report.synthesis)
        # Unsupported references rejected → conclusions match the findings.
        self.assertEqual(
            report.conclusions,
            ["Orchestrators converge.", "LangGraph scales to one million agents."],
        )

    def test_E_negation_rejected(self):
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        payload = json.dumps({
            "synthesis": [{"text": "Orchestrators converge.", "finding_indices": [0]}],
            "conclusions": [{"text": "Orchestrators do not converge.", "finding_indices": [0]}],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(report.conclusions, ["Orchestrators converge."])

    def test_F_unsupported_quantitative_rejected(self):
        item = evidence_fixture(
            claim_text="Teams increasingly adopt event-driven architectures.", source_url=URL_ONE
        )
        payload = json.dumps({
            "synthesis": [
                {"text": "Teams increasingly adopt event-driven architectures.", "finding_indices": [0]}
            ],
            "conclusions": [
                {"text": "Event-driven architecture adoption increased by 73% in 2026.", "finding_indices": [0]}
            ],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertEqual(report.conclusions, ["Teams increasingly adopt event-driven architectures."])

    def test_H_multiple_findings_combined_retained(self):
        items, analysis = self._two_finding_analysis()
        combined = "Orchestrators converge, and LangGraph scales to one million agents."
        payload = json.dumps({
            "synthesis": [{"text": combined, "finding_indices": [0, 1]}],
            "conclusions": [],
        })
        with patch("src.report.complete_text", return_value=payload):
            report = synthesize_report(question_fixture(), [], items, analysis=analysis)

        self.assertEqual(report.synthesis, combined)

    def test_I_llm_failure_uses_deterministic_fallback_not_fabrication(self):
        item = evidence_fixture(claim_text="Orchestrators converge.", source_url=URL_ONE)
        with patch("src.report.complete_text", side_effect=RuntimeError("boom")):
            report = synthesize_report(question_fixture(), [], [item])

        self.assertIn("Key findings:", report.synthesis)
        self.assertTrue(report.synthesis)
        self.assertEqual(report.conclusions, ["Orchestrators converge."])
        self.assertEqual(report.evidence, [item])


class AnalysisGroundingTest(unittest.TestCase):
    """Fix 2 – CriticalAnalysis evidence must exist in the supplied Evidence[]."""

    def test_analysis_referencing_outside_evidence_is_excluded(self):
        outside = evidence_fixture(claim_text="Claim from outside.", source_url=URL_ONE)
        finding = ClaimAnalysis(
            claim="Claim from outside.",
            classification=ClaimAnalysis.CLASSIFICATION_FACT,
            supporting_evidence=[outside],
        )
        analysis = critical_analysis_fixture(finding)
        supplied = evidence_fixture(claim_text="Claim from inside.", source_url=URL_TWO)

        with patch("src.report.complete_text", return_value=synthesis_json()):
            result = run_report(
                question_fixture(),
                [source_fixture(url=URL_TWO)],
                [supplied],
                analysis=analysis,
            )

        report = result["report"]
        # URL_A must not appear anywhere in the resulting report.
        self.assertNotIn(URL_ONE, json.dumps(report.to_dict()))
        self.assertEqual(report.findings, [])
        # Only evidence from the supplied set remains.
        self.assertEqual(report.evidence, [supplied])
        self.assertEqual(report.analysis.claims, [])

    def test_mixed_valid_and_invalid_analysis_evidence(self):
        valid = evidence_fixture(claim_text="X scales.", source_url=URL_ONE)
        stale = evidence_fixture(claim_text="Stale detail.", source_url=URL_TWO)
        finding = ClaimAnalysis(
            claim="X scales.",
            classification=ClaimAnalysis.CLASSIFICATION_INTERPRETATION,
            supporting_evidence=[valid, stale],
        )
        analysis = critical_analysis_fixture(finding)

        with patch("src.report.complete_text", return_value=synthesis_json()):
            report = synthesize_report(question_fixture(), [], [valid], analysis=analysis)

        # The invalid/stale reference is removed; the valid one is preserved.
        self.assertIs(report.findings[0].supporting_evidence[0], valid)
        self.assertEqual(report.findings[0].source_urls, [URL_ONE])
        serialized = json.dumps(report.to_dict())
        self.assertNotIn("Stale detail.", serialized)
        self.assertNotIn(URL_TWO, serialized)

    def test_original_analysis_is_not_mutated(self):
        valid = evidence_fixture(claim_text="X scales.", source_url=URL_ONE)
        stale = evidence_fixture(claim_text="Stale detail.", source_url=URL_TWO)
        finding = ClaimAnalysis(
            claim="X scales.",
            classification=ClaimAnalysis.CLASSIFICATION_INTERPRETATION,
            supporting_evidence=[valid, stale],
        )
        analysis = critical_analysis_fixture(finding)

        with patch("src.report.complete_text", return_value=synthesis_json()):
            report = synthesize_report(question_fixture(), [], [valid], analysis=analysis)

        # The caller's analysis object and its Evidence entries are untouched.
        self.assertEqual(len(analysis.claims[0].supporting_evidence), 2)
        self.assertIs(analysis.claims[0].supporting_evidence[0], valid)
        self.assertIs(analysis.claims[0].supporting_evidence[1], stale)


class RunReportTest(unittest.TestCase):
    def test_returns_structured_report(self):
        rq = question_fixture()
        src = source_fixture()
        item = evidence_fixture()
        with patch("src.report.complete_text", return_value=synthesis_json()):
            result = run_report(rq, [src], [item])

        self.assertIsInstance(result["report"], ResearchReport)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["sources"], [src])
        self.assertEqual(result["evidence"], [item])
        self.assertIsNone(result["analysis"])

    def test_empty_session_reports_empty_status(self):
        result = run_report(None)
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["report"].findings, [])

    def test_does_not_draft_or_publish(self):
        with patch("src.report.complete_text", return_value=synthesis_json()), \
                patch("src.agent.draft_post") as draft, \
                patch("src.agent.store_draft") as store:
            run_report(question_fixture(), [source_fixture()], [evidence_fixture()])

        draft.assert_not_called()
        store.assert_not_called()


class ResearchReportModelTest(unittest.TestCase):
    def test_new_sections_round_trip(self):
        item = evidence_fixture()
        finding = ClaimAnalysis(
            claim="C", classification=ClaimAnalysis.CLASSIFICATION_FACT, supporting_evidence=[item]
        )
        report = ResearchReport(
            topic="T",
            research_question="Q?",
            summary="Summary.",
            conclusions=["C1"],
            sources=[source_fixture()],
            evidence=[item],
            findings=[finding],
            supporting_evidence=[item],
            conflicting_evidence=[],
            counterarguments=[],
            uncertainties=["U"],
            unresolved_questions=["OQ"],
            synthesis="Synthesis.",
            analysis=critical_analysis_fixture(finding),
            confidence_score=0.7,
        )

        restored = ResearchReport.from_dict(report.to_dict())
        self.assertEqual(restored, report)
        self.assertIsNotNone(restored.analysis)
        self.assertEqual(
            restored.analysis.research_question,
            "Which orchestration framework scales best for production agents?",
        )

    def test_from_dict_with_legacy_shape_still_works(self):
        report = ResearchReport.from_dict({"topic": "T", "summary": "S", "conclusions": ["C"]})
        self.assertEqual(report.research_question, "")
        self.assertEqual(report.findings, [])
        self.assertIsNone(report.analysis)


if __name__ == "__main__":
    unittest.main()