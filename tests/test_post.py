"""
Unit tests for the Phase-5 LinkedIn/X post-generation stage (src/post.py) and
its entry point (src.agent.run_post).

The Phase-5 drafting stage is structurally grounded: the LLM returns a JSON
"segments" payload and every factual segment must reference valid research
findings. These tests cover:

Grounding (structural)
- unsupported factual claims are rejected whether or not they carry numbers
  or URLs
- valid paraphrases of findings are accepted
- multi-finding segments, reference validation (unknown/negative/boolean/
  non-integer/duplicate/missing), unknown URLs/numbers, report numbers/URLs,
  and original-report immutability
- non-factual framing (hooks, questions, CTA, hashtags) is allowed only when
  it introduces no facts

used_findings
- derived ONLY from the findings referenced by accepted grounded segments

Platform limits
- LinkedIn 1300-1900 chars enforced in code; X <= 280 enforced in code

Fallback semantics
- "status" ("ok" / "fallback") and "grounded" are separate: a report-derived
  fallback is grounded=True while "issues" explains the rejection

Approval workflow
- drafts are stored PENDING and a Telegram approval alert is sent; nothing is
  published automatically

LLM (complete_text), persistence and alert/publish boundaries are mocked, so
no network or API keys are needed. Dummy env vars are installed before
importing src.* so config validation passes without a populated .env file.
"""

from __future__ import annotations

import copy
import json
import os
import unittest
from itertools import cycle
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

from src.agent import node_draft, node_store_and_alert, run_post  # noqa: E402
from src.models import (  # noqa: E402
    ClaimAnalysis,
    Counterargument,
    Evidence,
    ResearchQuestion,
    ResearchReport,
    ResearchSource,
)
from src.post import (  # noqa: E402
    _validate_segments,
    draft_report_post,
    validate_draft_segments,
    validate_post_grounding,
)

URL_ONE = "https://ex.com/one"
URL_TWO = "https://ex.com/two"

# A grounded (finding-0) phrase used to pad posts to platform length limits.
_PHRASE = "Event-driven architectures grew, as teams adopting them grew by 73% in 2026. "
_GROWTH_CLAIM = "Teams adopting event-driven architectures grew by 73% in 2026."
_SCALE_CLAIM = "LangGraph handles large scale."


def payload(*segments) -> str:
    """Render an LLM segments payload as the JSON the mock returns."""
    return json.dumps({"segments": list(segments)})


def source_fixture(url: str = URL_ONE, title: str = "Vendor Report") -> ResearchSource:
    return ResearchSource(
        url=url,
        title=title,
        body="Adoption grew by 73% in 2026, per the vendor report.",
        published="2026-09-19",
        source="ex.com",
        score=0.9,
        source_type=ResearchSource.SOURCE_TYPE_SECONDARY,
    )


def evidence_fixture(
    claim_text: str = "Teams adopting event-driven architectures grew by 73% in 2026.",
    source_url: str = URL_ONE,
    quote="Adoption grew by 73% in 2026, per the vendor report.",
) -> Evidence:
    return Evidence(
        claim_text=claim_text,
        source_url=source_url,
        supporting_quote=quote,
        confidence=0.8,
        context="Benchmark report.",
        directly_supports=True,
        verification_status="unverified",
    )


def findings_fixture():
    """Two compatible findings with supporting evidence."""
    growth = ClaimAnalysis(
        claim=_GROWTH_CLAIM,
        classification=ClaimAnalysis.CLASSIFICATION_FACT,
        supporting_evidence=[evidence_fixture()],
    )
    scale = ClaimAnalysis(
        claim=_SCALE_CLAIM,
        classification=ClaimAnalysis.CLASSIFICATION_FACT,
        supporting_evidence=[
            Evidence(
                claim_text="LangGraph handles large scale.",
                source_url=URL_ONE,
                supporting_quote="LangGraph handles large scale in production.",
            )
        ],
    )
    return [growth, scale]


def report_fixture() -> ResearchReport:
    return ResearchReport(
        topic="Agentic orchestration",
        research_question="Which orchestration framework scales best for production agents?",
        summary="Event-driven adoption grew by 73%.",
        conclusions=["Event-driven adoption grew by 73%."],
        questions=[ResearchQuestion(
            topic="Agentic orchestration",
            question="Which orchestration framework scales best for production agents?",
            aspects=["reliability", "cost"],
        )],
        sources=[source_fixture()],
        findings=findings_fixture(),
        supporting_evidence=[evidence_fixture()],
        conflicting_evidence=[],
        counterarguments=[
            Counterargument(
                argument="Costs may outweigh gains.",
                evidence=[evidence_fixture(claim_text="Costs may outweigh gains.", quote="Costs may outweigh gains.")],
            )
        ],
        limitations=["Benchmarks come from vendor blogs."],
        uncertainties=["Production scale is not independently verified."],
        unresolved_questions=["How does it behave under partition?"],
        synthesis="Event-driven adoption grew by 73%; LangGraph handles large scale.",
        analysis=None,
        confidence_score=0.75,
    )


def conflicting_report_fixture() -> ResearchReport:
    claim = ClaimAnalysis(
        claim="X scales.",
        classification=ClaimAnalysis.CLASSIFICATION_FACT,
        supporting_evidence=[
            Evidence(claim_text="X scales.", source_url=URL_ONE, supporting_quote="X scales in production.")
        ],
        conflicting_evidence=[
            Evidence(
                claim_text="X does not scale under partition.",
                source_url=URL_TWO,
                supporting_quote="X does not scale under partition.",
            )
        ],
    )
    return ResearchReport(
        topic="Elastic search scalability",
        research_question="Does X keep scaling as the cluster grows?",
        summary="X scales, with caveats under partition.",
        sources=[source_fixture(), source_fixture(url=URL_TWO, title="Adversarial Blog")],
        evidence=[
            Evidence(claim_text="X scales.", source_url=URL_ONE, supporting_quote="X scales in production."),
            Evidence(claim_text="X does not scale under partition.", source_url=URL_TWO, supporting_quote="X does not scale under partition."),
        ],
        findings=[claim],
        counterarguments=[Counterargument(argument="X costs outweigh the gains.")],
        limitations=["Single vendor benchmark."],
        uncertainties=[],
        unresolved_questions=[],
        synthesis="X scales, but conflicts remain.",
        conclusions=["X scales with caveats."],
        confidence_score=0.6,
    )


def word_padded(target: int, phrase: str = _PHRASE) -> str:
    """Return the largest whole-word prefix of repeated `phrase` <= `target` chars.

    Words are never truncated mid-token, so every token stays in finding 0's
    vocabulary (73%, 2026 appear early in the phrase) and the text remains
    grounded. The result ends at a word boundary (no trailing whitespace).
    """
    text = ""
    for word in cycle(phrase.split()):
        candidate = f"{text} {word}".strip()
        if len(candidate) > target:
            break
        text = candidate
    return text


STYLE = {
    "tone": "professional yet conversational",
    "voice": "first-person, thought-leader",
    "topics_of_interest": ["AI", "engineering"],
    "avoid": ["hype"],
    "linkedin": {"max_hashtags": 4},
    "x": {"max_hashtags": 2},
}


class DraftReportPostTest(unittest.TestCase):
    """Valid report → grounded, in-range drafts that reflect the findings."""

    def linkedin_segments(self):
        return [
            {"text": word_padded(1500), "finding_indices": [0]},
            {"text": "LangGraph handles large scale.", "finding_indices": [1]},
        ]

    def x_segments(self):
        return [
            {"text": "Event-driven adoption grew by 73%, and LangGraph handles large scale.",
             "finding_indices": [0, 1]},
            {"text": "#AIEngineering #AgenticAI", "finding_indices": []},
        ]

    def test_valid_report_produces_grounded_draft(self):
        expected_x = "Event-driven adoption grew by 73%, and LangGraph handles large scale.\n#AIEngineering #AgenticAI"
        with patch("src.post.complete_text",
                   side_effect=[payload(*self.linkedin_segments()), payload(*self.x_segments())]) as mocked:
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["grounded"])
        self.assertEqual(result["issues"], [])
        self.assertEqual(mocked.call_count, 2)
        # The post reflects the research findings, within the linkedin range.
        self.assertTrue(len(result["linkedin_draft"]) >= 1300)
        self.assertTrue(len(result["linkedin_draft"]) <= 1900)
        self.assertIn("73%", result["linkedin_draft"])
        self.assertIn("LangGraph handles large scale.", result["linkedin_draft"])
        self.assertEqual(result["x_draft"], expected_x)
        self.assertLessEqual(len(result["x_draft"]), 280)
        # used_findings comes from the validated segments' references.
        self.assertEqual(result["used_findings"], [_GROWTH_CLAIM, _SCALE_CLAIM])

    def test_valid_framing_reaches_the_final_draft(self):
        linkedin = payload(
            {"text": word_padded(1500), "finding_indices": [0]},
            {"text": "Which orchestration framework scales best?", "finding_indices": []},
        )
        x = payload({"text": "Adoption grew by 73%.", "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[linkedin, x]):
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "ok")
        self.assertIn("Which orchestration framework scales best?", result["linkedin_draft"])
        # Framing contributes no used_findings of its own.
        self.assertEqual(result["used_findings"], [_GROWTH_CLAIM])

    def test_prompt_receives_report_not_article(self):
        report = report_fixture()
        with patch("src.post.complete_text",
                   side_effect=[payload(*self.x_segments()), payload(*self.x_segments())]) as mocked:
            draft_report_post(report, style_profile=STYLE)

        linkedin_prompt = mocked.call_args_list[0].args[1]
        self.assertIn(report.research_question, linkedin_prompt)
        self.assertIn(_GROWTH_CLAIM, linkedin_prompt)
        self.assertIn("Adoption grew by 73% in 2026, per the vendor report.", linkedin_prompt)
        self.assertIn("Costs may outweigh gains.", linkedin_prompt)
        self.assertIn("Benchmarks come from vendor blogs.", linkedin_prompt)
        self.assertNotIn("ARTICLE BODY", linkedin_prompt)
        self.assertNotIn("## ARTICLE TO TRANSFORM", linkedin_prompt)


class IncompleteReportTest(unittest.TestCase):
    def test_empty_report_is_bounded_and_does_not_call_llm(self):
        report = ResearchReport(topic="Agentic orchestration", research_question="Q?")

        with patch("src.post.complete_text") as mocked:
            result = draft_report_post(report, style_profile=STYLE)

        mocked.assert_not_called()
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["linkedin_draft"], "")
        self.assertEqual(result["x_draft"], "")
        self.assertFalse(result["grounded"])

    def test_llm_disabled_uses_report_derived_fallback(self):
        with patch("src.post.complete_text") as mocked:
            result = draft_report_post(report_fixture(), style_profile=STYLE, use_llm=False)

        mocked.assert_not_called()
        self.assertEqual(result["status"], "fallback")
        self.assertTrue(result["grounded"])
        self.assertIn("Key findings:", result["linkedin_draft"])
        self.assertIn("LangGraph handles large scale.", result["linkedin_draft"])
        self.assertTrue(result["linkedin_draft"])
        self.assertTrue(result["x_draft"])


class ConflictingFindingsTest(unittest.TestCase):
    def test_conflicts_reach_the_draft_and_fallback(self):
        report = conflicting_report_fixture()
        with patch("src.post.complete_text", side_effect=["not json", "not json"]) as mocked:
            draft_report_post(report, style_profile=STYLE, use_llm=True)

        prompt = mocked.call_args_list[0].args[1]
        self.assertIn("X does not scale under partition.", prompt)
        self.assertIn("X costs outweigh the gains.", prompt)

        with patch("src.post.complete_text") as mocked2:
            result = draft_report_post(report, style_profile=STYLE, use_llm=False)
        mocked2.assert_not_called()
        # The deterministic draft acknowledges the conflict honestly.
        self.assertIn("Counterarguments: X costs outweigh the gains.", result["linkedin_draft"])

    def test_llm_segment_may_echo_conflicting_evidence(self):
        report = conflicting_report_fixture()
        segments = [
            {"text": "X scales in production, but may not scale under partition.",
             "finding_indices": [0]},
        ]
        ok, issues, used = validate_draft_segments(segments, report)
        self.assertTrue(ok, issues)
        self.assertEqual(used, ["X scales."])


class DraftFailureTest(unittest.TestCase):
    def test_llm_failure_falls_back_without_fabrication(self):
        with patch("src.post.complete_text", side_effect=RuntimeError("boom")):
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "fallback")
        self.assertTrue(result["grounded"])
        self.assertEqual(result["issues"], ["LLM failure: RuntimeError"])
        self.assertEqual(result["used_findings"], [_GROWTH_CLAIM, _SCALE_CLAIM])
        # Deterministic restatement of the findings – nothing invented.
        self.assertIn("Key findings:", result["linkedin_draft"])
        self.assertIn("73%", result["linkedin_draft"])
        self.assertIn("LangGraph handles large scale.", result["linkedin_draft"])


class NumberUrlSafetyTest(unittest.TestCase):
    """Phase-4 number/URL safety check, kept as an additional safety net."""

    def test_material_numbers_are_traceable(self):
        report = report_fixture()
        ok, issues = validate_post_grounding("Adoption grew by 73% in 2026.", report)
        self.assertTrue(ok)
        self.assertEqual(issues, [])

    def test_unknown_number_rejects_the_draft(self):
        report = report_fixture()
        ok, issues = validate_post_grounding("Adoption jumped by 80%.", report)
        self.assertFalse(ok)
        self.assertIn("80", " ".join(issues))

    def test_unknown_url_rejects_the_draft(self):
        report = report_fixture()
        ok, issues = validate_post_grounding("See https://evil.example.net for details.", report)
        self.assertFalse(ok)
        self.assertIn("unknown URL", " ".join(issues))

    def test_report_url_is_allowed(self):
        report = report_fixture()
        ok, _ = validate_post_grounding(f"Read more: {URL_ONE}", report)
        self.assertTrue(ok)

    def test_empty_post_grounding_fails(self):
        ok, issues = validate_post_grounding("", report_fixture())
        self.assertFalse(ok)
        self.assertIn("empty draft", " ".join(issues))


class StructuralGroundingTest(unittest.TestCase):
    """Per-segment structural grounding against the report's findings."""

    def validate(self, segments, report=None):
        return validate_draft_segments(segments, report or report_fixture())

    def accepted_texts(self, segments, report=None):
        accepted, _, _ = _validate_segments(segments, report or report_fixture())
        return [segment["text"] for segment in accepted]

    def test_unsupported_claim_without_number_rejected(self):
        segments = [{"text": "Acme Corporation built a quantum computer.", "finding_indices": [0]}]
        ok, issues, used = self.validate(segments)
        self.assertFalse(ok)
        self.assertIn("acme", " ".join(issues).lower())
        self.assertEqual(self.accepted_texts(segments), [])
        self.assertEqual(used, [])

    def test_unsupported_claim_with_number_rejected(self):
        segments = [{"text": "Adoption grew by 99%.", "finding_indices": [0]}]
        ok, issues, used = self.validate(segments)
        self.assertFalse(ok)
        self.assertIn("99", " ".join(issues))
        self.assertEqual(used, [])

    def test_unsupported_claim_without_number_or_url_rejected(self):
        segments = [{"text": "LangGraph runs for ten thousand years.", "finding_indices": [1]}]
        ok, issues, used = self.validate(segments)
        self.assertFalse(ok)
        self.assertIn("ten", " ".join(issues))
        self.assertEqual(used, [])

    def test_valid_paraphrase_accepted(self):
        segments = [{"text": "Adoption grew by 73% in 2026.", "finding_indices": [0]}]
        ok, issues, used = self.validate(segments)
        self.assertTrue(ok, issues)
        self.assertEqual(used, [_GROWTH_CLAIM])
        self.assertEqual(self.accepted_texts(segments), ["Adoption grew by 73% in 2026."])

    def test_segment_referencing_multiple_findings_accepted(self):
        segments = [{"text": "Adoption grew by 73%, and LangGraph handles large scale.",
                     "finding_indices": [0, 1]}]
        ok, issues, used = self.validate(segments)
        self.assertTrue(ok, issues)
        self.assertEqual(used, [_GROWTH_CLAIM, _SCALE_CLAIM])

    def test_unknown_finding_index_rejected(self):
        for bad in ([5], [2]):
            segments = [{"text": "LangGraph handles large scale.", "finding_indices": bad}]
            ok, issues, _ = self.validate(segments)
            self.assertFalse(ok, f"expected rejection for {bad}")
            self.assertEqual(self.accepted_texts(segments), [])

    def test_out_of_range_index_at_upper_boundary_rejected(self):
        # 2 findings → valid indices are 0 and 1; 2 is out of range.
        segments = [{"text": "LangGraph handles large scale.", "finding_indices": [2]}]
        ok, _, _ = self.validate(segments)
        self.assertFalse(ok)

    def test_negative_finding_index_rejected(self):
        segments = [{"text": "LangGraph handles large scale.", "finding_indices": [-1]}]
        ok, _, _ = self.validate(segments)
        self.assertFalse(ok)

    def test_boolean_finding_index_rejected(self):
        segments = [{"text": "LangGraph handles large scale.", "finding_indices": [True]}]
        ok, _, _ = self.validate(segments)
        self.assertFalse(ok)

    def test_non_integer_finding_index_rejected(self):
        for bad in ([0.5], ["0"]):
            segments = [{"text": "LangGraph handles large scale.", "finding_indices": bad}]
            ok, _, _ = self.validate(segments)
            self.assertFalse(ok, f"expected rejection for {bad}")

    def test_duplicate_references_rejected(self):
        segments = [{"text": "Adoption grew by 73% in 2026.", "finding_indices": [0, 0]}]
        ok, issues, used = self.validate(segments)
        self.assertFalse(ok)
        self.assertIn("duplicate", " ".join(issues).lower())
        self.assertEqual(self.accepted_texts(segments), [])
        self.assertEqual(used, [])

    def test_missing_indices_on_factual_segment_rejected(self):
        for text in ("Adoption grew by 73% in 2026.", _SCALE_CLAIM):
            segments = [{"text": text}]
            ok, issues, used = self.validate(segments)
            self.assertFalse(ok, f"expected rejection for {text!r}")
            self.assertEqual(self.accepted_texts(segments), [])
            self.assertEqual(used, [])

    def test_unsupported_url_rejected(self):
        segments = [{"text": "LangGraph handles large scale. https://evil.example.net",
                     "finding_indices": [1]}]
        ok, issues, _ = self.validate(segments)
        self.assertFalse(ok)
        self.assertIn("evil", " ".join(issues))

    def test_unsupported_number_rejected(self):
        segments = [{"text": "LangGraph handles large scale in 2030.", "finding_indices": [1]}]
        ok, issues, _ = self.validate(segments)
        self.assertFalse(ok)
        self.assertIn("2030", " ".join(issues))

    def test_valid_report_urls_and_numbers_accepted(self):
        ok, _, _ = self.validate([
            {"text": "Adoption grew by 73% in 2026.", "finding_indices": [0]},
            {"text": "LangGraph handles large scale in production. https://ex.com/one",
             "finding_indices": [1]},
        ])
        self.assertTrue(ok)

    def test_original_report_is_not_mutated(self):
        report = report_fixture()
        before = copy.deepcopy(report)
        segments = [
            {"text": "Adoption grew by 73% in 2026.", "finding_indices": [0]},
            {"text": "Acme Corporation built a quantum computer.", "finding_indices": [0]},
            {"text": "Adoption jumped by 80%.", "finding_indices": [0]},
        ]
        validate_draft_segments(segments, report)
        self.assertEqual(report.to_dict(), before.to_dict())
        self.assertEqual(len(report.findings), 2)

    def test_empty_segments_list_is_valid(self):
        ok, issues, used = self.validate([])
        self.assertTrue(ok, issues)
        self.assertEqual(used, [])

    def test_empty_text_segment_rejected(self):
        ok, issues, _ = self.validate([{"text": "", "finding_indices": [0]}])
        self.assertFalse(ok)
        self.assertIn("empty", " ".join(issues))


class FramingSegmentTest(unittest.TestCase):
    """Non-factual writing is allowed only when it introduces no facts."""

    def test_question_framing_allowed(self):
        segments = [{"text": "Which orchestration framework scales best?", "finding_indices": []}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertTrue(ok, issues)

    def test_hashtag_framing_allowed(self):
        segments = [{"text": "#AIEngineering #AgenticAI", "finding_indices": []}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertTrue(ok, issues)

    def test_question_with_unsupported_vocabulary_rejected(self):
        # A question is not safe just because it ends with "?": its vocabulary
        # must be grounded in the report.
        segments = [{"text": "What do you think?", "finding_indices": []}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok)

    def test_framing_with_number_rejected(self):
        segments = [{"text": "Adoption jumped by 80%, so the bet is clear.", "finding_indices": []}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok)

    def test_framing_with_new_entity_rejected(self):
        segments = [{"text": "Acme dominates orchestration.", "finding_indices": []}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok)

    def test_framing_with_unsupported_vocabulary_rejected_conservatively(self):
        segments = [{"text": "Share your experience below.", "finding_indices": []}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok)


class QuestionGroundingTest(unittest.TestCase):
    """Questions are only framing AFTER their vocabulary is report-grounded.

    Ending with "?" is not an automatic pass: unsupported entities/concepts
    are rejected exactly like factual claims.
    """

    def test_unsupported_company_question_rejected(self):
        segments = [{"text": "What does QuantumCorp's quantum architecture mean for engineers?",
                     "finding_indices": []}]
        ok, issues, used = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok, issues)
        self.assertEqual(used, [])
        accepted, _, _ = _validate_segments(segments, report_fixture())
        self.assertEqual(accepted, [])

    def test_unsupported_concept_question_rejected(self):
        segments = [{"text": "How does reliability hold up in production?", "finding_indices": []}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok)

    def test_multiple_unsupported_terms_question_rejected(self):
        segments = [{"text": "Does QuantumCorp truly deliver reliability here?", "finding_indices": []}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok)

    def test_supported_question_with_empty_indices_accepted(self):
        ok, issues, used = validate_draft_segments(
            [{"text": "Which orchestration framework scales best?", "finding_indices": []}],
            report_fixture(),
        )
        self.assertTrue(ok, issues)
        self.assertEqual(used, [])

    def test_supported_question_without_indices_accepted(self):
        segments = [{"text": "Which orchestration framework scales best?"}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertTrue(ok, issues)

    def test_supported_question_with_invalid_refs_still_rejected(self):
        # Question text that WOULD be valid framing is still rejected when the
        # supplied finding_indices are malformed.
        segments = [{"text": "Which orchestration framework scales best?", "finding_indices": [999]}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok)


class InvalidRefsDoNotBecomeFramingTest(unittest.TestCase):
    """Malformed finding_indices must NEVER be rescued by framing detection.

    The distinction is explicit: missing/empty indices may be framing, valid
    indices require grounding, but invalid indices are always rejected – even
    when the text looks like a question/hook/CTA.
    """

    QUESTION = "Which orchestration framework scales best?"
    """A question using report-supported vocabulary → valid framing when the
    references themselves are sound (or absent)."""

    def uses(self, refs):
        return [{"text": self.QUESTION, "finding_indices": refs}]

    def asserted_accepted(self, refs):
        ok, issues, _ = validate_draft_segments(self.uses(refs), report_fixture())
        self.assertTrue(ok, issues)
        return ok

    def asserted_rejected(self, refs):
        ok, issues, used = validate_draft_segments(self.uses(refs), report_fixture())
        self.assertFalse(ok, f"expected rejection for refs={refs!r}")
        self.assertEqual(used, [])
        return ok

    def rejected_texts(self, refs):
        accepted, _, _ = _validate_segments(self.uses(refs), report_fixture())
        return [segment["text"] for segment in accepted]

    def test_question_with_out_of_range_index_rejected(self):
        self.assertFalse(self.asserted_rejected([999]))
        self.assertEqual(self.rejected_texts([999]), [])

    def test_question_with_negative_index_rejected(self):
        self.assertFalse(self.asserted_rejected([-1]))
        self.assertEqual(self.rejected_texts([-1]), [])

    def test_question_with_string_index_rejected(self):
        self.assertFalse(self.asserted_rejected(["0"]))
        self.assertEqual(self.rejected_texts(["0"]), [])

    def test_question_with_boolean_index_rejected(self):
        self.assertFalse(self.asserted_rejected([True]))
        self.assertEqual(self.rejected_texts([True]), [])

    def test_question_with_duplicate_indices_rejected(self):
        self.assertFalse(self.asserted_rejected([0, 0]))
        self.assertEqual(self.rejected_texts([0, 0]), [])

    def test_question_with_non_list_indices_rejected(self):
        for refs in (0, "0", 1.5, True):
            self.assertFalse(self.asserted_rejected(refs), f"refs={refs!r}")
            self.assertEqual(self.rejected_texts(refs), [])

    def test_question_with_indices_omitted_accepted_as_framing(self):
        segments = [{"text": self.QUESTION}]
        ok, issues, used = validate_draft_segments(segments, report_fixture())
        self.assertTrue(ok, issues)
        self.assertEqual(used, [])

    def test_explicit_empty_indices_accepted_when_framing(self):
        self.assertTrue(self.asserted_accepted([]))

    def test_explicit_empty_indices_rejected_when_factual(self):
        # [] + factual content is NOT rescued: the framing gate must reject it.
        segments = [{"text": "Adoption grew by 73%.", "finding_indices": []}]
        ok, issues, _ = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok)

    def test_valid_references_still_require_grounding(self):
        ok, issues, used = validate_draft_segments(
            [{"text": "Adoption grew by 73% in 2026.", "finding_indices": [0]}],
            report_fixture(),
        )
        self.assertTrue(ok, issues)
        accepted, _, _ = _validate_segments(
            [{"text": "Adoption grew by 73% in 2026.", "finding_indices": [0]}],
            report_fixture(),
        )
        self.assertEqual(accepted[0]["finding_indices"], [0])
        self.assertEqual(used, [_GROWTH_CLAIM])

    def test_duplicate_refs_do_not_reach_accepted_or_used_findings(self):
        segments = [{"text": "Adoption grew by 73% in 2026.", "finding_indices": [0, 0]}]
        ok, _, used = validate_draft_segments(segments, report_fixture())
        self.assertFalse(ok)
        self.assertEqual(used, [])
        accepted, _, _ = _validate_segments(segments, report_fixture())
        self.assertEqual(accepted, [])


class UsedFindingsTest(unittest.TestCase):
    """used_findings is derived ONLY from accepted grounded segments."""

    def test_used_findings_exactly_referenced_and_grounded(self):
        report = report_fixture()
        segments = [
            {"text": "Adoption grew by 73%, and LangGraph handles large scale.",
             "finding_indices": [0, 1]},
            {"text": "Which orchestration framework scales best?", "finding_indices": []},
        ]
        ok, _, used = validate_draft_segments(segments, report)
        self.assertTrue(ok)
        self.assertEqual(used, [_GROWTH_CLAIM, _SCALE_CLAIM])

    def test_unsupported_segment_does_not_contribute_to_used_findings(self):
        report = report_fixture()
        segments = [
            {"text": "Adoption grew by 73% in 2026.", "finding_indices": [0]},
            {"text": "LangGraph runs for ten thousand years.", "finding_indices": [1]},
        ]
        ok, _, used = validate_draft_segments(segments, report)
        self.assertFalse(ok)
        self.assertEqual(used, [_GROWTH_CLAIM])


class PlatformLimitsTest(unittest.TestCase):
    """Platform length limits are enforced in code, not just prompts."""

    def valid_linkedin(self):
        return payload(
            {"text": word_padded(1500), "finding_indices": [0]},
            {"text": "LangGraph handles large scale.", "finding_indices": [1]},
        )

    def valid_x(self):
        return payload(
            {"text": "Adoption grew by 73%, and LangGraph handles large scale.", "finding_indices": [0, 1]},
            {"text": "#AIEngineering", "finding_indices": []},
        )

    def test_linkedin_below_minimum_falls_back(self):
        short = payload({"text": "Adoption grew by 73%.", "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[short, self.valid_x()]):
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "fallback")
        self.assertIn("length", "; ".join(result["issues"]))
        # The deterministic fallback replaced the over-short LLM draft.
        self.assertIn("Key findings:", result["linkedin_draft"])
        self.assertNotEqual(result["linkedin_draft"], "Adoption grew by 73%.")

    def test_linkedin_above_maximum_falls_back(self):
        too_long = payload({"text": word_padded(2000), "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[too_long, self.valid_x()]):
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "fallback")
        self.assertIn("length", "; ".join(result["issues"]))
        self.assertIn("Key findings:", result["linkedin_draft"])

    def test_linkedin_minimum_and_maximum_boundaries_accepted(self):
        # Multiple in-range sizes are accepted (inclusive [1300, 1900]);
        # word_padded guarantees whole-word grounded text <= target chars.
        for target in (1504, 1898):
            linkedin = payload({"text": word_padded(target), "finding_indices": [0]})
            with patch("src.post.complete_text", side_effect=[linkedin, self.valid_x()]):
                result = draft_report_post(report_fixture(), style_profile=STYLE)
            self.assertEqual(result["status"], "ok", f"failed at {target} chars")
            self.assertEqual(len(result["linkedin_draft"]), word_padded(target).__len__())
            self.assertGreaterEqual(len(result["linkedin_draft"]), 1300)
            self.assertLessEqual(len(result["linkedin_draft"]), 1900)

    def test_linkedin_below_minimum_and_above_maximum_rejected(self):
        # One whole word under the minimum (1299 chars) → rejected.
        short = payload({"text": word_padded(1299), "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[short, self.valid_x()]):
            result = draft_report_post(report_fixture(), style_profile=STYLE)
        self.assertEqual(result["status"], "fallback")
        self.assertIn("length", "; ".join(result["issues"]))

        # Just over the maximum (1903 chars) → rejected.
        long = payload({"text": word_padded(1903), "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[long, self.valid_x()]):
            result = draft_report_post(report_fixture(), style_profile=STYLE)
        self.assertEqual(result["status"], "fallback")
        self.assertIn("length", "; ".join(result["issues"]))

    def test_x_above_280_falls_back(self):
        too_long = payload({"text": "Event-driven adoption grew by 73% in 2026. " * 8,
                            "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[self.valid_linkedin(), too_long]):
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "fallback")
        self.assertIn("exceeds", "; ".join(result["issues"]))
        # Fallback X is short and self-contained.
        self.assertLessEqual(len(result["x_draft"]), 280)
        self.assertIn("#", result["x_draft"])

    def test_x_at_exactly_280_accepted(self):
        x = payload({"text": word_padded(280), "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[self.valid_linkedin(), x]):
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "ok")
        self.assertLessEqual(len(result["x_draft"]), 280)

    def test_fallback_x_is_always_within_280(self):
        for fixture in (report_fixture(), conflicting_report_fixture()):
            result = draft_report_post(fixture, style_profile=STYLE, use_llm=False)
            self.assertLessEqual(len(result["x_draft"]), 280)


class FallbackStatusTest(unittest.TestCase):
    """status ("ok" / "fallback") and grounded are independent concepts."""

    def test_llm_failure_produces_grounded_fallback(self):
        with patch("src.post.complete_text", side_effect=RuntimeError("boom")):
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "fallback")
        self.assertTrue(result["grounded"])
        self.assertEqual(result["issues"], ["LLM failure: RuntimeError"])

    def test_invalid_grounding_produces_grounded_fallback(self):
        bad_linkedin = payload({"text": "Acme Corporation built a quantum computer.",
                                "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[bad_linkedin, self._valid_x()]):
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "fallback")
        self.assertTrue(result["grounded"])
        self.assertNotIn("Acme", result["linkedin_draft"])
        self.assertFalse(any("Acme" in issue for issue in result["issues"]))
        self.assertTrue(result["issues"])

    def test_invalid_framing_indices_produce_grounded_fallback(self):
        # An out-of-range index on question text must NOT become framing:
        # the segment is rejected and the deterministic fallback takes over.
        bad_linkedin = payload(
            {"text": "What does this mean for engineers?", "finding_indices": [999]}
        )
        with patch("src.post.complete_text", side_effect=[bad_linkedin, self._valid_x()]):
            result = draft_report_post(report_fixture(), style_profile=STYLE)

        self.assertEqual(result["status"], "fallback")
        self.assertTrue(result["grounded"])
        self.assertNotIn("What does this mean for engineers?", result["linkedin_draft"])
        # The fallback restates the findings, so all findings are flagged used.
        self.assertEqual(result["used_findings"], [_GROWTH_CLAIM, _SCALE_CLAIM])
        self.assertTrue(result["issues"])

    def test_status_distinguishes_ok_from_fallback(self):
        valid_linkedin = payload(
            {"text": word_padded(1500), "finding_indices": [0]},
            {"text": "LangGraph handles large scale.", "finding_indices": [1]},
        )
        valid_x = payload({"text": "Adoption grew by 73%.", "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[valid_linkedin, valid_x]):
            ok_result = draft_report_post(report_fixture(), style_profile=STYLE)
        self.assertEqual(ok_result["status"], "ok")
        self.assertTrue(ok_result["grounded"])

        bad_linkedin = payload({"text": "Acme Corporation built a quantum computer.",
                                "finding_indices": [0]})
        with patch("src.post.complete_text", side_effect=[bad_linkedin, valid_x]):
            fallback_result = draft_report_post(report_fixture(), style_profile=STYLE)
        self.assertEqual(fallback_result["status"], "fallback")
        self.assertTrue(fallback_result["grounded"])

        self.assertNotEqual(ok_result["status"], fallback_result["status"])

    @staticmethod
    def _valid_x():
        return payload({"text": "Adoption grew by 73%.", "finding_indices": [0]})


class ApprovalFlowCompatibilityTest(unittest.TestCase):
    """Report → drafts → PENDING store + Telegram approval alert. Never publish."""

    def setUp(self):
        self.report = report_fixture()
        self.linkedin = ("Event-driven adoption grew by 73%. LangGraph handles large scale. "
                         "#AIEngineering #AgenticAI")
        self.x = "Event-driven adoption grew by 73%. #AgenticAI"

    def test_run_post_stores_pending_and_alerts_for_approval(self):
        draft_result = {
            "status": "ok",
            "linkedin_draft": self.linkedin,
            "x_draft": self.x,
            "used_findings": ["LangGraph handles large scale."],
            "grounded": True,
            "issues": [],
        }
        with patch("src.agent.draft_report_post", return_value=draft_result) as m_draft, \
                patch("src.agent.get_style_profile", return_value=STYLE), \
                patch("src.agent.extract_og_image_with_url", return_value=(None, None)), \
                patch("src.agent.get_normalized_embedding", return_value=[0.1, 0.2, 0.3]), \
                patch("src.agent.store_draft", return_value="post-123") as m_store, \
                patch("src.agent.send_telegram_alert") as m_alert, \
                patch("src.publishers.publish_to_linkedin") as m_li, \
                patch("src.publishers.publish_to_x") as m_x:
            result = run_post(self.report)

        # The report was consumed by the drafting stage.
        m_draft.assert_called_once()
        report_arg = m_draft.call_args.args[0]
        self.assertIsInstance(report_arg, ResearchReport)
        self.assertEqual(report_arg.topic, self.report.topic)
        # Draft reflects the research findings.
        self.assertEqual(result["linkedin_draft"], self.linkedin)
        self.assertEqual(result["x_draft"], self.x)
        self.assertEqual(result["status"], "ok")

        # Stored as a PENDING LinkedIn/X draft with the drafted content.
        m_store.assert_called_once()
        store_kwargs = m_store.call_args.kwargs
        self.assertEqual(store_kwargs["platform"], "both")
        self.assertEqual(store_kwargs["topic"], self.report.topic)
        self.assertIn(self.linkedin, store_kwargs["content"])
        self.assertIn(self.x, store_kwargs["content"])

        # Telegram approval alert sent with the drafts + post id (approval flow).
        m_alert.assert_called_once()
        alert_kwargs = m_alert.call_args.kwargs
        self.assertEqual(alert_kwargs["post_id"], "post-123")
        self.assertEqual(alert_kwargs["linkedin_draft"], self.linkedin)
        self.assertEqual(alert_kwargs["x_draft"], self.x)

        # Nothing was published automatically.
        m_li.assert_not_called()
        m_x.assert_not_called()

    def test_run_post_accepts_dict_report(self):
        draft_result = {
            "status": "ok",
            "linkedin_draft": self.linkedin,
            "x_draft": self.x,
            "used_findings": [],
            "grounded": True,
            "issues": [],
        }
        with patch("src.agent.draft_report_post", return_value=draft_result), \
                patch("src.agent.get_style_profile", return_value=STYLE), \
                patch("src.agent.extract_og_image_with_url", return_value=(None, None)), \
                patch("src.agent.get_normalized_embedding", return_value=[0.1, 0.2, 0.3]), \
                patch("src.agent.store_draft", return_value="post-999"), \
                patch("src.agent.send_telegram_alert"):
            result = run_post(self.report.to_dict())

        self.assertIsInstance(result["report"], ResearchReport)
        self.assertEqual(result["report"].topic, "Agentic orchestration")
        self.assertEqual(result["post_id"], "post-999")

    def test_run_post_never_calls_publish_on_llm_failure(self):
        fallback = {
            "status": "fallback",
            "linkedin_draft": "Key findings: LangGraph handles large scale.",
            "x_draft": "LangGraph handles large scale. #AgenticAI",
            "used_findings": ["LangGraph handles large scale."],
            "grounded": True,
            "issues": ["LLM failure: RuntimeError"],
        }
        with patch("src.agent.draft_report_post", return_value=fallback), \
                patch("src.agent.get_style_profile", return_value=STYLE), \
                patch("src.agent.extract_og_image_with_url", return_value=(None, None)), \
                patch("src.agent.get_normalized_embedding", return_value=[0.1, 0.2, 0.3]), \
                patch("src.agent.store_draft", return_value="post-fallback"), \
                patch("src.agent.send_telegram_alert"), \
                patch("src.publishers.publish_to_linkedin") as m_li, \
                patch("src.publishers.publish_to_x") as m_x:
            result = run_post(self.report)

        self.assertEqual(result["status"], "fallback")
        self.assertIn("Key findings:", result["linkedin_draft"])
        m_li.assert_not_called()
        m_x.assert_not_called()


class AgentNodeTest(unittest.TestCase):
    """The production LangGraph nodes consume the report when present."""

    def test_node_draft_uses_report_when_present(self):
        style = {"tone": "t"}
        draft_result = {
            "status": "ok",
            "linkedin_draft": "LinkedIn from report.",
            "x_draft": "X from report.",
            "used_findings": [],
            "grounded": True,
            "issues": [],
        }
        state = {"report": self._report_dict(), "style_profile": None}
        with patch("src.agent.get_style_profile", return_value=style), \
                patch("src.agent.draft_report_post", return_value=draft_result) as m_draft, \
                patch("src.agent.draft_post") as m_article:
            result = node_draft(state)

        m_draft.assert_called_once()
        self.assertIsInstance(m_draft.call_args.args[0], ResearchReport)
        m_article.assert_not_called()
        self.assertEqual(result["linkedin_draft"], "LinkedIn from report.")
        self.assertEqual(result["x_draft"], "X from report.")

    def test_node_draft_falls_back_to_article_without_report(self):
        style = {"tone": "t"}
        state = {"article": {"title": "T", "body": "B", "url": "U", "source": "S"}, "style_profile": None}
        with patch("src.agent.get_style_profile", return_value=style), \
                patch("src.agent.draft_post", side_effect=["LI", "X"]) as m_article, \
                patch("src.agent.draft_report_post") as m_report:
            result = node_draft(state)

        self.assertEqual(m_article.call_count, 2)
        m_report.assert_not_called()
        self.assertEqual(result["linkedin_draft"], "LI")
        self.assertEqual(result["x_draft"], "X")

    def test_node_store_and_alert_uses_report_metadata(self):
        state = {
            "report": self._report_dict(),
            "embedding": [0.1],
            "linkedin_draft": "LI",
            "x_draft": "X",
        }
        with patch("src.agent.extract_og_image_with_url", return_value=("img", b"bytes")), \
                patch("src.agent.store_draft", return_value="post-1") as m_store, \
                patch("src.agent.send_telegram_alert") as m_alert:
            result = node_store_and_alert(state)

        store_kwargs = m_store.call_args.kwargs
        self.assertEqual(store_kwargs["topic"], "Agentic orchestration")
        self.assertEqual(store_kwargs["article_url"], URL_ONE)
        self.assertIn("LI", store_kwargs["content"])
        alert_kwargs = m_alert.call_args.kwargs
        self.assertEqual(alert_kwargs["article"]["title"], "Agentic orchestration")
        # The research digest travels with the approval message so the
        # approver can see what the drafts are based on.
        research = alert_kwargs["research"]
        self.assertEqual(research["source_count"], 1)
        self.assertEqual(research["research_question"], "Which orchestration framework scales best for production agents?")
        self.assertIn(_GROWTH_CLAIM.split(".")[0], " ".join(research["key_findings"]))
        self.assertEqual(result["post_id"], "post-1")

    def test_node_store_and_alert_legacy_article_path_has_no_research(self):
        # No report present -> the legacy article path must still alert, but
        # with research=None so the digest is omitted (no empty section).
        state = {
            "article": {
                "title": "Plain article",
                "body": "Body",
                "url": "https://ex.com/article",
                "source": "ex.com",
            },
            "embedding": [0.1],
            "linkedin_draft": "LI",
            "x_draft": "X",
        }
        with patch("src.agent.extract_og_image_with_url", return_value=("img", b"bytes")), \
                patch("src.agent.store_draft", return_value="post-2") as m_store, \
                patch("src.agent.send_telegram_alert") as m_alert:
            result = node_store_and_alert(state)

        store_kwargs = m_store.call_args.kwargs
        self.assertEqual(store_kwargs["topic"], "Plain article")
        alert_kwargs = m_alert.call_args.kwargs
        self.assertIsNone(alert_kwargs["research"])
        self.assertEqual(alert_kwargs["post_id"], "post-2")
        self.assertEqual(result["post_id"], "post-2")

    @staticmethod
    def _report_dict() -> dict:
        return report_fixture().to_dict()


if __name__ == "__main__":
    unittest.main()