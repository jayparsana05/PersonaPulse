"""
Unit tests for the Phase-3 evidence extraction stage (src/evidence.py) and
its entry point (src.agent.run_evidence).

Covers:
- supported claims (full metadata extraction)
- multiple sources supporting the same claim → corroborated
- conflicting claims are retained, never silently resolved
- missing / unknown source references
- unsupported claims (no text, no quote, quote not in body) are dropped
- malformed LLM output and LLM failures (per-source, non-fatal)
- empty inputs, body-less sources, claim caps
- run_evidence() boundary (extraction only – no synthesis/drafting)

LLM (complete_text) is mocked, so no network or API keys are needed. Dummy
env vars are installed before importing src.evidence so config validation
passes without a populated .env file (see tests/test_research.py).
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

from src.agent import run_evidence  # noqa: E402
from src.evidence import extract_evidence  # noqa: E402
from src.models import Evidence, ResearchQuestion, ResearchSource  # noqa: E402


def question_fixture() -> ResearchQuestion:
    return ResearchQuestion(
        topic="Agentic orchestration",
        question="Which orchestration framework scales best for production agents?",
        aspects=["reliability", "cost"],
    )


def source_fixture(
    url: str = "https://ex.com/one",
    title: str = "Framework Benchmarks",
    body: str = "LangGraph scaled to one million agents in production this year.",
) -> ResearchSource:
    return ResearchSource(
        url=url,
        title=title,
        body=body,
        published="2026-09-19",
        source="ex.com",
        score=0.9,
    )


def claims_json(*claims) -> str:
    return json.dumps({"claims": list(claims)})


def prompt_source_body(prompt: str) -> str:
    """Extract the 'source_body' the LLM actually received from a built prompt."""
    return json.loads(prompt[prompt.index("{"):])["source_body"]


def claim(
    text: str,
    quote: str,
    context: str = "Sales engineering report.",
    confidence: float = 0.8,
    directly_supports: bool = True,
    source_url: str = None,
) -> dict:
    data = {
        "claim_text": text,
        "supporting_quote": quote,
        "context": context,
        "confidence": confidence,
        "directly_supports": directly_supports,
    }
    if source_url is not None:
        data["source_url"] = source_url
    return data


# ---------------------------------------------------------------------------
# Supported claims
# ---------------------------------------------------------------------------

class SupportedClaimsTest(unittest.TestCase):
    def test_valid_claim_is_extracted(self):
        with patch("src.evidence.complete_text", return_value=claims_json(claim(
            "LangGraph scales to one million agents.",
            "LangGraph scaled to one million agents",
            context="Report on production deployments.",
        ))):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(len(evidence), 1)
        item = evidence[0]
        self.assertEqual(item.claim_text, "LangGraph scales to one million agents.")
        self.assertEqual(item.source_url, "https://ex.com/one")
        self.assertEqual(item.supporting_quote, "LangGraph scaled to one million agents")
        self.assertEqual(item.context, "Report on production deployments.")
        self.assertEqual(item.confidence, 0.8)
        self.assertTrue(item.directly_supports)
        self.assertEqual(item.verification_status, Evidence.VERIFICATION_CLAIMED)

    def test_missing_optional_fields_use_neutral_defaults(self):
        with patch("src.evidence.complete_text", return_value=claims_json(
            {"claim_text": "C", "supporting_quote": "one million agents"}
        )):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].confidence, 0.5)
        self.assertTrue(evidence[0].directly_supports)
        self.assertEqual(evidence[0].context, "")

    def test_non_claim_entries_are_ignored(self):
        with patch("src.evidence.complete_text", return_value=claims_json(
            {"claim_text": "C", "supporting_quote": "one million agents"}, "not-a-dict"
        )):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].claim_text, "C")

    def test_multiple_sources_supporting_same_claim_are_corroborated(self):
        source_a = source_fixture(url="https://ex.com/a", body="Orchestration is converging on standards.")
        source_b = source_fixture(url="https://ex.com/b", body="Standards are emerging in orchestration.")
        with patch("src.evidence.complete_text", side_effect=[
            claims_json(claim("Orchestration is converging on standards.", "Orchestration is converging on standards")),
            claims_json(claim("Orchestration is converging on standards.", "Standards are emerging in orchestration")),
        ]):
            evidence = extract_evidence(question_fixture(), [source_a, source_b])
        self.assertEqual(len(evidence), 2)
        for item in evidence:
            self.assertEqual(item.claim_text, "Orchestration is converging on standards.")
            self.assertEqual(item.verification_status, Evidence.VERIFICATION_CORROBORATED)

    def test_conflicting_claims_are_both_retained(self):
        source_a = source_fixture(url="https://ex.com/a", body="LangGraph supports one million agents.")
        source_b = source_fixture(url="https://ex.com/b", body="No framework has reached one million agents yet.")
        with patch("src.evidence.complete_text", side_effect=[
            claims_json(claim("LangGraph supports one million agents.", "supports one million agents")),
            claims_json(claim("No framework has reached one million agents.", "one million agents yet")),
        ]):
            evidence = extract_evidence(question_fixture(), [source_a, source_b])
        self.assertEqual(len(evidence), 2)
        urls = {item.source_url for item in evidence}
        self.assertEqual(urls, {"https://ex.com/a", "https://ex.com/b"})
        self.assertEqual({item.verification_status for item in evidence}, {Evidence.VERIFICATION_CLAIMED})


# ---------------------------------------------------------------------------
# Source attribution – never an unsupported or untraceable claim
# ---------------------------------------------------------------------------

class SourceAttributionTest(unittest.TestCase):
    def test_missing_source_reference_uses_processed_source(self):
        with patch("src.evidence.complete_text", return_value=claims_json(
            {"claim_text": "C", "supporting_quote": "one million agents"}
        )):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(evidence[0].source_url, "https://ex.com/one")

    def test_unknown_source_reference_is_dropped(self):
        with patch("src.evidence.complete_text", return_value=claims_json(claim(
            "C", "one million agents", source_url="https://other.com/x"
        ))):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(evidence, [])

    def test_claim_without_text_is_dropped(self):
        with patch("src.evidence.complete_text", return_value=claims_json(
            {"supporting_quote": "one million agents"}
        )):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(evidence, [])

    def test_claim_without_quote_is_dropped(self):
        with patch("src.evidence.complete_text", return_value=claims_json(
            {"claim_text": "C", "context": "ctx"}
        )):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(evidence, [])

    def test_quote_not_found_in_body_is_dropped(self):
        with patch("src.evidence.complete_text", return_value=claims_json(claim(
            "Invented claim", "This quote does not appear anywhere."
        ))):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(evidence, [])

    def test_unsupported_entries_do_not_consume_the_cap(self):
        body = "A. B. C. D. E."
        source = source_fixture(body=body)
        valid = [{"claim_text": f"Claim {i}", "supporting_quote": letter}
                 for i, letter in enumerate(["A.", "B.", "C.", "D.", "E."])]
        invalid = [{"claim_text": "Bogus", "supporting_quote": "not in body"},
                   {"claim_text": "No quote", "context": "ctx"}]
        with patch("src.evidence.complete_text", return_value=claims_json(*(valid + invalid))):
            evidence = extract_evidence(question_fixture(), [source], max_claims_per_source=5)
        self.assertEqual(len(evidence), 5)
        self.assertEqual([e.claim_text for e in evidence], [f"Claim {i}" for i in range(5)])


# ---------------------------------------------------------------------------
# Resilience – malformed LLM output, failures, empty inputs
# ---------------------------------------------------------------------------

class ResilienceTest(unittest.TestCase):
    def test_malformed_output_skips_only_that_source(self):
        sources = [
            source_fixture(url="https://ex.com/a", body="First source supports a claim."),
            source_fixture(url="https://ex.com/b", body="Second source supports another claim."),
        ]
        with patch("src.evidence.complete_text", side_effect=["not json at all", claims_json(
            claim("Second claim", "Second source supports another claim")
        )]):
            evidence = extract_evidence(question_fixture(), sources)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].source_url, "https://ex.com/b")

    def test_wrong_json_shape_skips_source(self):
        with patch("src.evidence.complete_text", return_value=json.dumps({"claims": "nope"})):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(evidence, [])

    def test_llm_failure_skips_source_but_continues(self):
        sources = [
            source_fixture(url="https://ex.com/a"),
            source_fixture(url="https://ex.com/b"),
        ]
        with patch("src.evidence.complete_text", side_effect=[
            RuntimeError("LLM down"),
            claims_json(claim("Second", "one million agents")),
        ]):
            evidence = extract_evidence(question_fixture(), sources)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].source_url, "https://ex.com/b")

    def test_all_sources_failing_returns_empty(self):
        with patch("src.evidence.complete_text", side_effect=RuntimeError("LLM down")):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(evidence, [])

    def test_empty_sources_returns_empty(self):
        with patch("src.evidence.complete_text") as llm:
            evidence = extract_evidence(question_fixture(), [])
        llm.assert_not_called()
        self.assertEqual(evidence, [])

    def test_none_question_returns_empty(self):
        with patch("src.evidence.complete_text") as llm:
            evidence = extract_evidence(None, [source_fixture()])
        llm.assert_not_called()
        self.assertEqual(evidence, [])

    def test_source_without_body_is_skipped(self):
        source = source_fixture(body="   ")
        with patch("src.evidence.complete_text") as llm:
            evidence = extract_evidence(question_fixture(), [source])
        llm.assert_not_called()
        self.assertEqual(evidence, [])

    def test_use_llm_false_produces_no_evidence(self):
        with patch("src.evidence.complete_text") as llm:
            evidence = extract_evidence(question_fixture(), [source_fixture()], use_llm=False)
        llm.assert_not_called()
        self.assertEqual(evidence, [])

    def test_max_claims_per_source_is_respected(self):
        raw = [claim(f"Claim {i}", "one million agents") for i in range(6)]
        with patch("src.evidence.complete_text", return_value=claims_json(*raw)):
            evidence = extract_evidence(question_fixture(), [source_fixture()], max_claims_per_source=3)
        self.assertEqual(len(evidence), 3)


# ---------------------------------------------------------------------------
# Source-body limit (configurable via EVIDENCE_SOURCE_BODY_CHARS)
# ---------------------------------------------------------------------------

class SourceBodyLimitTest(unittest.TestCase):
    def test_default_limit_is_substantially_larger_than_2000(self):
        from src.config import settings
        self.assertGreater(settings.EVIDENCE_SOURCE_BODY_CHARS, 2000)
        self.assertEqual(settings.EVIDENCE_SOURCE_BODY_CHARS, 8000)

    def test_short_source_is_passed_completely(self):
        body = "A short source that easily fits under the limit."
        with patch("src.evidence.complete_text", return_value=claims_json(
            claim("C", "fits under the limit")
        )) as llm:
            extract_evidence(question_fixture(), [source_fixture(body=body)])
        sent = prompt_source_body(llm.call_args.args[1])
        self.assertEqual(sent, body)

    def test_long_source_is_truncated_at_the_configured_limit(self):
        limit = 120
        body = "words " * 500
        with patch("src.evidence._MAX_SOURCE_BODY_CHARS", limit), \
             patch("src.evidence.complete_text", return_value=claims_json(
                 claim("C", "words words words")
             )) as llm:
            extract_evidence(question_fixture(), [source_fixture(body=body)])
        sent = prompt_source_body(llm.call_args.args[1])
        self.assertEqual(len(sent), limit)
        self.assertEqual(sent, body[:limit])

    def test_configured_value_is_actually_used_by_extraction(self):
        limit = 64
        body = "content " * 200
        with patch("src.evidence._MAX_SOURCE_BODY_CHARS", limit), \
             patch("src.evidence.complete_text", return_value=claims_json(
                 claim("C", "content content content")
             )) as llm:
            extract_evidence(question_fixture(), [source_fixture(body=body)])
        sent = prompt_source_body(llm.call_args.args[1])
        self.assertEqual(len(sent), limit)
        self.assertNotEqual(len(sent), 8000)

    def test_quote_validation_still_works_with_the_larger_limit(self):
        quote = "The framework handles concurrency efficiently."
        body = ("padding sentence. " * 100) + quote + ("trailing filler sentence. " * 500)
        # Source exceeds the limit, but the valid quote sits inside the window
        # the LLM (and therefore the validator) actually sees.
        self.assertGreater(len(body), 8000)
        self.assertLess(body.index(quote), 8000)
        with patch("src.evidence.complete_text", return_value=claims_json(
            claim("The framework handles concurrency.", quote),
        )):
            evidence = extract_evidence(question_fixture(), [source_fixture(body=body)])
        self.assertEqual(len(evidence), 1)
        self.assertIn("concurrency efficiently", evidence[0].supporting_quote)

    def test_quote_beyond_the_limit_is_rejected(self):
        limit = 200
        quote = "The framework handles concurrency efficiently."
        body = ("padding sentence. " * 20) + quote
        self.assertGreater(body.index(quote), limit)     # quote exists ONLY past the limit
        with patch("src.evidence._MAX_SOURCE_BODY_CHARS", limit), \
             patch("src.evidence.complete_text", return_value=claims_json(
                 claim("The framework handles concurrency.", quote)
             )) as llm:
            evidence = extract_evidence(question_fixture(), [source_fixture(body=body)])
        # The LLM only ever received the bounded body...
        sent = prompt_source_body(llm.call_args.args[1])
        self.assertEqual(len(sent), limit)
        self.assertNotIn(quote, sent)
        # ...and validation must apply the SAME bound, so the claim is rejected
        # even though the quote does exist in the full source body.
        self.assertEqual(evidence, [])

    def test_fabricated_quote_is_still_rejected_with_long_body(self):
        body = ("padding sentence. " * 500) + "The framework handles concurrency efficiently."
        with patch("src.evidence.complete_text", return_value=claims_json(
            claim("Invented claim", "This quote was never written in the source."),
        )):
            evidence = extract_evidence(question_fixture(), [source_fixture(body=body)])
        self.assertEqual(evidence, [])

    def test_per_source_cap_unchanged_with_larger_limit(self):
        body = "claim one. claim two. claim three. claim four. claim five. claim six."
        words = ["one", "two", "three", "four", "five", "six"]
        raw = [{"claim_text": f"C{i}", "supporting_quote": f"claim {words[i]}."} for i in range(6)]
        with patch("src.evidence.complete_text", return_value=claims_json(*raw)):
            evidence = extract_evidence(question_fixture(), [source_fixture(body=body)], max_claims_per_source=3)
        self.assertEqual(len(evidence), 3)


# ---------------------------------------------------------------------------
# Confidence validation
# ---------------------------------------------------------------------------

class ConfidenceValidationTest(unittest.TestCase):
    def _extract_with_confidence(self, value):
        with patch("src.evidence.complete_text", return_value=claims_json(claim(
            "C", "one million agents", confidence=value
        ))):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        return evidence[0].confidence

    def test_valid_numeric_confidences_are_accepted(self):
        accepted = {0: 0.0, 0.0: 0.0, 0.5: 0.5, 1: 1.0, 1.0: 1.0}
        for value, expected in accepted.items():
            with self.subTest(value=value):
                self.assertEqual(self._extract_with_confidence(value), expected)

    def test_numeric_strings_keep_existing_behavior(self):
        accepted = {"0": 0.0, "0.5": 0.5, "1": 1.0, "0.8": 0.8}
        for value, expected in accepted.items():
            with self.subTest(value=value):
                self.assertEqual(self._extract_with_confidence(value), expected)

    def test_invalid_confidences_fall_back_to_neutral(self):
        invalid = [True, False, -0.1, 1.1, None, "True", "False", "-0.1", "1.1", "None", "not-a-number"]
        for value in invalid:
            with self.subTest(value=value):
                self.assertEqual(self._extract_with_confidence(value), 0.5)

    def test_missing_confidence_uses_existing_neutral_fallback(self):
        with patch("src.evidence.complete_text", return_value=claims_json(
            {"claim_text": "C", "supporting_quote": "one million agents"}
        )):
            evidence = extract_evidence(question_fixture(), [source_fixture()])
        self.assertEqual(evidence[0].confidence, 0.5)


# ---------------------------------------------------------------------------
# Entry point: run_evidence#
# ---------------------------------------------------------------------------

class RunEvidenceTest(unittest.TestCase):
    def _sources(self):
        return [source_fixture(), source_fixture(url="https://ex.com/b")]

    def test_boundary_returns_evidence_only(self):
        with patch("src.agent.evidence_stage", return_value=[Evidence(
            claim_text="C", source_url="https://ex.com/one", supporting_quote="one million agents",
        )]) as extract, \
             patch("src.agent.draft_post") as draft, \
             patch("src.agent.store_draft") as store_post:
            result = run_evidence(
                question_fixture(),
                self._sources(),
                research_question_id="qid-1",
            )
        self.assertEqual(set(result.keys()),
                         {"research_question", "research_sources", "evidence", "status"})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["evidence"]), 1)
        self.assertEqual(len(result["research_sources"]), 2)
        extract.assert_called_once()
        self.assertEqual(extract.call_args.args[0], question_fixture())
        self.assertEqual(extract.call_args.args[1], self._sources())
        draft.assert_not_called()
        store_post.assert_not_called()

    def test_empty_evidence_returns_empty_status(self):
        with patch("src.agent.evidence_stage", return_value=[]):
            result = run_evidence(question_fixture(), self._sources())
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["status"], "empty")

    def test_no_inputs_returns_empty(self):
        with patch("src.agent.evidence_stage", return_value=[]) as extract:
            result = run_evidence(None, None)
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["status"], "empty")
        extract.assert_called_once()


if __name__ == "__main__":
    unittest.main()