"""
End-to-end tests for the AI Engineering Research Agent orchestrator
(src.agent.run_research_agent).

The orchestrator composes the 13-step workflow:
discovery → candidates → selection → ResearchQuestion → multi-source research
→ evidence/claims → critical analysis → report → LinkedIn/X drafts
→ Telegram approval request (PENDING, never published directly)
→ persist research history for future deduplication.

Everything external is mocked (LLM dispatcher keyed on system-prompt marker,
search provider, Supabase memory, draft storage, Telegram, publishers), so no
network or API keys are needed.

Covers:
- full happy path → status "posted", draft stored, Telegram approval alerted
- discovery/selection/research/evidence/analysis/post gates halt cleanly
- empty research session created this run is cleaned up (dedup hygiene)
- duplicate topic reuses the remembered session without a new search
- failures in one research source do not terminate the session
- the orchestrator NEVER publishes directly (approval is external)
"""

from __future__ import annotations

import json
import os
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

# Dummy env vars (see tests/test_research.py) so src.config imports cleanly.
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

from src.agent import run_research_agent  # noqa: E402
from src.models import ResearchSource, TopicCandidate  # noqa: E402

BODY = "A snippet about orchestration."
SELECTION_MARKER = "topic-selection agent"
FRAMING_MARKER = "question framer"
QUERY_MARKER = "research-query planner"
EVIDENCE_MARKER = "evidence extractor"
ANALYSIS_MARKER = "critical analyst"
REPORT_MARKER = "research synthesis writer"
POST_MARKER = "ghostwriter"


def make_candidate(
    title="Agentic orchestration",
    url="https://ex.com/cand",
    score=0.85,
) -> TopicCandidate:
    return TopicCandidate(
        title=title,
        description="Scaling agent orchestration frameworks in production.",
        url=url,
        keywords=["agents", "orchestration"],
        source="ex.com",
        published="2026-09-19",
        search_score=score,
    )


def raw_result(
    url="https://ex.com/a",
    title="First source",
    content=BODY,
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


def selection_payload(index=0):
    return {
        "selected_index": index,
        "reasoning": "This candidate directly supports reliable production agents.",
        "evaluations": [
            {
                "index": 0,
                "fit_score": 90,
                "reason": "Directly relevant engineering topic.",
                "strengths": "deep technical depth",
                "concerns": "",
            }
        ],
    }


def framing_payload():
    return {
        "question": "Which orchestration framework scales best for production agents?",
        "aspects": ["reliability", "cost"],
    }


def query_payload(queries=("orchestration scaling production agents",)):
    return {"queries": list(queries)}


def evidence_payload(source_url="https://ex.com/a"):
    return {
        "claims": [
            {
                "claim_text": "Orchestration frameworks scale horizontally in production.",
                "supporting_quote": "A snippet about orchestration",
                "context": "Orchestration scaling",
                "confidence": 0.9,
                "directly_supports": True,
            }
        ]
    }


def analysis_payload(source_url="https://ex.com/a"):
    return {
        "claims": [
            {
                "claim": "Agent orchestration scales horizontally for production workloads.",
                "classification": "documented_fact",
                "supporting_source_urls": [source_url],
                "conflicting_source_urls": [],
                "confidence": 0.85,
                "reasoning": "Stated directly by the evidence.",
                "limitations": [],
            }
        ],
        "counterarguments": [],
        "limitations": ["Small sample of sources."],
        "uncertainties": [],
        "unresolved_questions": [],
    }


def empty_analysis_payload():
    return {
        "claims": [],
        "counterarguments": [],
        "limitations": [],
        "uncertainties": [],
        "unresolved_questions": [],
    }


def default_payloads(source_url="https://ex.com/a") -> dict:
    """Valid payloads for every grounded LLM stage; report + post fall back
    deterministically (no LLM payload needed for those markers)."""
    return {
        SELECTION_MARKER: selection_payload(),
        FRAMING_MARKER: framing_payload(),
        QUERY_MARKER: query_payload(),
        EVIDENCE_MARKER: evidence_payload(source_url),
        ANALYSIS_MARKER: analysis_payload(source_url),
    }


@contextmanager
def mocked_pipeline(
    search_side_effect=None,
    payloads=None,
    check_result=None,
    hydrated_sources=None,
    store_draft_result="post-1",
    store_draft_exception=None,
    alert_exception=None,
    discover_result=None,
):
    """Install mocks for every external boundary the orchestrator touches.

    Yields a dict of mock handles for call assertions. ``complete_text`` is
    dispatched on the system-prompt marker; report/post markers receive
    invalid JSON so their deterministic, grounded fallbacks run.
    """
    if search_side_effect is None:
        search_side_effect = [[]]
    payloads = payloads or default_payloads()
    check_result = check_result if check_result is not None else {
        "matched": False,
        "reason": None,
        "question_id": None,
        "matched_question": None,
        "similarity": None,
    }
    hydrated_sources = hydrated_sources if hydrated_sources is not None else ([], [])
    discover_result = [make_candidate()] if discover_result is None else list(discover_result)

    def send_alert(**kwargs):
        if alert_exception is not None:
            raise alert_exception
        return None

    def responder(system_prompt, user_prompt, **_kwargs):
        for marker, payload in payloads.items():
            if marker in system_prompt:
                return json.dumps(payload)
        return "not-json"

    def store_draft(**kwargs):
        if store_draft_exception is not None:
            raise store_draft_exception
        return store_draft_result

    # Each stage module aliases complete_text at import time, so the mock is
    # installed at every stage-local name (a src.llm.complete_text patch would
    # not intercept the aliased calls). ExitStack avoids Python's nested-with
    # block limit.
    with ExitStack() as _stack:
        mocks = {
            "llm_selection": _stack.enter_context(
                patch("src.selection.complete_text", side_effect=responder)),
            "llm_research": _stack.enter_context(
                patch("src.research.complete_text", side_effect=responder)),
            "llm_evidence": _stack.enter_context(
                patch("src.evidence.complete_text", side_effect=responder)),
            "llm_analysis": _stack.enter_context(
                patch("src.analysis.complete_text", side_effect=responder)),
            "llm_report": _stack.enter_context(
                patch("src.report.complete_text", side_effect=responder)),
            "llm_post": _stack.enter_context(
                patch("src.post.complete_text", side_effect=responder)),
            "search": _stack.enter_context(
                patch("src.research.search_news", side_effect=search_side_effect)),
            "discover": _stack.enter_context(
                patch("src.agent.discover_topic_candidates", return_value=discover_result)),
            "check": _stack.enter_context(
                patch("src.agent.check_topic_researched", return_value=check_result)),
            "hydrate": _stack.enter_context(
                patch("src.agent.get_research_sources_for_question",
                      return_value=hydrated_sources)),
            "known": _stack.enter_context(
                patch("src.agent.get_known_source_urls", return_value=set())),
            "store_q": _stack.enter_context(
                patch("src.agent.store_research_question", return_value="qid-fresh")),
            "store_s": _stack.enter_context(
                patch("src.agent.store_research_sources", return_value=["src-1"])),
            "delete_q": _stack.enter_context(
                patch("src.agent.delete_research_question")),
            "style": _stack.enter_context(
                patch("src.agent.get_style_profile", return_value={})),
            "og": _stack.enter_context(
                patch("src.agent.extract_og_image_with_url", return_value=(None, None))),
            "embed": _stack.enter_context(
                patch("src.agent.get_normalized_embedding", return_value=[0.1, 0.2])),
            "store_draft": _stack.enter_context(
                patch("src.agent.store_draft", side_effect=store_draft)),
            "alert": _stack.enter_context(
                patch("src.agent.send_telegram_alert", side_effect=send_alert)),
            "publish_li": _stack.enter_context(
                patch("src.publishers.publish_to_linkedin")),
            "publish_x": _stack.enter_context(
                patch("src.publishers.publish_to_x")),
        }
        yield mocks


class RunResearchAgentTest(unittest.TestCase):

    def test_full_pipeline_dispatches_post_and_alerts_telegram(self):
        """Happy path: every stage runs, a PENDING draft is stored, Telegram
        is alerted, and nothing is ever published directly."""
        with mocked_pipeline(search_side_effect=[[raw_result()]]) as m:
            result = run_research_agent(candidates=[make_candidate()])

        self.assertEqual(result["status"], "posted")
        self.assertEqual(result["post_id"], "post-1")
        self.assertIsNone(result["halt_reason"])
        self.assertEqual(
            result["stages"],
            {
                "discovery": "ok",
                "selection": "ok",
                "research": "ok",
                "evidence": "ok",
                "analysis": "ok",
                "report": "ok",
                "post": "fallback",
            },
        )
        self.assertEqual(result["research_question_id"], "qid-fresh")
        self.assertEqual(result["already_researched"], False)
        self.assertEqual(len(result["research_sources"]), 1)
        self.assertEqual(len(result["research_source_ids"]), 1)
        self.assertEqual(len(result["evidence"]), 1)
        self.assertIsNotNone(result["analysis"])
        self.assertIsNotNone(result["report"])
        self.assertTrue(result["report"].findings)
        self.assertEqual(result["post"]["grounded"], True)

        # Research ran and persisted into the freshly created session.
        m["search"].assert_called_once()
        m["store_q"].assert_called_once()
        m["store_s"].assert_called_once()
        self.assertEqual(m["store_s"].call_args.args[0], "qid-fresh")
        self.assertEqual(len(m["store_s"].call_args.args[1]), 1)

        # Draft stored + Telegram approval requested; never published.
        m["store_draft"].assert_called_once()
        m["alert"].assert_called_once()
        self.assertEqual(m["alert"].call_args.kwargs["post_id"], "post-1")
        m["publish_li"].assert_not_called()
        m["publish_x"].assert_not_called()
        self.assertFalse(m["delete_q"].called)

    def test_discovery_pipeline_used_when_no_candidates_injected(self):
        with mocked_pipeline(search_side_effect=[[raw_result()]]) as m:
            result = run_research_agent(
                query="engineering topic",
                candidates=None,
            )
        self.assertEqual(result["status"], "posted")
        self.assertEqual(len(result["topic_candidates"]), 1)
        m["search"].assert_called_once()

    def test_no_candidates_halts_at_discovery(self):
        with mocked_pipeline() as m:
            result = run_research_agent(candidates=[])
        self.assertEqual(result["status"], "discovery_empty")
        self.assertIn("no topic candidates", result["halt_reason"].lower())
        m["search"].assert_not_called()
        m["alert"].assert_not_called()

    def test_discovery_empty_halts(self):
        with mocked_pipeline(discover_result=[]) as m:
            result = run_research_agent(query="engineering topic", candidates=None)
        self.assertEqual(result["status"], "discovery_empty")
        m["discover"].assert_called_once()
        m["search"].assert_not_called()

    def test_selection_empty_halts(self):
        payloads = {
            SELECTION_MARKER: {
                "selected_index": None,
                "reasoning": "Nothing fits.",
                "evaluations": [
                    {"index": 0, "fit_score": 10, "reason": "Not relevant.",
                     "strengths": "", "concerns": ""}
                ],
            },
            FRAMING_MARKER: {"question": "q", "aspects": ["a"]},
        }
        with mocked_pipeline(payloads=payloads) as m:
            result = run_research_agent(candidates=[make_candidate()])
        self.assertEqual(result["status"], "selection_empty")
        self.assertIn("research question", result["halt_reason"].lower())
        m["search"].assert_not_called()
        m["alert"].assert_not_called()

    def test_research_empty_halts_and_cleans_up_fresh_session(self):
        """No search results → the just-created session row is deleted so a
        dead question is never remembered as researched."""
        with mocked_pipeline(search_side_effect=[[]]) as m:
            result = run_research_agent(candidates=[make_candidate()])
        self.assertEqual(result["status"], "research_empty")
        self.assertIsNone(result["research_question_id"])
        m["search"].assert_called_once()
        m["delete_q"].assert_called_once_with("qid-fresh")
        m["alert"].assert_not_called()
        m["publish_li"].assert_not_called()
        m["publish_x"].assert_not_called()

    def test_evidence_empty_halts(self):
        payloads = {
            **default_payloads(),
            EVIDENCE_MARKER: {"claims": []},
        }
        with mocked_pipeline(
            payloads=payloads,
            search_side_effect=[[raw_result()]],
        ) as m:
            result = run_research_agent(candidates=[make_candidate()])
        self.assertEqual(result["status"], "evidence_empty")
        self.assertEqual(result["evidence"], [])
        m["alert"].assert_not_called()

    def test_analysis_empty_halts(self):
        payloads = {
            **default_payloads(),
            ANALYSIS_MARKER: empty_analysis_payload(),
        }
        with mocked_pipeline(
            payloads=payloads,
            search_side_effect=[[raw_result()]],
        ) as m:
            result = run_research_agent(candidates=[make_candidate()])
        self.assertEqual(result["status"], "analysis_empty")
        self.assertIsNone(result["report"])
        m["alert"].assert_not_called()

    def test_research_source_failure_does_not_terminate_session(self):
        """One research query raising must not kill the session; the other
        query still produces sources and the pipeline completes."""
        payloads = {
            **default_payloads(source_url="https://ex.com/b"),
            QUERY_MARKER: query_payload(("q1", "q2")),
        }
        side_effect = [RuntimeError("provider down"), [raw_result(url="https://ex.com/b")]]
        with mocked_pipeline(search_side_effect=side_effect, payloads=payloads) as m:
            result = run_research_agent(candidates=[make_candidate()])
        self.assertEqual(result["status"], "posted")
        self.assertEqual(len(result["research_sources"]), 1)
        self.assertEqual(m["search"].call_count, 2)
        m["alert"].assert_called_once()

    def test_duplicate_topic_reuses_session_without_new_search(self):
        """An already-researched topic reuses the remembered session (hydrated
        sources) and never re-searches or re-stores the question row."""
        known_source = ResearchSource(url="https://ex.com/known", title="Known", body=BODY)
        check_result = {
            "matched": True,
            "reason": "exact",
            "question_id": "qid-known",
            "matched_question": {"id": "qid-known"},
            "similarity": None,
        }
        payloads = default_payloads(source_url="https://ex.com/known")
        with mocked_pipeline(
            check_result=check_result,
            hydrated_sources=([known_source], ["src-0"]),
            payloads=payloads,
        ) as m:
            result = run_research_agent(candidates=[make_candidate()])

        self.assertEqual(result["status"], "posted")
        self.assertEqual(result["stages"]["research"], "reused")
        self.assertEqual(result["research_question_id"], "qid-known")
        self.assertEqual(result["reused_question_id"], "qid-known")
        self.assertEqual(result["duplicate_reason"], "exact")
        self.assertEqual(result["already_researched"], True)
        self.assertEqual(result["research_sources"], [known_source])
        m["search"].assert_not_called()
        m["store_q"].assert_not_called()
        m["store_s"].assert_not_called()
        m["delete_q"].assert_not_called()
        m["hydrate"].assert_called_once_with("qid-known")
        m["alert"].assert_called_once()

    def test_post_stage_failure_halts(self):
        with mocked_pipeline(
            store_draft_exception=RuntimeError("supabase down"),
            search_side_effect=[[raw_result()]],
        ) as m:
            result = run_research_agent(candidates=[make_candidate()])
        self.assertEqual(result["status"], "post_empty")
        self.assertIn("draft", result["halt_reason"].lower())
        self.assertIsNone(result["post_id"])
        m["alert"].assert_not_called()
        m["publish_li"].assert_not_called()
        m["publish_x"].assert_not_called()

    def test_approval_alert_failure_keeps_draft_and_reports_post_id(self):
        """When the draft is stored but the Telegram approval alert fails, the
        pipeline must NOT hide the stored draft or report 'post_empty': it
        returns the PENDING post_id with an explicit 'approval_alert_failed'
        status, and never publishes."""
        with mocked_pipeline(
            search_side_effect=[[raw_result()]],
            alert_exception=RuntimeError("telegram down"),
        ) as m:
            result = run_research_agent(candidates=[make_candidate()])

        self.assertEqual(result["status"], "approval_alert_failed")
        self.assertIsNotNone(result["halt_reason"])
        self.assertIn("telegram down", result["halt_reason"])
        # The stored PENDING draft is preserved and surfaced, not hidden.
        self.assertEqual(result["post_id"], "post-1")
        self.assertEqual(result["post"]["post_id"], "post-1")
        self.assertEqual(result["post"]["status"], "approval_alert_failed")
        self.assertIn("telegram down", result["post"]["approval_error"])
        self.assertTrue(result["post"]["grounded"])
        # Draft stored exactly once; alert attempted; nothing published.
        m["store_draft"].assert_called_once()
        m["alert"].assert_called_once()
        m["publish_li"].assert_not_called()
        m["publish_x"].assert_not_called()
        m["delete_q"].assert_not_called()
        # The session/question was not rolled back either.
        self.assertEqual(result["research_question_id"], "qid-fresh")

    def test_hydrated_session_without_sources_falls_back_to_fresh_research(self):
        """A matched session that holds no usable sources must run fresh
        research into the existing session instead of halting."""
        check_result = {
            "matched": True,
            "reason": "exact",
            "question_id": "qid-known",
            "matched_question": {"id": "qid-known"},
            "similarity": None,
        }
        with mocked_pipeline(
            check_result=check_result,
            hydrated_sources=([], []),
            search_side_effect=[[raw_result()]],
        ) as m:
            result = run_research_agent(candidates=[make_candidate()])
        self.assertEqual(result["status"], "posted")
        self.assertEqual(result["stages"]["research"], "ok")
        self.assertEqual(result["research_question_id"], "qid-known")
        m["search"].assert_called_once()
        m["store_s"].assert_called_once()
        self.assertEqual(m["store_s"].call_args.args[0], "qid-known")


if __name__ == "__main__":
    unittest.main()