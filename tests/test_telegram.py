"""
Tests for the Telegram approval message generation (src/llm.py caption
builders) and the report→research-digest wiring (src/agent.py).

The approval message must now:
  - include a concise research summary (research question, key findings, an
    important counterargument/limitation, source count and source references)
    next to the generated drafts and the existing Approve/Reject controls,
    without exposing raw source bodies and without breaking on long or empty
    research results;
  - be bounded *by construction* so the escaped caption never exceeds the
    Telegram limits (sendMessage 4096 / sendPhoto 1024), trimming the least
    important content first (drafts -> research digest -> header) while always
    keeping the approval context and the Draft ID footer;
  - escape all dynamic content for legacy Telegram Markdown (``* _ [``
    backtick and backslash) without escaping the code's own static format
    markers and without corrupting URLs.

All network calls (httpx) are mocked – no real Telegram traffic.
"""

from __future__ import annotations

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

from src.agent import _research_summary_for_report  # noqa: E402
from src.llm import (  # noqa: E402
    _HEADER_SOURCE_MAX,
    _HEADER_URL_MAX,
    _MAX_CAPTION_PHOTO,
    _MAX_CAPTION_MESSAGE,
    _RESEARCH_QUESTION_MAX,
    _RESEARCH_SECTION_MAX,
    _build_caption,
    _build_header,
    _build_inline_keyboard,
    _build_research_section,
    _clip_md_bounded,
    _clip_md_safe,
    _unescaped_backticks,
    send_telegram_alert,
    send_telegram_markdown,
    send_telegram_text,
)
from src.models import ClaimAnalysis, Counterargument, ResearchReport, ResearchSource  # noqa: E402

ARTICLE = {
    "source": "ex.com",
    "title": "Agentic orchestration",
    "url": "https://ex.com/article",
}

RESEARCH = {
    "research_question": "Which orchestration framework scales best for production agents?",
    "key_findings": [
        "Teams adopting event-driven architectures grew by 73% in 2026.",
        "LangGraph handles large scale.",
    ],
    "counterargument": "Costs may outweigh gains.",
    "limitation": "Benchmarks come from vendor blogs.",
    "source_count": 2,
    "source_urls": ["https://ex.com/one", "https://ex.com/two"],
}

SECRET_RAW_BODY = "Revenue tripled, and the vendor's ethics claims are unverified in 2026."

LINKEDIN_DRAFT = "Event-driven adoption grew by 73%. LangGraph handles large scale. #AIEngineering"
X_DRAFT = "Adoption grew by 73%. #AgenticAI"

_MD_ESCAPABLE = "\\*_[`"


def assert_markdown_safe(testcase: unittest.TestCase, text: str) -> None:
    """Assert *text* is well-formed legacy Telegram Markdown on every boundary:
    no dangling escape character and no unmatched backtick. A backslash must be
    followed by an escapeable character, and an escape consumes *both* chars
    (so ``\\\\``, an escaped literal backslash, is valid and the char after the
    pair is plain content). Avoids duplicating fragile per-test string checks.
    """
    index = 0
    while index < len(text):
        if text[index] != "\\":
            index += 1
            continue
        testcase.assertLess(
            index + 1, len(text), f"trailing dangling backslash in {text!r}"
        )
        testcase.assertIn(
            text[index + 1],
            _MD_ESCAPABLE,
            f"backslash not part of an escape sequence in {text!r}",
        )
        index += 2
    testcase.assertEqual(
        text.count("`") % 2, 0, f"unmatched backtick(s) in {text!r}"
    )


def assert_no_dangling_escape(testcase: unittest.TestCase, text: str) -> None:
    """Assert *text* is escaped/well-formed on every boundary: no dangling
    backslash and no backslash followed by a non-special, with ``\\`` (escaped
    literal backslash) counted as one valid two-char escape. Unlike
    assert_markdown_safe it does *not* require even backtick parity, because
    escaped literal backticks (``\\` ``) legitimately contribute single
    backtick characters.
    """
    index = 0
    while index < len(text):
        if text[index] != "\\":
            index += 1
            continue
        testcase.assertLess(
            index + 1, len(text), f"trailing dangling backslash in {text!r}"
        )
        testcase.assertIn(
            text[index + 1],
            _MD_ESCAPABLE,
            f"backslash not part of an escape sequence in {text!r}",
        )
        index += 2


def unescape_legacy(text: str) -> str:
    """Undo legacy Markdown escaping: each ``\\X`` pair becomes the character X.

    Used to prove a rendered (escaped) value reads back as its original visible
    form – e.g. that ``a\\\\b`` renders as ``a\\b``.
    """
    out = []
    index = 0
    while index < len(text):
        if text[index] == "\\" and index + 1 < len(text):
            out.append(text[index + 1])
            index += 2
        else:
            out.append(text[index])
            index += 1
    return "".join(out)


def make_report() -> ResearchReport:
    return ResearchReport(
        topic="Agentic orchestration",
        research_question="Which orchestration framework scales best for production agents?",
        findings=[
            ClaimAnalysis(
                claim="Teams adopting event-driven architectures grew by 73% in 2026.",
                classification=ClaimAnalysis.CLASSIFICATION_FACT,
            ),
            ClaimAnalysis(
                claim="LangGraph handles large scale.",
                classification=ClaimAnalysis.CLASSIFICATION_FACT,
            ),
        ],
        counterarguments=[
            Counterargument(argument="Costs may outweigh gains."),
        ],
        limitations=["Benchmarks come from vendor blogs."],
        sources=[
            ResearchSource(url="https://ex.com/one", title="Vendor Report"),
            ResearchSource(url="https://ex.com/two", title="Adversarial Blog"),
        ],
    )


class CaptionContentTest(unittest.TestCase):
    def test_caption_includes_research_summary(self):
        caption = _build_caption("draft-1", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH)

        self.assertIn("Which orchestration framework scales best", caption)
        self.assertIn("event-driven architectures grew by 73%", caption)
        self.assertIn("LangGraph handles large scale", caption)
        self.assertIn("Costs may outweigh gains.", caption)
        self.assertIn("Sources (2)", caption)
        self.assertIn("https://ex.com/one", caption)
        # The generated draft is present too – this is what is being approved.
        self.assertIn(LINKEDIN_DRAFT, caption)
        self.assertIn(X_DRAFT, caption)

    def test_caption_without_research_preserves_existing_shape(self):
        caption = _build_caption("draft-1", LINKEDIN_DRAFT, X_DRAFT, ARTICLE)

        self.assertNotIn("Research question", caption)
        self.assertNotIn("Key findings", caption)
        self.assertNotIn("Sources (", caption)
        self.assertIn("ex.com", caption)
        self.assertIn(LINKEDIN_DRAFT, caption)
        self.assertIn(X_DRAFT, caption)
        self.assertIn("draft-1", caption)

    def test_empty_research_is_omitted(self):
        caption = _build_caption("draft-1", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, {})
        self.assertNotIn("Research question", caption)
        self.assertNotIn("Sources (", caption)
        self.assertIn(LINKEDIN_DRAFT, caption)

    def test_full_drafts_kept_when_no_research(self):
        without = _build_caption("d", LINKEDIN_DRAFT * 4, X_DRAFT * 4, ARTICLE, None)
        self.assertGreater(without.count(LINKEDIN_DRAFT), 1)

    def test_raw_source_bodies_never_leak_into_caption(self):
        report = make_report()
        report.sources[0].body = SECRET_RAW_BODY
        research = _research_summary_for_report(report)
        caption = _build_caption("draft-1", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, research)
        # The digest exposes only claims/questions/caveats/URLs – never bodies.
        self.assertNotIn("Revenue tripled", caption)
        self.assertIn("https://ex.com/one", caption)


class ResearchSectionBoundsTest(unittest.TestCase):
    def test_long_research_results_are_bounded(self):
        long_finding = "A very long finding sentence repeated many times to inflate. " * 20
        long_caveat = "An extremely long caveat about vendor bias repeated. " * 20
        research = {
            "research_question": "Q?" * 200,
            "key_findings": [f"Finding #{i}: {long_finding}" for i in range(8)],
            "counterargument": long_caveat,
            "source_count": 20,
            "source_urls": [f"https://ex.com/a{i}/a-very-long-path" for i in range(20)],
        }
        section = _build_research_section(research)
        self.assertLessEqual(len(section), _RESEARCH_SECTION_MAX)
        # At most the first three findings survive the bounding, never more.
        self.assertIn("Finding #0:", section)
        self.assertIn("Finding #1:", section)
        self.assertIn("Finding #2:", section)
        self.assertNotIn("Finding #3:", section)
        self.assertEqual(section.count("Finding #"), 3)
        self.assertIn("Sources (20)", section)
        self.assertIn("https://ex.com/a0", section)
        self.assertNotIn("https://ex.com/a19", section)

    def test_caption_stays_within_telegram_limits(self):
        huge_findings = ["word " * 500] * 8
        research = {
            "research_question": "word " * 400,
            "key_findings": huge_findings,
            "counterargument": "word " * 400,
            "source_count": 50,
            "source_urls": ["https://ex.com/" + "x" * 300 for _ in range(50)],
        }
        caption = _build_caption(
            "draft-1", LINKEDIN_DRAFT * 30, X_DRAFT * 30, ARTICLE, research
        )
        self.assertLessEqual(len(caption), _MAX_CAPTION_MESSAGE)


class ReportSummaryTest(unittest.TestCase):
    def test_summary_builds_from_report(self):
        summary = _research_summary_for_report(make_report())

        self.assertEqual(
            summary["research_question"],
            "Which orchestration framework scales best for production agents?",
        )
        self.assertEqual(len(summary["key_findings"]), 2)
        self.assertIn("LangGraph handles large scale.", summary["key_findings"])
        self.assertEqual(summary["counterargument"], "Costs may outweigh gains.")
        self.assertEqual(summary["limitation"], "Benchmarks come from vendor blogs.")
        self.assertEqual(summary["source_count"], 2)
        self.assertEqual(summary["source_urls"], ["https://ex.com/one", "https://ex.com/two"])
        # No raw source body content leaks into the digest.
        self.assertNotIn("Revenue tripled", " ".join(summary["key_findings"]))
        self.assertNotIn(
            "per the vendor report",
            " ".join(summary["key_findings"]) + str(summary["counterargument"]),
        )

    def test_summary_findings_capped_at_three(self):
        report = make_report()
        report.findings = [
            ClaimAnalysis(claim=f"Finding {i}.", classification=ClaimAnalysis.CLASSIFICATION_FACT)
            for i in range(8)
        ]
        summary = _research_summary_for_report(report)
        self.assertEqual(len(summary["key_findings"]), 3)

    def test_empty_report_summary_is_graceful(self):
        summary = _research_summary_for_report(ResearchReport(topic="Agentic orchestration"))

        self.assertEqual(summary["key_findings"], [])
        self.assertIsNone(summary["counterargument"])
        self.assertIsNone(summary["limitation"])
        self.assertEqual(summary["source_count"], 0)
        self.assertEqual(summary["source_urls"], [])

        caption = _build_caption("draft-1", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, summary)
        self.assertIn("Sources (0)", caption)
        self.assertIn(LINKEDIN_DRAFT, caption)


class AlertControlsTest(unittest.TestCase):
    def test_inline_keyboard_has_approve_and_reject(self):
        keyboard = _build_inline_keyboard("post-42")
        buttons = keyboard["inline_keyboard"][0]
        self.assertEqual(len(buttons), 2)
        self.assertEqual(buttons[0]["text"], "✅ Approve & Publish")
        self.assertEqual(buttons[0]["callback_data"], "approve_post-42")
        self.assertEqual(buttons[1]["text"], "❌ Reject")
        self.assertEqual(buttons[1]["callback_data"], "reject_post-42")

    def test_alert_sends_message_with_research_and_controls(self):
        with patch("src.llm.httpx.Client") as mock_cls:
            client = mock_cls.return_value
            client.__enter__.return_value = client
            client.post.return_value.raise_for_status.return_value = None
            client.post.return_value.json.return_value = {"ok": True}

            send_telegram_alert(
                "post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH
            )

        (url,), kwargs = client.post.call_args
        self.assertIn("sendMessage", url)
        payload = kwargs["json"]
        self.assertEqual(payload["chat_id"], "12345")
        caption = payload["text"]
        self.assertLessEqual(len(caption), _MAX_CAPTION_MESSAGE)
        self.assertIn("Research question", caption)
        self.assertIn("Sources (2)", caption)
        self.assertIn(LINKEDIN_DRAFT, caption)
        markup = payload["reply_markup"]
        self.assertEqual(markup["inline_keyboard"][0][0]["callback_data"], "approve_post-42")
        self.assertEqual(markup["inline_keyboard"][0][1]["callback_data"], "reject_post-42")

    def test_alert_photo_caption_clamped_and_still_research_rich(self):
        image = b"\x00" * 32
        with patch("src.llm.httpx.Client") as mock_cls:
            client = mock_cls.return_value
            client.__enter__.return_value = client
            client.post.return_value.raise_for_status.return_value = None
            client.post.return_value.json.return_value = {"ok": True}

            send_telegram_alert(
                "post-42", LINKEDIN_DRAFT * 30, X_DRAFT * 30, ARTICLE, RESEARCH, image
            )

        (url,), kwargs = client.post.call_args
        self.assertIn("sendPhoto", url)
        caption = kwargs["data"]["caption"]
        self.assertLessEqual(len(caption), _MAX_CAPTION_PHOTO)
        # Research digest survives even when the drafts are huge.
        self.assertIn("Research question", caption)
        self.assertIn("Sources (2)", caption)


class CaptionLimitsTest(unittest.TestCase):
    """The caption is bounded to the transport limit *by construction*.

    Trim priority: draft content -> research digest -> header detail; the
    approval context header and the Draft ID footer always survive.
    """

    HUGE_LI = LINKEDIN_DRAFT * 30
    HUGE_X = X_DRAFT * 30

    def test_message_caption_bounded_to_4096(self):
        caption = _build_caption("post-42", self.HUGE_LI, self.HUGE_X, ARTICLE, RESEARCH)
        self.assertLessEqual(len(caption), _MAX_CAPTION_MESSAGE)

    def test_photo_caption_bounded_to_1024(self):
        caption = _build_caption(
            "post-42", self.HUGE_LI, self.HUGE_X, ARTICLE, RESEARCH,
            max_length=_MAX_CAPTION_PHOTO,
        )
        self.assertLessEqual(len(caption), _MAX_CAPTION_PHOTO)

    def test_photo_caption_keeps_header_and_draft_id(self):
        caption = _build_caption(
            "post-42", self.HUGE_LI, self.HUGE_X, ARTICLE, RESEARCH,
            max_length=_MAX_CAPTION_PHOTO,
        )
        self.assertIn("New Draft Ready for Approval", caption)
        self.assertIn("post-42", caption)

    def test_draft_id_pinned_under_tight_cap(self):
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=40)
        self.assertIn("Draft ID", caption)
        self.assertIn("`post-42`", caption)
        self.assertNotIn("New Draft", caption)
        self.assertNotIn("LinkedIn Draft", caption)
        self.assertNotIn("Research question", caption)

    def test_question_survives_when_findings_trimmed(self):
        # Inside the digest, the question outlasts findings...
        section = _build_research_section(RESEARCH, 150)
        self.assertIn("Research question", section)
        self.assertNotIn("Key findings", section)
        # ...and in the full caption, caveat and sources are trimmed before
        # the question is ever at risk.
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=400)
        self.assertIn("Research question", caption)
        self.assertNotIn("Caveat", caption)
        self.assertNotIn("Sources (", caption)

    def test_digest_keeps_caveat_longer_than_sources(self):
        section = _build_research_section(RESEARCH, 250)
        self.assertIn("Caveat", section)
        self.assertNotIn("Sources (", section)

    def test_sources_preserved_before_drafts_under_photo_cap(self):
        caption = _build_caption(
            "post-42", self.HUGE_LI, self.HUGE_X, ARTICLE, RESEARCH,
            max_length=_MAX_CAPTION_PHOTO,
        )
        self.assertIn("Sources (2)", caption)
        self.assertIn("https://ex.com/one", caption)
        # Drafts are the lowest-priority text: they get trimmed, the digest not.
        self.assertLess(caption.count("#AIEngineering"), 30)

    def test_tiny_cap_falls_back_to_clipped_footer(self):
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=5)
        self.assertLessEqual(len(caption), 5)
        self.assertTrue(caption.endswith("…"))

    def test_exact_boundary_is_respected(self):
        full = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH)
        self.assertLessEqual(len(full), _MAX_CAPTION_MESSAGE)
        self.assertEqual(
            _build_caption(
                "post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH,
                max_length=len(full),
            ),
            full,
        )


class MarkdownSafetyTest(unittest.TestCase):
    """Dynamic content is escaped for Telegram Markdown; static markers are not."""

    def test_static_markers_left_unescaped(self):
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH)
        self.assertIn("*LinkedIn Draft*", caption)
        self.assertNotIn("\\*LinkedIn Draft\\*", caption)
        self.assertIn("🔎 *Key findings:*", caption)

    def test_underscores_and_stars_escaped_in_drafts(self):
        caption = _build_caption("post-42", "growth_rate *stars* here", X_DRAFT, ARTICLE, RESEARCH)
        self.assertIn("growth\\_rate", caption)
        self.assertIn("\\*stars\\*", caption)

    def test_brackets_escaped_in_title(self):
        caption = _build_caption(
            "post-42", LINKEDIN_DRAFT, X_DRAFT,
            {**ARTICLE, "title": "A [bracketed] title"}, RESEARCH,
        )
        self.assertIn("A \\[bracketed] title", caption)

    def test_backticks_escaped_in_drafts(self):
        caption = _build_caption("post-42", "code: `x` now", X_DRAFT, ARTICLE, RESEARCH)
        self.assertIn("\\`x\\`", caption)

    def test_backslash_escaped_first(self):
        caption = _build_caption("post-42", r"path\to\file", X_DRAFT, ARTICLE, RESEARCH)
        self.assertIn(r"path\\to\\file", caption)

    def test_truncation_never_leaves_dangling_escape(self):
        snippet = _clip_md_bounded("a_b" * 200, 600)
        for index, char in enumerate(snippet):
            if char == "\\":
                self.assertLess(index + 1, len(snippet))
                self.assertIn(snippet[index + 1], "\\*_[`")

    def test_article_title_escaped(self):
        caption = _build_caption(
            "post-42", LINKEDIN_DRAFT, X_DRAFT,
            {**ARTICLE, "title": "Title _with_ [specials]"}, RESEARCH,
        )
        self.assertIn("Title \\_with\\_ \\[specials]", caption)

    def test_source_and_draft_id_escaped(self):
        caption = _build_caption(
            "post_42", LINKEDIN_DRAFT, X_DRAFT,
            {**ARTICLE, "source": "news_site.co"}, RESEARCH,
        )
        self.assertIn("news\\_site.co", caption)
        self.assertIn("post\\_42", caption)


class UrlHandlingTest(unittest.TestCase):
    """URLs are escaped in place – never corrupted, never turned into links."""

    def test_url_with_specials_escaped_but_preserved(self):
        article = {**ARTICLE, "url": "https://ex.com/a_b[c]d"}
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, article, RESEARCH)
        self.assertIn("https://ex.com/a\\_b\\[c]d", caption)
        # Stripping the escapes restores the original visible URL.
        self.assertIn("https://ex.com/a_b[c]d", caption.replace("\\", ""))

    def test_no_markdown_links_introduced(self):
        article = {**ARTICLE, "url": "https://ex.com/a_b"}
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, article, RESEARCH)
        self.assertNotIn("](http", caption)
        self.assertIn("https://ex.com/a\\_b", caption)

    def test_research_source_urls_escaped(self):
        research = {
            **RESEARCH,
            "source_urls": ["https://ex.com/first_report", "https://ex.com/two"],
        }
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, research)
        self.assertIn("https://ex.com/first\\_report", caption)
        self.assertIn("Sources (2)", caption)


class ResearchRegressionTest(unittest.TestCase):
    """The existing digest behavior is preserved alongside bounded captions."""

    def test_full_digest_present_with_escaping(self):
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH)
        for marker in ("Research question", "Key findings", "Caveat", "Sources (2)"):
            self.assertIn(marker, caption)

    def test_raw_source_bodies_with_specials_never_leak(self):
        report = make_report()
        report.sources[0].body = "SECRET_body[1] `query` *result*"
        research = _research_summary_for_report(report)
        caption = _build_caption("draft-1", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, research)
        self.assertNotIn("SECRET", caption)
        self.assertNotIn("Revenue tripled", caption)

    def test_long_digest_bounded_directly(self):
        research = {
            "research_question": "word " * 400,
            "key_findings": ["word " * 500] * 8,
            "counterargument": "word " * 400,
            "source_urls": ["https://ex.com/a" + "x" * 300 for _ in range(20)],
            "source_count": 20,
        }
        self.assertLessEqual(len(_build_research_section(research)), _RESEARCH_SECTION_MAX)

    def test_findings_still_capped_at_three(self):
        research = {
            "research_question": "Q",
            "key_findings": [f"Finding #{i} claim." for i in range(8)],
            "source_count": 0,
            "source_urls": [],
        }
        section = _build_research_section(research)
        self.assertEqual(section.count("Finding #"), 3)

    def test_sources_count_preserved_when_room(self):
        caption = _build_caption(
            "post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH,
            max_length=_MAX_CAPTION_PHOTO,
        )
        self.assertIn("Sources (2):", caption)
        self.assertIn("https://ex.com/one", caption)
        self.assertIn("https://ex.com/two", caption)

    def test_empty_research_graceful_under_photo_cap(self):
        caption = _build_caption(
            "post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, {}, max_length=_MAX_CAPTION_PHOTO
        )
        self.assertNotIn("Research question", caption)
        self.assertNotIn("Sources (", caption)
        self.assertLessEqual(len(caption), _MAX_CAPTION_PHOTO)


class CompatibilityTest(unittest.TestCase):
    """Alert flow, controls and outgoing payloads keep their contract."""

    def test_alert_without_research_keeps_classic_shape(self):
        with patch("src.llm.httpx.Client") as mock_cls:
            client = mock_cls.return_value
            client.__enter__.return_value = client
            client.post.return_value.raise_for_status.return_value = None
            client.post.return_value.json.return_value = {"ok": True}

            send_telegram_alert("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE)

        (url,), kwargs = client.post.call_args
        self.assertIn("sendMessage", url)
        text = kwargs["json"]["text"]
        self.assertNotIn("Research question", text)
        self.assertIn(LINKEDIN_DRAFT, text)
        self.assertIn("post-42", text)

    def test_approve_reject_controls_unchanged(self):
        keyboard = _build_inline_keyboard("post-42")
        row = keyboard["inline_keyboard"][0]
        self.assertEqual(
            [button["callback_data"] for button in row],
            ["approve_post-42", "reject_post-42"],
        )

    def test_send_message_payload_text_within_limit(self):
        with patch("src.llm.httpx.Client") as mock_cls:
            client = mock_cls.return_value
            client.__enter__.return_value = client
            client.post.return_value.raise_for_status.return_value = None
            client.post.return_value.json.return_value = {"ok": True}

            send_telegram_alert(
                "post-42", LINKEDIN_DRAFT * 30, X_DRAFT * 30, ARTICLE, RESEARCH
            )

        (url,), kwargs = client.post.call_args
        self.assertIn("sendMessage", url)
        self.assertLessEqual(len(kwargs["json"]["text"]), _MAX_CAPTION_MESSAGE)

    def test_send_photo_payload_caption_within_limit(self):
        image = b"\x00" * 32
        with patch("src.llm.httpx.Client") as mock_cls:
            client = mock_cls.return_value
            client.__enter__.return_value = client
            client.post.return_value.raise_for_status.return_value = None
            client.post.return_value.json.return_value = {"ok": True}

            send_telegram_alert(
                "post-42", LINKEDIN_DRAFT * 30, X_DRAFT * 30, ARTICLE, RESEARCH, image
            )

        (url,), kwargs = client.post.call_args
        self.assertIn("sendPhoto", url)
        self.assertLessEqual(len(kwargs["data"]["caption"]), _MAX_CAPTION_PHOTO)
        self.assertIn("Sources (2)", kwargs["data"]["caption"])


class TinyCaptionSafetyTest(unittest.TestCase):
    """_build_caption() must stay Markdown-valid and bounded for *every*
    supported positive max_length, including limits too small for the full
    Draft ID footer – no blind post-hoc truncation of formatted text."""

    def test_max_length_one_is_valid_and_bounded(self):
        caption = _build_caption(
            "post_42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=1
        )
        self.assertLessEqual(len(caption), 1)
        assert_markdown_safe(self, caption)

    def test_max_length_two_is_valid_and_bounded(self):
        caption = _build_caption(
            "post_42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=2
        )
        self.assertLessEqual(len(caption), 2)
        assert_markdown_safe(self, caption)

    def test_limit_smaller_than_footer_has_no_malformed_markdown(self):
        full_footer_len = len("🆔 Draft ID: `post\\_42`")
        for limit in range(1, full_footer_len):
            caption = _build_caption(
                "post_42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=limit
            )
            assert_markdown_safe(self, caption)
            self.assertLessEqual(len(caption), limit)

    def test_boundary_exactly_at_minimum_safe_footer(self):
        expected = "🆔 Draft ID: `post\\_42`"
        caption = _build_caption(
            "post_42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH,
            max_length=len(expected),
        )
        self.assertEqual(caption, expected)
        assert_markdown_safe(self, caption)

    def test_no_result_ends_inside_an_escape_sequence(self):
        # Specials everywhere so escape sequences are actually exercised at the
        # truncation boundary; a broken escape would fail assert_markdown_safe.
        for limit in range(1, 60):
            caption = _build_caption(
                "post\\_1[2]", "a_b*c`d", "x[y]_z",
                {**ARTICLE, "title": "t[a]_b", "url": "https://u.co/x_y"}, RESEARCH,
                max_length=limit,
            )
            assert_markdown_safe(self, caption)
            self.assertLessEqual(len(caption), limit)

    def test_no_unmatched_backticks(self):
        for limit in range(1, 40):
            caption = _build_caption(
                "post_42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=limit
            )
            self.assertEqual(caption.count("`") % 2, 0)

    def test_all_small_captions_satisfy_length_bound(self):
        for limit in range(1, 70):
            caption = _build_caption(
                "post_42", LINKEDIN_DRAFT * 3, X_DRAFT * 3, ARTICLE, RESEARCH,
                max_length=limit,
            )
            self.assertLessEqual(len(caption), limit)
            assert_markdown_safe(self, caption)


class GenericTextNotificationTest(unittest.TestCase):
    """send_telegram_text() is a plain-text (Markdown-free) bounded sender;
    the explicit Markdown path survives for callers like the canary."""

    SPECIALS_TEXT = "underscore_ star* [bracket] `tick` and \\backslash\\"

    def _send(self, message):
        with patch("src.llm.httpx.Client") as mock_cls:
            client = mock_cls.return_value
            client.__enter__.return_value = client
            client.post.return_value.raise_for_status.return_value = None
            client.post.return_value.json.return_value = {"ok": True}
            send_telegram_text(message)
        return client.post.call_args

    def test_special_characters_sent_as_plain_text(self):
        (url,), kwargs = self._send(self.SPECIALS_TEXT)
        self.assertIn("sendMessage", url)
        payload = kwargs["json"]
        # Sent verbatim – nothing escaped, nothing Markdown-interpreted.
        self.assertEqual(payload["text"], self.SPECIALS_TEXT)

    def test_long_message_bounded_to_4096(self):
        (url,), kwargs = self._send("word " * 1200)
        payload = kwargs["json"]
        self.assertLessEqual(len(payload["text"]), _MAX_CAPTION_MESSAGE)

    def test_outgoing_payload_is_asserted_not_intermediate(self):
        (url,), kwargs = self._send("x" * 9000)
        payload = kwargs["json"]
        self.assertEqual(payload["chat_id"], "12345")
        self.assertLessEqual(len(payload["text"]), _MAX_CAPTION_MESSAGE)
        self.assertNotEqual(payload["text"], "x" * 9000)

    def test_generic_path_sets_no_markdown_parsing(self):
        (url,), kwargs = self._send("plain *text* _not_ marked [verbatim]")
        self.assertNotIn("parse_mode", kwargs["json"])

    def test_canary_markdown_path_still_works(self):
        message = "🚨 *URGENT – expired on 2026-01-01*.\n_Pipeline halted._"
        with patch("src.llm.httpx.Client") as mock_cls:
            client = mock_cls.return_value
            client.__enter__.return_value = client
            client.post.return_value.raise_for_status.return_value = None
            client.post.return_value.json.return_value = {"ok": True}

            send_telegram_markdown(message)

        (url,), kwargs = client.post.call_args
        self.assertIn("sendMessage", url)
        payload = kwargs["json"]
        self.assertEqual(payload["parse_mode"], "Markdown")
        self.assertEqual(payload["text"], message)

    def test_markdown_bound_never_splits_escape_or_backtick(self):
        markdown = "x\\_y `inline` *bold* _em_ " * 500
        for limit in range(1, 120):
            clipped = _clip_md_safe(markdown, limit)
            self.assertLessEqual(len(clipped), limit)
            self.assertFalse(clipped.endswith("\\"))
            self.assertEqual(clipped.count("`") % 2, 0)

    def test_parentheses_and_specials_sent_verbatim_plain(self):
        message = "call foo(bar) with _x_ *y* [z] `t` (nested] now"
        (url,), kwargs = self._send(message)
        payload = kwargs["json"]
        self.assertEqual(payload["text"], message)
        self.assertNotIn("parse_mode", payload)

    def test_markdown_long_message_clipped_within_limit(self):
        markdown = "🚨 *URGENT* `code` _em_ [x](tg://user?id=123)\n" * 300
        with patch("src.llm.httpx.Client") as mock_cls:
            client = mock_cls.return_value
            client.__enter__.return_value = client
            client.post.return_value.raise_for_status.return_value = None
            client.post.return_value.json.return_value = {"ok": True}

            send_telegram_markdown(markdown)

        (url,), kwargs = client.post.call_args
        self.assertIn("sendMessage", url)
        payload = kwargs["json"]
        self.assertEqual(payload["parse_mode"], "Markdown")
        self.assertLessEqual(len(payload["text"]), _MAX_CAPTION_MESSAGE)
        self.assertNotEqual(payload["text"], markdown)
        assert_markdown_safe(self, payload["text"])


class EscapedLengthBoundsTest(unittest.TestCase):
    """The bounded clipper measures the *escaped* output, never the raw input:
    len(_clip_md_bounded(value, budget)) <= budget holds even when escaping
    doubles the string (the regression this test class locks in)."""

    def test_special_heavy_content_never_exceeds_budget(self):
        for budget in (4, 8, 12, 20, 30):
            for value in ("_" * 10, "*" * 10, "[" * 10):
                clipped = _clip_md_bounded(value, budget)
                self.assertLessEqual(len(clipped), budget)
                assert_markdown_safe(self, clipped)

    def test_mixed_specials_never_exceed_budget_across_sweep(self):
        value = "a_b*c[d]`e\\f_"
        for budget in range(1, 60):
            self.assertLessEqual(len(_clip_md_bounded(value, budget)), budget)

    def test_escaped_length_is_the_measured_amount(self):
        # 10 underscores escape to 20 chars; an 8-char budget must not yield 20.
        clipped = _clip_md_bounded("__________", 8)
        self.assertLessEqual(len(clipped), 8)
        self.assertIn("\\_", clipped)
        self.assertEqual(clipped.replace("\\", ""), "_" * 4)

    def test_clip_boundary_never_lands_inside_an_escape(self):
        for budget in (10, 11, 12, 13, 14):
            clipped = _clip_md_bounded("aa_bb*cc`", budget)
            self.assertLessEqual(len(clipped), budget)
            assert_no_dangling_escape(self, clipped)

    def test_short_values_not_truncated(self):
        self.assertEqual(_clip_md_bounded("post_42", 20), "post\\_42")
        self.assertEqual(_clip_md_bounded("plain", 20), "plain")

    def test_truncated_value_signals_with_ellipsis_within_budget(self):
        clipped = _clip_md_bounded("abc__def", 6)
        self.assertLessEqual(len(clipped), 6)
        self.assertTrue(clipped.endswith("…"))
        self.assertEqual(clipped.replace("\\", ""), "abc_…")


class ResearchSpecialsTest(unittest.TestCase):
    """Research fields with every legacy special are escaped, stay within their
    field budgets, and keep the digest within the section budget."""

    QUESTION = "AI_agent_system_[v2]_uses_*tools*_(MCP)_and `context`"
    EXPECTED_QUESTION = (
        "AI\\_agent\\_system\\_\\[v2]\\_uses\\_\\*tools\\*\\_(MCP)\\_and \\`context\\`"
    )

    def test_question_specials_escaped_and_field_bounded(self):
        field = _clip_md_bounded(self.QUESTION, _RESEARCH_QUESTION_MAX)
        self.assertLessEqual(len(field), _RESEARCH_QUESTION_MAX)
        self.assertEqual(field, self.EXPECTED_QUESTION)

        section = _build_research_section({**RESEARCH, "research_question": self.QUESTION})
        self.assertLessEqual(len(section), _RESEARCH_SECTION_MAX)
        self.assertIn(self.EXPECTED_QUESTION, section)
        assert_markdown_safe(self, section)

    def test_findings_and_caveat_with_specials_bounded(self):
        research = {
            **RESEARCH,
            "key_findings": ["Result_*:[snippet] `x` grew 73%"],
            "counterargument": "trade-off_(value) vs *complexity* might `regress`",
        }
        section = _build_research_section(research)
        self.assertLessEqual(len(section), _RESEARCH_SECTION_MAX)
        assert_markdown_safe(self, section)
        self.assertIn("Result\\_\\*:\\[snippet] \\`x\\` grew 73%", section)
        self.assertIn(
            "trade-off\\_(value) vs \\*complexity\\* might \\`regress\\`", section
        )

    def test_source_urls_escaped_never_form_links(self):
        urls = [
            "https://example.com/path_(test)",
            "https://example.com/search?q=a_b",
            "https://example.com/[section]/item",
            "https://example.com/a`b",
            "https://example.com/a\\b",
        ]
        section = _build_research_section(
            {**RESEARCH, "source_urls": urls, "source_count": len(urls)}
        )
        self.assertLessEqual(len(section), _RESEARCH_SECTION_MAX)
        # The backticks in URLs are escaped, so only single raw backtick
        # chars remain; parity is not a validity signal for escaped content.
        assert_no_dangling_escape(self, section)
        self.assertNotIn("](", section)
        for url in urls:
            self.assertIn(url, unescape_legacy(section))

    def test_header_url_with_parens_and_specials_bounded(self):
        article = {**ARTICLE, "url": "https://example.com/a_(b)_[c]"}
        caption = _build_caption(
            "post-42", LINKEDIN_DRAFT, X_DRAFT, article, RESEARCH,
            max_length=_MAX_CAPTION_PHOTO,
        )
        self.assertIn("https://example.com/a\\_(b)\\_\\[c]", caption)
        self.assertIn("https://example.com/a_(b)_[c]", caption.replace("\\", ""))
        assert_markdown_safe(self, caption)


class UrlDialectTest(unittest.TestCase):
    """Dynamic URL-shaped text cannot open a Markdown link; legacy dialect
    leaves ``(`` ``)`` ``]`` as literal characters (not escaped) so visible
    URLs and ``foo(bar)`` survive verbatim after rendering."""

    def test_link_like_draft_cannot_form_a_link(self):
        caption = _build_caption(
            "post-42", "[malicious](https://example.com)", X_DRAFT, ARTICLE, RESEARCH
        )
        self.assertIn("\\[malicious](https://example.com)", caption)
        self.assertIn("[malicious](https://example.com)", caption.replace("\\", ""))
        assert_markdown_safe(self, caption)

    def test_parentheses_are_literal_legacy_characters(self):
        caption = _build_caption("post-42", "foo(bar)_x", X_DRAFT, ARTICLE, RESEARCH)
        self.assertIn("foo(bar)\\_x", caption)
        self.assertIn("foo(bar)_x", caption.replace("\\", ""))
        assert_markdown_safe(self, caption)

    def test_windows_path_backslashes_escaped(self):
        caption = _build_caption("post-42", r"C:\Users\Test\file", X_DRAFT, ARTICLE, RESEARCH)
        self.assertIn(r"C:\\Users\\Test\\file", caption)
        assert_markdown_safe(self, caption)

    def test_backticks_in_drafts_escaped_and_paired(self):
        caption = _build_caption("post-42", "use `code` now", X_DRAFT, ARTICLE, RESEARCH)
        self.assertIn("use \\`code\\` now", caption)
        self.assertEqual(caption.count("`") % 2, 0)
        assert_markdown_safe(self, caption)

    def test_url_inventory_escaped_bounded_and_readable(self):
        urls = [
            "https://example.com/path_(test)",
            "https://example.com/search?q=a_b",
            "https://example.com/[section]/item",
            "https://example.com/a`b",
            "https://example.com/a*b",
            "https://example.com/a\\b",
        ]
        for url in urls:
            escaped = _clip_md_bounded(url, _HEADER_URL_MAX)
            self.assertLessEqual(len(escaped), _HEADER_URL_MAX)
            # The escaped form is well-formed and reads back as the original.
            assert_no_dangling_escape(self, escaped)
            self.assertEqual(unescape_legacy(escaped), url)
            # An escaped `[` keeps the URL inert: it cannot start a link.
            self.assertNotIn("](", escaped)


class GlobalLimitExpansionTest(unittest.TestCase):
    """Markdown-heavy content grows when escaped; both global Telegram caps
    still bind the fully-escaped caption."""

    HEAVY = "_star_ [bracket] `tick` \\path\\ *emphasis*"

    def test_photo_cap_1024_with_escaping_expansion(self):
        caption = _build_caption(
            "post-42", self.HEAVY * 60, self.HEAVY * 60, ARTICLE, RESEARCH,
            max_length=_MAX_CAPTION_PHOTO,
        )
        self.assertLessEqual(len(caption), _MAX_CAPTION_PHOTO)
        assert_no_dangling_escape(self, caption)

    def test_message_cap_4096_with_escaping_expansion(self):
        caption = _build_caption(
            "post-42", self.HEAVY * 60, self.HEAVY * 60, ARTICLE, RESEARCH,
        )
        self.assertLessEqual(len(caption), _MAX_CAPTION_MESSAGE)
        assert_no_dangling_escape(self, caption)

    def test_research_section_expansion_stays_bounded(self):
        research = {**RESEARCH, "research_question": "a_b*c[d]`e`(" * 60}
        section = _build_research_section(research)
        self.assertLessEqual(len(section), _RESEARCH_SECTION_MAX)
        assert_no_dangling_escape(self, section)


class TinyAndEdgeLimitTest(unittest.TestCase):
    """max_length is honored at its very edges (0, 1, 2, 3, 4, 5, 10, 20)."""

    def test_zero_and_negative_limits_return_empty(self):
        self.assertEqual(
            _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=0),
            "",
        )
        self.assertEqual(
            _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=-1),
            "",
        )

    def test_documented_tiny_limits_all_bounded_and_safe(self):
        for limit in (1, 2, 3, 4, 5, 10, 20):
            caption = _build_caption(
                "post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH, max_length=limit
            )
            self.assertLessEqual(len(caption), limit)
            assert_markdown_safe(self, caption)


class HeaderSourceBoundsTest(unittest.TestCase):
    """Every dynamic header field (source, title, URL) is escaped AND bounded;
    the source line must satisfy len(rendered) <= _HEADER_SOURCE_MAX."""

    LONG_SOURCE = "incredibly_long_source_name_with_*stars*`ticks`\\paths " * 40
    SPECIALS_SOURCE = "news_site_[v2]*stars*`tick`\\path_(sub)"

    @staticmethod
    def _rendered_source(caption: str) -> str:
        lines = caption.split("\n")
        source_line = next(line for line in lines if line.startswith("📰 *Source:*"))
        return source_line[len("📰 *Source:* "):]

    def test_extremely_long_source_is_bounded(self):
        caption = _build_caption(
            "post-42", LINKEDIN_DRAFT, X_DRAFT, {**ARTICLE, "source": self.LONG_SOURCE}, RESEARCH
        )
        self.assertLessEqual(len(caption), _MAX_CAPTION_MESSAGE)
        rendered = self._rendered_source(caption)
        self.assertLessEqual(len(rendered), _HEADER_SOURCE_MAX)
        assert_no_dangling_escape(self, caption)

    def test_source_with_specials_escaped_and_bounded(self):
        caption = _build_caption(
            "post-42", LINKEDIN_DRAFT, X_DRAFT, {**ARTICLE, "source": self.SPECIALS_SOURCE}, RESEARCH
        )
        rendered = self._rendered_source(caption)
        self.assertLessEqual(len(rendered), _HEADER_SOURCE_MAX)
        # legacy dialect: only `_ * [ ` \` are escaped; ( ) and ] are literal
        self.assertEqual(unescape_legacy(rendered), self.SPECIALS_SOURCE)
        self.assertIn("news\\_site\\_\\[v2]\\*stars\\*\\`tick\\`\\\\path\\_(sub)", rendered)
        assert_markdown_safe(self, caption)

    def test_source_only_header_mode_bounded(self):
        # Direct _build_header mode 3 = source only; still bounded.
        header = _build_header({**ARTICLE, "source": self.LONG_SOURCE}, 3)
        source_line = next(line for line in header.split("\n") if line.startswith("📰 *Source:*"))
        rendered = source_line[len("📰 *Source:* "):]
        self.assertLessEqual(len(rendered), _HEADER_SOURCE_MAX)
        assert_no_dangling_escape(self, header)


class ClipMdSafeScannerTest(unittest.TestCase):
    """_clip_md_safe() clips *already-formatted* Markdown with an escape-aware
    scanner: ``\\` `` is a literal backtick, never a code delimiter."""

    def test_escaped_backtick_is_never_a_delimiter(self):
        markdown = "a \\`b\\` z " * 200
        for limit in (7, 15, 40, 100):
            clipped = _clip_md_safe(markdown, limit)
            self.assertLessEqual(len(clipped), limit)
            self.assertEqual(len(_unescaped_backticks(clipped)) % 2, 0)
            self.assertIn("\\`", clipped)

    def test_escaped_backtick_pair_preserved_when_fits(self):
        markdown = "text \\`code\\` text"
        self.assertEqual(_clip_md_safe(markdown, len(markdown) + 10), markdown)

    def test_real_backtick_pair_left_intact_when_fits(self):
        markdown = "text `code` text"
        self.assertEqual(_clip_md_safe(markdown, len(markdown)), markdown)

    def test_unmatched_real_backtick_removed_on_clip(self):
        markdown = "text `code" + " pad" * 50
        clipped = _clip_md_safe(markdown, 9)
        self.assertLessEqual(len(clipped), 9)
        self.assertNotIn("`", clipped)
        self.assertEqual(len(_unescaped_backticks(clipped)) % 2, 0)

    def test_clip_before_backtick_keeps_clean_prefix(self):
        self.assertEqual(_clip_md_safe("abc`def", 3), "abc")

    def test_clip_exactly_on_backtick_drops_delimiter(self):
        self.assertEqual(_clip_md_safe("abc`def", 4), "abc")

    def test_clip_just_after_backtick_drops_opened_span(self):
        self.assertEqual(_clip_md_safe("abc`def", 5), "abc")

    def test_clip_after_escaped_backtick_keeps_literal(self):
        self.assertEqual(_clip_md_safe("ab\\`cd", 4), "ab\\`")

    def test_dangling_backslash_at_edge_dropped(self):
        self.assertEqual(_clip_md_safe("ab\\cdef", 3), "ab")

    def test_escaped_backslash_pair_kept_at_edge(self):
        self.assertEqual(_clip_md_safe("ab\\\\cd", 4), "ab\\\\")

    def test_odd_trailing_backslash_run_trimmed_to_couples(self):
        self.assertEqual(_clip_md_safe("ab\\\\\\cd", 5), "ab\\\\")

    def test_escaped_char_units_survive_clipping(self):
        self.assertEqual(_clip_md_safe("x\\_y", 6), "x\\_y")
        self.assertEqual(_clip_md_safe("x\\*y", 6), "x\\*y")
        self.assertEqual(_clip_md_safe("x\\[y", 6), "x\\[y")
        self.assertEqual(_clip_md_safe("x\\]y", 6), "x\\]y")

    def test_escape_never_split_across_boundary(self):
        self.assertEqual(_clip_md_safe("ab\\_cd", 3), "ab")
        self.assertEqual(_clip_md_safe("ab\\_cd", 4), "ab\\_")

    def test_mixed_escapes_and_backticks_clip_cleanly(self):
        markdown = "a\\_b `c\\`d` *e* " * 60
        for limit in range(1, 200):
            clipped = _clip_md_safe(markdown, limit)
            self.assertLessEqual(len(clipped), limit)
            run = len(clipped) - len(clipped.rstrip("\\"))
            self.assertEqual(run % 2, 0, f"dangling backslash in {clipped!r}")
            self.assertEqual(len(_unescaped_backticks(clipped)) % 2, 0, clipped)

    def test_nonpositive_limits(self):
        self.assertEqual(_clip_md_safe("`code` xyz *x*", 0), "")
        self.assertEqual(_clip_md_safe("`code` xyz", -3), "")
        self.assertEqual(_clip_md_safe("`code` xyz", 1), "")


class GeneratedVsDynamicTest(unittest.TestCase):
    """Generated Markdown markers are preserved; dynamic values are escaped
    exactly once when inserted – the assembled caption is never re-escaped."""

    def test_dynamic_escaped_generated_kept(self):
        article = {**ARTICLE, "title": "Price_Jump *2026* `rev` [x]"}
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, article, RESEARCH)
        for marker in ("*Source:*", "*Topic:*", "*Research question:*", "*Key findings:*"):
            self.assertIn(marker, caption)
        self.assertIn("Price\\_Jump \\*2026\\* \\`rev\\` \\[x]", caption)
        assert_markdown_safe(self, caption)

    def test_dynamic_never_escapes_generated_markers(self):
        caption = _build_caption("post-42", LINKEDIN_DRAFT, X_DRAFT, ARTICLE, RESEARCH)
        self.assertIn("*Topic:* Agentic orchestration", caption)
        self.assertNotIn("\\*Topic:", caption)


class ClipMdSafeWithinLimitTest(unittest.TestCase):
    """Validation is not bypassed just because the text already fits the limit:
    dangling trailing backslashes and unmatched real backticks are repaired
    even under the limit, and valid Markdown is returned unchanged."""

    def test_valid_markdown_below_limit_unchanged(self):
        markdown = "**bold** _em_ `code` text\\_with\\*escapes\\*"
        self.assertEqual(_clip_md_safe(markdown, len(markdown) + 20), markdown)

    def test_trailing_backslash_repaired_within_limit(self):
        self.assertEqual(_clip_md_safe("c:\\path\\", 20), "c:\\path")

    def test_escaped_backticks_preserved_within_limit(self):
        markdown = r"a \`b\` z"
        self.assertEqual(_clip_md_safe(markdown, 20), markdown)
        self.assertEqual(_unescaped_backticks(markdown), [])

    def test_unmatched_real_backtick_repaired_within_limit(self):
        self.assertEqual(_clip_md_safe("text `code", 20), "text ")

    def test_supported_escapes_preserved_within_limit(self):
        markdown = r"a\_b \*c\* \[d\] \`e\`"
        self.assertEqual(_clip_md_safe(markdown, 40), markdown)

    def test_arbitrary_backslashes_are_not_escapes(self):
        markdown = r"\a \q \( x"
        self.assertEqual(_clip_md_safe(markdown, 20), markdown)
        # \q is an arbitrary backslash: the following backtick is a real one.
        self.assertEqual(_unescaped_backticks(r"a\q`b"), [3])

    def test_unescaped_backtick_scanner_escape_set(self):
        # `\`` is escaped; `\\`` (escaped backslash + bare backtick) is real.
        self.assertEqual(_unescaped_backticks(r"a \`b"), [])
        self.assertEqual(_unescaped_backticks(r"a \\`b"), [4])
        self.assertEqual(_unescaped_backticks("plain `code` z"), [6, 11])

    def test_tiny_limits_stay_safe(self):
        markdown = r"a\_b `c` \q` *e*"
        for limit in (0, 1, 2, 3, 5, 10):
            clipped = _clip_md_safe(markdown, limit)
            self.assertLessEqual(len(clipped), limit)
            run = len(clipped) - len(clipped.rstrip("\\"))
            self.assertEqual(run % 2, 0, f"dangling backslash at limit {limit}")
            self.assertEqual(
                len(_unescaped_backticks(clipped)) % 2, 0, clipped
            )

    def test_output_always_stays_within_limit(self):
        markdown = r"text \`tick\` \q real` and \\ and \* ends"
        for limit in range(0, 40):
            clipped = _clip_md_safe(markdown, limit)
            self.assertLessEqual(len(clipped), limit)


if __name__ == "__main__":
    unittest.main()