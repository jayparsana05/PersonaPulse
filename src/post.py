"""
PersonaPulse – Post Drafting Module (Phase 5)
=============================================
Turns a *ResearchReport* into platform-tailored LinkedIn & X drafts.

Phase 4 introduced report-driven drafting. Phase 5 makes the grounding
STRUCTURAL: the LLM returns the draft as JSON segments, and each factual
segment MUST declare which research findings it is drawn from. A segment is
retained only when its references are valid AND its content is confined to
what the referenced findings carry. This rejects unsupported facts even when
they contain no numbers or URLs.

Flow
----
ResearchReport (validated research)
    ↓ build report prompt (findings + evidence + conflicts + style profile)
    ↓ LLM draft per platform → JSON { "segments": [{"text", "finding_indices"}] }
    ↓ structural segment validation (references + conservative grounding)
    ↓ optional non-factual framing (hooks / questions / CTA / hashtags)
    ↓ number/URL safety check on the joined draft (kept from Phase 4)
    ↓ platform length enforcement in code (LinkedIn 1300–1900, X ≤ 280)
    ↓ deterministic fallback when the LLM fails or the draft is invalid
    ↓ linkedin_draft + x_draft + used_findings (derived from validated segments)

Design guarantees
-----------------
- Structural grounding: every factual segment must reference one or more valid
  RANGE index into ResearchReport.findings. Invalid/missing/non-integer/
  boolean/negative/out-of-range references reject the segment. References are
  resolved ONLY against rpt.findings – arbitrary LLM claims are not trusted.
- Grounding check is per-segment and conservative: the segment's content words
  must all be present (after light stemming) in the referenced findings' claims
  or their evidence; negations must be carried by that evidence; digits must
  exist in the referenced findings' evidence; URLs must belong to the report.
- Non-factual writing (hooks, questions, transitions, CTA, framing, hashtags)
  is allowed WITHOUT references ONLY when it introduces no facts: no digits,
  no URLs, no negation, and every content word is already report vocabulary.
  A non-referenced segment that echoes a single finding's claim is treated as
  a factual restatement and REJECTED (it must carry references).
- used_findings is derived from the findings actually referenced by the
  accepted, grounded factual segments – never from token-overlap heuristics.
- Platform length limits are enforced in code (not just prompts): LinkedIn must
  be within 1300–1900 chars; X must be ≤ 280; violations fall back.
- Fallback is report-derived and therefore report-grounded: "status" and
  "grounded" are separate concepts – a fallback draft reports grounded=True
  while "issues" explains why the LLM result was rejected.
- No publishing: this module never calls publishers / stores / alerts. The
  Telegram approval flow lives in src/agent.py + src/llm.py unchanged.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from src.llm import complete_text
from src.models import ResearchReport
from src.report import (
    _URL_RE,
    _NEGATION_MARKERS,
    _content_tokens,
    _function_stemmed,
    _raw_tokens,
    _url_key,
)
from src.selection import _extract_json

log = logging.getLogger(__name__)

_LINKEDIN_MIN_CHARS = 1300
_LINKEDIN_MAX_CHARS = 1900
_X_MAX_CHARS = 280
_MAX_RESPONSE_TOKENS = 2048
_X_MAX_RESPONSE_TOKENS = 500
_TEMPERATURE = 0.7
# Fraction of a framing segment's tokens that may come from one finding's claim
# before it is treated as a factual restatement (and must carry references).
_MIN_FINDING_COVERAGE = 0.5

_HASHTAG_RE = re.compile(r"#\w+")


# ---------------------------------------------------------------------------
# System prompts (report-driven, per platform, JSON output contract)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_LINKEDIN = """You are a ghostwriter for a senior technology professional with a strong personal brand on LinkedIn.
Your task: turn a RESEARCH REPORT into a high-performing LinkedIn post that feels AUTHENTIC and PERSONAL, not like a news summary.

The report contains validated research findings, evidence, and counterarguments. Base EVERY FACTUAL block of the post ONLY on the FINDINGS under RESEARCH MATERIAL (each finding has a 0-based index in brackets). Do NOT summarise a single source – synthesise the findings.

STRUCTURE (follow this exactly, each section separated by a blank line):
1. HOOK (1-2 lines): A provocative question or a concrete metric drawn from a specific finding. Lead with hard numbers whenever they are present in the findings.
2. CONTEXT (2-3 lines): Ground the post in the research question. What was researched and why does it matter?
3. PERSONAL TAKE (3-5 lines): Your perspective as a tech professional, using "I think…", "In my view…".
4. KEY INSIGHT / LESSONS (3-4 bullet points using ▸ or →): Concrete takeaways, one per finding you reference.
5. CALL TO ACTION (1-2 lines): A thought-provoking question.
6. HASHTAGS (3-5 relevant hashtags on the last line).

OUTPUT CONTRACT – return VALID JSON ONLY. No markdown, no code fences, no preamble.
{{
  "segments": [
    {{"text": "<one block of the post>", "finding_indices": [0]}},
    {{"text": "<a block combining multiple findings>", "finding_indices": [0, 1]}},
    {{"text": "<a hook / question / transition / CTA / hashtag line with no factual content>", "finding_indices": []}}
  ]
}}
RULES:
- "finding_indices" are 0-based indexes into the FINDINGS listed in RESEARCH MATERIAL.
- A block that states a FACT (numbers, metrics, dates, claims, people, institutions, URLs) MUST reference the finding(s) it is drawn from and may ONLY paraphrase the referenced findings.
- Blocks with NO factual content – hooks, questions, transitions, framing, calls to action, hashtags – use "finding_indices": [].
- NEVER invent numbers, percentages, dates, people, institutions, quotes, or URLs. If a detail is not in the referenced findings, omit it.
- If the report lists counterarguments or conflicting evidence, reflect that honestly instead of presenting one side as settled.
- The FINAL post is the segments joined in order with a blank line; it MUST be between {min_chars} and {max_chars} characters measured across all segments combined.

STYLE & LENGTH RULES:
- First-person, conversational yet professional voice. 1-2 emojis per section, not every line.
- Max 3 lines per paragraph; blank lines between sections.
- NEVER use clickbait, excessive exclamation marks, unsubstantiated claims.
"""

_SYSTEM_PROMPT_X = """You are a ghostwriter for a senior technology professional with a strong presence on X (Twitter).
Turn the RESEARCH REPORT's findings into a single punchy, high-engagement tweet.

OUTPUT CONTRACT – return VALID JSON ONLY. No markdown, no code fences, no preamble.
{
  "segments": [
    {"text": "<one part of the tweet>", "finding_indices": [0]},
    {"text": "<hashtag line, no factual content>", "finding_indices": []}
  ]
}
RULES:
- "finding_indices" are 0-based indexes into the FINDINGS listed in RESEARCH MATERIAL.
- Blocks that state a FACT MUST reference the finding(s) they are drawn from and may ONLY paraphrase the referenced findings.
- Blocks with NO factual content – hooks, questions, hashtags – use "finding_indices": [].
- Never invent numbers, quotes, or facts. Lead with the most surprising or important FINDING.
- The FINAL tweet is the segments joined in order with a single newline; it MUST be at most 280 characters in total.
- 1-2 relevant emojis and 1-2 hashtags at the end. First-person, direct voice.
"""


# ---------------------------------------------------------------------------
# Public: Draft both platforms from a ResearchReport
# ---------------------------------------------------------------------------

def draft_report_post(
    report,
    style_profile: Optional[dict] = None,
    use_llm: bool = True,
) -> dict:
    """
    Generate LinkedIn and X drafts from a ResearchReport.

    Parameters
    ----------
    report        : ResearchReport (or its dict serialization).
    style_profile : dict from memory.get_style_profile() (empty when absent).
    use_llm       : False forces the deterministic fallback (no API call).

    Returns
    -------
    dict with keys:
        status         : "ok" (LLM drafts retained, grounded) |
                         "fallback" (LLM failed / draft invalid) | "empty"
        linkedin_draft : str
        x_draft        : str
        used_findings  : list[str] – finding claims referenced by the accepted,
                         grounded factual segments (both platforms)
        grounded       : bool – content is built from the report only
        issues         : list[str] – dismantled reasons (empty when status "ok")
    """
    rpt = report if isinstance(report, ResearchReport) else ResearchReport.from_dict(report or {})
    style = style_profile or {}

    if not rpt.findings and not rpt.evidence and not rpt.synthesis:
        log.warning("[Post] Report has no findings/evidence – producing empty-bounded post.")
        return _empty_result(rpt)

    if use_llm:
        try:
            linkedin = _draft_platform(rpt, style, "linkedin")
            x = _draft_platform(rpt, style, "x")
        except Exception as exc:  # pylint: disable=broad-except
            log.warning(
                "[Post] LLM draft failed (%s: %s) – using deterministic fallback.",
                type(exc).__name__, exc,
            )
            return _fallback_result(rpt, reason=f"LLM failure: {type(exc).__name__}")

        issues = [
            f"linkedin: {issue}" for issue in linkedin["issues"]
        ] + [
            f"x: {issue}" for issue in x["issues"]
        ]
        if not linkedin["ok"] or not x["ok"]:
            log.warning(
                "[Post] Draft invalid (%s) – using deterministic fallback.",
                "; ".join(issues) or "unknown",
            )
            return _fallback_result(rpt, reason="; ".join(issues) or "invalid draft")

        used = _index_claims(rpt, linkedin["refs"] + x["refs"])
        log.info(
            "[Post] Drafted from report: %d finding(s) used | LinkedIn=%d chars | X=%d chars",
            len(used), len(linkedin["text"]), len(x["text"]),
        )
        return {
            "status": "ok",
            "linkedin_draft": linkedin["text"],
            "x_draft": x["text"],
            "used_findings": used,
            "grounded": True,
            "issues": [],
        }

    return _fallback_result(rpt, reason="LLM disabled")


# ---------------------------------------------------------------------------
# Public: validate a structured segments payload
# ---------------------------------------------------------------------------

def validate_draft_segments(segments, report) -> tuple[bool, list[str], list[str]]:
    """
    Structurally validate an LLM "segments" payload against a ResearchReport.

    Returns (ok, issues, used_findings):
      ok           – True when every segment is either a grounded factual
                     segment or an allowed non-factual framing segment
      issues       – list[str] of reasons (empty when ok)
      used_findings– list[str] of finding claims referenced by the accepted
                     grounded factual segments

    Parameters
    ----------
    segments : the "segments" list parsed from the LLM JSON, or
               {"segments": [...]} for convenience.
    report   : ResearchReport (or its dict serialization).
    """
    rpt = report if isinstance(report, ResearchReport) else ResearchReport.from_dict(report or {})
    if isinstance(segments, dict):
        segments = segments.get("segments")
    if not isinstance(segments, list):
        return False, ["'segments' must be a list"], []

    _, issues, used_indices = _validate_segments(segments, rpt)
    return (not issues), issues, _index_claims(rpt, used_indices)


# ---------------------------------------------------------------------------
# Public: number/URL safety check on a rendered draft (kept from Phase 4)
# ---------------------------------------------------------------------------

def validate_post_grounding(post: str, report) -> tuple[bool, list[str]]:
    """
    Reject drafts whose concrete details are not traceable to the report.

    Phase 4's additional safety check, applied on top of the structural
    segment validation: every numeric token in the final draft must exist in
    the report's material, and every URL must belong to a report source.

    Returns (grounded, list_of_issues).
    """
    rpt = report if isinstance(report, ResearchReport) else ResearchReport.from_dict(report or {})
    if not post:
        return False, ["empty draft"]

    material_digits = _report_material_digits(rpt)
    allowed_urls = _report_source_urls(rpt)

    issues: list[str] = []
    post_digits = {t for t in _raw_tokens(post) if any(ch.isdigit() for ch in t)}
    if post_digits and not post_digits <= material_digits:
        issues.append(f"numbers absent from the report: {sorted(post_digits - material_digits)}")

    for url in _URL_RE.findall(post):
        if _url_key(url) not in allowed_urls:
            issues.append(f"unknown URL: {url}")

    return not issues, issues


def _draft_platform(report: ResearchReport, style: dict, platform: str) -> dict:
    """Run one LLM completion for a platform and validate the outcome.

    Returns {"ok", "text", "issues", "refs"}: ok True only when the parsed
    segments validate structurally, pass the number/URL safety check, and
    respect the platform length limits (linkedin 1300-1900, x <= 280).
    """
    if platform == "linkedin":
        system = _SYSTEM_PROMPT_LINKEDIN.format(
            min_chars=_LINKEDIN_MIN_CHARS, max_chars=_LINKEDIN_MAX_CHARS
        )
        max_tokens = _MAX_RESPONSE_TOKENS
        separator = "\n\n"
    else:
        system = _SYSTEM_PROMPT_X
        max_tokens = _X_MAX_RESPONSE_TOKENS
        separator = "\n"

    raw = complete_text(
        system,
        _build_report_prompt(report, style, platform),
        max_tokens=max_tokens,
        temperature=_TEMPERATURE,
    )
    data = _extract_json(raw)
    if not isinstance(data, dict):
        raise ValueError("LLM response is not a JSON object")
    segments = data.get("segments")
    if not isinstance(segments, list):
        raise ValueError("LLM response missing 'segments' list")

    accepted, issues, used_indices = _validate_segments(segments, report)
    text = _render_draft(accepted, separator)

    ok_numbers, number_issues = validate_post_grounding(text, report)
    if not ok_numbers:
        issues.extend(number_issues)

    if platform == "linkedin":
        if not (_LINKEDIN_MIN_CHARS <= len(text) <= _LINKEDIN_MAX_CHARS):
            issues.append(
                f"length {len(text)} outside {_LINKEDIN_MIN_CHARS}-{_LINKEDIN_MAX_CHARS} chars"
            )
    else:
        if len(text) > _X_MAX_CHARS:
            issues.append(f"length {len(text)} exceeds {_X_MAX_CHARS} chars")

    return {"ok": not issues, "text": text, "issues": issues, "refs": used_indices}


# ---------------------------------------------------------------------------
# Structural segment validation
# ---------------------------------------------------------------------------

def _resolve_segment_refs(value, count: int) -> tuple[str, list[int], str]:
    """Classify a segment's ``finding_indices`` into an explicit status.

    Returns a ``(status, refs, reason)`` triple:

    - ``("missing", [], "")``   field absent; the segment MAY be non-factual
                                framing (hook, transition, CTA, question)
    - ``("empty", [], "")``     explicit ``[]``; MAY be non-factual framing
    - ``("invalid", [], reason)`` references supplied but malformed; the
                                segment MUST be rejected and never rescued by
                                framing
    - ``("valid", refs, "")``   references resolve; factual grounding required

    The status is explicit so callers never infer intent from a truthiness
    test (``if refs:`` could silently turn invalid references into framing).
    Malformed inputs – non-list/tuple values, booleans, floats or strings,
    negatives, out-of-range indices and duplicates – are ``invalid`` rather
    than silently normalized.
    """
    if value is None:
        return "missing", [], ""
    if not isinstance(value, (list, tuple)):
        return "invalid", [], f"finding_indices must be a list, got {type(value).__name__}"
    refs: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return "invalid", [], f"finding index must be an integer, got bool {item!r}"
        if not isinstance(item, int):
            return "invalid", [], f"finding index must be an integer, got {item!r}"
        if item < 0 or item >= count:
            return "invalid", [], f"finding index {item} out of range (0-{count - 1})"
        if item in refs:
            return "invalid", [], f"duplicate finding index {item}"
        refs.append(item)
    return ("empty", [], "") if not refs else ("valid", refs, "")


def _validate_segments(segments: list, report: ResearchReport) -> tuple[list[dict], list[str], list[int]]:
    """Validate an LLM "segments" list against the report's findings.

    Returns (accepted_segments, issues, used_indices):
      accepted_segments – normalized {"text", "finding_indices"} dicts, in order
      issues            – reasons each other segment was rejected
      used_indices      – finding indices referenced by accepted factual
                          segments (first-occurrence order, deduplicated)
    """
    accepted: list[dict] = []
    issues: list[str] = []
    used_indices: list[int] = []
    allowed_urls = _report_source_urls(report)

    for position, segment in enumerate(segments):
        if not isinstance(segment, dict):
            issues.append(f"segment #{position}: not an object")
            continue

        text = _normalize_text(segment.get("text"))
        if not text:
            issues.append(f"segment #{position}: empty text")
            continue

        ref_status, refs, ref_reason = _resolve_segment_refs(
            segment.get("finding_indices"), len(report.findings)
        )
        if ref_status == "invalid":
            # Supplied but malformed references are always rejected – they
            # must never silently fall back to framing.
            issues.append(f"segment #{position}: invalid refs ({ref_reason})")
            continue
        if ref_status == "valid":
            supported, reason = _segment_supported(text, refs, report, allowed_urls)
            if supported:
                accepted.append({"text": text, "finding_indices": refs})
                for ref in refs:
                    if ref not in used_indices:
                        used_indices.append(ref)
            else:
                issues.append(f"segment #{position}: {reason}")
        else:  # "missing" or "empty" → framing candidate only
            if _is_framing(text, report):
                accepted.append({"text": text, "finding_indices": []})
            else:
                issues.append(
                    f"segment #{position}: invalid finding reference or unsupported "
                    f"content (not grounded in the report)"
                )

    return accepted, issues, used_indices


def _segment_supported(
    text: str,
    refs: list[int],
    report: ResearchReport,
    allowed_urls: set,
) -> tuple[bool, str]:
    """True only when *text* is confined to what the referenced findings carry.

    Content may only use the referenced findings' claim/evidence vocabulary
    (after light stemming, minus function words). Negation must be present in
    the referenced findings' evidence. Digits must exist in that evidence.
    URLs must belong to the report.
    """
    referenced = [report.findings[idx] for idx in refs]

    for url in _URL_RE.findall(text):
        if _url_key(url) not in allowed_urls:
            return False, f"references unknown URL: {url}"

    # URLs carry no semantic content of their own – drop them before the
    # vocabulary / negation / digit checks so their internal tokens (schemes,
    # hosts, paths) do not count as unsupported content.
    scrubbed = _URL_RE.sub(" ", text)

    claim_tokens: set[str] = set()
    claim_raw: set[str] = set()
    evidence_tokens: set[str] = set()
    evidence_raw: set[str] = set()
    allowed_digits: set[str] = set()
    for finding in referenced:
        claim_tokens |= _content_tokens(finding.claim)
        claim_raw |= _raw_tokens(finding.claim)
        for item in finding.supporting_evidence + finding.conflicting_evidence:
            evidence_tokens |= _content_tokens(item.supporting_quote)
            evidence_tokens |= _content_tokens(item.context)
            evidence_tokens |= _content_tokens(item.claim_text)
            evidence_raw |= _raw_tokens(item.supporting_quote)
            evidence_raw |= _raw_tokens(item.claim_text)
            allowed_digits |= {t for t in _raw_tokens(item.supporting_quote)
                               if any(ch.isdigit() for ch in t)}
            allowed_digits |= {t for t in _raw_tokens(item.context)
                               if any(ch.isdigit() for ch in t)}
            allowed_digits |= {t for t in _raw_tokens(item.claim_text)
                               if any(ch.isdigit() for ch in t)}

    text_tokens = _content_tokens(scrubbed)

    # Content words absent from the referenced findings (claims + their
    # evidence) introduce unsupported facts – numbers or not.
    unsupported = (
        text_tokens - claim_tokens - evidence_tokens - _function_stemmed()
    )
    if unsupported:
        return False, (
            f"introduces content absent from the referenced findings: "
            f"{sorted(unsupported)[:8]}"
        )

    # Negation the referenced findings do not carry → contradiction.
    if _NEGATION_MARKERS & (_raw_tokens(scrubbed) - claim_raw - evidence_raw):
        return False, "negates a referenced finding"

    # Quantitative specifics the referenced findings do not carry.
    text_digits = {t for t in _raw_tokens(scrubbed) if any(ch.isdigit() for ch in t)}
    if text_digits and not text_digits <= allowed_digits:
        return False, (
            f"asserts numbers absent from the referenced findings: "
            f"{sorted(text_digits - allowed_digits)}"
        )

    return True, ""


def _is_framing(text: str, report: ResearchReport) -> bool:
    """True when *text* is non-factual writing (hook/question/CTA/framing).

    A framing segment must not introduce facts: no digits, no URLs, no
    negation, and every content word must already be report vocabulary.
    A question is only ever accepted AFTER that same vocabulary gate passes –
    ending with "?" is not a free pass for unsupported entities or concepts.
    A non-question segment that mainly echoes one finding's claim is a
    factual restatement and must carry references.
    """
    tokens = _raw_tokens(text)
    if any(any(ch.isdigit() for ch in token) for token in tokens):
        return False
    if _NEGATION_MARKERS & tokens:
        return False
    if _URL_RE.findall(text):
        return False

    stripped = _HASHTAG_RE.sub(" ", text)
    is_question = stripped.strip().endswith("?")

    content = _content_tokens(stripped) - _function_stemmed()
    if not content:
        return True

    if content - _report_vocab(report):
        return False

    # Vocabulary is grounded: a question introduces no assertion and is
    # allowed as framing.
    if is_question:
        return True

    for finding in report.findings:
        claim_tokens = _content_tokens(finding.claim) - _function_stemmed()
        if claim_tokens and (
            len(content & claim_tokens) / len(content) >= _MIN_FINDING_COVERAGE
        ):
            return False

    return True


# ---------------------------------------------------------------------------
# Report prompt builder
# ---------------------------------------------------------------------------

def _build_report_prompt(report: ResearchReport, style: dict, platform: str) -> str:
    """Render the report as the ONLY research material the LLM may use."""
    topic = report.topic or ""
    question = report.research_question or ""

    lines: list[str] = ["## RESEARCH MATERIAL", f"Topic: {topic}"]
    if question:
        lines.append(f"Research question: {question}")

    findings = report.findings
    if findings:
        lines.append("")
        lines.append("## FINDINGS (0-based indices)")
        for idx, finding in enumerate(findings):
            lines.append(f"[{idx}] [{finding.classification}] {finding.claim}")
            for item in finding.supporting_evidence:
                quote = (item.supporting_quote or "").strip()
                if quote:
                    lines.append(f"   – Evidence: {quote}  ({item.source_url or 'no url'})")
            for item in finding.conflicting_evidence:
                quote = (item.supporting_quote or "").strip()
                if quote:
                    lines.append(f"   − Conflicting: {quote}  ({item.source_url or 'no url'})")
    else:
        lines.append("")
        lines.append("## FINDINGS")
        lines.append("(No structured findings were produced.)")

    if report.counterarguments:
        lines.append("")
        lines.append("## COUNTERARGUMENTS")
        for index, counter in enumerate(report.counterarguments, start=1):
            lines.append(f"{index}. {counter.argument}" + (f" – rebuttal: {counter.rebuttal}" if counter.rebuttal else ""))

    if report.limitations:
        lines.append("")
        lines.append("## LIMITATIONS")
        lines.extend(f"- {lim}" for lim in report.limitations)

    if report.conclusions:
        lines.append("")
        lines.append("## CONCLUSIONS")
        lines.extend(f"- {conclusion}" for conclusion in report.conclusions)

    lines.append("")
    lines.append("## AUTHOR'S WRITING STYLE")
    lines.append(f"- Tone:   {style.get('tone', 'professional yet conversational')}")
    lines.append(f"- Voice:  {style.get('voice', 'first-person, thought-leader')}")
    lines.append(
        "- Topics: " + ", ".join(style.get("topics_of_interest", ["AI", "technology"]))
    )
    avoid = style.get("avoid", [])
    lines.append(f"- Things to AVOID: {', '.join(avoid) if avoid else 'none'}")
    lines.append(f"- Platform rules: {json.dumps(style.get(platform, {}), indent=2)}")

    platform_tasks = {
        "linkedin": (
            "Write the LINKEDIN post now as one JSON object with a"
            f" \"segments\" array. Stay strictly between {_LINKEDIN_MIN_CHARS}"
            f" and {_LINKEDIN_MAX_CHARS} characters across all segments."
            " Ground every factual block in a 0-based FINDING index."
        ),
        "x": (
            "Write the X (Twitter) tweet now as one JSON object with a"
            f" \"segments\" array. Maximum {_X_MAX_CHARS} characters across"
            " all segments."
        ),
    }
    lines.append("")
    lines.append("## YOUR TASK")
    lines.append(platform_tasks.get(platform, platform_tasks["linkedin"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic fallback (never fabricates)
# ---------------------------------------------------------------------------

def _fallback_post(report: ResearchReport, platform: str) -> str:
    """Deterministic open post restating the report's findings."""
    topic = report.topic or "Research brief"
    question = report.research_question or ""
    claims = [f.claim for f in report.findings if f.claim]
    counterarguments = [c.argument for c in report.counterarguments if c.argument]
    limitations = list(report.limitations)

    if platform == "linkedin":
        sentences: list[str] = []
        sentences.append(f"Topic: {topic}.")
        if question:
            sentences.append(f"Research question: {question}")
        if claims:
            sentences.append("Key findings: " + "; ".join(claims))
        if counterarguments:
            sentences.append("Counterarguments: " + "; ".join(counterarguments))
        if limitations:
            sentences.append("Limitations: " + "; ".join(limitations))
        if not claims and not counterarguments:
            sentences.append("This research produced no findings yet – here is where it stands.")
        sentences.append(f"Weighing the evidence on {topic}, the picture is still forming.")
        sentences.append("What have your own experiments shown?")
        return "\n\n".join(sentences) + "\n\n#AIEngineering #AgenticAI #LLM"

    first = claims[0] if claims else f"Research on {topic} produced no confirmed findings yet."
    tweet = f"{first}"
    if counterarguments:
        tweet += f" Note: {counterarguments[0]}"
    return (tweet + "\n\n#AIEngineering #AgenticAI")[:_X_MAX_CHARS]


def _fallback_result(report: ResearchReport, reason: str) -> dict:
    """Report-derived fallback: grounded in the report by construction."""
    linkedin = _fallback_post(report, "linkedin")
    x = _fallback_post(report, "x")
    log.info("[Post] Fallback draft used (%s).", reason or "unknown reason")
    return {
        "status": "fallback",
        "linkedin_draft": linkedin,
        "x_draft": x,
        "used_findings": [f.claim for f in report.findings if f.claim],
        "grounded": True,
        "issues": [reason] if reason else [],
    }


def _empty_result(report: ResearchReport) -> dict:
    return {
        "status": "empty",
        "linkedin_draft": "",
        "x_draft": "",
        "used_findings": [],
        "grounded": False,
        "issues": ["report has no findings or evidence to draft from"],
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _render_draft(accepted: list[dict], separator: str) -> str:
    return separator.join(segment["text"] for segment in accepted)


def _index_claims(report: ResearchReport, indices: list[int]) -> list[str]:
    """Finding claim strings for deduplicated, sorted, in-range indices."""
    return [
        report.findings[idx].claim
        for idx in sorted(set(indices))
        if 0 <= idx < len(report.findings)
    ]


def _normalize_text(value) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _report_material_corpus(report: ResearchReport) -> list[str]:
    """Every text field a draft may draw from (for vocab + digit grounding)."""
    corpus: list[str] = [report.topic, report.research_question]
    corpus.append(report.synthesis)
    corpus.append(report.summary)
    corpus.extend(report.conclusions)
    corpus.extend(report.limitations)
    corpus.extend(report.uncertainties)
    corpus.extend(report.unresolved_questions)
    for finding in report.findings:
        corpus.append(finding.claim)
        corpus.append(finding.reasoning)
        corpus.extend(finding.limitations)
        for item in finding.supporting_evidence + finding.conflicting_evidence:
            corpus.append(item.supporting_quote)
            corpus.append(item.context)
            corpus.append(item.claim_text)
    for counter in report.counterarguments:
        corpus.append(counter.argument)
        corpus.append(counter.rebuttal)
    for item in report.evidence:
        corpus.append(item.supporting_quote)
        corpus.append(item.context)
        corpus.append(item.claim_text)
    for source in report.sources:
        corpus.append(source.title)
        corpus.append(source.url)
    return corpus


def _report_material_digits(report: ResearchReport) -> set[str]:
    """Every numeric token the report carries; anything else in a draft is an
    unsupported assertion."""
    digits: set[str] = set()
    for text in _report_material_corpus(report):
        if not isinstance(text, str):
            continue
        digits |= {t for t in _raw_tokens(text) if any(ch.isdigit() for ch in t)}
    return digits


def _report_vocab(report: ResearchReport) -> frozenset:
    """Stemmed content vocabulary of the whole report (for framing segments)."""
    vocab: set[str] = set()
    for text in _report_material_corpus(report):
        if not isinstance(text, str):
            continue
        vocab |= _content_tokens(text)
    return frozenset(vocab)


def _report_source_urls(report: ResearchReport) -> set[str]:
    """The set of URLs the report is allowed to reference (sources + evidence)."""
    urls: set[str] = {_url_key(s.url) for s in report.sources if s.url}
    for item in report.evidence:
        if item.source_url:
            urls.add(_url_key(item.source_url))
    for finding in report.findings:
        for item in finding.supporting_evidence + finding.conflicting_evidence:
            if item.source_url:
                urls.add(_url_key(item.source_url))
    for counter in report.counterarguments:
        for item in counter.evidence:
            if item.source_url:
                urls.add(_url_key(item.source_url))
    return urls


__all__ = [
    "draft_report_post",
    "validate_draft_segments",
    "validate_post_grounding",
]