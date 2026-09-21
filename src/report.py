"""
PersonaPulse – ResearchReport Synthesis Module (Phase 3, step 3)
================================================================
Combines the topic, research question, research sources, extracted
evidence/claims and the CriticalAnalysis into a structured ResearchReport.

Flow
----
ResearchQuestion + ResearchSource[] + Evidence[] + CriticalAnalysis
    ↓ deterministic assembly (findings, evidence sections, confidence)
    ↓ grounded LLM synthesis (optional) → synthesis prose + conclusions
    ↓ deterministic fallback when the LLM is unavailable / malformed
    ↓ ResearchReport

Design guarantees
-----------------
- Report evidence ⊆ supplied evidence: the CriticalAnalysis is grounded
  against the report's Evidence[] (matched by normalized source URL + claim
  text); stale or externally injected evidence can never enter the report.
- The synthesis is grounded ONLY in the collected research: the LLM receives
  the collected material, its conclusions must match an existing finding's
  claim, its narrative must reference the material and introduce no new URLs,
  and any unparseable/ungrounded output falls back to a deterministic
  restatement of the structured (already validated) findings.
- No new Evidence / ClaimAnalysis / source records are ever created during
  synthesis; original Evidence objects are never mutated.
- Traceability is preserved: every finding carries its Evidence objects
  (source URL + supporting quote verbatim); the source list and full evidence
  list are embedded in the report.
- The report is research-agnostic: it contains no LinkedIn formatting,
  hashtags, or social copy (those live in the drafting phase, untouched).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from src.ingestion import _normalize_url
from src.llm import complete_text
from src.models import (
    ClaimAnalysis,
    Counterargument,
    CriticalAnalysis,
    Evidence,
    ResearchQuestion,
    ResearchReport,
    ResearchSource,
)
from src.selection import _extract_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RESPONSE_TOKENS = 2048
_TEMPERATURE = 0.2
_MAX_FALLBACK_CONCLUSIONS = 3

_URL_RE = re.compile(r"https?://[^\s)\]}]+")

# ---------------------------------------------------------------------------
# Synthesis prompt
# ---------------------------------------------------------------------------

_SYNTHESIS_SYSTEM_PROMPT = """You are the research synthesis writer for an "AI Engineering Research Agent".
Write the final research synthesis for a document that has already been analysed and validated.
Ground EVERYTHING exclusively in the findings provided under MATERIAL.findings.

OUTPUT CONTRACT - return VALID JSON ONLY. No markdown, no code fences, no hashtags.
Shape:
{
  "synthesis": [
    {"text": "<sentence or two synthesising the referenced findings>", "finding_indices": [0, 2]}
  ],
  "conclusions": [
    {"text": "<a conclusion drawn from the referenced finding>", "finding_indices": [0]}
  ]
}

RULES:
- "finding_indices" MUST be 0-based indexes into MATERIAL.findings.
- Every synthesis segment and every conclusion MUST reference at least one
  finding it is DIRECTLY based on (use multiple indexes to combine findings).
- Only paraphrase the referenced findings. Do NOT add facts, numbers, URLs,
  negations, causal/comparative/recommendation claims that the referenced
  findings do not already carry. Omit an idea instead of inventing it.
- This is a research document, NOT social media copy: no hashtags, emojis,
  hooks, or calls-to-action."""


# ---------------------------------------------------------------------------
# Public: report synthesis
# ---------------------------------------------------------------------------

def synthesize_report(
    research_question: Optional[ResearchQuestion] = None,
    sources: Optional[list[ResearchSource]] = None,
    evidence: Optional[list[Evidence]] = None,
    analysis: Optional[CriticalAnalysis] = None,
    use_llm: bool = True,
) -> ResearchReport:
    """
    Assemble a complete ResearchReport from the collected research inputs.

    ``analysis`` may be None/empty (an incomplete research session): the report
    still builds from the evidence deterministically and never fabricates.

    Returns
    -------
    ResearchReport with: topic, research_question, findings, evidence sections,
    counterarguments, limitations/uncertainties/open questions, sources,
    evidence references, conclusions, and a synthesis narrative.
    """
    rq = research_question if isinstance(research_question, ResearchQuestion) else None
    srcs = list(sources or [])
    items = list(evidence or [])
    analysis = analysis if isinstance(analysis, CriticalAnalysis) else None

    topic = getattr(rq, "topic", "") or ""
    question_text = getattr(rq, "question", "") or ""

    allowed_urls = {
        _url_key(s.url) for s in srcs if s.url
    } | {
        _url_key(e.source_url) for e in items if e.source_url
    }
    allowed_keys = {_evidence_key(e) for e in items}

    # Ground the CriticalAnalysis against the report's supplied Evidence[]:
    # stale / externally injected evidence never enters the report.
    grounded_analysis = _ground_analysis(analysis, allowed_keys) if analysis is not None else None

    if analysis is not None and analysis.claims:
        findings = list(grounded_analysis.claims)
    else:
        # No usable analysis (absent, or produced no claims): derive findings
        # deterministically from the supplied evidence.
        findings = _findings_from_evidence(items)
        if grounded_analysis is not None:
            grounded_analysis = _discard_claims(grounded_analysis)

    supporting = _merge_evidence(e for f in findings for e in f.supporting_evidence)
    conflicting = _merge_evidence(e for f in findings for e in f.conflicting_evidence)
    counterarguments = list(grounded_analysis.counterarguments) if grounded_analysis else []
    limitations = list(grounded_analysis.limitations) if grounded_analysis else []
    uncertainties = list(grounded_analysis.uncertainties) if grounded_analysis else []
    unresolved = list(grounded_analysis.unresolved_questions) if grounded_analysis else []
    confidence = _confidence_score(items)

    fallback_synthesis, fallback_conclusions = _fallback_narrative(
        topic, question_text, findings, counterarguments, limitations
    )

    synthesis_text = ""
    synthesis_segments: list[str] = []
    conclusions: list[str] = []
    if use_llm:
        try:
            text = complete_text(
                _SYNTHESIS_SYSTEM_PROMPT,
                _build_synthesis_prompt(
                    topic, question_text, srcs, findings,
                    counterarguments, limitations, uncertainties, unresolved,
                ),
                max_tokens=_MAX_RESPONSE_TOKENS,
                temperature=_TEMPERATURE,
            )
            data = _extract_json(text)
            if isinstance(data, dict):
                synthesis_text, synthesis_segments, conclusions = _validate_llm_output(
                    data, findings, allowed_urls
                )
        except Exception as exc:  # pylint: disable=broad-except
            log.warning(
                "[Report] LLM synthesis rejected (%s: %s) – using deterministic fallback.",
                type(exc).__name__, exc,
            )

    if not synthesis_text:
        synthesis_text = fallback_synthesis
    if not conclusions:
        conclusions = fallback_conclusions

    report = ResearchReport(
        topic=topic,
        research_question=question_text,
        summary=synthesis_text,  # legacy narrative mirrors the synthesis
        conclusions=conclusions,
        questions=[rq] if rq else [],
        sources=srcs,
        evidence=items,
        findings=findings,
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        counterarguments=counterarguments,
        limitations=limitations,
        uncertainties=uncertainties,
        unresolved_questions=unresolved,
        synthesis=synthesis_text,
        synthesis_segments=synthesis_segments,
        analysis=grounded_analysis,
        confidence_score=confidence,
        created_at=datetime.now(timezone.utc),
    )

    log.info(
        "[Report] Synthesised: %d finding(s), %d source(s), %d evidence item(s), %d conclusion(s).",
        len(findings), len(srcs), len(items), len(conclusions),
    )
    return report


# ---------------------------------------------------------------------------
# Findings
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CriticalAnalysis grounding (against the supplied evidence)
# ---------------------------------------------------------------------------

def _ground_analysis(analysis: Optional[CriticalAnalysis], allowed_keys: set) -> CriticalAnalysis:
    """Return a NEW CriticalAnalysis whose findings/counterarguments only
    reference evidence present in the supplied set.

    Evidence is matched with the project's stable identity mechanism
    (normalized source URL + claim text) rather than object identity, so stale
    or externally injected evidence is rejected. The original Evidence objects
    are left unmutated; findings/counterarguments that lose all grounding are
    dropped (an unresolved_question may stand without evidence).
    """
    claims = [
        _grounded_claim(c, allowed_keys)
        for c in analysis.claims
    ]
    claims = [c for c in claims if c is not None]

    counterarguments = [
        c for c in (
            _grounded_counterargument(c, allowed_keys)
            for c in analysis.counterarguments
        )
        if c is not None
    ]

    return CriticalAnalysis(
        topic=analysis.topic,
        research_question=analysis.research_question,
        claims=claims,
        counterarguments=counterarguments,
        limitations=list(analysis.limitations),
        uncertainties=list(analysis.uncertainties),
        unresolved_questions=list(analysis.unresolved_questions),
        status=analysis.status,
        created_at=analysis.created_at,
    )


def _grounded_claim(claim: ClaimAnalysis, allowed_keys: set) -> Optional[ClaimAnalysis]:
    supporting = [e for e in claim.supporting_evidence if _evidence_key(e) in allowed_keys]
    conflicting = [e for e in claim.conflicting_evidence if _evidence_key(e) in allowed_keys]

    if (
        not supporting
        and not conflicting
        and claim.classification != ClaimAnalysis.CLASSIFICATION_UNRESOLVED
    ):
        log.debug("[Report] Dropping analysis finding '%s' (no supplied evidence matches).", claim.claim)
        return None

    return ClaimAnalysis(
        claim=claim.claim,
        classification=claim.classification,
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        confidence=claim.confidence,
        reasoning=claim.reasoning,
        limitations=list(claim.limitations),
    )


def _grounded_counterargument(counter: Counterargument, allowed_keys: set) -> Optional[Counterargument]:
    evidence = [e for e in counter.evidence if _evidence_key(e) in allowed_keys]
    if not evidence:
        log.debug("[Report] Dropping counterargument with no supplied evidence reference.")
        return None
    return Counterargument(argument=counter.argument, evidence=evidence, rebuttal=counter.rebuttal)


def _discard_claims(analysis: CriticalAnalysis) -> CriticalAnalysis:
    """Keep the analysis' context sections but drop any claims (their findings
    were derived from the supplied evidence instead)."""
    return CriticalAnalysis(
        topic=analysis.topic,
        research_question=analysis.research_question,
        claims=[],
        counterarguments=list(analysis.counterarguments),
        limitations=list(analysis.limitations),
        uncertainties=list(analysis.uncertainties),
        unresolved_questions=list(analysis.unresolved_questions),
        status=analysis.status,
        created_at=analysis.created_at,
    )


def _evidence_key(item: Evidence) -> tuple[str, str]:
    """Stable identity for an evidence entry: normalized source URL + claim text."""
    url = _url_key(item.source_url)
    claim = " ".join((item.claim_text or "").split()).casefold()
    return (url, claim)


def _url_key(url: Optional[str]) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    return (_normalize_url(url) or url).casefold()


def _findings_from_evidence(evidence: list[Evidence]) -> list[ClaimAnalysis]:
    grouped: dict[str, list[Evidence]] = {}
    for item in evidence:
        key = item.claim_text.casefold()
        grouped.setdefault(key, []).append(item)

    findings: list[ClaimAnalysis] = []
    for items in grouped.values():
        sources = {e.source_url for e in items if e.source_url}
        corroborated = len(sources) >= 2
        findings.append(
            ClaimAnalysis(
                claim=items[0].claim_text,
                classification=(
                    ClaimAnalysis.CLASSIFICATION_FACT
                    if corroborated
                    else ClaimAnalysis.CLASSIFICATION_INTERPRETATION
                ),
                supporting_evidence=list(items),
                confidence=max((e.confidence for e in items), default=0.5),
            )
        )
    return findings


def _merge_evidence(items) -> list[Evidence]:
    seen: set[tuple] = set()
    merged: list[Evidence] = []
    for item in items:
        key = (item.source_url, item.claim_text)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _confidence_score(evidence: list[Evidence]) -> float:
    if not evidence:
        return 0.0
    return round(sum(e.confidence for e in evidence) / len(evidence), 4)


# ---------------------------------------------------------------------------
# Structural LLM narrative validation
# ---------------------------------------------------------------------------
# The LLM response contract is structured: every synthesis segment and every
# conclusion carries its own "finding_indices" into the validated findings.
# A statement is retained only when it references valid findings AND its
# content is confined to what those findings carry. This makes grounding
# structural, not a substring scan over the whole narrative.

_NEGATION_MARKERS = {
    "no", "not", "never", "nor", "without", "lacks", "lacking",
    "cannot", "can't", "doesn't", "don't", "didn't", "isn't", "aren't",
    "won't", "wouldn't", "hasn't", "haven't",
    "fails", "fail", "failed", "refutes", "refute", "rejects", "reject",
    "contradicts", "denies", "disputes", "opposes", "impossible", "unlikely",
}

# Function words add no factual content: a statement may add them to a
# referenced claim without making an unsupported assertion. Anything else the
# statement introduces beyond the referenced findings is rejected. Causal /
# comparative / recommendation words ("because", "better", "should", ...) are
# deliberately NOT here, so they surface as unsupported content.
_FUNCTION_WORDS = frozenset({
    "the", "a", "an", "and", "but", "or", "nor", "so", "for", "of", "to",
    "in", "on", "at", "by", "with", "without", "from", "up", "down", "about",
    "into", "over", "under", "around", "between", "among", "through",
    "during", "within", "along", "against", "across", "toward", "towards",
    "is", "are", "was", "were", "been", "being", "be", "am", "do", "does",
    "did", "have", "has", "had", "would", "will", "shall", "can", "could",
    "may", "might",
    "i", "you", "he", "she", "it", "we", "they", "them", "us", "him", "her",
    "me", "its", "their", "theirs", "our", "ours", "your", "yours", "my",
    "mine", "his", "hers",
    "this", "that", "these", "those", "there", "here", "which", "who",
    "whom", "whose", "when", "where", "why", "how", "what", "as", "than",
    "if", "then", "too", "very", "also", "still", "even", "just", "only",
    "such", "same", "own", "each", "every", "either", "neither", "several",
})

_FUNCTION_STEMMED_CACHE: list = []  # lazily filled once _stem is defined


def _function_stemmed() -> frozenset:
    if not _FUNCTION_STEMMED_CACHE:
        _FUNCTION_STEMMED_CACHE.append(frozenset(_stem(w) for w in _FUNCTION_WORDS))
    return _FUNCTION_STEMMED_CACHE[0]


def _validate_llm_output(
    data: dict,
    findings: list[ClaimAnalysis],
    allowed_urls: set,
) -> tuple[str, list[str], list[str]]:
    """Structurally validate the LLM synthesis + conclusions.

    Each item must carry valid finding_indices and its text must be supported
    by the referenced findings. Returns
    (synthesis_text, retained_segment_texts, retained_conclusion_texts);
    ''.join of nothing / [] signals "use the deterministic fallback".
    """
    retained_segments: list[str] = []

    raw_segments = data.get("synthesis")
    if isinstance(raw_segments, list):
        for index, segment in enumerate(raw_segments):
            if not isinstance(segment, dict):
                log.warning("[Report] Rejecting malformed synthesis segment #%d.", index)
                continue
            text = _normalize_text(segment.get("text"))
            refs = _resolve_refs(segment.get("finding_indices"), len(findings))
            if text and refs and _statement_supported(text, refs, findings, allowed_urls):
                retained_segments.append(text)
            else:
                log.warning("[Report] Rejecting synthesis segment #%d: %r", index, text)

    retained_conclusions: list[str] = []
    raw_conclusions = data.get("conclusions")
    if isinstance(raw_conclusions, list):
        for index, conclusion in enumerate(raw_conclusions):
            if not isinstance(conclusion, dict):
                log.warning("[Report] Rejecting malformed conclusion #%d.", index)
                continue
            text = _normalize_text(conclusion.get("text"))
            refs = _resolve_refs(conclusion.get("finding_indices"), len(findings))
            if text and refs and _statement_supported(text, refs, findings, allowed_urls):
                retained_conclusions.append(text)
            else:
                log.warning("[Report] Rejecting conclusion #%d: %r", index, text)

    return " ".join(retained_segments), retained_segments, retained_conclusions


def _resolve_refs(value, count: int) -> Optional[list[int]]:
    """Resolve finding_indices; None when missing/invalid/empty.

    A single invalid index rejects the whole item (missing references must not
    be silently ignored).
    """
    if not isinstance(value, (list, tuple)):
        return None
    refs: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        if item < 0 or item >= count:
            return None
        if item not in refs:
            refs.append(item)
    return refs or None


def _statement_supported(
    text: str,
    refs: list[int],
    findings: list[ClaimAnalysis],
    allowed_urls: set,
) -> bool:
    """True only when *text* is confined to what the referenced findings carry."""
    referenced = [findings[i] for i in refs]

    # 1. Unknown source URLs introduced by the statement → unsupported.
    for url in _URL_RE.findall(text):
        if _url_key(url) not in allowed_urls:
            log.warning("[Report] Statement references unknown URL: %r", url)
            return False

    claim_tokens: set[str] = set()
    claim_raw: set[str] = set()
    allowed_digits: set[str] = set()
    for finding in referenced:
        claim_tokens |= _content_tokens(finding.claim)
        claim_raw |= _raw_tokens(finding.claim)
        for item in finding.supporting_evidence + finding.conflicting_evidence:
            allowed_digits |= {t for t in _raw_tokens(item.supporting_quote)
                               if any(ch.isdigit() for ch in t)}
            allowed_digits |= {t for t in _raw_tokens(item.context)
                               if any(ch.isdigit() for ch in t)}

    text_tokens = _content_tokens(text)

    # 2. Content words absent from the referenced findings → unsupported
    #    (this is the structural core: no reliance on keyword markers).
    if text_tokens - claim_tokens - _function_stemmed():
        log.debug("[Report] Statement introduces content absent from its findings: %r", text)
        return False

    # 3. Negation of a referenced finding.
    if _NEGATION_MARKERS & (_raw_tokens(text) - claim_raw):
        log.debug("[Report] Statement negates a referenced finding: %r", text)
        return False

    # 4. Quantitative specifics the referenced findings do not carry.
    text_digits = {t for t in _raw_tokens(text) if any(ch.isdigit() for ch in t)}
    if text_digits and not text_digits <= allowed_digits:
        log.debug("[Report] Statement asserts numbers absent from its findings: %r", text)
        return False

    return True


def _raw_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", text.casefold()))


def _content_tokens(text: str) -> set[str]:
    """Lightly-stemmed content tokens for paraphrase-tolerant coverage."""
    return {_stem(tok) for tok in _raw_tokens(text)}


def _stem(token: str) -> str:
    """Very light suffix stripping (ties, ing, ly, ed, es, ion, s)."""
    token = token.casefold()
    if any(ch.isdigit() for ch in token):
        return token
    changed = True
    while changed:
        changed = False
        for suffix in ("lies", "ies", "ing", "ly", "ed", "'s", "es", "s", "ion"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                changed = True
                break
    return token


# ---------------------------------------------------------------------------
# Narrative (LLM + deterministic fallback)
# ---------------------------------------------------------------------------

def _build_synthesis_prompt(
    topic: str,
    question_text: str,
    sources: list[ResearchSource],
    findings: list[ClaimAnalysis],
    counterarguments: list[Counterargument],
    limitations: list[str],
    uncertainties: list[str],
    unresolved: list[str],
) -> str:
    payload = {
        "topic": topic,
        "research_question": question_text,
        "sources": [
            {"url": s.url, "title": s.title} for s in sources if s.url
        ],
        "findings": [
            {
                "claim": f.claim,
                "classification": f.classification,
                "source_urls": f.source_urls,
                "evidence": [
                    {"source_url": e.source_url, "supporting_quote": e.supporting_quote}
                    for e in f.supporting_evidence + f.conflicting_evidence
                ],
            }
            for f in findings
        ],
        "counterarguments": [
            {"argument": c.argument, "source_urls": c.source_urls} for c in counterarguments
        ],
        "limitations": limitations,
        "uncertainties": uncertainties,
        "unresolved_questions": unresolved,
    }
    return f"MATERIAL (the only research this synthesis may use):\n{json.dumps(payload, indent=2)}"


def _fallback_narrative(
    topic: str,
    question_text: str,
    findings: list[ClaimAnalysis],
    counterarguments: list[Counterargument],
    limitations: list[str],
) -> tuple[str, list[str]]:
    """Deterministic, fully-grounded restatement of the structured findings."""
    sentences: list[str] = []
    if topic:
        sentences.append(f"Topic: {topic}.")
    if question_text:
        sentences.append(f"Research question: {question_text}")
    if findings:
        sentences.append("Key findings: " + "; ".join(f.claim for f in findings))
    if counterarguments:
        sentences.append("Counterarguments: " + "; ".join(c.argument for c in counterarguments))
    if limitations:
        sentences.append("Limitations: " + "; ".join(limitations))
    if not findings and not limitations:
        sentences.append("No analysis was produced for this research.")

    conclusions = [
        f.claim
        for f in findings
        if f.classification in (ClaimAnalysis.CLASSIFICATION_FACT, ClaimAnalysis.CLASSIFICATION_INTERPRETATION)
    ][:_MAX_FALLBACK_CONCLUSIONS]
    return " ".join(sentences), conclusions


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _normalize_text(value) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _normalize_string_list(value) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in value:
        text = _normalize_text(item)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


__all__ = ["synthesize_report", "ResearchReport"]