"""
PersonaPulse – Evidence Extraction Module (Phase 3, step 1)
============================================================
Extracts structured, source-attributed claims from each ResearchSource
relevant to a ResearchQuestion.

Flow
----
ResearchQuestion + ResearchSource[]
    ↓ (per source)
LLM extraction   → JSON claims about THIS source only (no cross-source reasoning)
    ↓
validate         → each claim must carry text, a supporting quote found in the
                   source body, and a source reference resolving to the source
    ↓
corroborate      → identical claim text seen in ≥2 distinct sources is marked
                   corroborated; conflicting claims are both kept (never merged,
                   never silently resolved)
    ↓
Evidence[]

Key Functions
-------------
- extract_evidence(research_question, research_sources, ...) → list[Evidence]

Design guarantees
-----------------
- Every claim is traceable to ≥1 source: claims reference the source they were
  extracted from (source_url), and are dropped when they cannot be attributed.
- No unsupported claims: a claim is discarded unless its supporting quote is
  actually found in that source's body.
- Extraction is separate from synthesis: this module never writes prose,
  reports, or LinkedIn content (that is a later phase).
- Contradictory claims from different sources are preserved side by side,
  each with its own attribution; no source is silently preferred.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from src.config import settings
from src.ingestion import _clean_body
from src.llm import complete_text
from src.models import Evidence, ResearchQuestion, ResearchSource
from src.selection import _extract_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RESPONSE_TOKENS = 1024
_TEMPERATURE = 0.2
# Slice of each source body sent to the LLM. Configurable via
# EVIDENCE_SOURCE_BODY_CHARS (see src/config.py); sources short of this pass
# completely, longer sources are trimmed at the limit while token usage stays
# bounded.
_MAX_SOURCE_BODY_CHARS = settings.EVIDENCE_SOURCE_BODY_CHARS
_MAX_QUOTE_CHARS = 300               # keep supporting quotes bounded


# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EVIDENCE_SYSTEM_PROMPT = """You are the evidence extractor for an "AI Engineering Research Agent".
You extract FACTUAL CLAIMS from a single source document that are relevant to a research question.
Only extract claims that are actually stated in the provided source text - never invent or guess.

OUTPUT CONTRACT - return VALID JSON ONLY. No markdown, no code fences.
Shape: {{"claims": [{{"claim_text": "<string>", "supporting_quote": "<string>", "context": "<string>", "confidence": <0.0-1.0>, "directly_supports": <true|false>}}]}}

RULES:
- Return up to {max_claims} claims.
- claim_text is a short, factual statement stated by the source.
- supporting_quote is a VERBATIM slice of the source text that supports the claim.
- context is the surrounding context that makes the claim meaningful (may be short).
- confidence reflects how strongly the source states the claim, 0.0-1.0.
- directly_supports is true when the source states the claim outright, false when it only implies it.
- If the source does not support any claims relevant to the question, return {{"claims": []}}.
- Return only fact extraction; do not analyze, compare sources, or summarize across documents."""


# ---------------------------------------------------------------------------
# Public: extract evidence
# ---------------------------------------------------------------------------

def extract_evidence(
    research_question: Optional[ResearchQuestion],
    research_sources: list[ResearchSource],
    use_llm: bool = True,
    max_claims_per_source: Optional[int] = None,
) -> list[Evidence]:
    """
    Extract source-attributed Evidence for *research_question* from
    *research_sources*.

    Each source is processed independently: extraction never reasons across
    sources, so conflicts surface as separate Evidence rows instead of being
    silently resolved. A failing/malformed source contributes no evidence but
    never aborts the stage or invents claims.

    Returns a deterministic list of Evidence: source order is preserved, only
    validated claims are kept, identical claims across ≥2 sources are marked
    corroborated, and the per-source claim cap (default
    settings.EVIDENCE_MAX_CLAIMS_PER_SOURCE) is respected.
    """
    if research_question is None or not research_sources:
        return []

    if not use_llm:
        log.warning("[Evidence] LLM extraction disabled (use_llm=False) – no evidence produced.")
        return []

    cap = max(0, int(
        max_claims_per_source
        if max_claims_per_source is not None
        else settings.EVIDENCE_MAX_CLAIMS_PER_SOURCE
    ))

    collected: list[Evidence] = []
    for source in research_sources:
        try:
            claims = _extract_claims_from_source(
                research_question, source, max_claims=cap
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.warning(
                "[Evidence] Source '%s' failed (%s: %s) – continuing with remaining sources.",
                source.url, type(exc).__name__, exc,
            )
            continue
        collected.extend(claims)

    _mark_corroboration(collected)
    return collected


# ---------------------------------------------------------------------------
# Per-source extraction
# ---------------------------------------------------------------------------

def _extract_claims_from_source(
    research_question: ResearchQuestion,
    source: ResearchSource,
    max_claims: int,
) -> list[Evidence]:
    """Extract + validate claims for ONE source. Returns only supported claims."""
    body = _clean_body(source.body or "")
    if not body:
        log.debug("[Evidence] Source '%s' has no usable body – skipping.", source.url)
        return []
    if max_claims <= 0:
        return []

    # One bounded body drives BOTH the LLM prompt and quote validation, so a
    # quote that the LLM never saw can never be accepted as supporting evidence.
    bounded_body = body[:_MAX_SOURCE_BODY_CHARS]

    try:
        text = complete_text(
            _EVIDENCE_SYSTEM_PROMPT.format(max_claims=max_claims),
            _build_source_prompt(research_question, source, bounded_body),
            max_tokens=_MAX_RESPONSE_TOKENS,
            temperature=_TEMPERATURE,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("[Evidence] LLM extraction failed for '%s' (%s) – source skipped.", source.url, type(exc).__name__)
        return []

    try:
        data = _extract_json(text)
        raw = data.get("claims")
        if not isinstance(raw, list):
            raise ValueError("'claims' must be a list")
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Evidence] Malformed LLM output for '%s' (%s) – no claims extracted from this source.",
            source.url, type(exc).__name__,
        )
        return []

    claims: list[Evidence] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        claim = _validate_claim(entry, source, bounded_body)
        if claim is not None:
            claims.append(claim)
            if len(claims) >= max_claims:
                break
    return claims


def _build_source_prompt(research_question: ResearchQuestion, source: ResearchSource, body: str) -> str:
    payload = {
        "research_question": {
            "topic": research_question.topic,
            "question": research_question.question,
            "aspects": list(research_question.aspects),
        },
        "source": {
            "url": source.url,
            "title": source.title,
            "published": source.published,
            "source": source.source,
        },
        "source_body": body,    # already bounded by the caller
    }
    return f"Extract claims relevant to the question from this source (JSON):\n{json.dumps(payload, indent=2)}"


# ---------------------------------------------------------------------------
# Claim validation – nothing unsupported gets out
# ---------------------------------------------------------------------------

def _validate_claim(entry: dict, source: ResearchSource, body: str) -> Optional[Evidence]:
    """Turn one raw LLM claim into a supported Evidence, or None.

    A claim is only accepted when it (a) has non-empty text, (b) quotes text
    actually present in the source body, and (c) resolves to the source it was
    extracted from. Missing source_url defaults to the processed source (never
    fabricated); a source_url naming any other/nonexistent source is rejected.
    """
    claim_text = _normalize_text(entry.get("claim_text"))
    if not claim_text:
        log.debug("[Evidence] Dropping claim without text.")
        return None

    supporting_quote = _normalize_text(entry.get("supporting_quote"))
    if not supporting_quote:
        log.debug("[Evidence] Dropping unsupported claim: no quote.")
        return None

    normalized_body = _normalize_text(body)
    if supporting_quote not in normalized_body:
        log.debug("[Evidence] Dropping claim whose quote is not found in source '%s'.", source.url)
        return None

    source_url = (entry.get("source_url") or "").strip() or source.url
    if not _urls_match(source_url, source.url):
        log.debug("[Evidence] Dropping claim referencing unknown source '%s'.", source_url)
        return None

    return Evidence(
        claim_text=claim_text,
        source_url=source.url,
        supporting_quote=_trim_quote(entry.get("supporting_quote")),
        confidence=_validated_confidence(entry.get("confidence")),
        context=_normalize_text(entry.get("context")),
        directly_supports=_validated_direct_support(entry.get("directly_supports")),
    )


def _normalize_text(value) -> str:
    """Collapse whitespace and strip; returns '' for non-string input."""
    if not isinstance(value, str):
        return ""
    return " ".join(value.split()).strip()


def _urls_match(a: str, b: str) -> bool:
    a = (a or "").strip().strip("/")
    b = (b or "").strip().strip("/")
    return a.casefold() == b.casefold()


def _trim_quote(quote) -> str:
    text = _normalize_text(quote)
    if len(text) <= _MAX_QUOTE_CHARS:
        return text
    return text[: _MAX_QUOTE_CHARS - 1].rstrip() + "…"


def _validated_confidence(value) -> float:
    """Return a bounded confidence in [0.0, 1.0], or the neutral 0.5 fallback.

    Booleans are rejected (True/False are not valid scores), values outside
    the 0.0–1.0 range are rejected rather than silently clamped, and
    non-numeric/missing values keep the existing neutral fallback. Numeric
    strings ("0.8") are still accepted, matching the existing contract.
    """
    if isinstance(value, bool):
        return 0.5
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    if 0.0 <= confidence <= 1.0:
        return confidence
    return 0.5


def _validated_direct_support(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().casefold() in ("true", "1", "yes")
    return True


# ---------------------------------------------------------------------------
# Cross-source verification (corroboration / conflict visibility)
# ---------------------------------------------------------------------------

def _mark_corroboration(evidence: list[Evidence]) -> None:
    """Mark claims corroborated when identical text appears across ≥2 sources.

    Conflicting claims are intentionally preserved: each keeps its own source
    attribution and no source is silently preferred here. Contradiction
    adjudication belongs to the critical-analysis/synthesis phase.
    """
    by_text: dict[str, list[Evidence]] = {}
    for item in evidence:
        by_text.setdefault(_claim_key(item), []).append(item)

    for group in by_text.values():
        distinct_sources = {item.source_url for item in group}
        if len(distinct_sources) >= 2:
            for item in group:
                item.verification_status = Evidence.VERIFICATION_CORROBORATED


def _claim_key(item: Evidence) -> str:
    return _normalize_text(item.claim_text).casefold()


__all__ = ["extract_evidence", "Evidence"]