"""
PersonaPulse – Critical Analysis Module (Phase 3, step 2)
=========================================================
Turns a research question plus its extracted Evidence into a structured
CriticalAnalysis: major claims, their epistemic classification, the evidence
that supports or contradicts them, grounded counterarguments, limitations,
uncertainties, and unresolved questions.

Flow
----
ResearchQuestion + Evidence[]
    ↓
LLM analysis      → JSON: claims / counterarguments / limitations / ...
    ↓
ground + validate → every referenced source must resolve to an input Evidence
                    (nothing invented); counterarguments without grounding are
                    dropped; classifications normalised
    ↓
CriticalAnalysis

Key Functions
-------------
- analyze_evidence(research_question, evidence, ...) → CriticalAnalysis

Design guarantees
-----------------
- No invention: supporting/conflicting evidence entries are the ACTUAL input
  Evidence objects, selected by source reference – quotes and attribution are
  never rewritten by the analysis.
- Counterarguments must be grounded in ≥1 known source or they are dropped.
- Claims classed as documented_fact / interpretation / opinion are grounded:
  they must resolve to ≥1 input Evidence item (an unmatched URL alone is
  dropped). Only unresolved_question may stand without evidence.
- Claims are classified as documented_fact / interpretation / opinion /
  unresolved_question.
- The output is structured data only; it contains no prose report and no
  LinkedIn content (synthesis/drafting are later phases).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from src.evidence import _validated_confidence
from src.ingestion import _normalize_url
from src.llm import complete_text
from src.models import (
    ClaimAnalysis,
    Counterargument,
    CriticalAnalysis,
    Evidence,
    ResearchQuestion,
)
from src.selection import _extract_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RESPONSE_TOKENS = 2048
_TEMPERATURE = 0.2
_MAX_QUOTE_CHARS = 300               # keep the analysis prompt bounded

_CLASSIFICATION_ALIASES = {
    "documented_fact": ClaimAnalysis.CLASSIFICATION_FACT,
    "documented": ClaimAnalysis.CLASSIFICATION_FACT,
    "fact": ClaimAnalysis.CLASSIFICATION_FACT,
    "verified_fact": ClaimAnalysis.CLASSIFICATION_FACT,
    "interpretation": ClaimAnalysis.CLASSIFICATION_INTERPRETATION,
    "analysis": ClaimAnalysis.CLASSIFICATION_INTERPRETATION,
    "inference": ClaimAnalysis.CLASSIFICATION_INTERPRETATION,
    "opinion": ClaimAnalysis.CLASSIFICATION_OPINION,
    "viewpoint": ClaimAnalysis.CLASSIFICATION_OPINION,
    "unresolved_question": ClaimAnalysis.CLASSIFICATION_UNRESOLVED,
    "unresolved": ClaimAnalysis.CLASSIFICATION_UNRESOLVED,
    "open_question": ClaimAnalysis.CLASSIFICATION_UNRESOLVED,
    "question": ClaimAnalysis.CLASSIFICATION_UNRESOLVED,
}


# ---------------------------------------------------------------------------
# Analysis prompt
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM_PROMPT = """You are the critical analyst for an "AI Engineering Research Agent".
You are given a research question and the evidence extracted from multiple sources.
Produce a structured critical analysis grounded ONLY in that evidence.

OUTPUT CONTRACT - return VALID JSON ONLY. No markdown, no code fences.
Shape:
{
  "claims": [
    {
      "claim": "<major claim>",
      "classification": "documented_fact|interpretation|opinion|unresolved_question",
      "supporting_source_urls": ["<url from the evidence>"],
      "conflicting_source_urls": ["<url from the evidence>"],
      "confidence": <0.0-1.0>,
      "reasoning": "<why this classification / how the evidence relates>",
      "limitations": ["<limitation of this claim>"]
    }
  ],
  "counterarguments": [
    {"argument": "<counterargument>", "source_urls": ["<url from the evidence>"], "rebuttal": "<optional>"}
  ],
  "limitations": ["<overall limitation>"],
  "uncertainties": ["<important uncertainty>"],
  "unresolved_questions": ["<question the evidence does not answer>"]
}

RULES:
- Use ONLY the provided evidence. Never invent facts, sources, or URLs.
- Every supporting_source_urls / conflicting_source_urls / counterargument source_urls MUST be a URL taken from the provided evidence.
- Do NOT invent counterarguments that are not grounded in the evidence; omit them instead.
- classification: documented_fact (well-evidenced), interpretation (reasoned reading), opinion (value judgement), unresolved_question (evidence insufficient).
- Distinguish conflicting evidence explicitly via conflicting_source_urls.
- Return only the structured analysis; do not write a report or summary prose."""


# ---------------------------------------------------------------------------
# Public: critical analysis
# ---------------------------------------------------------------------------

def analyze_evidence(
    research_question: Optional[ResearchQuestion],
    evidence: list[Evidence],
    use_llm: bool = True,
) -> CriticalAnalysis:
    """
    Produce a CriticalAnalysis of *evidence* for *research_question*.

    Returns an empty (status="empty") analysis when there is no question, no
    evidence, when the LLM is disabled, or when the LLM output is malformed/
    unavailable – never fabricated content.
    """
    if research_question is None:
        return _empty_analysis(None)

    if not evidence:
        log.warning("[Analysis] No evidence supplied for '%s' – nothing to analyse.", research_question.question)
        return _empty_analysis(research_question)

    if not use_llm:
        log.warning("[Analysis] LLM analysis disabled (use_llm=False) – no analysis produced.")
        return _empty_analysis(research_question)

    index = _evidence_index(evidence)

    try:
        text = complete_text(
            _ANALYSIS_SYSTEM_PROMPT,
            _build_analysis_prompt(research_question, evidence),
            max_tokens=_MAX_RESPONSE_TOKENS,
            temperature=_TEMPERATURE,
        )
        data = _extract_json(text)
        if not isinstance(data, dict):
            raise ValueError("analysis response must be a JSON object")
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Analysis] LLM analysis rejected (%s: %s) – returning empty analysis.",
            type(exc).__name__, exc,
        )
        return _empty_analysis(research_question)

    try:
        analysis = _build_analysis(data, research_question, index)
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Analysis] Could not build analysis (%s: %s) – returning empty analysis.",
            type(exc).__name__, exc,
        )
        return _empty_analysis(research_question)

    log.info(
        "[Analysis] %d claim(s), %d counterargument(s), %d limitation(s) for '%s'.",
        len(analysis.claims), len(analysis.counterarguments), len(analysis.limitations),
        research_question.question,
    )
    return analysis


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_analysis_prompt(research_question: ResearchQuestion, evidence: list[Evidence]) -> str:
    payload = {
        "research_question": {
            "topic": research_question.topic,
            "question": research_question.question,
            "aspects": list(research_question.aspects),
        },
        "evidence": [
            {
                "claim_text": item.claim_text,
                "source_url": item.source_url,
                "supporting_quote": _trim_quote(item.supporting_quote),
                "confidence": item.confidence,
                "verification_status": item.verification_status,
                "context": item.context,
            }
            for item in evidence
        ],
    }
    return f"Analyse this evidence critically (JSON):\n{json.dumps(payload, indent=2)}"


# ---------------------------------------------------------------------------
# Validation – grounded, no invention
# ---------------------------------------------------------------------------

def _build_analysis(data: dict, research_question: ResearchQuestion, index: dict) -> CriticalAnalysis:
    claims = [
        claim for claim in (
            _validate_claim(item, index)
            for item in (data.get("claims") or [])
            if isinstance(item, dict)
        )
        if claim is not None
    ]
    counterarguments = [
        counter for counter in (
            _validate_counterargument(item, index)
            for item in (data.get("counterarguments") or [])
            if isinstance(item, dict)
        )
        if counter is not None
    ]
    limitations = _normalize_string_list(data.get("limitations"))
    uncertainties = _normalize_string_list(data.get("uncertainties"))
    unresolved = _normalize_string_list(data.get("unresolved_questions"))

    has_content = bool(claims or counterarguments or limitations or uncertainties or unresolved)
    return CriticalAnalysis(
        topic=research_question.topic,
        research_question=research_question.question,
        claims=claims,
        counterarguments=counterarguments,
        limitations=limitations,
        uncertainties=uncertainties,
        unresolved_questions=unresolved,
        status=CriticalAnalysis.STATUS_OK if has_content else CriticalAnalysis.STATUS_EMPTY,
        created_at=datetime.now(timezone.utc),
    )


def _validate_claim(item: dict, index: dict) -> Optional[ClaimAnalysis]:
    claim_text = _normalize_text(item.get("claim"))
    if not claim_text:
        log.debug("[Analysis] Dropping claim without text.")
        return None

    classification = _normalize_classification(item.get("classification"))
    supporting = _select_evidence(item.get("supporting_source_urls"), index)
    conflicting = _select_evidence(item.get("conflicting_source_urls"), index)

    if (
        classification != ClaimAnalysis.CLASSIFICATION_UNRESOLVED
        and not supporting
        and not conflicting
    ):
        # A fact/interpretation/opinion claim must be grounded in ≥1 resolved
        # Evidence item – an LLM-supplied URL is not enough. Only an
        # unresolved_question may stand without evidence (the material simply
        # may not answer it).
        log.debug("[Analysis] Dropping ungrounded '%s' claim (no resolved evidence).", classification)
        return None

    return ClaimAnalysis(
        claim=claim_text,
        classification=classification,
        supporting_evidence=supporting,
        conflicting_evidence=conflicting,
        confidence=_validated_confidence(item.get("confidence")),
        reasoning=_normalize_text(item.get("reasoning")),
        limitations=_normalize_string_list(item.get("limitations")),
    )


def _validate_counterargument(item: dict, index: dict) -> Optional[Counterargument]:
    argument = _normalize_text(item.get("argument"))
    if not argument:
        return None

    grounding = _select_evidence(item.get("source_urls"), index)
    if not grounding:
        # Never surface an ungrounded counterargument: it would be invented.
        log.debug("[Analysis] Dropping counterargument with no known source reference.")
        return None

    return Counterargument(
        argument=argument,
        evidence=grounding,
        rebuttal=_normalize_text(item.get("rebuttal")),
    )


def _evidence_index(evidence: list[Evidence]) -> dict[str, list[Evidence]]:
    """Map normalized source URL → the Evidence entries from that source."""
    index: dict[str, list[Evidence]] = {}
    for item in evidence:
        key = _url_key(item.source_url)
        if not key:
            continue
        index.setdefault(key, []).append(item)
    return index


def _select_evidence(values, index: dict) -> list[Evidence]:
    """Resolve LLM-supplied source references to the actual input Evidence.

    Unknown URLs are ignored (grounding); duplicates are collapsed.
    """
    selected: list[Evidence] = []
    seen: set[tuple] = set()
    for url in _as_url_list(values):
        for item in index.get(_url_key(url), []):
            key = (item.source_url, item.claim_text)
            if key in seen:
                continue
            seen.add(key)
            selected.append(item)
    return selected


def _as_url_list(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        return [v for v in values if isinstance(v, str)]
    return []


def _url_key(url: Optional[str]) -> str:
    url = (url or "").strip()
    if not url:
        return ""
    return (_normalize_url(url) or url).casefold()


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


def _normalize_classification(value) -> str:
    text = _normalize_text(value).casefold().replace(" ", "_").replace("-", "_")
    return _CLASSIFICATION_ALIASES.get(text, ClaimAnalysis.CLASSIFICATION_INTERPRETATION)


def _trim_quote(quote: str) -> str:
    text = _normalize_text(quote)
    if len(text) <= _MAX_QUOTE_CHARS:
        return text
    return text[: _MAX_QUOTE_CHARS - 1].rstrip() + "…"


def _empty_analysis(research_question: Optional[ResearchQuestion]) -> CriticalAnalysis:
    return CriticalAnalysis(
        topic=getattr(research_question, "topic", "") or "",
        research_question=getattr(research_question, "question", "") or "",
        status=CriticalAnalysis.STATUS_EMPTY,
        created_at=datetime.now(timezone.utc),
    )


__all__ = ["analyze_evidence", "CriticalAnalysis"]