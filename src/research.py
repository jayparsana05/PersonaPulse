"""
PersonaPulse – Research Module (Prompt 4)
==========================================
Turns a framed ResearchQuestion into a collection of normalized,
deduplicated ResearchSource objects via multi-source web search.

Flow
----
ResearchQuestion
    ↓
generate_research_queries()   → complementary search queries
    ↓
multi-source search           → one provider search per query (failures tolerated)
    ↓
normalize                     → provider results → ResearchSource
    ↓
deduplicate                   → normalized-URL persistence, strongest metadata wins
    ↓
ResearchSource[]

Key Functions
-------------
- generate_research_queries(research_question, ...) → list[str]
- research_question(research_question, ...)         → list[ResearchSource]

The stage deliberately STOPS at ResearchSource[]: evidence extraction,
synthesis, and research reports are future phases.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from src.config import settings
from src.ingestion import _clean_body, _extract_domain, _normalize_url, search_news
from src.llm import complete_text
from src.models import ResearchQuestion, ResearchSource
from src.selection import _extract_json

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RESPONSE_TOKENS = 1024
_TEMPERATURE = 0.2
_RECENCY_DAYS = 7                 # match the discovery recency window
_MAX_SNIPPET_CHARS = 800          # cap on the stored content snippet/body


# ---------------------------------------------------------------------------
# Research query generation
# ---------------------------------------------------------------------------

_RESEARCH_QUERY_SYSTEM_PROMPT = """You are the research-query planner for an "AI Engineering Research Agent".
Given a research question and its aspects, produce a small set of complementary, specific search queries that together cover the question and its most important aspects for multi-source research.

OUTPUT CONTRACT - return VALID JSON ONLY. No markdown, no code fences.
Shape: {{"queries": ["<string>", ...]}}

RULES:
- Return up to {max_queries} queries.
- Each query must be a non-empty string with useful specificity (not a single generic word).
- Queries must be complementary – no duplicates.
- The first query should capture the primary research question.
- Return only search queries; do not research, answer, or summarize."""


def generate_research_queries(
    research_question: Optional[ResearchQuestion],
    use_llm: bool = True,
    max_queries: Optional[int] = None,
) -> list[str]:
    """
    Produce the search queries for *research_question*.

    Uses the LLM to draft complementary queries from the question + aspects.
    On malformed output, LLM failure, or use_llm=False, falls back to a
    deterministic set (question + aspect-anchored queries) so research never
    blocks on the LLM.

    Never returns an empty query, never returns the same query twice, and
    respects the configured maximum count. Does NOT run any search itself.

    Returns
    -------
    list[str] of whitespace-normalized, unique queries (may be [] only when
    the research question carries no usable text at all).
    """
    limit = max(1, int(max_queries if max_queries is not None else settings.RESEARCH_MAX_QUERIES))

    if research_question is None:
        return []

    if use_llm:
        try:
            text = complete_text(
                _RESEARCH_QUERY_SYSTEM_PROMPT.format(max_queries=limit),
                _build_query_prompt(research_question, limit),
                max_tokens=_MAX_RESPONSE_TOKENS,
                temperature=_TEMPERATURE,
            )
            data = _extract_json(text)
            raw = data.get("queries")
            if not isinstance(raw, list):
                raise ValueError("'queries' must be a list")
            queries = _normalize_queries(raw, limit)
            if queries:
                log.info("[Research] Generated %d query(ies) for '%s'.", len(queries), research_question.question)
                return queries
        except Exception as exc:  # pylint: disable=broad-except
            log.warning(
                "[Research] Query generation LLM rejected (%s: %s) – using deterministic fallback queries.",
                type(exc).__name__, exc,
            )

    return _fallback_queries(research_question, limit)


def _build_query_prompt(research_question: ResearchQuestion, max_queries: int) -> str:
    payload = {
        "topic": research_question.topic,
        "question": research_question.question,
        "aspects": list(research_question.aspects),
    }
    return (
        f"Research question (JSON):\n{json.dumps(payload, indent=2)}\n\n"
        f"Generate up to {max_queries} complementary search queries."
    )


def _normalize_queries(queries, max_queries: int) -> list[str]:
    """Whitespace-normalize, drop empties/non-strings, deduplicate (case-fold),
    and cap at *max_queries*, preserving first-occurrence order."""
    seen: set[str] = set()
    result: list[str] = []
    for raw in queries:
        if not isinstance(raw, str):
            continue
        query = " ".join(raw.split())
        if not query:
            continue
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(query)
        if len(result) >= max_queries:
            break
    return result


def _fallback_queries(research_question: ResearchQuestion, max_queries: int) -> list[str]:
    """Deterministic fallback: the question itself, then each aspect anchored
    to the topic/subject. No LLM involved."""
    anchor = (research_question.topic or "").strip()
    question = (research_question.question or "").strip()

    raw_queries: list[str] = []
    if question:
        raw_queries.append(question)
    for aspect in research_question.aspects:
        if not isinstance(aspect, str):
            continue
        aspect = aspect.strip()
        if not aspect:
            continue
        raw_queries.append(f"{aspect} {anchor}".strip() if anchor else aspect)

    queries = _normalize_queries(raw_queries, max_queries)
    if queries:
        return queries
    # Last resort: a bare topic anchor keeps research actionable without
    # fabricating specificity. Returns [] when there is literally no text.
    return [anchor] if anchor else []


# ---------------------------------------------------------------------------
# Multi-source search
# ---------------------------------------------------------------------------

def research_question(
    research_question: Optional[ResearchQuestion],
    use_llm: bool = True,
    max_queries: Optional[int] = None,
    max_sources_per_query: Optional[int] = None,
    max_sources: Optional[int] = None,
    min_score: Optional[float] = None,
) -> list[ResearchSource]:
    """
    Research *research_question* across multiple queries and sources.

    Steps: generate queries → search each query → normalize provider results
    into ResearchSource → URL-normalize + dedupe → apply the final source cap.

    Resilient by design: a failing query is logged and skipped; other
    queries still contribute. When everything fails or yields nothing, an
    empty list is returned (never fabricated sources).

    Parameters
    ----------
    research_question    : the framed question to research (None → [])
    use_llm              : use the LLM for query generation (fallback otherwise)
    max_queries          : override for settings.RESEARCH_MAX_QUERIES
    max_sources_per_query: override for settings.RESEARCH_MAX_SOURCES_PER_QUERY
    max_sources          : override for settings.RESEARCH_MAX_SOURCES
    min_score            : override for settings.RESEARCH_MIN_SCORE

    Returns
    -------
    list[ResearchSource], deduplicated by normalized URL, deterministically
    ordered by relevance (provider search score), never exceeding the limit.
    """
    if research_question is None:
        return []

    per_query = max(1, int(max_sources_per_query if max_sources_per_query is not None else settings.RESEARCH_MAX_SOURCES_PER_QUERY))
    final_cap = max(0, int(max_sources if max_sources is not None else settings.RESEARCH_MAX_SOURCES))
    threshold = float(min_score if min_score is not None else settings.RESEARCH_MIN_SCORE)

    queries = generate_research_queries(
        research_question,
        use_llm=use_llm,
        max_queries=max_queries,
    )
    log.info("[Research] %d query/queries for '%s'.", len(queries), research_question.question or "(no question)")

    raw_sources: list[ResearchSource] = []
    for query in queries:
        try:
            results = _search_query(query, per_query, threshold)
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("[Research] Query '%s' failed (%s) – continuing with remaining queries.", query, type(exc).__name__)
            continue
        if not results:
            log.debug("[Research] Query '%s' returned no results – continuing.", query)
            continue
        for item in results:
            source = _source_from_result(item)
            if source is not None:
                raw_sources.append(source)

    if not raw_sources:
        log.warning("[Research] No usable sources for '%s'.", research_question.question or "(no question)")
        return []

    return _dedupe_sources(raw_sources, final_cap)


def _search_query(query: str, max_results: int, min_score: float) -> list[dict]:
    """Run one provider search and apply the optional relevance floor.

    Provider results without a score are kept irrespective of the floor
    (metadata is never invented); scored results must meet it.
    """
    results = search_news(
        query=query,
        days=_RECENCY_DAYS,
        max_results=max_results,
        include_raw_content=False,      # snippets are enough for source collection
    )
    if min_score and min_score > 0:
        kept = [item for item in results if _passes_min_score(item, min_score)]
        log.debug("[Research] Relevance floor %.2f kept %d/%d results.", min_score, len(kept), len(results))
        return kept
    return results


def _passes_min_score(item: dict, min_score: float) -> bool:
    score = _provider_score(item)
    if score is None:
        return True                     # no provider score → don't drop on missing data
    return score >= min_score


# ---------------------------------------------------------------------------
# Normalization into ResearchSource
# ---------------------------------------------------------------------------

def _source_from_result(item: dict) -> Optional[ResearchSource]:
    """Map one raw provider result into a ResearchSource.

    Only the provider-supplied metadata is preserved; missing optional fields
    degrade to model defaults rather than being invented. Results without a
    usable URL are skipped.
    """
    url = (item.get("url") or "").strip()
    if not url:
        log.debug("[Research] Skipping result without a URL: %r", item.get("title"))
        return None

    snippet = item.get("content") or item.get("raw_content") or ""
    score = _provider_score(item)
    provider_published = (item.get("published_date") or "").strip()

    return ResearchSource(
        url=url,
        title=(item.get("title") or "").strip(),
        body=_truncate_snippet(snippet),
        published=provider_published,
        source=_extract_domain(url),
        score=max(0.0, min(1.0, score or 0.0)),
        source_type=ResearchSource.SOURCE_TYPE_SECONDARY,
        accessed_at=datetime.now(timezone.utc),
    )


def _provider_score(item: dict) -> Optional[float]:
    """Tavily relevance score (0.0–1.0) as float, or None when absent/garbage."""
    try:
        score = float(item.get("score"))
    except (TypeError, ValueError):
        return None
    return score


def _truncate_snippet(text: str, max_chars: int = _MAX_SNIPPET_CHARS) -> str:
    cleaned = _clean_body(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# URL normalization + deduplication
# ---------------------------------------------------------------------------

def _dedupe_sources(sources: list[ResearchSource], max_sources: int) -> list[ResearchSource]:
    """Keep one ResearchSource per normalized URL, then apply the final cap.

    For a duplicated URL the strongest entry wins: highest provider score,
    then a non-empty body, then the longer title (deterministic tie-breaks).
    Final ordering is deterministic: relevance score desc, title length desc,
    normalized URL asc.
    """
    best: dict[str, ResearchSource] = {}
    for source in sources:
        key = _normalize_url(source.url) or source.url
        current = best.get(key)
        if current is None or _higher_quality(source, current):
            best[key] = source

    ranked = sorted(
        best.values(),
        key=lambda s: (-s.score, -len(s.title), _normalize_url(s.url)),
    )
    return ranked[: max(0, max_sources)]


def _higher_quality(candidate: ResearchSource, current: ResearchSource) -> bool:
    if candidate.score != current.score:
        return candidate.score > current.score
    candidate_has_body = bool((candidate.body or "").strip())
    current_has_body = bool((current.body or "").strip())
    if candidate_has_body != current_has_body:
        return candidate_has_body
    if len(candidate.title) != len(current.title):
        return len(candidate.title) > len(current.title)
    return False          # fully equivalent → keep the earlier entry


__all__ = [
    "generate_research_queries",
    "research_question",
    "ResearchSource",
]