"""
PersonaPulse – Topic Selection Module
======================================
Implements the Select Topic stage of the research agent workflow:

    Discover → TopicCandidates → Select Topic → ResearchQuestion

Given the candidates surfaced by discovery, this module:
  - evaluates each candidate against engineering-research relevance criteria,
  - picks ONE selected topic plus the reasoning and criteria used,
  - frames the ResearchQuestion for the selected topic.

Design notes
------------
- Selection is deliberately SEPARATE from research: this module never
  performs research, reference gathering, or any multi-source work.
- It never drafts or publishes a LinkedIn/X post (that is the MVP social
  pipeline, untouched by this phase).
- It is fully testable without a network: LLM calls flow through
  src.llm.complete_text (mock/disable via `use_llm=False`), and a
  deterministic heuristic fallback covers LLM failures.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from src.config import TOPIC_SELECTION_CRITERIA
from src.llm import complete_text
from src.models import ResearchQuestion, TopicCandidate, TopicSelection

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RESPONSE_TOKENS = 1024
_TEMPERATURE = 0.2
_MAX_ASPECTS = 6
_MAX_REASONING_WORDS = 200

# The selector's role and output contract only. Evaluation criteria are NOT
# hardcoded here: they come from TOPIC_SELECTION_CRITERIA (config) and are
# injected into the runtime/user prompt by _build_selection_prompt(). This
# keeps config the single source of truth for what the agent should prioritize.
_SELECTION_SYSTEM_PROMPT = """You are the topic-selection agent for an "AI Engineering Research Agent" that publishes technical analysis about agentic AI engineering.
Evaluate the provided candidates using the evaluation criteria supplied in the request, and choose the single most valuable topic for a deep engineering research brief.

OUTPUT CONTRACT - return VALID JSON ONLY. No markdown, no prose, no code fences.
Shape:
{"selected_index": <int or null>, "reasoning": "<string>", "evaluations": [{"index": <int>, "fit_score": <int 0-100>, "reason": "<string>", "strengths": "<string>", "concerns": "<string>"}]}

RULES:
- "selected_index" must be the 0-based integer index of your chosen candidate, or null when NO candidate fits the criteria.
- Provide EXACTLY ONE "evaluations" entry per candidate index 0..N-1 (evaluate every candidate).
- "fit_score" is an integer 0-100.
- "reason" is a short non-empty explanation that references the supplied criteria ("strengths" and "concerns" are optional extras).
- "reasoning" (max 200 words) must summarize the choice and reference at least two of the supplied criteria."""

_QUESTION_SYSTEM_PROMPT = """You are the question framer for an "AI Engineering Research Agent".
Given a selected topic, produce ONE sharp, researchable framing question plus 3-6 concrete aspects (sub-questions) to investigate.

REQUIREMENTS:
- Return VALID JSON ONLY. No markdown, no code fences.
- Shape: {"question": "<string>", "aspects": ["<string>", ...]}
- The question must be a non-empty string, answerable from public sources, and specific to agentic AI engineering.
- Each aspect must be a non-empty string.
- Do not propose running experiments or multi-source research plans; just frame the question."""


# ---------------------------------------------------------------------------
# Public: Select Topic
# ---------------------------------------------------------------------------

def select_topic(
    candidates: list[TopicCandidate],
    query: str = "",
    criteria: Optional[list[str]] = None,
    use_llm: bool = True,
) -> TopicSelection:
    """
    Select the single best topic candidate for the Engineering Research Agent.

    Parameters
    ----------
    candidates : list[TopicCandidate] from discovery (may be empty).
    query      : the search query that sparked the discovery (log context).
    criteria   : evaluation criteria; defaults to
                 settings TOPIC_SELECTION_CRITERIA.
    use_llm    : when False, skip the LLM and use the deterministic
                 heuristic (highest search_score). Useful for tests and
                 offline runs.

    Returns
    -------
    TopicSelection with:
      - selected  : the chosen candidate (None when nothing fit / empty)
      - reasoning : why it was chosen (or why nothing fit)
      - criteria  : the criteria actually used
      - evaluations : per-candidate verdicts when the LLM ran
      - mode      : "llm" | "heuristic" | "none_fit" | "empty"
      - created_at: selection time

    If the LLM call fails or returns an out-of-range index, falls back to a
    deterministic heuristic so the workflow can continue.
    """
    candidates = list(candidates or [])
    criteria = list(criteria or TOPIC_SELECTION_CRITERIA)
    now = datetime.now(timezone.utc)

    if not candidates:
        log.warning("[Selection] No candidates to select from.")
        return TopicSelection(
            reasoning="Discovery surfaced no topic candidates.",
            criteria=criteria,
            mode=TopicSelection.MODE_EMPTY,
            created_at=now,
        )

    if not use_llm:
        log.info("[Selection] LLM selection disabled – using heuristic.")
        return _heuristic_result(candidates, criteria, "LLM selection disabled", now)

    try:
        text = complete_text(
            _SELECTION_SYSTEM_PROMPT,
            _build_selection_prompt(candidates, criteria, query),
            max_tokens=_MAX_RESPONSE_TOKENS,
            temperature=_TEMPERATURE,
        )
        selected, reasoning, evaluations = _parse_llm_selection(text, candidates)
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Selection] LLM selection failed (%s) – falling back to heuristic.",
            type(exc).__name__,
        )
        return _heuristic_result(candidates, criteria, f"LLM selection failed ({type(exc).__name__})", now)

    if selected is None:
        log.warning("[Selection] LLM judged no candidate fit.")
        return TopicSelection(
            reasoning=reasoning or "No candidate met the selection criteria.",
            criteria=criteria,
            evaluations=evaluations,
            mode=TopicSelection.MODE_NONE_FIT,
            created_at=now,
        )

    log.info(
        "[Selection] Selected '%s' (%s) – mode=llm",
        selected.title, selected.source or "unknown source",
    )
    return TopicSelection(
        selected=selected,
        reasoning=reasoning,
        criteria=criteria,
        evaluations=evaluations,
        mode=TopicSelection.MODE_LLM,
        created_at=now,
    )


# ---------------------------------------------------------------------------
# Public: Frame Research Question
# ---------------------------------------------------------------------------

def frame_question(
    topic: Optional[TopicCandidate],
    query: str = "",
    use_llm: bool = True,
) -> Optional[ResearchQuestion]:
    """
    Frame the ResearchQuestion for the selected topic (status "proposed").

    Parameters
    ----------
    topic    : the selected TopicCandidate; None returns None.
    query    : the originating search query (log context).
    use_llm  : when False (or on LLM failure), falls back to a default
               framing question so the workflow never blocks.

    Returns
    -------
    ResearchQuestion with question + aspects, or None when *topic* is None.
    """
    if topic is None:
        return None

    now = datetime.now(timezone.utc)
    question = ""
    aspects: list[str] = []

    if use_llm:
        try:
            text = complete_text(
                _QUESTION_SYSTEM_PROMPT,
                _build_question_prompt(topic, query),
                max_tokens=_MAX_RESPONSE_TOKENS,
                temperature=_TEMPERATURE,
            )
            data = _extract_json(text)

            raw_question = data.get("question")
            if not isinstance(raw_question, str) or not raw_question.strip():
                raise ValueError("question missing, empty, or not a string")

            normalized_aspects = _normalize_aspects(data.get("aspects"))
            if not normalized_aspects:
                raise ValueError("no usable aspects")

            question = raw_question.strip()
            aspects = normalized_aspects
        except Exception as exc:  # pylint: disable=broad-except
            log.warning(
                "[Selection] Question framing LLM rejected (%s: %s) – using default question for '%s'.",
                type(exc).__name__, exc, topic.title,
            )

    if not question:
        question = _default_question(topic)
        log.info("[Selection] Using default framing question for '%s'.", topic.title)
    else:
        log.info("[Selection] Framed research question for '%s' (%d aspects).", topic.title, len(aspects))

    return ResearchQuestion(
        topic=topic.title,
        question=question,
        aspects=aspects,
        status=ResearchQuestion.STATUS_PROPOSED,
        priority=ResearchQuestion.PRIORITY_NORMAL,
        created_at=now,
    )


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_selection_prompt(
    candidates: list[TopicCandidate],
    criteria: list[str],
    query: str,
) -> str:
    rows = [
        {
            "index": idx,
            "title": c.title,
            "url": c.url,
            "source": c.source,
            "published": c.published,
            "description": c.description[:320],
            "keywords": list(c.keywords),
            "search_score": c.search_score,
        }
        for idx, c in enumerate(candidates)
    ]
    criteria_text = "\n".join(f"{i + 1}. {c.strip()}" for i, c in enumerate(criteria))
    query_line = f'The research brief was sparked by the query "{query}".' if query else ""
    return (
        f"{query_line}\n\n"
        f"Evaluation criteria:\n{criteria_text}\n\n"
        f"Candidates (JSON):\n{json.dumps(rows, indent=2)}"
    ).strip()


def _build_question_prompt(topic: TopicCandidate, query: str) -> str:
    payload = {
        "title": topic.title,
        "url": topic.url,
        "source": topic.source,
        "description": topic.description[:320],
        "keywords": list(topic.keywords),
    }
    context = f'\nResearch context query: "{query}"' if query else ""
    return (
        f"Selected topic (JSON):\n{json.dumps(payload, indent=2)}"
        f"{context}\n\nFrame one sharp, researchable question with concrete aspects."
    )


# ---------------------------------------------------------------------------
# LLM output parsing
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Best-effort parse of a JSON object from an LLM response."""
    if not text:
        raise ValueError("Empty LLM response")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end < start:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(cleaned[start:end + 1])


def _parse_llm_selection(
    text: str,
    candidates: list[TopicCandidate],
) -> tuple[Optional[TopicCandidate], str, list[dict]]:
    """
    Parse and STRICTLY validate the selector's JSON into
    (selected, reasoning, evaluations).

    Raises ValueError on any contract violation – malformed JSON, a
    non-integer/out-of-range selected_index, an invalid evaluation, an
    invalid fit_score, invalid reason, or incomplete/duplicate evaluation
    coverage – so select_topic falls back to the deterministic heuristic.
    No TopicSelection is constructed until every part validates.
    """
    data = _extract_json(text)
    reasoning = _validate_reasoning(data.get("reasoning"))
    # Evaluations are validated first so a response with a valid index but
    # invalid/incomplete evaluations is still rejected as a whole.
    evaluations = _validate_evaluations(data.get("evaluations"), candidates)
    selected = _validate_selected_index(data.get("selected_index"), candidates)
    return selected, reasoning, evaluations


def _validate_reasoning(raw) -> str:
    """
    Validate the LLM's "reasoning" string.

    Accepts only a non-empty string of at most 200 whitespace-separated
    words. Rejects (raises ValueError): numbers, booleans, lists, dicts,
    nulls, empty/whitespace-only strings, and anything over 200 words.
    No coercion (str()) is performed.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"reasoning must be a non-empty string, got {raw!r}")
    reasoning = raw.strip()
    word_count = len(reasoning.split())
    if not 0 < word_count <= _MAX_REASONING_WORDS:
        raise ValueError(f"reasoning must be 1-{_MAX_REASONING_WORDS} words, got {word_count}")
    return reasoning


def _is_strict_int(value) -> bool:
    """True only for a plain Python ``int``. Rejects ``bool`` (a subclass of
    int), floats, strings, and everything else."""
    return type(value) is int


def _validate_selected_index(
    raw,
    candidates: list[TopicCandidate],
) -> Optional[TopicCandidate]:
    """
    Validate the LLM's "selected_index".

    Accepts only:
      - null  → no candidate fits (caller uses mode "none_fit")
      - int   with 0 <= value < len(candidates) → that candidate

    Rejects (raises ValueError): floats, strings, booleans, negatives,
    out-of-range integers. No coercion is performed.
    """
    if raw is None:
        return None
    if not _is_strict_int(raw):
        raise ValueError(f"selected_index must be an integer or null, got {raw!r}")
    if not (0 <= raw < len(candidates)):
        raise ValueError(f"selected_index {raw} out of range (0..{len(candidates) - 1})")
    return candidates[raw]


def _validate_fit_score(value) -> int:
    """Validate an evaluation "fit_score": a strict int in [0, 100].

    Rejects floats, booleans, strings, and nulls. No coercion is performed
    (a violated contract invalidates the whole LLM response).
    """
    if type(value) is not int:
        raise ValueError(f"fit_score must be an integer, got {value!r}")
    if not (0 <= value <= 100):
        raise ValueError(f"fit_score out of range (0-100): {value!r}")
    return value


def _validate_evaluations(raw, candidates: list[TopicCandidate]) -> list[dict]:
    """
    Strictly validate the LLM's "evaluations".

    Contract: exactly ONE evaluation per candidate index (0..N-1), each with:
      - index      : plain int, 0 <= index < len(candidates)
      - fit_score  : numeric (not bool), 0 <= fit_score <= 100
      - reason     : non-empty string (trimmed)
      - strengths/concerns : optional strings (kept when present)

    Returns a normalized list ordered by candidate index. Raises ValueError
    for missing/duplicated/out-of-range/negative/non-integer indices,
    invalid fit_scores, or missing/empty reasons, so the caller falls back.
    """
    if not isinstance(raw, list):
        raise ValueError("evaluations must be a list")
    if len(candidates) == 0:
        raise ValueError("cannot validate evaluations without candidates")

    by_index: dict[int, dict] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError(f"evaluation must be a JSON object, got {item!r}")

        idx = item.get("index")
        if not _is_strict_int(idx):
            raise ValueError(f"evaluation index must be an integer, got {idx!r}")
        if not (0 <= idx < len(candidates)):
            raise ValueError(f"evaluation index {idx} out of range (0..{len(candidates) - 1})")
        if idx in by_index:
            raise ValueError(f"duplicate evaluation for candidate index {idx}")

        fit_score = _validate_fit_score(item.get("fit_score"))

        reason = item.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"evaluation {idx} missing a non-empty 'reason'")

        strengths = item.get("strengths")
        concerns = item.get("concerns")
        by_index[idx] = {
            "index": idx,
            "fit_score": fit_score,
            "reason": reason.strip(),
            "strengths": strengths.strip() if isinstance(strengths, str) else "",
            "concerns": concerns.strip() if isinstance(concerns, str) else "",
        }

    if len(by_index) != len(candidates):
        missing = sorted(set(range(len(candidates))) - set(by_index))
        raise ValueError(
            f"evaluation coverage incomplete; missing candidate indices {missing}"
        )

    return [by_index[i] for i in range(len(candidates))]


# ---------------------------------------------------------------------------
# Heuristic fallback (deterministic, no LLM)
# ---------------------------------------------------------------------------

def _heuristic_select(candidates: list[TopicCandidate]) -> Optional[TopicCandidate]:
    """Pick the candidate with the highest search_score (title tie-break)."""
    ranked = sorted(candidates, key=lambda c: (c.search_score, c.title.casefold()), reverse=True)
    return ranked[0] if ranked else None


def _heuristic_result(
    candidates: list[TopicCandidate],
    criteria: list[str],
    reason: str,
    now: datetime,
) -> TopicSelection:
    selected = _heuristic_select(candidates)
    reasoning = (
        f"{reason}. Fallback chosen by provider search_score "
        f"({selected.search_score:.2f}); no LLM evaluation used."
        if selected is not None
        else f"{reason}. No candidates available."
    )
    log.warning(
        "[Selection] Heuristic fallback selected '%s'",
        selected.title if selected is not None else None,
    )
    return TopicSelection(
        selected=selected,
        reasoning=reasoning,
        criteria=criteria,
        mode=TopicSelection.MODE_HEURISTIC,
        created_at=now,
    )


def _default_question(topic: TopicCandidate) -> str:
    """Deterministic fallback framing question when the LLM is unavailable."""
    title = topic.title.strip() or "this topic"
    return (
        f"What is the current engineering state of the art for {title}, "
        f"and what are its practical implications for building production "
        f"agentic AI systems?"
    )


def _normalize_aspects(raw) -> Optional[list[str]]:
    """
    Normalize the LLM's "aspects" field.

    Accepts only a list. Each entry must be a non-empty string: items are
    trimmed, empty entries dropped, duplicates removed (order preserved),
    and the list capped at _MAX_ASPECTS. Non-string entries (numbers,
    dicts, lists, booleans) are rejected individually and skipped.

    Returns None when *raw* is not a list (invalid structure), and an
    (possibly empty) normalized list otherwise.
    """
    if not isinstance(raw, list):
        return None
    seen: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
        if len(seen) >= _MAX_ASPECTS:
            break
    return seen