"""
PersonaPulse – Domain Models
============================
Foundational data structures for the AI Engineering Research Agent phase.

They map onto the target workflow:

    Discover → Question → Research → Evidence → Critical Analysis → Synthesize

Models
------
- TopicCandidate   : a candidate topic surfaced during Discovery
- TopicSelection   : the outcome of Selecting one candidate (topic + rationale)
- ResearchQuestion : the framing question (+ aspects) for a topic
- ResearchSource   : one source gathered during Research (backward compatible
                     with the existing ingestion article dict shape)
- Evidence         : a claim pulled from a source, with attribution
                     (Claim is provided as an alias)
- ResearchReport   : the assembled, analysed document synthesizing the above

Every model provides to_dict()/from_dict() so the existing LangGraph state
(pure dicts) and Supabase JSONB storage patterns keep working unchanged.

This is intentionally minimal: only data + serialization + naming.
Workflow logic arrives in later phases.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _iso(value: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime to ISO-8601 string (or None)."""
    return value.isoformat() if value is not None else None


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse a datetime from ISO-8601 string, passing datetimes through."""
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        log.warning("[Models] Unrecognized datetime value: %r", value)
        return None


# ---------------------------------------------------------------------------
# TopicCandidate – Discovery output
# ---------------------------------------------------------------------------

@dataclass
class TopicCandidate:
    """
    A single candidate topic surfaced during the Discovery stage.

    Scope is deliberately limited to discovery output: title/URL/keywords
    plus the raw search-provider relevance hint. Selection scoring happens
    in a later phase and is NOT stored here.

    Fields
    ------
    title         : short human-readable topic label
    url           : canonical URL of the source article (may be empty)
    description   : short description/summary of the topic (1–2 sentences)
    keywords      : search/discovery keywords preserved from the provider
                    ([] when the provider returned none)
    source        : publisher/domain the candidate was surfaced from
    published     : publication date string when available (may be empty)
    search_score  : 0.0–1.0 raw search relevance from the discovery
                    provider; a ranking hint only, not the selection score
    discovered_at : when the topic was surfaced
    """
    title: str
    description: str = ""
    url: str = ""
    keywords: list = field(default_factory=list)
    source: Optional[str] = None
    published: str = ""
    search_score: float = 0.0
    discovered_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "url": self.url,
            "description": self.description,
            "keywords": list(self.keywords),
            "source": self.source,
            "published": self.published,
            "search_score": self.search_score,
            "discovered_at": _iso(self.discovered_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TopicCandidate":
        # Accept the pre-rename "relevance_score" key for backward compatibility.
        score_key = "search_score" if "search_score" in data else "relevance_score"
        return cls(
            title=str(data.get("title", "")),
            url=str(data.get("url", "")),
            description=str(data.get("description", "")),
            keywords=list(data.get("keywords") or []),
            source=data.get("source"),
            published=str(data.get("published", "")),
            search_score=float(data.get(score_key, 0.0)),
            discovered_at=_parse_iso(data.get("discovered_at")),
        )


# ---------------------------------------------------------------------------
# TopicSelection – Selection output
# ---------------------------------------------------------------------------

@dataclass
class TopicSelection:
    """
    The outcome of the Selection stage: the single chosen topic plus the
    reasoning and criteria used to choose it.

    Selection is deliberately separate from research: this model records
    WHAT was chosen and WHY, never the research itself.

    Fields
    ------
    selected     : the chosen TopicCandidate, or None when nothing fit
    reasoning    : human-readable explanation of the selection
    criteria     : list[str] of criteria the candidates were evaluated against
    evaluations  : list[dict] of per-candidate verdicts (LLM row shape:
                   {index, fit_score, strengths, concerns})
    mode         : how the choice was made – "llm" | "heuristic" |
                   "none_fit" | "empty"
    created_at   : when the selection was made
    """
    selected: Optional["TopicCandidate"] = None
    reasoning: str = ""
    criteria: list = field(default_factory=list)
    evaluations: list = field(default_factory=list)
    mode: str = "llm"
    created_at: Optional[datetime] = None

    MODE_LLM = "llm"
    MODE_HEURISTIC = "heuristic"
    MODE_NONE_FIT = "none_fit"
    MODE_EMPTY = "empty"

    def to_dict(self) -> dict:
        return {
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "reasoning": self.reasoning,
            "criteria": list(self.criteria),
            "evaluations": list(self.evaluations),
            "mode": self.mode,
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TopicSelection":
        selected = data.get("selected")
        return cls(
            selected=(
                TopicCandidate.from_dict(selected) if isinstance(selected, dict) else None
            ),
            reasoning=str(data.get("reasoning", "")),
            criteria=list(data.get("criteria") or []),
            evaluations=list(data.get("evaluations") or []),
            mode=str(data.get("mode", cls.MODE_LLM)),
            created_at=_parse_iso(data.get("created_at")),
        )


# ---------------------------------------------------------------------------
# ResearchQuestion – Question output
# ---------------------------------------------------------------------------

@dataclass
class ResearchQuestion:
    """
    The framing question derived from a topic, plus concrete aspects to
    research. Status moves proposed → researching → answered / dropped.
    """
    topic: str
    question: str
    aspects: list = field(default_factory=list)
    status: str = "proposed"
    priority: str = "normal"
    created_at: Optional[datetime] = None

    STATUS_PROPOSED = "proposed"
    STATUS_RESEARCHING = "researching"
    STATUS_ANSWERED = "answered"
    STATUS_DROPPED = "dropped"

    PRIORITY_HIGH = "high"
    PRIORITY_NORMAL = "normal"
    PRIORITY_LOW = "low"

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "question": self.question,
            "aspects": list(self.aspects),
            "status": self.status,
            "priority": self.priority,
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchQuestion":
        return cls(
            topic=str(data.get("topic", "")),
            question=str(data.get("question", "")),
            aspects=list(data.get("aspects") or []),
            status=str(data.get("status", cls.STATUS_PROPOSED)),
            priority=str(data.get("priority", cls.PRIORITY_NORMAL)),
            created_at=_parse_iso(data.get("created_at")),
        )


# ---------------------------------------------------------------------------
# ResearchSource – Research output (backward compatible with ingestion dict)
# ---------------------------------------------------------------------------

@dataclass
class ResearchSource:
    """
    A single source gathered during the Research stage.

    to_dict() emits the same keys as the existing ingestion article dict
    (url, title, body, published, source) followed by research-only fields,
    so a ResearchSource can be used anywhere an article dict was expected.
    """
    url: str
    title: str = ""
    body: str = ""
    published: str = ""
    source: str = ""
    score: float = 0.0              # relevance 0.0–1.0
    source_type: str = "secondary"  # primary | secondary
    accessed_at: Optional[datetime] = None

    SOURCE_TYPE_PRIMARY = "primary"
    SOURCE_TYPE_SECONDARY = "secondary"

    @staticmethod
    def from_article(article: dict, **overrides) -> "ResearchSource":
        """
        Wrap the exact dict produced by ingestion.fetch_trending_tech_news()
        so existing Discovery output can be reused as a ResearchSource.
        """
        base = {
            "url": article.get("url", ""),
            "title": article.get("title", ""),
            "body": article.get("body", ""),
            "published": article.get("published", ""),
            "source": article.get("source", ""),
        }
        base.update(overrides)
        return ResearchSource(**base)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "body": self.body,
            "published": self.published,
            "source": self.source,
            "score": self.score,
            "source_type": self.source_type,
            "accessed_at": _iso(self.accessed_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchSource":
        return cls(
            url=str(data.get("url", "")),
            title=str(data.get("title", "")),
            body=str(data.get("body", "")),
            published=str(data.get("published", "")),
            source=str(data.get("source", "")),
            score=float(data.get("score", 0.0)),
            source_type=str(data.get("source_type", cls.SOURCE_TYPE_SECONDARY)),
            accessed_at=_parse_iso(data.get("accessed_at")),
        )


# ---------------------------------------------------------------------------
# Evidence / Claim – Evidence output
# ---------------------------------------------------------------------------

@dataclass
class Evidence:
    """
    A single claim pulled from a source during the Evidence stage, including
    attribution and a verification status:
      claimed → corroborated | contradicted

    Fields
    ------
    claim_text         : the extracted claim (factual assertion)
    source_url         : the URL the claim is traceable to (required in practice;
                         defaults to None so empty dicts still deserialize)
    supporting_quote   : verbatim snippet from the source backing the claim
    confidence         : 0.0–1.0 extraction confidence
    context            : surrounding context that makes the claim meaningful
    directly_supports  : True when the source states the claim directly,
                         False when it only implies/relates to it
    verification_status: claimed | corroborated | contradicted
    """
    claim_text: str
    source_url: Optional[str] = None
    supporting_quote: str = ""
    confidence: float = 0.0         # 0.0–1.0
    context: str = ""
    directly_supports: bool = True
    verification_status: str = "claimed"

    VERIFICATION_CLAIMED = "claimed"
    VERIFICATION_CORROBORATED = "corroborated"
    VERIFICATION_CONTRADICTED = "contradicted"

    def to_dict(self) -> dict:
        return {
            "claim_text": self.claim_text,
            "source_url": self.source_url,
            "supporting_quote": self.supporting_quote,
            "confidence": self.confidence,
            "context": self.context,
            "directly_supports": self.directly_supports,
            "verification_status": self.verification_status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Evidence":
        return cls(
            claim_text=str(data.get("claim_text", "")),
            source_url=data.get("source_url"),
            supporting_quote=str(data.get("supporting_quote", "")),
            confidence=float(data.get("confidence", 0.0)),
            context=str(data.get("context", "")),
            directly_supports=bool(data.get("directly_supports", True)),
            verification_status=str(
                data.get("verification_status", cls.VERIFICATION_CLAIMED)
            ),
        )


# Claim is an alias for Evidence – the two names refer to the same concept.
Claim = Evidence


# ---------------------------------------------------------------------------
# ResearchReport – Critical Analysis / Synthesize output
# ---------------------------------------------------------------------------

@dataclass
class ResearchReport:
    """
    The assembled output document combining questions, sources, and evidence
    with the critical analysis/synthesis narrative and conclusions.
    """
    topic: str
    summary: str = ""                       # Critical Analysis + Synthesis narrative
    conclusions: list = field(default_factory=list)
    questions: list = field(default_factory=list)   # list[ResearchQuestion]
    sources: list = field(default_factory=list)     # list[ResearchSource]
    evidence: list = field(default_factory=list)    # list[Evidence]
    confidence_score: float = 0.0
    created_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "conclusions": list(self.conclusions),
            "questions": [q.to_dict() for q in self.questions],
            "sources": [s.to_dict() for s in self.sources],
            "evidence": [e.to_dict() for e in self.evidence],
            "confidence_score": self.confidence_score,
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResearchReport":
        return cls(
            topic=str(data.get("topic", "")),
            summary=str(data.get("summary", "")),
            conclusions=list(data.get("conclusions") or []),
            questions=[
                ResearchQuestion.from_dict(item)
                for item in (data.get("questions") or [])
            ],
            sources=[
                ResearchSource.from_dict(item)
                for item in (data.get("sources") or [])
            ],
            evidence=[
                Evidence.from_dict(item)
                for item in (data.get("evidence") or [])
            ],
            confidence_score=float(data.get("confidence_score", 0.0)),
            created_at=_parse_iso(data.get("created_at")),
        )


__all__ = [
    "TopicCandidate",
    "TopicSelection",
    "ResearchQuestion",
    "ResearchSource",
    "Evidence",
    "Claim",
    "ResearchReport",
]