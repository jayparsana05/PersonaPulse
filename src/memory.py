"""
PersonaPulse – Memory Module
============================
Handles vector embeddings (Gemini gemini-embedding-001) with
L2 normalization, and semantic deduplication against Supabase.

Two kinds of memory live here:

1. Post memory (production pipeline) – embeddings + pgvector cosine
   distance against the ``posts`` table via ``match_posts_by_embedding``.

2. Research-session memory (research agent) – remembers researched
   topics/questions so the same topic is never re-searched. Exact
   (normalized text) matches are decided first; semantic (embedding
   cosine) matches reuse the same threshold machinery. Sources already
   recorded in earlier sessions are kept source-level deduplicated
   across sessions as well.

Key Functions
-------------
- get_normalized_embedding(text)  → list[float]
- check_is_duplicate(embedding, threshold)  → bool
- store_draft(platform, topic, content, embedding, article_url, image_url) → str  (UUID)
- check_topic_researched(topic, question, threshold, use_embedding)  → dict
- get_known_research_questions(limit)  → list[dict]
- get_research_sources_for_question(question_id)  → (list[ResearchSource], list[str])
- store_research_question(question, embedding)  → str  (UUID)
- store_research_sources(question_id, sources)  → list[str]  (UUIDs)
- delete_research_question(question_id)          → None
- get_known_source_urls()  → set[str]
- filter_repeated_sources(sources, known_urls)  → (list, list)
- update_post_status(post_id, status)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from google import genai
from google.genai import types as genai_types
from supabase import create_client, Client

from src.config import settings
from src.ingestion import _normalize_url
from src.models import ResearchQuestion, ResearchSource

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Clients (initialized lazily at module level for reuse)
# ---------------------------------------------------------------------------

_gemini_client: genai.Client | None = None
_supabase_client: Client | None = None


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


def _get_supabase() -> Client:
    global _supabase_client
    if _supabase_client is None:
        _supabase_client = create_client(
            settings.SUPABASE_URL,
            settings.SUPABASE_SERVICE_ROLE_KEY,
        )
    return _supabase_client


def _vector_literal(embedding: list[float]) -> str:
    """Serialize a float embedding as a PostgreSQL vector literal."""
    return "[" + ",".join(f"{x:.8f}" for x in embedding) + "]"


def _normalize_research_text(text) -> str:
    """Canonical dedup key for research topics/questions.

    Case-folds and collapses whitespace so "Agentic   AI " and "agentic ai"
    compare equal, without inventing meaning (unlike a fuzzy match).
    """
    return " ".join((text or "").split()).casefold()


# ---------------------------------------------------------------------------
# Public: Embedding with L2 Normalization
# ---------------------------------------------------------------------------

def get_normalized_embedding(text: str) -> list[float]:
    """
    Generate a 768-dimensional, L2-normalized embedding for *text*
    using Google's gemini-embedding-001 model.

    Why normalize?
    --------------
    The Gemini API truncates the raw output vector to 768 dims,
    which breaks the original unit-length guarantee.  Dividing by
    the L2 norm (||v|| = sqrt(Σ xᵢ²)) restores magnitude 1.0, which
    is a precondition for meaningful cosine-similarity comparisons.
    """
    client = _get_gemini()

    response = client.models.embed_content(
        model=settings.EMBEDDING_MODEL,
        contents=text,
        config=genai_types.EmbedContentConfig(
            output_dimensionality=settings.EMBEDDING_DIMENSIONS,
        ),
    )

    raw_vector: list[float] = response.embeddings[0].values

    # ── L2 Normalization ────────────────────────────────────────────────
    vec = np.array(raw_vector, dtype=np.float64)
    norm = np.linalg.norm(vec)                   # sqrt(Σ xᵢ²)

    if norm < 1e-10:
        log.warning("[Memory] Embedding norm is near-zero – returning zero vector.")
        return raw_vector                        # edge-case: empty / garbage text

    normalized = (vec / norm).tolist()
    log.debug("[Memory] Embedding norm before=%.6f  after=%.6f", norm, np.linalg.norm(normalized))
    return normalized


# ---------------------------------------------------------------------------
# Public: Semantic Deduplication
# ---------------------------------------------------------------------------

def check_is_duplicate(
    embedding: list[float],
    threshold: float = 0.85,
) -> bool:
    """
    Return True if a post with cosine similarity > *threshold* already
    exists in the `posts` table (covers PUBLISHED and PENDING statuses).

    Cosine *distance* in pgvector: 0 = identical, 2 = opposite.
    Cosine *similarity* = 1 - cosine_distance.
    We filter where  (1 - distance) > threshold  ↔  distance < (1 - threshold).
    """
    supabase = _get_supabase()
    distance_cutoff = 1.0 - threshold            # e.g. 0.15 for threshold=0.85

    # Format the embedding as a PostgreSQL vector literal
    vector_literal = _vector_literal(embedding)

    # Use raw SQL via RPC to leverage the pgvector operator
    result = supabase.rpc(
        "match_posts_by_embedding",
        {
            "query_embedding": vector_literal,
            "match_threshold": distance_cutoff,
            "match_count": 1,
        },
    ).execute()

    if result.data:
        log.info(
            "[Memory] Duplicate detected – similar post id=%s",
            result.data[0].get("id", "unknown"),
        )
        return True

    return False


# ---------------------------------------------------------------------------
# Public: Store Draft Post
# ---------------------------------------------------------------------------

def store_draft(
    platform: str,
    topic: str,
    content: str,
    embedding: list[float],
    article_url: Optional[str] = None,
    image_url: Optional[str] = None,
) -> str:
    """
    Insert a new PENDING draft into the `posts` table.
    Returns the generated UUID of the new row.

    Per-platform status columns (linkedin_status / x_status) start as PENDING
    alongside the top-level `status` so publication can be tracked and retired
    per platform (see the Telegram approval edge function).
    """
    supabase = _get_supabase()

    vector_literal = _vector_literal(embedding)

    row = {
        "platform": platform,
        "topic": topic,
        "content": content,
        "article_url": article_url,
        "image_url": image_url,
        "embedding": vector_literal,
        "status": "PENDING",
        "linkedin_status": "PENDING",
        "x_status": "PENDING",
    }

    result = supabase.table("posts").insert(row).execute()
    post_id: str = result.data[0]["id"]
    log.info("[Memory] Draft stored – id=%s  platform=%s", post_id, platform)
    return post_id


# ---------------------------------------------------------------------------
# Public: Store Research Question
# ---------------------------------------------------------------------------

def store_research_question(
    question: ResearchQuestion,
    embedding: Optional[list[float]] = None,
) -> str:
    """
    Insert the framing research question into the `research_questions`
    table, keeping topic + question + created_at together for traceability.

    When *embedding* is provided (embedding of topic + question), it is stored
    in the ``embedding`` column so later runs can recognise the same topic or
    question semantically. The vector column is nullable: old rows without an
    embedding simply never match semantically (exact matches still work).
    Returns the generated UUID of the new row.
    """
    supabase = _get_supabase()

    row = {
        "topic": question.topic,
        "question": question.question,
        "aspects": json.dumps(list(question.aspects), ensure_ascii=False),
        "status": question.status,
        "priority": question.priority,
    }
    if embedding is not None:
        row["embedding"] = _vector_literal(embedding)

    result = supabase.table("research_questions").insert(row).execute()
    question_id: str = result.data[0]["id"]
    log.info("[Memory] Research question stored – id=%s  topic=%s", question_id, question.topic)
    return question_id


# ---------------------------------------------------------------------------
# Public: Store Research Sources
# ---------------------------------------------------------------------------

def store_research_sources(
    research_question_id: Optional[str],
    sources: list[ResearchSource],
) -> list[str]:
    """
    Insert the normalized, deduplicated research sources into the
    ``research_sources`` table, each row linked to the originating research
    question (when its id is known). Returns the generated UUIDs.

    Persistence here is intentionally per-source so a partial failure never
    silently drops rows; callers wrap the whole call in try/except to keep
    the in-memory research result intact.
    """
    supabase = _get_supabase()
    source_ids: list[str] = []

    for source in sources:
        accessed_at = source.accessed_at or datetime.now(timezone.utc)
        row = {
            "research_question_id": research_question_id,
            "url": source.url,
            "title": source.title,
            "body": source.body,
            "source": source.source,
            "published": source.published,
            "score": source.score,
            "source_type": source.source_type,
            "accessed_at": accessed_at.isoformat(),
        }
        result = supabase.table("research_sources").insert(row).execute()
        source_ids.append(result.data[0]["id"])

    log.info("[Memory] Stored %d research source(s) for question_id=%s", len(source_ids), research_question_id)
    return source_ids


def link_research_sources(
    research_question_id: Optional[str],
    sources: list[ResearchSource],
) -> list[str]:
    """
    Persist lightweight per-session *link rows* for already-known sources.

    When ``skip_known_sources`` filters EVERY discovered source out of
    :func:`store_research_sources` (all of them already stored in earlier
    sessions), a brand-new session would otherwise own no source rows at all
    and could never be reused. ``research_sources`` links each row to exactly
    one research question (no many-to-many / sharing table), so re-inserting
    the known sources would create duplicate full records. Instead this
    records, per session, a row carrying only ``url`` (+ ``title``) — a
    reference back to the existing full record.

    Hydration (:func:`get_research_sources_for_question`) recognises these
    link rows (empty ``body``) and re-loads the full content from the newest
    stored record with the same normalized URL, so a reused session returns
    fully usable sources without either re-running Tavily or duplicating
    source content. Returns the generated UUIDs of the link rows.
    """
    supabase = _get_supabase()
    link_ids: list[str] = []

    for source in sources:
        accessed_at = source.accessed_at or datetime.now(timezone.utc)
        row = {
            "research_question_id": research_question_id,
            "url": source.url,
            "title": source.title,
            "score": source.score,
            "source_type": source.source_type,
            "accessed_at": accessed_at.isoformat(),
        }
        result = supabase.table("research_sources").insert(row).execute()
        link_ids.append(result.data[0]["id"])

    log.info("[Memory] Linked %d known source(s) to question_id=%s", len(link_ids), research_question_id)
    return link_ids


# ---------------------------------------------------------------------------
# Research-session memory (remember researched topics/questions)
# ---------------------------------------------------------------------------

def get_known_research_questions(limit: int = 200) -> list[dict]:
    """
    Fetch the most recent research_questions rows (id, topic, question,
    status, embedding) so callers can recognise previously researched
    topics/questions. Non-blocking: an unavailable store returns [].
    """
    try:
        supabase = _get_supabase()
        result = (
            supabase.table("research_questions")
            .select("id", "topic", "question", "status", "embedding")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(result.data or [])
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Memory] Could not load known research questions (%s: %s) – memory empty.",
            type(exc).__name__, exc,
        )
        return []


def _cosine(a, b) -> float:
    """Cosine similarity between two embedding vectors (0.0 when unusable).

    Embeddings are L2-normalized, so this is a plain dot product; the
    normalization guard keeps ragged/empty inputs safe regardless.
    """
    try:
        va = np.asarray(a, dtype=np.float64).reshape(-1)
        vb = np.asarray(b, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return 0.0
    if va.size == 0 or vb.size == 0 or va.size != vb.size:
        return 0.0
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _as_similarity_embedding(value) -> Optional[list[float]]:
    """Vector column value → list[float], handling both arrays and literals."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def check_topic_researched(
    topic: str,
    question: str,
    threshold: Optional[float] = None,
    use_embedding: bool = True,
) -> dict:
    """
    Decide whether *topic* + *question* were already researched.

    Two passes, both against the previously stored research sessions:

    1. Exact pass (no network calls beyond the memory fetch): a session is a
       duplicate when the SAME normalized topic AND question was asked, or the
       SAME normalized question was asked (regardless of topic wording). An
       equal topic with a *different* question is a follow-up, never a
       duplicate – related but meaningfully different research is allowed.

    2. Semantic pass (embedding cosine vs stored session embeddings): used
       only when no exact match fired, so lightly reworded duplicates still
       trip the memory while unrelated/follow-up questions stay below the
       threshold.

    Dead sessions (status ``dropped``) are skipped in both passes, so a
    discarded/incomplete session never suppresses later research on the same
    question.

    Non-blocking: embedding or store failures degrade to "not matched"
    (research proceeds) rather than halting the pipeline.

    Parameters
    ----------
    topic / question : the proposed research topic and question.
    threshold        : minimum cosine similarity for a semantic match
                       (defaults to settings.RESEARCH_DUPLICATE_THRESHOLD).
    use_embedding    : skip the semantic pass entirely when False (pure
                       exact-text matching; deterministic, no API calls).

    Returns
    -------
    dict with keys:
        matched           : bool
        reason            : "exact" | "semantic" | None
        question_id       : UUID of the matched research session (None otherwise)
        matched_question  : {id, topic, question, status} of the match (or None)
        similarity        : best cosine found (None when no semantic pass ran)
    """
    result: dict = {
        "matched": False,
        "reason": None,
        "question_id": None,
        "matched_question": None,
        "similarity": None,
    }

    try:
        known = get_known_research_questions()
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Memory] Research-memory unavailable (%s: %s) – treating as new research.",
            type(exc).__name__, exc,
        )
        return result
    if not known:
        return result

    norm_topic = _normalize_research_text(topic)
    norm_question = _normalize_research_text(question)

    # Dropped sessions are dead end states: a run produced nothing usable and
    # the row was discarded (or could not be deleted). They must never match
    # again, or they would suppress re-research forever.
    def _alive(row: dict) -> bool:
        return (row.get("status") or "").strip() != ResearchQuestion.STATUS_DROPPED

    # ── Exact pass ────────────────────────────────────────────────────────
    for row in known:
        if not _alive(row):
            continue
        same_topic = bool(norm_topic) and _normalize_research_text(row.get("topic")) == norm_topic
        same_question = bool(norm_question) and _normalize_research_text(row.get("question")) == norm_question
        if same_topic and same_question:
            result.update(
                matched=True,
                reason="exact",
                question_id=row.get("id"),
                matched_question={k: row.get(k) for k in ("id", "topic", "question", "status")},
            )
            return result
        if same_question:
            result.update(
                matched=True,
                reason="exact",
                question_id=row.get("id"),
                matched_question={k: row.get(k) for k in ("id", "topic", "question", "status")},
            )
            return result

    # ── Semantic pass ─────────────────────────────────────────────────────
    if not use_embedding:
        return result

    try:
        query_embedding = get_normalized_embedding(f"{topic} {question}".strip())
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Memory] Embedding unavailable (%s) – skipping semantic research match.",
            type(exc).__name__,
        )
        return result

    t = float(threshold if threshold is not None else settings.RESEARCH_DUPLICATE_THRESHOLD)

    best_sim = float("-inf")
    best_row = None
    for row in known:
        if not _alive(row):
            continue
        value = _as_similarity_embedding(row.get("embedding"))
        if value is None:
            continue
        sim = _cosine(query_embedding, value)
        if sim > best_sim:
            best_sim, best_row = sim, row

    result["similarity"] = best_sim if best_row is not None else None
    if best_row is not None and best_sim >= t:
        result.update(
            matched=True,
            reason="semantic",
            question_id=best_row.get("id"),
            matched_question={k: best_row.get(k) for k in ("id", "topic", "question", "status")},
        )
    return result


def _load_latest_sources_by_url(urls: list[str]) -> dict[str, ResearchSource]:
    """Newest full-source row per normalized URL, across all sessions.

    Used to enrich per-session *link rows* (see :func:`link_research_sources`)
    with the full content that the original (earlier) session stored for the
    same normalized URL. Only rows with usable source content qualify as the
    canonical source — lightweight link rows (``body=''``), even when they are
    NEWER than the full record, are never selected as the canonical source.
    Non-blocking: an unavailable store returns {}.
    """
    urls = [u for u in urls if (u or "").strip()]
    if not urls:
        return {}

    try:
        supabase = _get_supabase()
        result = (
            supabase.table("research_sources")
            .select("id", "url", "title", "body", "source", "published",
                    "score", "source_type", "accessed_at")
            .in_("url", urls)
            .order("accessed_at", desc=True)
            .execute()
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Memory] Could not load existing source records for enrichment (%s: %s).",
            type(exc).__name__, exc,
        )
        return {}

    latest: dict[str, ResearchSource] = {}
    for row in list(result.data or []):
        body = (row.get("body") or "").strip()
        if not body:
            # Lightweight link rows carry no source content and must never be
            # treated as the canonical record, even if they were written later.
            continue
        url = (row.get("url") or "").strip()
        key = (_normalize_url(url) or url).casefold()
        if not key or key in latest:
            continue
        latest[key] = ResearchSource.from_dict(dict(row))
    return latest


def get_research_sources_for_question(
    research_question_id: Optional[str],
) -> tuple[list[ResearchSource], list[str]]:
    """
    Hydrate the sources (+ their row UUIDs) recorded for a research session.

    Used to *reuse* a remembered session: when a topic/question was already
    researched, the stored sources come back without re-searching them.
    Per-session *link rows* (recorded by :func:`link_research_sources` when a
    run discovered only already-known sources) are enriched in place with the
    full content of the existing stored record for the same normalized URL,
    so a successful session always hydrates usable sources without duplicating
    source content. Non-blocking: on store failure or unknown id, returns
    empty lists.
    """
    if not research_question_id:
        return [], []

    try:
        supabase = _get_supabase()
        result = (
            supabase.table("research_sources")
            .select("id", "url", "title", "body", "source", "published",
                    "score", "source_type", "accessed_at")
            .eq("research_question_id", research_question_id)
            .order("score", desc=True)
            .execute()
        )
        sources: list[ResearchSource] = []
        ids: list[str] = []
        for row in list(result.data or []):
            row = dict(row)
            row_id = row.pop("id", None)
            sources.append(ResearchSource.from_dict(row))
            ids.append(row_id)

        # Enrich link rows (empty body) with the full content already stored
        # for the same URL in an earlier session.
        link_urls = [s.url for s in sources if not (s.body or "").strip()]
        if link_urls:
            latest = _load_latest_sources_by_url(link_urls)
            if latest:
                enriched: list[ResearchSource] = []
                for source in sources:
                    key = (_normalize_url(source.url) or source.url).casefold()
                    full = latest.get(key)
                    if not (source.body or "").strip() and full is not None:
                        enriched.append(full)
                    else:
                        enriched.append(source)
                sources = enriched
                sources.sort(key=lambda s: s.score, reverse=True)

        return sources, ids
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Memory] Could not load sources for question %s (%s: %s).",
            research_question_id, type(exc).__name__, exc,
        )
        return [], []


def delete_research_question(research_question_id: Optional[str]) -> None:
    """
    Remove a research_questions row (its research_sources rows cascade).

    Used to clean up a session that was created for a research run which
    produced nothing persistable, so a dead/incomplete row never becomes a
    reusable duplicate for a later run. Non-blocking: an unavailable store
    only logs a warning (the caller keeps its in-memory result intact).
    """
    if not research_question_id:
        return
    try:
        supabase = _get_supabase()
        supabase.table("research_questions").delete().eq("id", research_question_id).execute()
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Memory] Could not delete research session %s (%s: %s).",
            research_question_id, type(exc).__name__, exc,
        )


# Research-question lifecycle state machine. A session moves through these
# states explicitly:
#
#   proposed ──► researching ──► answered        (successful run)
#      │              │
#      │              └────────► dropped          (run produced nothing usable)
#      └──────────────► dropped
#
# answered ──► researching is allowed for the re-research fallback: a matched
# session whose stored sources cannot be hydrated is re-researched in place
# rather than reusing an empty result. dropped is terminal: dead sessions are
# never matched or reused again (see check_topic_researched).
_RESEARCH_LIFECYCLE_NEXT: dict[str, set] = {
    ResearchQuestion.STATUS_PROPOSED: {
        ResearchQuestion.STATUS_RESEARCHING,
        ResearchQuestion.STATUS_DROPPED,
    },
    ResearchQuestion.STATUS_RESEARCHING: {
        ResearchQuestion.STATUS_ANSWERED,
        ResearchQuestion.STATUS_DROPPED,
    },
    ResearchQuestion.STATUS_ANSWERED: {
        ResearchQuestion.STATUS_RESEARCHING,
        ResearchQuestion.STATUS_DROPPED,
    },
    ResearchQuestion.STATUS_DROPPED: set(),
}


def set_research_question_status(
    research_question_id: Optional[str],
    status: str,
) -> bool:
    """
    Best-effort lifecycle transition for a research session.

    Applies the explicit state machine defined by
    ``ResearchQuestion.STATUS_*`` / ``_RESEARCH_LIFECYCLE_NEXT``: the current
    status is read, an illegal transition (e.g. ``answered`` → ``proposed`` or
    any move out of ``dropped``) is rejected with only a log line, and a
    valid transition is written back to the ``research_questions`` row.

    Never raises and never halts the pipeline: an unavailable store or a
    missing session row only logs a warning and returns False (the caller
    keeps its in-memory result intact). Returns True only when the transition
    was both valid and issued.
    """
    if not research_question_id or not status:
        return False
    if status not in _RESEARCH_LIFECYCLE_NEXT:
        log.warning("[Memory] Unknown research-question status '%s' – ignoring.", status)
        return False

    try:
        supabase = _get_supabase()
        result = (
            supabase.table("research_questions")
            .select("status")
            .eq("id", research_question_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Memory] Could not read research session %s status (%s: %s) – skipping transition.",
            research_question_id, type(exc).__name__, exc,
        )
        return False

    rows = list(result.data or [])
    if not rows:
        log.warning(
            "[Memory] No research session %s – skipping status transition.",
            research_question_id,
        )
        return False

    current = str(rows[0].get("status") or ResearchQuestion.STATUS_PROPOSED)
    allowed = _RESEARCH_LIFECYCLE_NEXT.get(current, set())
    if status not in allowed:
        log.info(
            "[Memory] Skipping invalid research-session transition %s → %s for %s.",
            current, status, research_question_id,
        )
        return False

    try:
        supabase.table("research_questions").update({"status": status}).eq("id", research_question_id).execute()
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Memory] Could not update research session %s to %s (%s: %s).",
            research_question_id, status, type(exc).__name__, exc,
        )
        return False

    log.info("[Memory] Research session %s: status=%s → %s", research_question_id, current, status)
    return True


def get_known_source_urls() -> set[str]:
    """
    Normalized URLs already recorded in research_sources across all sessions.

    Non-blocking: an unavailable store returns an empty set (callers then
    persist everything, as today).
    """
    try:
        supabase = _get_supabase()
        result = supabase.table("research_sources").select("url").execute()
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "[Memory] Could not load known source URLs (%s: %s) – skipping cross-session source dedup.",
            type(exc).__name__, exc,
        )
        return set()

    known: set[str] = set()
    for row in list(result.data or []):
        url = (row.get("url") or "").strip()
        if not url:
            continue
        known.add(_normalize_url(url) or url)
    return known


def filter_repeated_sources(
    sources: list[ResearchSource],
    known_urls: Optional[set] = None,
) -> tuple[list[ResearchSource], list[ResearchSource]]:
    """
    Split *sources* into fresh vs already-known by normalized URL.

    "Known" means either previously stored across earlier sessions (pass in
    the result of :func:`get_known_source_urls`) or repeated within this
    batch (first occurrence wins). Pure function – no I/O.

    Returns ``(to_store, repeated)`` where *repeated* keeps the exact source
    objects so callers can log/measure them without losing the in-memory
    research result.
    """
    known = set(known_urls or set())
    seen: set[str] = set()
    to_store: list[ResearchSource] = []
    repeated: list[ResearchSource] = []

    for source in sources:
        norm = _normalize_url(source.url) or source.url
        key = norm.casefold()
        if key in known or key in seen:
            repeated.append(source)
            continue
        seen.add(key)
        to_store.append(source)

    return to_store, repeated


# ---------------------------------------------------------------------------
# Public: Update Post Status
# ---------------------------------------------------------------------------

def update_post_status(post_id: str, status: str) -> None:
    """Update the status column for a given post UUID."""
    supabase = _get_supabase()
    supabase.table("posts").update({"status": status}).eq("id", post_id).execute()
    log.info("[Memory] Post %s → status=%s", post_id, status)


# ---------------------------------------------------------------------------
# Public: Retrieve Style Profile
# ---------------------------------------------------------------------------

def get_style_profile() -> dict:
    """
    Fetch the master style profile JSON from the `style_profile` table.
    Returns an empty dict if no profile exists yet.
    """
    supabase = _get_supabase()
    result = supabase.table("style_profile").select("profile").eq("id", 1).execute()

    if result.data:
        return result.data[0]["profile"]

    log.warning("[Memory] No style_profile found – using empty defaults.")
    return {}


# ---------------------------------------------------------------------------
# Supabase RPC helper SQL (run once during setup)
# ---------------------------------------------------------------------------
# The check_is_duplicate function calls an RPC named match_posts_by_embedding.
# Add this SQL function to your Supabase project via the SQL editor:
#
#   CREATE OR REPLACE FUNCTION match_posts_by_embedding(
#       query_embedding   VECTOR(768),
#       match_threshold   FLOAT,
#       match_count       INT
#   )
#   RETURNS TABLE (id UUID, cosine_distance FLOAT) AS $$
#   BEGIN
#       RETURN QUERY
#       SELECT
#           p.id,
#           (p.embedding <=> query_embedding)::FLOAT AS cosine_distance
#       FROM posts p
#       WHERE p.status IN ('PUBLISHED', 'PENDING')
#         AND (p.embedding <=> query_embedding) < match_threshold
#       ORDER BY cosine_distance ASC
#       LIMIT match_count;
#   END;
#   $$ LANGUAGE plpgsql;
