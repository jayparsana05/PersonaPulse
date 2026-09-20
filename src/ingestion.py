"""
PersonaPulse – Ingestion Module
================================
Discovers trending tech news via Tavily and extracts og:image
metadata for rich media posts.

Key Functions
-------------
- fetch_trending_tech_news(query)         → dict                 (legacy single-article)
- discover_topic_candidates(query, limit) → list[TopicCandidate] (Phase-1 discovery)
- extract_og_image(article_url)           → bytes | None
- extract_og_image_with_url(article_url)  → (image_url, bytes) | (None, None)
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timezone
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient

from src.config import AGENTIC_SEARCH_QUERIES, EXCLUDED_SEARCH_DOMAINS, settings
from src.models import TopicCandidate

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
_REQUEST_TIMEOUT = 10          # seconds for HTTP requests
_IMAGE_MAX_BYTES = 8 * 1024 * 1024   # 8 MB cap for downloaded images

# ---------------------------------------------------------------------------
# Tavily Client (lazy singleton)
# ---------------------------------------------------------------------------

_tavily_client: TavilyClient | None = None


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
    return _tavily_client


# ---------------------------------------------------------------------------
# Tavily search helper (shared by single-article and discovery flows)
# ---------------------------------------------------------------------------

def _search_tavily(
    query: str,
    days: int = 7,
    max_results: int = 1,
    include_raw_content: bool = True,
) -> list[dict]:
    """Run a Tavily search and return the raw results list (may be empty)."""
    client = _get_tavily()
    response = client.search(
        query=query,
        search_depth="advanced",       # deeper crawl for full body text
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic="news",
        days=days,
        exclude_domains=EXCLUDED_SEARCH_DOMAINS,
    )
    return response.get("results", [])


def _resolve_query(query: Optional[str]) -> str:
    """
    Return the query to use, rotating through AGENTIC_SEARCH_QUERIES
    when no explicit query is supplied.
    """
    if query:
        return query
    effective_query = _rotate_search_query()
    log.info("[Ingestion] Using rotated agentic-AI query: '%s'", effective_query)
    return effective_query


# ---------------------------------------------------------------------------
# Public: Fetch Trending Tech News (legacy single-article flow)
# ---------------------------------------------------------------------------

def fetch_trending_tech_news(
    query: Optional[str] = None,
    days: int = 7,
) -> dict:
    """
    Query Tavily for the single most-relevant trending agentic-AI article.

    Parameters
    ----------
    query : str, optional
        Search query string. If None/empty, rotates through
        AGENTIC_SEARCH_QUERIES (one per pipeline run).
    days : int
        Strict recency filter (default 7 days) to only return recent news.

    Returns a dict with keys:
        url         – canonical article URL
        title       – article headline
        body        – cleaned markdown / plain-text body
        published   – publication date string (may be empty)
        source      – domain / publisher name

    Raises RuntimeError if Tavily returns no results.
    """
    effective_query = _resolve_query(query)
    log.info("[Ingestion] Searching Tavily: query='%s', days=%d", effective_query, days)

    results = _search_tavily(effective_query, days=days, max_results=1, include_raw_content=True)

    if not results:
        raise RuntimeError(
            f"[Ingestion] Tavily returned no results for query: '{effective_query}'"
        )

    top = results[0]

    # Prefer raw_content (full article) over snippet
    raw_body: str = top.get("raw_content") or top.get("content") or ""
    cleaned_body = _clean_body(raw_body)

    article = {
        "url":       top.get("url", ""),
        "title":     top.get("title", "").strip(),
        "body":      cleaned_body,
        "published": top.get("published_date", ""),
        "source":    _extract_domain(top.get("url", "")),
    }

    log.info(
        "[Ingestion] Found article: '%s' from %s (%d chars body)",
        article["title"],
        article["source"],
        len(article["body"]),
    )
    return article


# ---------------------------------------------------------------------------
# Public: Discover Topic Candidates (Phase-1 discovery output)
# ---------------------------------------------------------------------------

def discover_topic_candidates(
    query: Optional[str] = None,
    days: int = 7,
    limit: Optional[int] = None,
) -> list[TopicCandidate]:
    """
    Discover a configurable number of topic candidates from the existing
    Tavily discovery mechanism.

    Rather than returning a single article, this phase surfaces N candidates
    (default: settings.DISCOVERY_CANDIDATE_COUNT) with enough metadata for a
    later selection stage (title, url, source, published, description,
    discovered_at). No deep research is performed here.

    Parameters
    ----------
    query : str, optional
        Search query string. If None/empty, rotates through
        AGENTIC_SEARCH_QUERIES.
    days : int
        Strict recency filter (default 7 days).
    limit : int, optional
        Maximum number of candidates to return. Defaults to
        settings.DISCOVERY_CANDIDATE_COUNT.

    Returns
    -------
    list[TopicCandidate]
        A clean, deduplicated list of candidates (empty if discovery
        returned nothing usable). Sorted by Tavily search score
        (candidate.search_score, descending).
    """
    effective_query = _resolve_query(query)
    count = limit if limit is not None else settings.DISCOVERY_CANDIDATE_COUNT
    count = max(0, int(count))

    log.info(
        "[Ingestion] Discovering up to %d topic candidates (query='%s', days=%d)",
        count, effective_query, days,
    )

    if count == 0:
        return []

    results = _search_tavily(
        effective_query,
        days=days,
        max_results=count,
        include_raw_content=False,      # snippets are enough for candidates
    )

    if not results:
        log.warning("[Ingestion] Discovery returned no results for query: '%s'", effective_query)
        return []

    candidates = [
        candidate
        for item in results
        if (candidate := _candidate_from_result(item)) is not None
    ]

    candidates = _dedupe_candidates(candidates)

    # Re-sort by relevance after dedup so the strongest candidates come first.
    candidates.sort(key=lambda c: c.search_score, reverse=True)

    log.info(
        "[Ingestion] Discovery produced %d candidate(s) from %d raw result(s).",
        len(candidates), len(results),
    )
    return candidates


# ---------------------------------------------------------------------------
# Public: Extract og:image
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public: Extract og:image
# ---------------------------------------------------------------------------

def extract_og_image(article_url: str) -> Optional[bytes]:
    """
    Backward-compatible wrapper returning the og:image bytes only.

    Equivalent to extract_og_image_with_url(...)[1]. Returns None on any
    failure so callers can fall back to a text-only post.
    """
    return _extract_og_image(article_url)[1]


def extract_og_image_with_url(article_url: str) -> tuple[Optional[str], Optional[bytes]]:
    """
    Attempt to:
      1. Fetch the article's HTML page.
      2. Parse the <meta property="og:image"> tag with BeautifulSoup.
      3. Download and return the raw image bytes.

    Returns a ``(image_url, image_bytes)`` tuple; either element may be
    None on failure (403, missing tag, network timeout, etc.) so callers
    can gracefully fall back to a text-only post.
    """
    return _extract_og_image(article_url)


def _extract_og_image(article_url: str) -> tuple[Optional[str], Optional[bytes]]:
    """Shared implementation of og:image extraction (see public wrappers)."""
    if not article_url:
        return None, None

    headers = {"User-Agent": _BROWSER_UA}

    try:
        # ── Step 1: Fetch article HTML ──────────────────────────────────
        log.debug("[Ingestion] Fetching HTML for og:image: %s", article_url)
        with httpx.Client(timeout=_REQUEST_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(article_url, headers=headers)
            resp.raise_for_status()
            html = resp.text

        # ── Step 2: Parse og:image ──────────────────────────────────────
        soup = BeautifulSoup(html, "html.parser")
        og_tag = soup.find("meta", property="og:image")

        if og_tag is None:
            log.info("[Ingestion] No og:image tag found at %s", article_url)
            return None, None

        image_url: str = og_tag.get("content", "").strip()
        if not image_url:
            log.info("[Ingestion] og:image tag present but empty at %s", article_url)
            return None, None

        log.info("[Ingestion] Found og:image URL: %s", image_url)

        # ── Step 3: Download image binary ───────────────────────────────
        with httpx.Client(timeout=_REQUEST_TIMEOUT, follow_redirects=True) as client:
            img_resp = client.get(image_url, headers=headers)
            img_resp.raise_for_status()

            # Safety cap to avoid downloading massive images
            image_bytes = img_resp.content
            if len(image_bytes) > _IMAGE_MAX_BYTES:
                log.warning(
                    "[Ingestion] Image too large (%d bytes > %d) – skipping.",
                    len(image_bytes), _IMAGE_MAX_BYTES,
                )
                return None, None

            log.info("[Ingestion] Image downloaded: %d bytes", len(image_bytes))
            return image_url, image_bytes

    except httpx.HTTPStatusError as exc:
        log.warning(
            "[Ingestion] HTTP %d fetching %s – falling back to no image.",
            exc.response.status_code, article_url,
        )
        return None, None
    except httpx.TimeoutException:
        log.warning("[Ingestion] Timeout fetching image for %s – skipping.", article_url)
        return None, None
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("[Ingestion] Unexpected error extracting og:image: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rotate_search_query() -> str:
    """
    Pick today's query from AGENTIC_SEARCH_QUERIES.

    Day-of-year modulo list length gives a deterministic daily rotation
    with no persistent state, so cron runs cycle through all terms.
    """
    if not AGENTIC_SEARCH_QUERIES:
        return "agentic AI"
    day_of_year = date.today().timetuple().tm_yday
    return AGENTIC_SEARCH_QUERIES[day_of_year % len(AGENTIC_SEARCH_QUERIES)]


def _candidate_from_result(item: dict) -> Optional[TopicCandidate]:
    """
    Map a raw Tavily result dict into a TopicCandidate.

    A result is only kept if it has a usable URL (required for later
    evaluation). Missing title / description / published / source degrade
    gracefully to empty strings.
    """
    url = (item.get("url") or "").strip()
    if not url:
        log.debug("[Ingestion] Skipping result without a URL: %r", item.get("title"))
        return None

    title = (item.get("title") or "").strip()
    snippet = item.get("content") or item.get("raw_content") or ""

    try:
        score = float(item.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0

    # Preserve provider-supplied keywords when present; otherwise keep [].
    # Keyword enrichment is left to the later selection phase.
    raw_keywords = item.get("keywords")
    keywords = (
        [k.strip() for k in raw_keywords if isinstance(k, str) and k.strip()]
        if isinstance(raw_keywords, list)
        else []
    )

    return TopicCandidate(
        title=title,
        url=url,
        description=_truncate(snippet, 320),
        keywords=keywords,
        source=_extract_domain(url),
        published=(item.get("published_date") or "").strip(),
        search_score=max(0.0, min(1.0, score)),
        discovered_at=datetime.now(timezone.utc),
    )


def _dedupe_candidates(candidates: list[TopicCandidate]) -> list[TopicCandidate]:
    """
    Deduplicate candidates at the topic/source level (not the final post).

    Drops candidates that share the same normalized URL, or that share the
    same publisher and a normalized (case/whitespace-folded) title.
    """
    seen_urls: set[str] = set()
    seen_source_titles: set[str] = set()
    deduped: list[TopicCandidate] = []

    for candidate in candidates:
        norm_url = _normalize_url(candidate.url)
        if norm_url and norm_url in seen_urls:
            log.debug("[Ingestion] Deduping candidate (same URL): %s", candidate.url)
            continue

        source = (candidate.source or "").strip().casefold()
        title = _normalize_title(candidate.title)
        source_title_key = f"{source}\u0001{title}" if title else ""
        if source_title_key and source_title_key in seen_source_titles:
            log.debug("[Ingestion] Deduping candidate (same source+title): %s", candidate.title)
            continue

        seen_urls.add(norm_url)
        if source_title_key:
            seen_source_titles.add(source_title_key)
        deduped.append(candidate)

    return deduped


def _normalize_title(title: str) -> str:
    """Fold title case and collapse whitespace for stable dedup keys."""
    folded = " ".join(title.split()).casefold()
    return re.sub(r"[^\w\s]", "", folded)


def _normalize_url(url: str) -> str:
    """
    Canonicalize a URL for dedup: lowercase host, strip www., drop fragment,
    drop tracking params (utm_*), and normalize the trailing slash.
    """
    if not url:
        return ""
    try:
        scheme, netloc, path, query, _fragment = urlsplit(url)
        host = netloc.casefold()
        if host.startswith("www."):
            host = host[4:]
        path = path.rstrip("/") or "/"

        keep = [(k, v) for k, v in parse_qsl(query, keep_blank_values=True)
                if not k.casefold().startswith("utm_")]
        normalized_query = urlencode(keep, doseq=True)

        return urlunsplit((scheme.casefold() or "https", host, path, normalized_query, ""))
    except Exception:  # pylint: disable=broad-except
        return url


def _truncate(text: str, max_chars: int) -> str:
    """Collapse whitespace and truncate to *max_chars* characters."""
    text = _clean_body(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _clean_body(raw: str) -> str:
    """
    Remove excessive whitespace, JavaScript snippets, and cookie banners
    commonly returned by Tavily's raw_content field.
    """
    # Collapse multiple newlines → two at most
    text = re.sub(r"\n{3,}", "\n\n", raw)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _extract_domain(url: str) -> str:
    """Extract the domain name from a URL for display purposes."""
    import urllib.parse
    try:
        parsed = urllib.parse.urlparse(url)
        # Remove 'www.' prefix if present
        host = parsed.netloc.replace("www.", "")
        return host or url
    except Exception:  # pylint: disable=broad-except
        return url
