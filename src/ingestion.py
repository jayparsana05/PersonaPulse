"""
PersonaPulse – Ingestion Module
================================
Discovers trending tech news via Tavily and extracts og:image
metadata for rich media posts.

Key Functions
-------------
- fetch_trending_tech_news(query) → dict
- extract_og_image(article_url)   → bytes | None
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient

from src.config import settings

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
# Public: Fetch Trending Tech News
# ---------------------------------------------------------------------------

def fetch_trending_tech_news(query: str = "latest AI and tech news today") -> dict:
    """
    Query Tavily for the single most-relevant trending tech article.

    Returns a dict with keys:
        url         – canonical article URL
        title       – article headline
        body        – cleaned markdown / plain-text body
        published   – publication date string (may be empty)
        source      – domain / publisher name

    Raises RuntimeError if Tavily returns no results.
    """
    client = _get_tavily()
    log.info("[Ingestion] Searching Tavily: query='%s'", query)

    response = client.search(
        query=query,
        search_depth="advanced",       # deeper crawl for full body text
        max_results=1,
        include_raw_content=True,      # full article body
        topic="news",
    )

    results: list[dict] = response.get("results", [])

    if not results:
        raise RuntimeError(
            f"[Ingestion] Tavily returned no results for query: '{query}'"
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
# Public: Extract og:image
# ---------------------------------------------------------------------------

def extract_og_image(article_url: str) -> Optional[bytes]:
    """
    Attempt to:
      1. Fetch the article's HTML page.
      2. Parse the <meta property="og:image"> tag with BeautifulSoup.
      3. Download and return the raw image bytes.

    Returns None on any failure (403, missing tag, network timeout, etc.)
    to allow graceful text-only fallback in the publisher.
    """
    if not article_url:
        return None

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
            return None

        image_url: str = og_tag.get("content", "").strip()
        if not image_url:
            log.info("[Ingestion] og:image tag present but empty at %s", article_url)
            return None

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
                return None

            log.info("[Ingestion] Image downloaded: %d bytes", len(image_bytes))
            return image_bytes

    except httpx.HTTPStatusError as exc:
        log.warning(
            "[Ingestion] HTTP %d fetching %s – falling back to no image.",
            exc.response.status_code, article_url,
        )
        return None
    except httpx.TimeoutException:
        log.warning("[Ingestion] Timeout fetching image for %s – skipping.", article_url)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("[Ingestion] Unexpected error extracting og:image: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
