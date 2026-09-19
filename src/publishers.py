"""
PersonaPulse – Publishers Module
==================================
Handles all outbound social media publishing:
  - LinkedIn: 3-step image upload + UGC post creation
  - X (Twitter): tweepy v2 plain-text tweet

Key Functions
-------------
- upload_linkedin_image(access_token, author_urn, image_bytes) → str | None
- publish_to_linkedin(text, image_bytes) → None
- publish_to_x(text) → None
- check_linkedin_token_health() → dict
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx
import tweepy

from src.config import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LinkedIn API constants
# ---------------------------------------------------------------------------

_LI_API_BASE  = "https://api.linkedin.com/v2"
_LI_REGISTER  = f"{_LI_API_BASE}/assets?action=registerUpload"
_LI_UGC_POSTS = f"{_LI_API_BASE}/ugcPosts"
_LI_HEADERS   = lambda token: {            # noqa: E731
    "Authorization": f"Bearer {token}",
    "Content-Type":  "application/json",
    "X-Restli-Protocol-Version": "2.0.0",
}

# ---------------------------------------------------------------------------
# LinkedIn: Image Upload (3-Step)
# ---------------------------------------------------------------------------

def upload_linkedin_image(
    access_token: str,
    author_urn: str,
    image_bytes: bytes,
) -> Optional[str]:
    """
    Register → Upload → Return asset URN.

    Step 1: POST to /v2/assets?action=registerUpload
            Returns an uploadUrl + asset URN.

    Step 2: PUT raw image bytes to the uploadUrl (no Content-Type header
            as LinkedIn requires a plain binary PUT).

    Step 3: Return the asset URN (urn:li:digitalmediaAsset:…).

    Returns None on any failure.
    """
    headers = _LI_HEADERS(access_token)

    # ── Step 1: Register Upload ─────────────────────────────────────────
    register_payload = {
        "registerUploadRequest": {
            "recipes": ["urn:li:digitalmediaRecipe:feedshare-image"],
            "owner": author_urn,
            "serviceRelationships": [{
                "relationshipType": "OWNER",
                "identifier":       "urn:li:userGeneratedContent",
            }],
        }
    }

    try:
        with httpx.Client(timeout=20) as client:
            reg_resp = client.post(_LI_REGISTER, headers=headers, json=register_payload)
            reg_resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        log.error("[Publisher] LinkedIn register upload failed: %s – %s",
                  exc.response.status_code, exc.response.text)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        log.error("[Publisher] LinkedIn register upload error: %s", exc)
        return None

    reg_data    = reg_resp.json()
    upload_url  = reg_data["value"]["uploadMechanism"][
        "com.linkedin.digitalmedia.uploading.MediaUploadHttpRequest"
    ]["uploadUrl"]
    asset_urn   = reg_data["value"]["asset"]
    log.info("[Publisher] LinkedIn asset registered: %s", asset_urn)

    # ── Step 2: PUT image bytes ─────────────────────────────────────────
    try:
        with httpx.Client(timeout=60) as client:
            put_resp = client.put(
                upload_url,
                content=image_bytes,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    # No Content-Type – LinkedIn requires binary PUT
                },
            )
            put_resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        log.error("[Publisher] LinkedIn image upload PUT failed: %s – %s",
                  exc.response.status_code, exc.response.text[:200])
        return None
    except Exception as exc:  # pylint: disable=broad-except
        log.error("[Publisher] LinkedIn image PUT error: %s", exc)
        return None

    log.info("[Publisher] LinkedIn image uploaded successfully.")
    return asset_urn  # Step 3: return the URN


# ---------------------------------------------------------------------------
# LinkedIn: Publish UGC Post
# ---------------------------------------------------------------------------

def publish_to_linkedin(
    text: str,
    image_bytes: Optional[bytes] = None,
    article_url: Optional[str] = None,
) -> None:
    """
    Publish a UGC post to LinkedIn.

    If image_bytes is provided:
      - Calls upload_linkedin_image() to get the asset URN.
      - Creates a RICH media post.

    If image_bytes is None (fallback):
      - Appends article_url to post body.
      - Creates a NONE shareMediaCategory (link preview) post.

    Raises RuntimeError on publish failure.
    """
    token      = settings.LINKEDIN_ACCESS_TOKEN
    author_urn = settings.LINKEDIN_AUTHOR_URN
    headers    = _LI_HEADERS(token)

    if image_bytes:
        asset_urn = upload_linkedin_image(token, author_urn, image_bytes)
        if asset_urn:
            payload = _build_linkedin_media_payload(author_urn, text, asset_urn)
        else:
            log.warning("[Publisher] Image upload failed – falling back to text post.")
            payload = _build_linkedin_text_payload(author_urn, text, article_url)
    else:
        payload = _build_linkedin_text_payload(author_urn, text, article_url)

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(_LI_UGC_POSTS, headers=headers, json=payload)
            resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"[Publisher] LinkedIn UGC post failed: {exc.response.status_code} – "
            f"{exc.response.text[:300]}"
        ) from exc

    log.info("[Publisher] ✅ LinkedIn post published: %s", resp.headers.get("x-linkedin-id", "n/a"))


def _build_linkedin_media_payload(author_urn: str, text: str, asset_urn: str) -> dict:
    return {
        "author":          author_urn,
        "lifecycleState":  "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary":   {"text": text},
                "shareMediaCategory": "IMAGE",
                "media": [{
                    "status":    "READY",
                    "media":     asset_urn,
                }],
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        },
    }


def _build_linkedin_text_payload(
    author_urn: str,
    text: str,
    article_url: Optional[str],
) -> dict:
    body = text if not article_url else f"{text}\n\n🔗 {article_url}"
    return {
        "author":          author_urn,
        "lifecycleState":  "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary":    {"text": body},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {
            "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
        },
    }


# ---------------------------------------------------------------------------
# X (Twitter): Publish Tweet
# ---------------------------------------------------------------------------

def publish_to_x(text: str) -> None:
    """
    Publish a plain-text tweet via the official X API v2 using tweepy.

    Uses OAuth 1.0a User Context (required for write operations on free tier).
    Raises RuntimeError on failure.
    """
    client = tweepy.Client(
        consumer_key=settings.X_API_KEY,
        consumer_secret=settings.X_API_SECRET,
        access_token=settings.X_ACCESS_TOKEN,
        access_token_secret=settings.X_ACCESS_SECRET,
    )

    try:
        response = client.create_tweet(text=text)
        tweet_id = response.data["id"]
        log.info("[Publisher] ✅ X tweet published: id=%s", tweet_id)
    except tweepy.TweepyException as exc:
        raise RuntimeError(f"[Publisher] X (Twitter) publish failed: {exc}") from exc


# ---------------------------------------------------------------------------
# LinkedIn Token Health Check
# ---------------------------------------------------------------------------

def check_linkedin_token_health() -> dict:
    """
    Inspect the LinkedIn access token validity.

    Returns a dict:
        {
          "days_remaining": int,
          "expires_on":     str,   # YYYY-MM-DD
          "is_critical":    bool,  # True if < TOKEN_WARN_DAYS days left
          "is_expired":     bool,
        }
    """
    days = settings.linkedin_token_days_remaining
    expiry = settings.LINKEDIN_TOKEN_EXPIRY_DATE

    result = {
        "days_remaining": days,
        "expires_on":     expiry,
        "is_critical":    days < settings.TOKEN_WARN_DAYS,
        "is_expired":     days <= 0,
    }

    if result["is_expired"]:
        log.error("[Publisher] ❌ LinkedIn token EXPIRED on %s!", expiry)
    elif result["is_critical"]:
        log.warning(
            "[Publisher] ⚠️  LinkedIn token expires in %d days (%s).",
            days, expiry,
        )
    else:
        log.info("[Publisher] ✅ LinkedIn token valid for %d more days.", days)

    return result
