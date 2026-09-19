"""
PersonaPulse – LLM Module
==========================
Interfaces with Google Gemini Flash 2.0 for:
  - Platform-tailored post drafting (LinkedIn & X)
  - Telegram-formatted draft summaries

Key Functions
-------------
- draft_post(article, style_profile, platform) → str
- send_telegram_alert(post_id, drafts, image_bytes, article_url)
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import httpx
from google import genai
from google.genai import types as genai_types

from src.config import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Gemini Client (lazy singleton)
# ---------------------------------------------------------------------------

_gemini_client: genai.Client | None = None


def _get_gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
    return _gemini_client


# ---------------------------------------------------------------------------
# System Prompt Template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a social media ghostwriter for a senior tech professional.
Your task is to transform a news article into an authentic, engaging social media post
that exactly matches the provided writing style profile.

Rules:
- Write in first-person voice as the profile owner.
- Match the tone, emoji usage, hashtag density, and structure described.
- NEVER fabricate statistics or quotes not present in the article.
- End with a question or call-to-action to drive engagement.
- Output ONLY the final post text – no preamble, no explanation, no quotes.
"""


# ---------------------------------------------------------------------------
# Public: Draft Post
# ---------------------------------------------------------------------------

def draft_post(
    article: dict,
    style_profile: dict,
    platform: str,
) -> str:
    """
    Use Gemini Flash 2.0 to draft a platform-tailored post.

    Parameters
    ----------
    article       : dict from ingestion.fetch_trending_tech_news()
    style_profile : dict from memory.get_style_profile()
    platform      : 'linkedin' or 'x'

    Returns
    -------
    Draft post text as a plain string.
    """
    client = _get_gemini()

    # ── Platform-specific constraints ───────────────────────────────────
    if platform == "linkedin":
        platform_rules = style_profile.get("linkedin", {})
        char_limit = "up to 1300 characters"
        format_hint = "Use line breaks for readability. Suitable for a professional audience."
    else:  # 'x' / Twitter
        platform_rules = style_profile.get("x", {})
        char_limit = "under 280 characters"
        format_hint = "Be punchy and concise. A single impactful tweet."

    user_prompt = f"""
## Article Details
- Title:     {article.get('title', 'N/A')}
- Source:    {article.get('source', 'N/A')}
- Published: {article.get('published', 'N/A')}
- URL:       {article.get('url', 'N/A')}

## Article Body
{article.get('body', '')[:3000]}

## Writing Style Profile
{json.dumps(style_profile, indent=2)}

## Platform-Specific Rules
Platform: {platform.upper()}
Character limit: {char_limit}
Format: {format_hint}
Additional rules: {json.dumps(platform_rules, indent=2)}

Write the {platform.upper()} post now:
""".strip()

    response = client.models.generate_content(
        model=settings.LLM_MODEL,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            temperature=0.85,
            max_output_tokens=800,
        ),
    )

    draft = response.text.strip()
    log.info("[LLM] Drafted %s post (%d chars)", platform, len(draft))
    return draft


# ---------------------------------------------------------------------------
# Public: Send Telegram Approval Alert
# ---------------------------------------------------------------------------

def send_telegram_alert(
    post_id: str,
    linkedin_draft: str,
    x_draft: str,
    article: dict,
    image_bytes: Optional[bytes] = None,
) -> None:
    """
    Send a Telegram message to TELEGRAM_CHAT_ID with:
      - Article info + both drafts as caption text
      - Inline keyboard: [✅ Approve] [❌ Reject]
      - Photo attachment (sendPhoto) if image_bytes is available,
        otherwise plain text (sendMessage)
    """
    caption = _build_caption(post_id, linkedin_draft, x_draft, article)
    keyboard = _build_inline_keyboard(post_id)

    if image_bytes:
        _send_photo(caption, keyboard, image_bytes)
    else:
        _send_message(caption, keyboard)


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

def _build_caption(
    post_id: str,
    linkedin_draft: str,
    x_draft: str,
    article: dict,
) -> str:
    """Build the message caption (max 1024 chars for sendPhoto)."""
    source = article.get("source", "Unknown")
    title = article.get("title", "")
    url = article.get("url", "")

    header = f"🔔 *New Draft Ready for Approval*\n\n"
    header += f"📰 *Source:* {source}\n"
    header += f"📌 *Topic:* {title[:120]}\n"
    header += f"🔗 {url}\n\n"

    li_section = f"━━━ *LinkedIn Draft* ━━━\n{linkedin_draft[:600]}"
    x_section  = f"\n\n━━━ *X (Twitter) Draft* ━━━\n{x_draft[:260]}"

    footer = f"\n\n🆔 Draft ID: `{post_id}`"

    return header + li_section + x_section + footer


def _build_inline_keyboard(post_id: str) -> dict:
    """Construct the Telegram InlineKeyboardMarkup payload."""
    return {
        "inline_keyboard": [[
            {
                "text": "✅ Approve & Publish",
                "callback_data": f"approve_{post_id}",
            },
            {
                "text": "❌ Reject",
                "callback_data": f"reject_{post_id}",
            },
        ]]
    }


def _send_photo(caption: str, keyboard: dict, image_bytes: bytes) -> None:
    """Send a Telegram photo with caption and inline keyboard."""
    url = f"{settings.telegram_api_base}/sendPhoto"

    with httpx.Client(timeout=20) as client:
        resp = client.post(
            url,
            data={
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "caption": caption[:1024],          # Telegram limit
                "parse_mode": "Markdown",
                "reply_markup": json.dumps(keyboard),
            },
            files={"photo": ("article_image.jpg", image_bytes, "image/jpeg")},
        )

    _handle_telegram_response(resp, "sendPhoto")


def _send_message(caption: str, keyboard: dict) -> None:
    """Send a plain Telegram text message with inline keyboard."""
    url = f"{settings.telegram_api_base}/sendMessage"

    with httpx.Client(timeout=20) as client:
        resp = client.post(
            url,
            json={
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": caption[:4096],             # Telegram limit
                "parse_mode": "Markdown",
                "reply_markup": keyboard,
                "disable_web_page_preview": False,
            },
        )

    _handle_telegram_response(resp, "sendMessage")


def _handle_telegram_response(resp: httpx.Response, method: str) -> None:
    try:
        resp.raise_for_status()
        data = resp.json()
        if not data.get("ok"):
            log.error("[LLM] Telegram %s failed: %s", method, data)
        else:
            log.info("[LLM] Telegram %s sent successfully.", method)
    except httpx.HTTPStatusError as exc:
        log.error("[LLM] Telegram %s HTTP error: %s", method, exc)


# ---------------------------------------------------------------------------
# Public: Send plain Telegram text notification (used for canary alerts)
# ---------------------------------------------------------------------------

def send_telegram_text(message: str) -> None:
    """Send a simple Telegram text message (no keyboard, no image)."""
    url = f"{settings.telegram_api_base}/sendMessage"
    with httpx.Client(timeout=15) as client:
        resp = client.post(
            url,
            json={
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": message[:4096],
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
        )
    _handle_telegram_response(resp, "sendMessage[plain]")
