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
import time
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
# System Prompts (per-platform)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_LINKEDIN = """You are a ghostwriter for a senior technology professional with a strong personal brand on LinkedIn.
Your task: transform a news article into a high-performing LinkedIn post that feels AUTHENTIC and PERSONAL, not like a news summary.

STRUCTURE (follow this exactly, each section separated by a blank line):

1. HOOK (1-2 lines): A provocative question, surprising stat, or bold statement that stops the scroll. Lead with hard numbers or concrete metrics whenever possible.
2. CONTEXT (2-3 lines): Briefly explain the news in simple, non-jargon terms. What happened? Why does it matter?
3. PERSONAL TAKE (3-5 lines): Share YOUR perspective as a tech professional. What are the implications? What does this mean for the industry, developers, or businesses? Use "I think…", "In my view…", "This tells me…"
4. KEY INSIGHT / LESSONS (3-4 bullet points using ▸ or →): Concrete, actionable takeaways.
5. CALL TO ACTION (1-2 lines): End with a thought-provoking question to spark discussion in the comments.
6. HASHTAGS (3-5 relevant hashtags on the last line).

CRITICAL THINKING & ATTRIBUTION RULES:
- Concrete Numbers: You MUST prioritize extracting and surfacing concrete data, metrics, and statistics present in the source text (e.g., '10x improvement' or 'reduced from 30 days to 3 days'). Lead with these hard numbers to create a strong hook instead of using abstract thought-leader language.
- Specific Attribution: Never use generic attributions like 'researchers have discovered'. 
  Extract and name the actual people/institutions from the article text 
  (e.g., '[Researcher Name] and team at [Institution]').
- Anti-SEO / Anti-Hype Filter: Don't inherit an exaggerated headline uncritically 
  (e.g., a headline claiming a "revolution" for what the body describes as an 
  incremental improvement). Read the body to find the real achievement and calibrate accordingly.
- NEVER fabricate statistics or quotes not present in the article.

STYLE & LENGTH RULES:
- Write in first-person, conversational, yet professional voice.
- Use emojis strategically (1-2 per section, not every line).
- Short paragraphs — max 3 lines per paragraph. Use blank lines between every section.
- Strict Length Constraint: The final drafted LinkedIn post must be strictly between 1300 and 1900 characters in length. Adjust the depth of your analysis based on the source content to naturally fit within this window. Do not generate overly brief posts or massive walls of text.
- NEVER use: clickbait, excessive exclamation marks, unsubstantiated claims.
- Output ONLY the final post text. No preamble, no markdown code fences, no "Here is the post:" prefix.
"""

_SYSTEM_PROMPT_X = """You are a ghostwriter for a senior technology professional with a strong presence on X (Twitter).
Your task: transform a news article into a single punchy, high-engagement tweet.

RULES:
- Maximum 280 characters (hard limit — count carefully).
- Lead with the most surprising or important point.
- Use 1-2 relevant emojis.
- Include 1-2 hashtags at the end.
- Write in first-person, direct voice.
- NEVER fabricate stats or quotes not in the article.
- Output ONLY the tweet text. No preamble.
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
    Use Gemini to draft a platform-tailored post.

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

    # ── Platform-specific config ─────────────────────────────────────────
    if platform == "linkedin":
        system_prompt = _SYSTEM_PROMPT_LINKEDIN
        max_tokens    = 2048   # enough for a full LinkedIn post within 1300-1900 chars
        temperature   = 0.80
        platform_rules = style_profile.get("linkedin", {})
        output_spec = """Write a FULL structured LinkedIn post following the HOOK → CONTEXT → PERSONAL TAKE → KEY INSIGHTS → CTA → HASHTAGS format.

The post MUST be strictly between 1300 and 1900 characters in length.
Prioritize concrete metrics, specific researcher/institution attribution, and calibrated technical reality (anti-hype).
Do NOT truncate. Do NOT summarize. Write the complete post."""

    else:  # 'x' / Twitter
        system_prompt = _SYSTEM_PROMPT_X
        max_tokens    = 500    # ample tokens for 280-char tweet without truncation
        temperature   = 0.75
        platform_rules = style_profile.get("x", {})
        output_spec = "Write a single tweet. Maximum 280 characters. One punchy sentence + 1-2 hashtags."

    # ── Tone/voice from style profile ────────────────────────────────────
    tone   = style_profile.get("tone", "professional yet conversational")
    voice  = style_profile.get("voice", "first-person, thought-leader")
    topics = ", ".join(style_profile.get("topics_of_interest", ["AI", "technology"]))
    avoid  = ", ".join(style_profile.get("avoid", []))

    user_prompt = f"""## ARTICLE TO TRANSFORM
Title:     {article.get('title', 'N/A')}
Source:    {article.get('source', 'N/A')}
Published: {article.get('published', 'N/A')}
URL:       {article.get('url', 'N/A')}

## ARTICLE BODY
{article.get('body', '')[:4000]}

## AUTHOR'S WRITING STYLE
- Tone:   {tone}
- Voice:  {voice}
- Topics this person cares about: {topics}
- Things to AVOID: {avoid}
- Platform rules: {json.dumps(platform_rules, indent=2)}

## YOUR TASK
{output_spec}

Write the {platform.upper()} post now:""".strip()

    # Build model sequence: primary model followed by fallback models
    models_to_try = [settings.LLM_MODEL] + settings.fallback_models

    last_error: Exception | None = None
    for attempt_idx, model_name in enumerate(models_to_try):
        try:
            if attempt_idx > 0:
                log.warning(
                    "[LLM] Retrying %s draft using fallback model '%s'...",
                    platform,
                    model_name,
                )
            chat = client.chats.create(
                model=model_name,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            response = chat.send_message(user_prompt)
            draft = response.text.strip() if response.text else ""
            if draft:
                if attempt_idx > 0:
                    log.warning(
                        "[LLM] Successfully drafted %s post using fallback model '%s' (%d chars)",
                        platform,
                        model_name,
                        len(draft),
                    )
                else:
                    log.info("[LLM] Drafted %s post using '%s' (%d chars)", platform, model_name, len(draft))
                return draft
        except Exception as exc:
            last_error = exc
            next_model = models_to_try[attempt_idx + 1] if attempt_idx + 1 < len(models_to_try) else None
            log.warning(
                "[LLM] Model '%s' failed for %s draft: %s. %s",
                model_name,
                platform,
                exc,
                f"Trying fallback model '{next_model}'..." if next_model else "No further fallback models.",
            )
            if next_model:
                time.sleep(1)

    if last_error:
        raise last_error
    raise RuntimeError(f"Failed to generate draft for {platform}: model returned empty response.")


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
