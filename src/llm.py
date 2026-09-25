"""
PersonaPulse – LLM Module
==========================
Interfaces with Google Gemini Flash 2.0 for:
  - Platform-tailored post drafting (LinkedIn & X)
  - Telegram-formatted draft summaries

Key Functions
-------------
- draft_post(article, style_profile, platform) → str
- complete_text(system_prompt, user_prompt, max_tokens, temperature) → str
- send_telegram_alert(post_id, drafts, article, research, image_bytes)
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
# Public: Generic text completion (reused by the research-agent phases)
# ---------------------------------------------------------------------------

def complete_text(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> str:
    """
    Run a single LLM completion through the shared Gemini client, retrying
    across settings.LLM_MODEL and the configured fallback models.

    Returns the trimmed response text. Raises if every model fails or all
    return empty responses. Reused by the research-agent phases (topic
    selection, question framing) so they share the same client/fallback
    behavior as draft_post.
    """
    client = _get_gemini()

    last_error: Exception | None = None
    models_to_try = [settings.LLM_MODEL] + settings.fallback_models

    for attempt_idx, model_name in enumerate(models_to_try):
        try:
            if attempt_idx > 0:
                log.warning(
                    "[LLM] Retrying completion using fallback model '%s'...",
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
            text = response.text.strip() if response.text else ""
            if text:
                if attempt_idx > 0:
                    log.warning(
                        "[LLM] Completion succeeded using fallback model '%s'",
                        model_name,
                    )
                return text
        except Exception as exc:
            last_error = exc
            next_model = models_to_try[attempt_idx + 1] if attempt_idx + 1 < len(models_to_try) else None
            log.warning(
                "[LLM] Model '%s' failed for completion: %s. %s",
                model_name,
                exc,
                f"Trying fallback model '{next_model}'..." if next_model else "No further fallback models.",
            )
            if next_model:
                time.sleep(1)

    if last_error:
        raise last_error
    raise RuntimeError("Failed to generate completion: model returned empty response.")


# ---------------------------------------------------------------------------
# Public: Send Telegram Approval Alert
# ---------------------------------------------------------------------------

def send_telegram_alert(
    post_id: str,
    linkedin_draft: str,
    x_draft: str,
    article: dict,
    research: Optional[dict] = None,
    image_bytes: Optional[bytes] = None,
) -> None:
    """
    Send a Telegram message to TELEGRAM_CHAT_ID with:
      - A concise research digest (question, key findings, caveat, source
        references) when *research* is provided, so the approver can judge
        the drafting against the research behind it
      - Article info + both drafts as caption text
      - Inline keyboard: [✅ Approve] [❌ Reject]
      - Photo attachment (sendPhoto) if image_bytes is available,
        otherwise plain text (sendMessage)
    """
    keyboard = _build_inline_keyboard(post_id)

    if image_bytes:
        # sendPhoto caps captions at 1024 – build the caption to that bound.
        caption = _build_caption(
            post_id, linkedin_draft, x_draft, article, research,
            max_length=_MAX_CAPTION_PHOTO,
        )
        _send_photo(caption, keyboard, image_bytes)
    else:
        caption = _build_caption(
            post_id, linkedin_draft, x_draft, article, research,
            max_length=_MAX_CAPTION_MESSAGE,
        )
        _send_message(caption, keyboard)


# ---------------------------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------------------------

_MAX_CAPTION_PHOTO = 1024      # Telegram sendPhoto caption limit
_MAX_CAPTION_MESSAGE = 4096    # Telegram sendMessage text limit
_DRAFT_LINKEDIN_MAX = 600      # raw budget for the LinkedIn snippet
_DRAFT_X_MAX = 260             # raw budget for the X (Twitter) snippet
_DRAFT_LINKEDIN_STEP = 80      # draft content is the lowest-priority text and
_DRAFT_X_STEP = 40             # is trimmed first, in these decrements
_RESEARCH_SECTION_MAX = 1000   # worksafe ceiling for the whole digest (all
                              # required bullets must still fit a typical load)
_RESEARCH_SHRINK_STEPS = (1000, 800, 600, 400, 200, 0)
_RESEARCH_QUESTION_MAX = 160
_RESEARCH_FINDINGS_MAX = 3
_RESEARCH_LINE_MAX = 120
_RESEARCH_CAVEAT_MAX = 160
_RESEARCH_SOURCES_MAX = 5
_RESEARCH_URL_MAX = 64
_HEADER_SOURCE_MAX = 160  # escaped-output budget for the dynamic source line
_HEADER_TITLE_MAX = 120
_HEADER_TITLE_SLIM = 60
_HEADER_URL_MAX = 200

# The characters Telegram's *legacy* Markdown parser (parse_mode="Markdown")
# treats as special: per the Bot API docs, only `_`, `*`, '`' and `[` are
# escaped outside an entity (plus backslash itself), i.e. MarkdownV2-only
# escapes like `( ) ~ > # + - = | { } . !` must NOT be applied here. A
# parentheses or `]` is inert once every `[` is escaped, because we never
# generate Markdown links. Only dynamic content is escaped; the static format
# markers we write in captions are left untouched.
_MARKDOWN_SPECIALS = ("\\", "*", "_", "[", "`")


def _clip(value, limit: int) -> str:
    """Normalize to text and truncate to *limit* chars, appending "…" when cut."""
    if value is None:
        return ""
    value = str(value).strip()
    if len(value) <= limit:
        return value
    if limit <= 1:
        return value[:limit]
    return value[: limit - 1] + "…"


def _escape_markdown(value) -> str:
    """Escape Telegram legacy-Markdown specials in *dynamic* text.

    Applies exactly the five characters Telegram's legacy ``Markdown`` parser
    treats as special (backslash, ``*``, ``_``, ``[``, backtick) – nothing else
    (so no ``( )``, ``]``, ``~``, ``#``, ... escaping, which would be a
    MarkdownV2 rule). Backslash is escaped first so an input backslash cannot
    re-mark an escape we just inserted. URLs are escaped in place (e.g.
    ``a_b`` -> ``a\\_b``) so they render as the same visible URL; no Markdown
    links are introduced.
    """
    text = value if isinstance(value, str) else str(value)
    for special in _MARKDOWN_SPECIALS:
        text = text.replace(special, "\\" + special)
    return text


def _clip_md_bounded(value, budget: int) -> str:
    """Longest raw prefix of *value* whose *escaped* form fits in *budget*.

    Escaping can double a string's length, so the *escaped* output is measured
    against the limit, not the raw input: the returned string always satisfies
    ``len(result) <= budget`` (the binary search maximizes the raw prefix whose
    escaped length fits). A truncation boundary therefore can never leave a
    lone escape character or cut through an escape sequence, and Markdown stays
    valid by construction. When the value is cut short and a single char is
    left over, an ellipsis is appended to signal the truncation.
    """
    if value is None:
        return ""
    raw = str(value).strip()
    if budget <= 0 or not raw:
        return ""

    low, high, best = 1, min(len(raw), budget), 0
    while low <= high:
        mid = (low + high) // 2
        if len(_escape_markdown(raw[:mid])) <= budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    result = _escape_markdown(raw[:best])
    if best < len(raw) and len(result) + 1 <= budget and not result.endswith("\\"):
        result += "…"
    return result


def _draft_id_footer(post_id: str, max_length: int) -> str:
    """Render the Draft ID line bounded to *max_length* with valid Markdown.

    The static label is plain text (no Markdown specials), so any prefix of it
    is safe to emit. The Draft ID value is escaped *after* raw-clipping so
    truncation can never split an escape sequence, and the inline-code
    backticks are only emitted when both can fit – otherwise the value is
    shown as safe plain text, never as an unmatched backtick.
    """
    label = "🆔 Draft ID: "
    if max_length <= 0:
        return ""
    if max_length <= len(label):
        return _clip(label, max_length)
    room = max_length - len(label)
    if room <= 2:
        return label + _clip_md_bounded(post_id, room)
    return f"{label}`{_clip_md_bounded(post_id, room - 2)}`"


def _unescaped_backticks(text: str) -> list:
    """Indexes of backticks that are *not* escaped by a preceding backslash.

    An escape is only ``\\`` followed by a character the legacy Telegram
    ``Markdown`` dialect can actually escape (``\\ * _ [`` and backtick).
    Arbitrary backslashes such as ``\\a``, ``\\q`` or ``\\(`` are plain
    characters, not escapes, so they are walked past one char at a time. This
    mirrors the legacy parser and stops the scanner from assuming every
    ``\\x`` means "x is escaped".
    """
    ticks = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\":
            if (
                index + 1 < len(text)
                and text[index + 1] in _MARKDOWN_SPECIALS
            ):
                index += 2
                continue
        elif char == "`":
            ticks.append(index)
        index += 1
    return ticks


def _clip_md_safe(markdown_text, limit: int) -> str:
    """Longest prefix of *already-formatted* Markdown that stays well-formed.

    NOT interchangeable with :func:`_clip_md_bounded`: that helper escapes raw
    dynamic values and measures the escaped output; this helper receives text
    whose Markdown is already intended (e.g. a canary string with ``*bold*``
    markers) and only ever clips it, preserving that formatting.

    Validation never bypasses the length check: it runs whether or not the
    text had to be reduced, so valid Markdown that fits is returned unchanged
    while malformed input is repaired even under the limit. Guarantees:
      - ``len(result) <= limit``;
      - no dangling trailing backslash -- a lone ``\\`` whose escapee was cut
        away is dropped, while a complete ``\\\\`` pair or ``\\X`` escape
        stays intact: in a trailing run only an odd count leaves one dangling
        marker, which is the one removed;
      - no real backtick is left unmatched -- backticks preceded by a backslash
        (``\\` ``) are literal characters and are never treated as delimiters,
        and the input is cut back to the last *real* unbalanced backtick.
    """
    raw = str(markdown_text)
    if limit <= 0:
        return ""
    candidate = raw if len(raw) <= limit else raw[:limit]

    # A backslash alone at the edge is an escape whose escapee was truncated;
    # in a trailing run, only an odd count leaves one such dangling marker.
    if candidate.endswith("\\"):
        run = len(candidate) - len(candidate.rstrip("\\"))
        if run % 2 == 1:
            candidate = candidate[:-1]

    ticks = _unescaped_backticks(candidate)
    if len(ticks) % 2 == 1:
        candidate = candidate[: ticks[-1]] or ""

    return candidate


def _build_header(article: dict, mode: int) -> str:
    """Render the approval-context header at a trim level.

    mode 0: source + topic + url (full)
    mode 1: source + topic
    mode 2: source + slim topic
    mode 3: source only
    mode 4: the bare title line (the last piece of context kept)
    """
    source = _clip_md_bounded(article.get("source") or "Unknown", _HEADER_SOURCE_MAX)
    lines = ["🔔 *New Draft Ready for Approval*"]
    if mode >= 4:
        return lines[0]

    lines.append(f"📰 *Source:* {source}")

    if mode <= 1:
        topic = _clip_md_bounded(article.get("title"), _HEADER_TITLE_MAX)
    elif mode == 2:
        topic = _clip_md_bounded(article.get("title"), _HEADER_TITLE_SLIM)
    else:
        topic = ""
    if topic:
        lines.append(f"📌 *Topic:* {topic}")

    if mode == 0:
        url = _clip_md_bounded(article.get("url"), _HEADER_URL_MAX)
        if url:
            lines.append(f"🔗 {url}")

    return "\n".join(lines)


def _build_caption(
    post_id: str,
    linkedin_draft: str,
    x_draft: str,
    article: dict,
    research: Optional[dict] = None,
    max_length: int = _MAX_CAPTION_MESSAGE,
) -> str:
    """Build the approval caption, escaped and bounded to *max_length*.

    The escaped result is guaranteed not to exceed *max_length* by construction
    (no post-hoc truncation). Trimming follows a fixed priority ladder – draft
    content is sacrificed before the research digest, which is sacrificed
    before header detail; the header and the Draft ID footer always survive.
    For a limit too small for any assembled caption, a Markdown-safe bounded
    footer is returned instead of a blind truncation of formatted text.
    """
    def assemble(li_budget: int, x_budget: int, research_max: int, header_mode: int) -> str:
        header = _build_header(article, header_mode)
        research_section = _build_research_section(research, research_max)

        li_snippet = _clip_md_bounded(linkedin_draft, li_budget)
        x_snippet = _clip_md_bounded(x_draft, x_budget)
        li_section = f"━━━ *LinkedIn Draft* ━━━\n{li_snippet}" if li_snippet else ""
        x_section = f"━━━ *X (Twitter) Draft* ━━━\n{x_snippet}" if x_snippet else ""
        footer = f"🆔 Draft ID: `{_clip_md_bounded(post_id, 64)}`"

        sections = [header, research_section, li_section, x_section, footer]
        return "\n\n".join(section for section in sections if section)

    # Drafts are the least-important text: shrink them first.
    for li_budget in range(_DRAFT_LINKEDIN_MAX, -1, -_DRAFT_LINKEDIN_STEP):
        for x_budget in range(_DRAFT_X_MAX, -1, -_DRAFT_X_STEP):
            caption = assemble(li_budget, x_budget, _RESEARCH_SECTION_MAX, 0)
            if len(caption) <= max_length:
                return caption

    # Then the research digest...
    for research_max in _RESEARCH_SHRINK_STEPS:
        caption = assemble(0, 0, research_max, 0)
        if len(caption) <= max_length:
            return caption

    # ...then header detail; the footer is the absolute last resort.
    for header_mode in (1, 2, 3, 4):
        caption = assemble(0, 0, 0, header_mode)
        if len(caption) <= max_length:
            return caption

    return _draft_id_footer(post_id, max_length)


def _build_research_section(research, max_length: int = _RESEARCH_SECTION_MAX) -> str:
    """Render a concise, self-contained research digest for the approval message.

    Exposes only the question, finding claims, a single caveat
    (counterargument or limitation) and short source links – never raw source
    bodies. Every part is tightly bounded and the whole digest shrinks, in
    priority order (question > findings > caveat > sources), to fit
    *max_length*, so long research results cannot balloon the caption.
    """
    if not isinstance(research, dict) or not research:
        return ""

    raw_question = research.get("research_question")
    findings = [f for f in (research.get("key_findings") or []) if f]
    caveat = research.get("counterargument")
    if not caveat:
        caveat = research.get("limitation")
    urls = [u for u in (research.get("source_urls") or []) if u]
    raw_count = research.get("source_count")
    count = (
        raw_count
        if isinstance(raw_count, int) and not isinstance(raw_count, bool)
        else len(urls)
    )

    def sources_line() -> str:
        label = f"📚 *Sources ({count}):*"
        if urls:
            label += " " + " · ".join(
                _clip_md_bounded(u, _RESEARCH_URL_MAX)
                for u in urls[:_RESEARCH_SOURCES_MAX]
            )
        else:
            label += " none"
        return label

    def render(findings_n: int, caveat_on: bool, sources_on: int) -> str:
        lines = []
        if raw_question:
            lines.append(
                f"🧭 *Research question:* "
                f"{_clip_md_bounded(raw_question, _RESEARCH_QUESTION_MAX)}"
            )
        if findings_n and findings:
            finding_lines = [
                f" {index}. {_clip_md_bounded(finding, _RESEARCH_LINE_MAX)}"
                for index, finding in enumerate(findings[:findings_n], start=1)
            ]
            lines.append("🔎 *Key findings:*\n" + "\n".join(finding_lines))
        if caveat_on and caveat:
            lines.append(
                f"⚠️ *Caveat:* {_clip_md_bounded(caveat, _RESEARCH_CAVEAT_MAX)}"
            )
        if sources_on:
            lines.append(sources_line())
        return "\n\n".join(lines)

    # Largest digest that still fits the budget: drop sources before the
    # caveat before individual findings; the question is kept to the end.
    candidates = [
        (_RESEARCH_FINDINGS_MAX, True, 1),
        (_RESEARCH_FINDINGS_MAX, True, 0),
        (_RESEARCH_FINDINGS_MAX, False, 0),
        (2, False, 0),
        (1, False, 0),
        (0, False, 0),
    ]
    for findings_n, caveat_on, sources_on in candidates:
        section = render(findings_n, caveat_on, sources_on)
        if section and len(section) <= max_length:
            return section
    return ""


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
                "caption": caption,
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
                "text": caption,
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
# Public: Simple Telegram text notifications / canary alerts
# ---------------------------------------------------------------------------

def send_telegram_text(message: str) -> None:
    """Send a plain-text Telegram notification (no Markdown parsing).

    The message is arbitrary dynamic text, so Markdown interpretation is never
    requested and the text is sent verbatim (nothing to escape). It is bounded
    to Telegram's 4096-character limit with the same safe closer used for the
    approval captions.
    """
    url = f"{settings.telegram_api_base}/sendMessage"
    with httpx.Client(timeout=15) as client:
        resp = client.post(
            url,
            json={
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": _clip(message, _MAX_CAPTION_MESSAGE),
                "disable_web_page_preview": True,
            },
        )
    _handle_telegram_response(resp, "sendMessage[plain]")


def send_telegram_markdown(message: str) -> None:
    """Send a Markdown-formatted Telegram notification, bounded safely.

    Only for callers that intentionally build Markdown (e.g. the canary
    alerts). The text is truncated to a well-formed prefix so a length
    boundary can never split an escape sequence or leave an unmatched
    backtick.
    """
    url = f"{settings.telegram_api_base}/sendMessage"
    with httpx.Client(timeout=15) as client:
        resp = client.post(
            url,
            json={
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": _clip_md_safe(message, _MAX_CAPTION_MESSAGE),
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
        )
    _handle_telegram_response(resp, "sendMessage[markdown]")
