"""
PersonaPulse – Local Research & Drafting Tester
================================================
Quickly test what news Tavily finds, verify og:image extraction,
and preview Gemini Flash 2.0 drafts right in your terminal
(without publishing anything to LinkedIn or X).

Usage:
    # 1. Run with auto-rotated agentic-AI query (recommended):
    python test_research.py

    # 2. Run with custom query:
    python test_research.py "AI agents production issues"
    python test_research.py "agent orchestration frameworks vs single agents"
"""

import sys
import json
from src.config import EXCLUDED_SEARCH_DOMAINS
from src.ingestion import (
    _rotate_search_query,
    fetch_trending_tech_news,
    extract_og_image,
)
from src.memory import get_style_profile
from src.llm import draft_post

def test_research(query: str = ""):
    print("\n" + "=" * 65)
    print("🔍 STEP 1: Researching Web via Tavily API")

    effective_query = query or _rotate_search_query()
    print(f"   Query: \"{effective_query}\"")
    if not query:
        print("   (auto-rotated agentic-AI query for today)")
    print(f"   Excluding domains: {', '.join(EXCLUDED_SEARCH_DOMAINS)}")
    print("=" * 65)

    try:
        article = fetch_trending_tech_news(query or None)
    except Exception as e:
        print(f"❌ Ingestion error: {e}")
        return

    print(f"\n📰 Title:     {article['title']}")
    print(f"🌐 Source:    {article['source']}")
    print(f"🔗 URL:       {article['url']}")
    print(f"📅 Published: {article.get('published') or 'N/A'}")
    print(f"📄 Body Size: {len(article['body'])} characters")
    print("\n--- Article Content Snippet (first 500 chars) ---")
    print(article['body'][:500] + ("..." if len(article['body']) > 500 else ""))
    print("-" * 65)

    # ── Image extraction ─────────────────────────────────────────
    print("\n🖼️ STEP 2: Extracting og:image from article URL...")
    image_bytes = extract_og_image(article['url'])
    if image_bytes:
        print(f"✅ Image extracted successfully! ({len(image_bytes)} bytes)")
    else:
        print("ℹ️ No og:image found or extraction skipped (will use text/link-preview fallback).")

    # ── Style profile ────────────────────────────────────────────
    print("\n🎨 STEP 3: Loading Style Profile from Supabase...")
    try:
        style = get_style_profile()
        print(f"✅ Style profile loaded: Tone = \"{style.get('tone', 'default')}\"")
        topics = style.get('topics_of_interest', [])
        if topics:
            print(f"   Topics of interest: {', '.join(topics)}")
    except Exception as e:
        print(f"⚠️ Could not load style from Supabase ({e}), using fallback.")
        style = {
            "tone": "thoughtful, engaging, professional",
            "linkedin": {"structure": "Hook -> Insight -> Takeaway -> Question"},
            "x": {"style": "punchy, concise"}
        }

    # ── Gemini Drafting ──────────────────────────────────────────
    print("\n✍️ STEP 4: Generating Drafts with Google Gemini Flash 2.0...")
    
    print("\n" + "─" * 40 + " LinkedIn Draft Preview " + "─" * 40)
    try:
        linkedin_draft = draft_post(article, style, platform="linkedin")
        print(linkedin_draft)
    except Exception as e:
        print(f"❌ Gemini drafting error (LinkedIn): {e}")

    # print("\n" + "─" * 40 + " X (Twitter) Draft Preview " + "─" * 40)
    # try:
    #     x_draft = draft_post(article, style, platform="x")
    #     print(x_draft)
    # except Exception as e:
    #     print(f"❌ Gemini drafting error (X): {e}")

    print("\n" + "=" * 65)
    print("🎉 Test completed! No posts were published.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    search_query = sys.argv[1] if len(sys.argv) > 1 else ""
    test_research(search_query)
