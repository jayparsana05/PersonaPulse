"""
PersonaPulse – LangGraph Agent Pipeline
=========================================
Orchestrates the full end-to-end social media automation workflow
as a directed state graph using LangGraph.

Pipeline Nodes (sequential)
---------------------------
1. canary_check      – Verify LinkedIn token health; halt if critical.
2. search            – Fetch trending tech story via Tavily.
3. deduplicate        – Embed story and check against Supabase posts.
4. draft             – Generate LinkedIn + X drafts via Gemini Flash 2.0.
5. store_and_alert   – Save draft to Supabase; send Telegram approval alert.

State is passed between nodes as a TypedDict (AgentState).

Entry Points
------------
    python -m src.agent                          # run with default query
    python -m src.agent "quantum computing news" # custom query
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.canary import run_canary_check
from src.ingestion import extract_og_image, fetch_trending_tech_news
from src.llm import draft_post, send_telegram_alert
from src.memory import (
    check_is_duplicate,
    get_normalized_embedding,
    get_style_profile,
    store_draft,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent State Definition
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    """
    Shared mutable state passed between all LangGraph nodes.
    All fields are optional at construction; populated progressively.
    """
    # Input
    search_query: str

    # Canary
    token_ok: bool

    # Search
    article: dict                       # keys: url, title, body, published, source

    # Deduplication
    embedding: list[float]
    is_duplicate: bool

    # Drafting
    style_profile: dict
    linkedin_draft: str
    x_draft: str

    # Storage & Alert
    post_id: str
    image_bytes: Optional[bytes]

    # Control flow
    pipeline_halted: bool
    halt_reason: str


# ---------------------------------------------------------------------------
# Node 1: Canary Check
# ---------------------------------------------------------------------------

def node_canary_check(state: AgentState) -> AgentState:
    """Verify LinkedIn OAuth token health before any API calls."""
    log.info("━━━ [Node 1/5] Canary Check ━━━")

    token_ok = run_canary_check()

    if not token_ok:
        return {
            **state,
            "token_ok":        False,
            "pipeline_halted": True,
            "halt_reason":     "LinkedIn token expired – pipeline halted.",
        }

    return {**state, "token_ok": True, "pipeline_halted": False}


def _should_halt_after_canary(state: AgentState) -> str:
    """Conditional edge: route to END if token is expired."""
    if state.get("pipeline_halted"):
        log.warning("[Agent] Halting pipeline: %s", state.get("halt_reason"))
        return "halt"
    return "continue"


# ---------------------------------------------------------------------------
# Node 2: Search
# ---------------------------------------------------------------------------

def node_search(state: AgentState) -> AgentState:
    """Fetch the top trending tech news article via Tavily."""
    log.info("━━━ [Node 2/5] Search ━━━")

    query = state.get("search_query", "latest AI and machine learning breakthroughs")
    article = fetch_trending_tech_news(query=query)

    log.info("[Search] Article: '%s' from %s", article["title"], article["source"])
    return {**state, "article": article}


# ---------------------------------------------------------------------------
# Node 3: Deduplication
# ---------------------------------------------------------------------------

def node_deduplicate(state: AgentState) -> AgentState:
    """Embed the article and check for semantic duplicates in Supabase."""
    log.info("━━━ [Node 3/5] Deduplication ━━━")

    article = state["article"]
    # Embed title + first 500 chars of body for dedup signal
    dedup_text = f"{article['title']} {article['body'][:500]}"

    embedding = get_normalized_embedding(dedup_text)
    is_dup    = check_is_duplicate(embedding, threshold=0.85)

    if is_dup:
        log.info("[Dedup] Similar content already posted – skipping.")
        return {
            **state,
            "embedding":       embedding,
            "is_duplicate":    True,
            "pipeline_halted": True,
            "halt_reason":     "Duplicate content detected – skipping this story.",
        }

    log.info("[Dedup] No duplicate found – proceeding.")
    return {**state, "embedding": embedding, "is_duplicate": False}


def _should_halt_after_dedup(state: AgentState) -> str:
    """Conditional edge: stop if duplicate detected."""
    if state.get("is_duplicate"):
        return "halt"
    return "continue"


# ---------------------------------------------------------------------------
# Node 4: Drafting
# ---------------------------------------------------------------------------

def node_draft(state: AgentState) -> AgentState:
    """
    Retrieve the style profile and generate both LinkedIn and X drafts
    using Gemini Flash 2.0.
    """
    log.info("━━━ [Node 4/5] Drafting ━━━")

    style_profile = get_style_profile()
    article       = state["article"]

    linkedin_draft = draft_post(article, style_profile, platform="linkedin")
    x_draft        = draft_post(article, style_profile, platform="x")

    log.info("[Draft] LinkedIn: %d chars | X: %d chars", len(linkedin_draft), len(x_draft))

    return {
        **state,
        "style_profile":  style_profile,
        "linkedin_draft": linkedin_draft,
        "x_draft":        x_draft,
    }


# ---------------------------------------------------------------------------
# Node 5: Store & Alert
# ---------------------------------------------------------------------------

def node_store_and_alert(state: AgentState) -> AgentState:
    """
    1. Extract og:image from article URL.
    2. Store the draft to Supabase (status=PENDING).
    3. Send Telegram photo/text message with approval inline keyboard.
    """
    log.info("━━━ [Node 5/5] Store & Alert ━━━")

    article        = state["article"]
    embedding      = state["embedding"]
    linkedin_draft = state["linkedin_draft"]
    x_draft        = state["x_draft"]

    # ── Extract og:image (best-effort) ─────────────────────────────────
    image_bytes = extract_og_image(article.get("url", ""))

    # ── Store in Supabase ───────────────────────────────────────────────
    combined_content = f"LINKEDIN:\n{linkedin_draft}\n\nX:\n{x_draft}"
    post_id = store_draft(
        platform   = "both",
        topic      = article["title"][:256],
        content    = combined_content,
        embedding  = embedding,
        image_url  = article.get("url"),
    )

    log.info("[Store] Draft saved – id=%s", post_id)

    # ── Send Telegram Approval Alert ────────────────────────────────────
    send_telegram_alert(
        post_id        = post_id,
        linkedin_draft = linkedin_draft,
        x_draft        = x_draft,
        article        = article,
        image_bytes    = image_bytes,
    )

    log.info("[Alert] Telegram approval request sent.")
    return {**state, "post_id": post_id, "image_bytes": image_bytes}


# ---------------------------------------------------------------------------
# Build LangGraph State Graph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the PersonaPulse LangGraph pipeline.

    Graph topology:
        canary_check ──(halt?)──► END
                    ──(ok)────► search ──► deduplicate ──(dup?)──► END
                                                       ──(ok)───► draft ──► store_and_alert ──► END
    """
    graph = StateGraph(AgentState)

    # ── Register Nodes ───────────────────────────────────────────────────
    graph.add_node("canary_check",    node_canary_check)
    graph.add_node("search",          node_search)
    graph.add_node("deduplicate",     node_deduplicate)
    graph.add_node("draft",           node_draft)
    graph.add_node("store_and_alert", node_store_and_alert)

    # ── Entry Point ──────────────────────────────────────────────────────
    graph.set_entry_point("canary_check")

    # ── Edges ────────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "canary_check",
        _should_halt_after_canary,
        {"halt": END, "continue": "search"},
    )

    graph.add_edge("search", "deduplicate")

    graph.add_conditional_edges(
        "deduplicate",
        _should_halt_after_dedup,
        {"halt": END, "continue": "draft"},
    )

    graph.add_edge("draft",           "store_and_alert")
    graph.add_edge("store_and_alert", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def run_pipeline(query: str = "latest AI and technology news") -> dict:
    """
    Execute the full PersonaPulse pipeline.

    Parameters
    ----------
    query : str
        Tavily search query for discovering trending news.

    Returns
    -------
    Final agent state dict.
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    log.info("🚀 PersonaPulse pipeline starting | query='%s'", query)

    app    = build_graph()
    result = app.invoke({"search_query": query})

    if result.get("pipeline_halted"):
        log.info("🛑 Pipeline halted: %s", result.get("halt_reason", "unknown reason"))
    elif result.get("post_id"):
        log.info("✅ Pipeline complete – draft id=%s awaiting Telegram approval.", result["post_id"])
    else:
        log.warning("⚠️  Pipeline ended in unexpected state: %s", result)

    return result


if __name__ == "__main__":
    print("Starting PersonaPulse Pipeline...")
    query_arg = sys.argv[1] if len(sys.argv) > 1 else "latest AI and machine learning news"
    final_state = run_pipeline(query=query_arg)
    sys.exit(0 if not final_state.get("pipeline_halted") else 1)
