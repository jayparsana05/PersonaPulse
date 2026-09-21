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
    python -m src.agent --discover               # Phase-1 discovery only
    python -m src.agent --select                 # discover → select → frame question
    python -m src.agent --research "question"    # Prompt-4 multi-source research (stores question + linked sources)
    python -m src.agent --evidence "question"    # Phase-3 evidence extraction (research → claims), no drafting
    python -m src.agent --analyze "question"     # Phase-3 critical analysis (research → evidence → analysis), no drafting
    python -m src.agent --report "question"      # Phase-3 report synthesis (research → evidence → analysis → report), no drafting
"""

from __future__ import annotations

import logging
import sys
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from src.analysis import analyze_evidence as analysis_stage
from src.canary import run_canary_check
from src.evidence import extract_evidence as evidence_stage
from src.ingestion import (
    discover_topic_candidates,
    extract_og_image_with_url,
    fetch_trending_tech_news,
)
from src.llm import draft_post, send_telegram_alert
from src.memory import (
    check_is_duplicate,
    get_normalized_embedding,
    get_style_profile,
    store_draft,
    store_research_question,
    store_research_sources,
)
from src.models import ResearchQuestion
from src.report import synthesize_report as report_stage
from src.research import research_question as research_stage
from src.selection import frame_question, select_topic

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

    # Discovery (research-agent phase)
    topic_candidates: list              # list[TopicCandidate]

    # Selection (research-agent phase)
    selection: dict                     # TopicSelection dict (selected + reasoning)
    research_question: dict             # ResearchQuestion dict for the selected topic
    research_question_id: str           # persisted research_questions row id

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
    """Fetch the top trending agentic-AI article via Tavily."""
    log.info("━━━ [Node 2/5] Search ━━━")

    query = state.get("search_query")
    article = fetch_trending_tech_news(query=query or None)

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
    # Returns both the og:image URL (for the posts table) and the raw
    # image bytes (for the Telegram photo). Either may be None.
    image_url, image_bytes = extract_og_image_with_url(article.get("url", ""))

    # ── Store in Supabase ───────────────────────────────────────────────
    combined_content = f"LINKEDIN:\n{linkedin_draft}\n\nX:\n{x_draft}"
    post_id = store_draft(
        platform    = "both",
        topic       = article["title"][:256],
        content     = combined_content,
        embedding   = embedding,
        article_url = article.get("url"),
        image_url   = image_url,
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

def run_pipeline(query: str = "") -> dict:
    """
    Execute the full PersonaPulse pipeline.

    Parameters
    ----------
    query : str
        Tavily search query for discovering trending news. Empty string
        selects an agentic-AI query, rotated per run.

    Returns
    -------
    Final agent state dict.
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    log.info(
        "🚀 PersonaPulse pipeline starting | query='%s'",
        query or "(auto-rotated agentic-AI query)",
    )

    app    = build_graph()
    result = app.invoke({"search_query": query})

    if result.get("pipeline_halted"):
        log.info("🛑 Pipeline halted: %s", result.get("halt_reason", "unknown reason"))
    elif result.get("post_id"):
        log.info("✅ Pipeline complete – draft id=%s awaiting Telegram approval.", result["post_id"])
    else:
        log.warning("⚠️  Pipeline ended in unexpected state: %s", result)

    return result


# ---------------------------------------------------------------------------
# Phase-1 Discovery Entry Point (Research Agent)
# ---------------------------------------------------------------------------
# Discovery → Topic Candidates → (Selection is Phase 2).
# Returns the clean, deduplicated list of TopicCandidate objects so the
# next phase can consume them. Does not draft or publish anything.

def run_discovery(
    query: str = "",
    limit: Optional[int] = None,
) -> list:
    """
    Run the Phase-1 discovery stage of the research agent.

    Parameters
    ----------
    query : str
        Optional Tavily search query. Empty string rotates through the
        agentic-AI query list.
    limit : int, optional
        Maximum number of candidates to return (defaults to the
        configured DISCOVERY_CANDIDATE_COUNT).

    Returns
    -------
    list[TopicCandidate]
        The clean, deduplicated list of candidates for the next phase.
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    log.info(
        "🔎 Research Agent – Discovery | query='%s' | limit=%s",
        query or "(auto-rotated agentic-AI query)", limit,
    )

    candidates = discover_topic_candidates(query=query or None, limit=limit)

    if not candidates:
        log.warning("🧭 Discovery returned no usable topic candidates.")
    else:
        for idx, candidate in enumerate(candidates, start=1):
            log.info(
                "  %d. [%.2f] %s — %s (%s)",
                idx,
                candidate.search_score,
                candidate.title,
                candidate.source,
                candidate.url,
            )

    return candidates


# ---------------------------------------------------------------------------
# Phase-2 Selection Entry Point (Research Agent)
# ---------------------------------------------------------------------------
# Discover → TopicCandidates → Select → ResearchQuestion.
# Selection stays separate from research: this stage chooses ONE topic and
# frames the research question, but does NOT research, draft, or publish.

def run_selection(
    query: str = "",
    limit: Optional[int] = None,
    candidates: Optional[list] = None,
) -> dict:
    """
    Run the Select Topic stage of the research agent.

    Parameters
    ----------
    query : str
        Optional Tavily search query. Empty string rotates through the
        agentic-AI query list.
    limit : int, optional
        Maximum number of candidates to discover (defaults to the
        configured DISCOVERY_CANDIDATE_COUNT).
    candidates : list[TopicCandidate], optional
        Pre-discovered candidates; when provided, discovery is skipped
        (useful for tests and for running selection against existing
        candidates).

    Returns
    -------
    dict with keys:
        topic_candidates   : list[TopicCandidate] (discovered, may be empty)
        selection          : TopicSelection (selected + reasoning + criteria)
        research_question  : ResearchQuestion or None (when nothing was selected)
        research_question_id : UUID of the persisted research_questions row,
                              or None when nothing was selected (or persistence failed)
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    log.info(
        "🔎 Research Agent – Select Topic | query='%s' | limit=%s | candidates=%s",
        query or "(auto-rotated agentic-AI query)", limit,
        "provided" if candidates is not None else "discover",
    )

    if candidates is None:
        candidates = discover_topic_candidates(query=query or None, limit=limit)

    selection = select_topic(candidates, query=query)

    if selection.selected is not None:
        log.info(
            "  ✔ Selected: [%s] %s — %s",
            selection.mode, selection.selected.title, selection.selected.url,
        )
        log.info("  Reasoning: %s", selection.reasoning[:200])
    else:
        log.warning("  ✖ No topic selected (mode=%s): %s", selection.mode, selection.reasoning)

    question = frame_question(selection.selected, query=query)

    question_id = None
    if question is not None:
        log.info("  ❓ Research question: %s", question.question)
        if question.aspects:
            log.info("  Aspects: %s", ", ".join(question.aspects))
        question_id = persist_research_question(question)

    return {
        "topic_candidates": list(candidates),
        "selection": selection,
        "research_question": question,
        "research_question_id": question_id,
    }


def persist_research_question(rq) -> Optional[str]:
    """Persist a ResearchQuestion and return its UUID, or None when the store
    is unavailable. Non-blocking: callers continue in-memory either way."""
    try:
        question_id = store_research_question(rq)
        log.info("  💾 Research question stored – id=%s", question_id)
        return question_id
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "⚠️  Could not persist research question (%s: %s) – continuing.",
            type(exc).__name__, exc,
        )
        return None


# ---------------------------------------------------------------------------
# Prompt-4 Research Entry Point (Research Agent)
# ---------------------------------------------------------------------------
# ResearchQuestion → research queries → multi-source search → normalize →
# dedupe → ResearchSource[]. Stop. Independent of the production LangGraph
# pipeline: evidence extraction, synthesis, and reporting are later phases.

def run_research(
    rq,
    research_question_id: Optional[str] = None,
    use_llm: bool = True,
    max_queries: Optional[int] = None,
    max_sources_per_query: Optional[int] = None,
    max_sources: Optional[int] = None,
    min_score: Optional[float] = None,
) -> dict:
    """
    Run the Prompt-4 research stage for an existing ResearchQuestion.

    Parameters
    ----------
    rq : ResearchQuestion
        The framed question to research (from frame_question / run_selection).
    research_question_id : str, optional
        UUID of the persisted research_questions row, used to link the
        collected sources back to their question. Sources are NOT persisted
        when this is None/empty (the in-memory results are still returned);
        a falsy flag such as None keeps the stage non-blocking but avoids
        creating unlinked research_sources rows.
    use_llm / max_queries / max_sources_per_query / max_sources / min_score :
        passed through to the research stage (see src.research.research_question).

    Returns
    -------
    dict with keys:
        research_question    : the ResearchQuestion researched
        research_sources     : list[ResearchSource] (normalized + deduplicated)
        status               : "ok" when ≥1 source, else "empty"
        research_source_ids  : list[str] of persisted row UUIDs (may be empty
                               when nothing was persisted or persistence failed)
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    log.info("🔎 Research Agent – Research | question='%s'", getattr(rq, "question", rq))

    sources = research_stage(
        rq,
        use_llm=use_llm,
        max_queries=max_queries,
        max_sources_per_query=max_sources_per_query,
        max_sources=max_sources,
        min_score=min_score,
    )

    source_ids: list[str] = []
    if sources:
        if research_question_id:
            try:
                source_ids = store_research_sources(research_question_id, sources)
            except Exception as exc:  # pylint: disable=broad-except
                log.warning(
                    "⚠️  Could not persist research sources (%s: %s) – continuing with in-memory result.",
                    type(exc).__name__, exc,
                )
        else:
            log.warning(
                "🧭 No research_question_id – skipping source persistence (%d source(s) kept in memory).",
                len(sources),
            )
    else:
        log.warning("🧭 Research returned no usable sources for the question.")

    log.info("  📚 Research complete: %d source(s) | status=%s", len(sources), "ok" if sources else "empty")
    return {
        "research_question": rq,
        "research_sources": sources,
        "status": "ok" if sources else "empty",
        "research_source_ids": source_ids,
    }


# ---------------------------------------------------------------------------
# Phase-3 Evidence Entry Point (Research Agent)
# ---------------------------------------------------------------------------
# ResearchQuestion + ResearchSource[] → evidence_stage() → Evidence[].
# Evidence extraction stays strictly separated from synthesis: no prose, no
# report, no LinkedIn drafting here (that is a later phase).

def run_evidence(
    research_question=None,
    research_sources=None,
    research_question_id: Optional[str] = None,
    use_llm: bool = True,
    max_claims_per_source: Optional[int] = None,
) -> dict:
    """
    Run the Phase-3 evidence stage for a research question + its sources.

    Parameters
    ----------
    research_question : ResearchQuestion, optional
        The framed question the claims should relate to (None → no evidence).
    research_sources  : list[ResearchSource], optionally
        The collected, deduplicated sources to extract claims from.
    research_question_id : str, optional
        Kept for traceability parity with run_research; not required for
        extraction and reserved for a future persistence step.
    use_llm / max_claims_per_source : passed through to evidence_stage.

    Returns
    -------
    dict with keys:
        research_question : the ResearchQuestion the evidence relates to
        research_sources  : list[ResearchSource] the evidence came from
        evidence          : list[Evidence] (supported, source-attributed claims)
        status            : "ok" when ≥1 claim extracted, else "empty"
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    sources = list(research_sources or [])
    log.info("🧠 Research Agent – Evidence | question='%s' | sources=%d",
             getattr(research_question, "question", research_question), len(sources))

    evidence = evidence_stage(
        research_question,
        sources,
        use_llm=use_llm,
        max_claims_per_source=max_claims_per_source,
    )
    log.info("  📎 Evidence extraction complete: %d claim(s) | status=%s",
             len(evidence), "ok" if evidence else "empty")
    return {
        "research_question": research_question,
        "research_sources": sources,
        "evidence": evidence,
        "status": "ok" if evidence else "empty",
    }


# ---------------------------------------------------------------------------
# Phase-3 Critical Analysis Entry Point (Research Agent)
# ---------------------------------------------------------------------------
# ResearchQuestion + Evidence[] → analysis_stage() → CriticalAnalysis.
# Produces structured analysis only (claims, counterarguments, limitations,
# uncertainties); synthesis and LinkedIn drafting are later phases.

def run_critical_analysis(
    research_question=None,
    evidence=None,
    research_question_id: Optional[str] = None,
    use_llm: bool = True,
) -> dict:
    """
    Run the Phase-3 critical analysis stage for a question + its evidence.

    Parameters
    ----------
    research_question : ResearchQuestion, optional
        The framed question the analysis relates to (None → empty analysis).
    evidence          : list[Evidence], optional
        The extracted, source-attributed claims to analyse.
    research_question_id : str, optional
        Kept for traceability parity with run_research/run_evidence; not
        required for analysis and reserved for a future persistence step.
    use_llm : bool
        Passed through to analysis_stage; False → empty analysis.

    Returns
    -------
    dict with keys:
        research_question : the ResearchQuestion the analysis relates to
        evidence          : list[Evidence] the analysis was built from
        analysis          : CriticalAnalysis (structured, for later synthesis)
        status            : "ok" when analysis has content, else "empty"
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    items = list(evidence or [])
    log.info("🧠 Research Agent – Critical Analysis | question='%s' | evidence=%d",
             getattr(research_question, "question", research_question), len(items))

    analysis = analysis_stage(research_question, items, use_llm=use_llm)
    log.info("  🔎 Analysis complete: %d claim(s) | status=%s",
             len(analysis.claims), analysis.status)
    return {
        "research_question": research_question,
        "evidence": items,
        "analysis": analysis,
        "status": analysis.status,
    }


# ---------------------------------------------------------------------------
# Phase-3 ResearchReport Synthesis Entry Point (Research Agent)
# ---------------------------------------------------------------------------
# ResearchQuestion + ResearchSource[] + Evidence[] + CriticalAnalysis
# → report_stage() → ResearchReport. Assembles the structured report; the
# product lives in a research document. LinkedIn drafting is a later phase.

def run_report(
    research_question=None,
    research_sources=None,
    evidence=None,
    analysis=None,
    research_question_id: Optional[str] = None,
    use_llm: bool = True,
) -> dict:
    """
    Run the Phase-3 report synthesis stage for a research session.

    Parameters
    ----------
    research_question : ResearchQuestion, optional
    research_sources  : list[ResearchSource], optional  – collected sources
    evidence          : list[Evidence], optional        – extracted claims
    analysis          : CriticalAnalysis, optional      – critical analysis
    research_question_id : str, optional
        Traceability parity with the earlier stages; reserved for persistence.
    use_llm : bool
        Passed through to report_stage; False → deterministic report.

    Returns
    -------
    dict with keys:
        research_question : the ResearchQuestion the report relates to
        sources           : list[ResearchSource] in the report
        evidence          : list[Evidence] in the report
        analysis          : CriticalAnalysis carried into the report (or None)
        report            : ResearchReport (structured, LinkedIn-agnostic)
        status            : "ok" when the report has content, else "empty"
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    sources = list(research_sources or [])
    items = list(evidence or [])
    log.info("🧠 Research Agent – Report Synthesis | question='%s' | sources=%d evidence=%d",
             getattr(research_question, "question", research_question), len(sources), len(items))

    report = report_stage(
        research_question,
        sources,
        items,
        analysis=analysis,
        use_llm=use_llm,
    )
    status = "ok" if (report.findings or report.evidence or report.sources) else "empty"
    log.info("  📄 Report assembled: %d finding(s), %d conclusion(s) | status=%s",
             len(report.findings), len(report.conclusions), status)
    return {
        "research_question": research_question,
        "sources": sources,
        "evidence": items,
        "analysis": analysis,
        "report": report,
        "status": status,
    }


if __name__ == "__main__":
    print("Starting PersonaPulse Pipeline...")
    args = sys.argv[1:]

    if args and args[0] == "--discover":
        discovery = run_discovery(query=args[1] if len(args) > 1 else "")
        sys.exit(0)

    if args and args[0] == "--select":
        result = run_selection(query=args[1] if len(args) > 1 else "")
        sys.exit(0)

    if args and args[0] == "--research":
        rq = ResearchQuestion(
            topic=args[1] if len(args) > 1 else "",
            question=args[1] if len(args) > 1 else "",
            aspects=[],
        )
        question_id = persist_research_question(rq)
        result = run_research(rq, research_question_id=question_id)
        sys.exit(0)

    if args and args[0] == "--evidence":
        rq = ResearchQuestion(
            topic=args[1] if len(args) > 1 else "",
            question=args[1] if len(args) > 1 else "",
            aspects=[],
        )
        question_id = persist_research_question(rq)
        research = run_research(rq, research_question_id=question_id)
        result = run_evidence(
            research["research_question"],
            research["research_sources"],
            research_question_id=question_id,
        )
        sys.exit(0)

    if args and args[0] == "--analyze":
        rq = ResearchQuestion(
            topic=args[1] if len(args) > 1 else "",
            question=args[1] if len(args) > 1 else "",
            aspects=[],
        )
        question_id = persist_research_question(rq)
        research = run_research(rq, research_question_id=question_id)
        evidence_result = run_evidence(
            research["research_question"],
            research["research_sources"],
            research_question_id=question_id,
        )
        result = run_critical_analysis(
            evidence_result["research_question"],
            evidence_result["evidence"],
            research_question_id=question_id,
        )
        sys.exit(0)

    if args and args[0] == "--report":
        rq = ResearchQuestion(
            topic=args[1] if len(args) > 1 else "",
            question=args[1] if len(args) > 1 else "",
            aspects=[],
        )
        question_id = persist_research_question(rq)
        research = run_research(rq, research_question_id=question_id)
        evidence_result = run_evidence(
            research["research_question"],
            research["research_sources"],
            research_question_id=question_id,
        )
        analysis_result = run_critical_analysis(
            evidence_result["research_question"],
            evidence_result["evidence"],
            research_question_id=question_id,
        )
        result = run_report(
            research["research_question"],
            research["research_sources"],
            evidence_result["evidence"],
            analysis_result["analysis"],
            research_question_id=question_id,
        )
        sys.exit(0)

    query_arg = args[0] if args else ""
    final_state = run_pipeline(query=query_arg)
    sys.exit(0 if not final_state.get("pipeline_halted") else 1)
