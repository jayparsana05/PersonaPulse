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
    python -m src.agent --post "question"        # Phase-4 LinkedIn draft (research → ... → report → drafts → Telegram approval), no publishing
    python -m src.agent --run "query"            # End-to-end research agent
                                                 # (discovery → selection → research → evidence → analysis → report
                                                 #  → drafts → Telegram approval), stops at PENDING; publication
                                                 # happens later via the Telegram edge function after human approval.
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
    check_topic_researched,
    delete_research_question,
    filter_repeated_sources,
    get_known_source_urls,
    get_normalized_embedding,
    get_research_sources_for_question,
    get_style_profile,
    link_research_sources,
    store_draft,
    store_research_question,
    store_research_sources,
)
from src.models import ResearchQuestion, ResearchReport
from src.post import draft_report_post
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

    # ResearchReport (Phase-4: LinkedIn drafting consumes the report)
    report: dict                        # ResearchReport dict (topic, findings, ...)

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
    Draft LinkedIn + X posts.

    Primary path (Phase 4): a ResearchReport in the state drives the drafts via
    the report-grounded drafting stage – the single article is no longer
    required. Legacy path: falls back to the article-based drafter when no
    report is present (old canary → search → dedup flow).
    """
    log.info("━━━ [Node 4/5] Drafting ━━━")

    style_profile = get_style_profile()

    report = state.get("report")
    if report is not None:
        report_obj = report if isinstance(report, ResearchReport) else ResearchReport.from_dict(report)
        result = draft_report_post(report_obj, style_profile=style_profile)
        linkedin_draft = result["linkedin_draft"]
        x_draft        = result["x_draft"]
        log.info("[Draft] From ResearchReport | LinkedIn: %d chars | X: %d chars | status=%s",
                 len(linkedin_draft), len(x_draft), result["status"])
    else:
        article       = state["article"]
        linkedin_draft = draft_post(article, style_profile, platform="linkedin")
        x_draft        = draft_post(article, style_profile, platform="x")
        log.info("[Draft] From article | LinkedIn: %d chars | X: %d chars", len(linkedin_draft), len(x_draft))

    return {
        **state,
        "style_profile":  style_profile,
        "linkedin_draft": linkedin_draft,
        "x_draft":        x_draft,
    }


# ---------------------------------------------------------------------------
# Node 5: Store & Alert
# ---------------------------------------------------------------------------

def _article_view_for_report(report_obj) -> dict:
    """A minimal article-shaped dict for the Telegram alert caption + storage."""
    source = report_obj.sources[0] if report_obj.sources else None
    return {
        "source": (source.source if source else "") or "",
        "title":  report_obj.topic or "Research brief",
        "url":    (source.url if source else "") or "",
    }


def _research_summary_for_report(report_obj) -> dict:
    """A concise research digest for the Telegram approval message.

    Deliberately bounded: only the research question, up to three finding
    claims, a single counterargument (or limitation) and short source URLs.
    Raw source bodies are never exposed.
    """
    findings = [
        finding.claim
        for finding in report_obj.findings
        if (finding.claim or "").strip()
    ][:3]

    counterargument = None
    for counter in report_obj.counterarguments:
        if (counter.argument or "").strip():
            counterargument = counter.argument.strip()
            break

    limitation = None
    for lim in report_obj.limitations or []:
        if (lim or "").strip():
            limitation = lim.strip()
            break

    sources = [source for source in report_obj.sources if (source.url or "").strip()]
    return {
        "research_question": (report_obj.research_question or "").strip(),
        "key_findings": findings,
        "counterargument": counterargument,
        "limitation": limitation,
        "source_count": len(sources),
        "source_urls": [source.url.strip() for source in sources],
    }


def node_store_and_alert(state: AgentState) -> AgentState:
    """
    1. Extract og:image from the source (article URL, or the report's first
       source when drafting from a ResearchReport).
    2. Store the draft to Supabase (status=PENDING).
    3. Send Telegram photo/text message with approval inline keyboard.

    The report-driven path replaces the single-article dependency: when the
    drafts came from a ResearchReport, the report's topic + first source stand
    in for the article metadata. Publishing is untouched: it happens only via
    the Telegram approval flow.
    """
    log.info("━━━ [Node 5/5] Store & Alert ━━━")

    article_is_report = state.get("report") is not None
    if article_is_report:
        report_obj = state["report"]
        report_obj = report_obj if isinstance(report_obj, ResearchReport) else ResearchReport.from_dict(report_obj)
        article_ref = _article_view_for_report(report_obj)
    else:
        article_ref = state["article"]

    embedding      = state["embedding"]
    linkedin_draft = state["linkedin_draft"]
    x_draft        = state["x_draft"]

    # ── Extract og:image (best-effort) ─────────────────────────────────
    image_url, image_bytes = extract_og_image_with_url(article_ref.get("url", ""))

    # ── Store in Supabase ───────────────────────────────────────────────
    if embedding is None:
        embedding = get_normalized_embedding(
            f"{article_ref.get('title', '')} {linkedin_draft}"
        )

    combined_content = f"LINKEDIN:\n{linkedin_draft}\n\nX:\n{x_draft}"
    post_id = store_draft(
        platform    = "both",
        topic       = article_ref.get("title", "Research brief")[:256],
        content     = combined_content,
        embedding   = embedding,
        article_url = article_ref.get("url") or None,
        image_url   = image_url,
    )

    log.info("[Store] Draft saved – id=%s", post_id)

    # ── Send Telegram Approval Alert ────────────────────────────────────
    send_telegram_alert(
        post_id        = post_id,
        linkedin_draft = linkedin_draft,
        x_draft        = x_draft,
        article        = article_ref,
        research       = _research_summary_for_report(report_obj) if article_is_report else None,
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
    check_researched: bool = False,
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
    check_researched : bool
        When True, the framed question is checked against remembered research
        sessions BEFORE a new row is stored: an exact/semantic duplicate
        reuses the existing session id instead of duplicating it, and the
        result gains an ``already_researched`` bool. Follow-up questions
        (same topic, different angle) are never blocked. Default False keeps
        the historical behavior. Persistence stays non-blocking either way.

    Returns
    -------
    dict with keys:
        topic_candidates   : list[TopicCandidate] (discovered, may be empty)
        selection          : TopicSelection (selected + reasoning + criteria)
        research_question  : ResearchQuestion or None (when nothing was selected)
        research_question_id : UUID of the persisted research_questions row,
                              or None when nothing was selected (or persistence failed)
        already_researched : bool (only present when check_researched=True)
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
    already_researched = None
    if question is not None:
        log.info("  ❓ Research question: %s", question.question)
        if question.aspects:
            log.info("  Aspects: %s", ", ".join(question.aspects))
        if check_researched:
            dup = check_topic_researched(question.topic, question.question)
            if dup["matched"]:
                already_researched = dup
                question_id = dup["question_id"]
                log.info(
                    "  ♻️  Already researched (reason=%s, session=%s) – reusing session, not storing a duplicate.",
                    dup["reason"], question_id,
                )
        if question_id is None:
            question_id = persist_research_question(question)

    result = {
        "topic_candidates": list(candidates),
        "selection": selection,
        "research_question": question,
        "research_question_id": question_id,
    }
    if check_researched:
        result["already_researched"] = bool(already_researched is not None and already_researched.get("matched"))
    return result


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
    skip_known_sources: bool = False,
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
    skip_known_sources : bool
        When True, sources whose normalized URL was already recorded in an
        earlier research session are kept in the in-memory result but skipped
        on persistence (the existing per-run _dedupe_sources is unchanged).
        If that filter would leave NOTHING to persist while the run still
        discovered usable sources (all already known), NO duplicate full
        records are re-inserted: the session stays retained and only
        lightweight per-session *link rows* are recorded (see
        ``memory.link_research_sources``), so hydration re-loads the sources
        from the existing records and the session stays reusable. Non-blocking:
        memory failures degrade to persisting everything / leaving the session
        unlinked.

    Returns
    -------
    dict with keys:
        research_question    : the ResearchQuestion researched
        research_sources     : list[ResearchSource] (normalized + deduplicated)
        status               : "ok" when ≥1 source, else "empty"
        research_source_ids  : list[str] of persisted row UUIDs (full rows or
                               link rows; may be empty when nothing was
                               persisted or persistence failed)
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
        to_store = list(sources)
        if skip_known_sources:
            try:
                known_urls = get_known_source_urls()          # non-blocking
            except Exception as exc:  # pylint: disable=broad-except
                log.warning(
                    "⚠️  Could not check known sources (%s: %s) – persisting everything.",
                    type(exc).__name__, exc,
                )
                known_urls = set()
            to_store, repeated = filter_repeated_sources(sources, known_urls)
            if repeated:
                log.info(
                    "📚 Skipped persisting %d source(s) already stored in earlier sessions.",
                    len(repeated),
                )
        # Cross-session dedup may have filtered EVERY discovered source out
        # (all already known). That is not a failure: the run still discovered
        # valid sources and the fresh session must stay retained so a later
        # identical question reuses it instead of triggering Tavily again. We
        # must NOT re-insert duplicates of those already-known sources; their
        # per-session *link rows* are enough to keep the session hydratable.
        if research_question_id:
            if to_store:
                try:
                    source_ids = store_research_sources(research_question_id, to_store)
                except Exception as exc:  # pylint: disable=broad-except
                    log.warning(
                        "⚠️  Could not persist research sources (%s: %s) – continuing with in-memory result.",
                        type(exc).__name__, exc,
                    )
            elif sources and skip_known_sources:
                try:
                    source_ids = link_research_sources(research_question_id, sources)
                except Exception as exc:  # pylint: disable=broad-except
                    log.warning(
                        "⚠️  Could not link known research sources to session %s (%s: %s) – staying unlinked.",
                        research_question_id, type(exc).__name__, exc,
                    )
                else:
                    log.info(
                        "♻️  All %d discovered source(s) already stored in earlier sessions – "
                        "recorded %d link row(s) for session %s (no source content duplicated).",
                        len(sources), len(source_ids), research_question_id,
                    )
            else:
                log.warning(
                    "🧭 Nothing to persist for session %s (%d source(s) kept in memory).",
                    research_question_id, len(sources),
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
# Research-session memory entry point
# ---------------------------------------------------------------------------
# Remembered topics/questions gate redundant re-research: an exact or semantic
# duplicate reuses the stored session; a related but meaningfully different
# question always runs (follow-up research is never blocked here).

def run_research_or_reuse(
    rq,
    research_question_id: Optional[str] = None,
    use_llm: bool = True,
    reuse_researched: bool = True,
    skip_known_sources: bool = True,
    threshold: Optional[float] = None,
    max_queries: Optional[int] = None,
    max_sources_per_query: Optional[int] = None,
    max_sources: Optional[int] = None,
    min_score: Optional[float] = None,
) -> dict:
    """
    Research *rq* unless the topic/question is already remembered.

    Adds a memory gate in front of :func:`run_research`:

    - Exact or semantic duplicate (see ``check_topic_researched``): the
      stored research session is reused — its sources are hydrated from the
      database and NO new search/persistence happens, so the same topic is
      never re-researched (or duplicated on disk). Reuse only fires when the
      remembered session actually has usable persisted sources; a session
      whose sources cannot be hydrated (DB failure) or that recorded no
      sources at all falls back to fresh research instead of returning an
      empty "reused" result.
    - Related but meaningfully different question (same topic, new angle):
      NOT a duplicate — research runs normally. Follow-up research is allowed.
    - Memory unavailable / embedding failure: research runs normally
      (non-blocking, matching the rest of the research agent).

    When *research_question_id* is None and a brand-new question proceeds to
    research, the new question row is persisted first so the collected sources
    stay traceable (same mechanism the CLI entry points already used). The
    freshly created session is discarded ONLY when the run produced no usable
    research sources at all (failure / empty results); a run that discovered
    valid sources is retained even when persistence recorded no NEW full rows
    (e.g. every source was already stored in an earlier session — in which
    case lightweight per-session link rows are recorded instead, and
    hydration re-loads the source content from the existing records), so a
    successful session is always reusable later.

    Returns
    -------
    dict – a :func:`run_research` result, plus:
        reused_question_id   : UUID of the reused session (duplicate path only)
        duplicate_reason     : "exact" | "semantic" | None
        research_question_id : the session UUID used (reused or newly stored)
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    if reuse_researched:
        try:
            dup = check_topic_researched(
                getattr(rq, "topic", rq),
                getattr(rq, "question", rq),
                threshold=threshold,
            )
        except Exception as exc:  # pylint: disable=broad-except
            log.warning(
                "⚠️  Research-memory check failed (%s: %s) – proceeding with fresh research.",
                type(exc).__name__, exc,
            )
            dup = {"matched": False}
        if dup["matched"]:
            known_id = dup["question_id"] or research_question_id
            try:
                reused_sources, reused_ids = get_research_sources_for_question(known_id)
            except Exception as exc:  # pylint: disable=broad-except
                log.warning(
                    "⚠️  Could not hydrate remembered session %s (%s: %s) – falling back to fresh research.",
                    known_id, type(exc).__name__, exc,
                )
                reused_sources, reused_ids = [], []
            if reused_sources:
                log.info(
                    "♻️  Already researched (reason=%s, session=%s) – reusing '%s' instead of re-searching.",
                    dup["reason"], known_id, getattr(rq, "question", rq),
                )
                return {
                    "research_question": rq,
                    "research_sources": reused_sources,
                    "status": "reused",
                    "research_source_ids": reused_ids,
                    "reused_question_id": known_id,
                    "duplicate_reason": dup["reason"],
                    "research_question_id": known_id,
                }
            log.warning(
                "🧭 Remembered session %s has no usable stored sources – falling back to fresh research.",
                known_id,
            )

    created_id: Optional[str] = None
    if research_question_id is None:
        research_question_id = persist_research_question(rq)
        created_id = research_question_id

    result = run_research(
        rq,
        research_question_id=research_question_id,
        use_llm=use_llm,
        max_queries=max_queries,
        max_sources_per_query=max_sources_per_query,
        max_sources=max_sources,
        min_score=min_score,
        skip_known_sources=skip_known_sources,
    )
    result["reused_question_id"] = None
    result["duplicate_reason"] = None
    result["research_question_id"] = research_question_id

    # A brand-new session that yielded NO usable research sources must not
    # linger as a remembered (but empty) session: the next run for the same
    # question would otherwise hint a reuse and then fall back to research
    # anyway. Successful research is always kept – even when "research_source_ids"
    # is empty because every discovered source was already stored in an earlier
    # session (cross-session dedup) or the source rows could not be persisted.
    usable = bool(result.get("research_sources"))
    if created_id is not None and not usable:
        log.warning(
            "🧽 Research produced nothing persistable for new session %s – discarding it (not remembered).",
            created_id,
        )
        try:
            delete_research_question(created_id)
        except Exception as exc:  # pylint: disable=broad-except
            log.warning(
                "⚠️  Could not delete incomplete research session %s (%s: %s).",
                created_id, type(exc).__name__, exc,
            )
        else:
            result["research_question_id"] = None
    return result


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


# ---------------------------------------------------------------------------
# Phase-4 LinkedIn Draft / Approval Entry Point (Research Agent)
# ---------------------------------------------------------------------------
# ResearchReport → drafts → store PENDING → Telegram approval alert → stop.
# The drafts are generated from the ResearchReport (not a single article).
# Publishing is NOT performed here: it happens only via the existing Telegram
# approval edge function after the author approves the PENDING post.

def run_post(
    report=None,
    style_profile=None,
    research_question_id: Optional[str] = None,
    use_llm: bool = True,
) -> dict:
    """
    Run the Phase-4 LinkedIn Draft stage for an existing ResearchReport.

    Parameters
    ----------
    report          : ResearchReport (or its dict serialization) to draft from.
    style_profile   : dict from get_style_profile(); fetched when not given.
    research_question_id : str, optional
        Traceability parity with the earlier stages; reserved for persistence.
    use_llm         : bool – passed through to the drafting stage; False uses
                      the deterministic fallback (no API call).

    Returns
    -------
    dict with keys:
        report         : the ResearchReport that was drafted from
        linkedin_draft : str
        x_draft        : str
        used_findings  : list[str] finding claims restated by the post
        grounded       : bool
        issues         : list[str] grounding problems ("" when clean)
        status         : "ok" | "fallback" | "empty" | "approval_alert_failed"
        post_id        : UUID of the PENDING post awaiting Telegram approval
        approval_error : str – set only when status=="approval_alert_failed":
                         the draft was stored (post_id remains usable) but the
                         Telegram approval alert could not be delivered. The
                         PENDING draft is deliberately kept for recovery.
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )

    report_obj = report
    if not isinstance(report_obj, ResearchReport):
        report_obj = ResearchReport.from_dict(report) if isinstance(report, dict) else ResearchReport(topic="")

    log.info(
        "📝 Research Agent – LinkedIn Draft | topic='%s' | findings=%d",
        report_obj.topic, len(report_obj.findings),
    )

    profile = style_profile if style_profile is not None else get_style_profile()
    draft_result = draft_report_post(report_obj, style_profile=profile, use_llm=use_llm)

    article_ref = _article_view_for_report(report_obj)
    combined_content = (
        f"LINKEDIN:\n{draft_result['linkedin_draft']}\n\nX:\n{draft_result['x_draft']}"
    )

    # ── og:image (best effort, existing mechanism) ─────────────────────
    image_url, image_bytes = extract_og_image_with_url(article_ref.get("url", ""))

    # ── Dedup embedding of the drafted content (existing mechanism) ────
    embedding = get_normalized_embedding(f"{article_ref['title']} {report_obj.synthesis}")

    # ── Store PENDING + send approval alert (existing mechanism) ───────
    post_id = store_draft(
        platform    = "both",
        topic       = article_ref["title"][:256],
        content     = combined_content,
        embedding   = embedding,
        article_url = article_ref.get("url") or None,
        image_url   = image_url,
    )
    log.info("[Post] Draft stored – id=%s (status=PENDING)", post_id)

    approval_error: Optional[str] = None
    status = draft_result.get("status", "ok")
    try:
        send_telegram_alert(
            post_id        = post_id,
            linkedin_draft = draft_result["linkedin_draft"],
            x_draft        = draft_result["x_draft"],
            article        = article_ref,
            research       = _research_summary_for_report(report_obj),
            image_bytes    = image_bytes,
        )
        log.info("[Post] Telegram approval request sent – draft id=%s awaiting approval.", post_id)
    except Exception as exc:  # pylint: disable=broad-except
        approval_error = f"{type(exc).__name__}: {exc}"
        status = "approval_alert_failed"
        log.warning(
            "[Post] Telegram approval alert failed (%s) – draft %s kept as PENDING for recovery.",
            approval_error, post_id,
        )

    return {
        **draft_result,
        "status": status,
        "report": report_obj,
        "post_id": post_id,
        "approval_error": approval_error,
    }


def run_research_agent(
    query: str = "",
    candidates: Optional[list] = None,
    limit: Optional[int] = None,
    use_llm: bool = True,
    check_researched: bool = True,
    reuse_researched: bool = True,
    skip_known_sources: bool = True,
    max_queries: Optional[int] = None,
    max_sources_per_query: Optional[int] = None,
    max_sources: Optional[int] = None,
    min_score: Optional[float] = None,
    max_claims_per_source: Optional[int] = None,
) -> dict:
    """Run the full end-to-end AI Engineering Research Agent.

    This is a thin composition of the already-tested stage entry points
    (run_discovery / run_selection / run_research_or_reuse / run_evidence /
    run_critical_analysis / run_report / run_post) wrapped with the 13-step
    workflow gating:

    1. discover multiple engineering topics            → run_discovery (or injected candidates)
    2. create TopicCandidates                          → run_discovery
    3. select a topic                                  → run_selection
    4. create a ResearchQuestion                       → run_selection (persisted for dedup)
    5. discover multiple ResearchSources               → run_research_or_reuse (memory-aware)
    6. extract Evidence / Claims                       → run_evidence
    7. CriticalAnalysis                                → run_critical_analysis
    8. ResearchReport                                  → run_report
    9. LinkedIn draft (+ X)                            → run_post
    10. send research summary + draft to Telegram      → run_post (send_telegram_alert)
    11. wait for existing approval                      → handled by the Telegram edge function (external)
    12. publish to LinkedIn AFTER approval              → handled by the Telegram edge function (external)
    13. persist research history for future dedup       → selection persists the question; research persists sources

    The agent never publishes directly: it always stops after storing a
    PENDING draft and alerting Telegram (step 10). Publication (step 12) is
    the responsibility of the existing supabase/functions/telegram-webhook
    edge function after a human approves.

    Gating / failure isolation
    --------------------------
    * Each stage yields a structured result; if a downstream stage receives
      nothing usable, execution halts with a ``halt_reason`` and a final
      ``status`` naming the empty stage (never a crash).
    * Failures inside one research source are absorbed by the stage layers
      (per-source/per-query try/except) and never terminate the session.
    * If research produces nothing persistable **and** this run created the
      session row, the empty row is best-effort deleted so future runs do not
      treat a dead question as researched.

    Parameters
    ----------
    query          : str – discovery search query (ignored when ``candidates`` given).
    candidates     : optional pre-built TopicCandidate list to skip discovery.
    limit          : discovery candidate limit override.
    use_llm        : pass-through to the research/evidence/analysis/report/post stages.
    check_researched/reuse_researched/skip_known_sources : memory-axis flags.
    max_queries / max_sources_per_query / max_sources / min_score : research stage controls.
    max_claims_per_source : evidence stage control.

    Returns
    -------
    dict with keys:
        status             : "posted" | "approval_alert_failed"
                             | "discovery_empty" | "selection_empty"
                             | "research_empty" | "evidence_empty"
                             | "analysis_empty" | "report_empty" | "post_empty"
        query, stages      : per-stage status map ("ok" | "empty" | "reused" | "fallback")
        topic_candidates   : list[TopicCandidate]
        selection          : TopicSelection | None
        research_question  : ResearchQuestion | None
        research_question_id / reused_question_id / duplicate_reason
        research_sources   : list[ResearchSource]
        research_source_ids: list[str]
        evidence           : list[Evidence]
        analysis           : CriticalAnalysis | None
        report             : ResearchReport | None
        post               : run_post() result dict | None
        post_id            : PENDING draft id, when "posted"
        halt_reason        : str when halted
    """
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s %(levelname)-8s │ %(message)s",
        datefmt= "%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    stages: dict[str, str] = {}
    result: dict = {
        "status": "started",
        "query": query,
        "stages": stages,
        "topic_candidates": [],
        "selection": None,
        "research_question": None,
        "research_question_id": None,
        "already_researched": False,
        "reused_question_id": None,
        "duplicate_reason": None,
        "research_sources": [],
        "research_source_ids": [],
        "evidence": [],
        "analysis": None,
        "report": None,
        "post": None,
        "post_id": None,
        "halt_reason": None,
    }

    def _halt(status: str, reason: str) -> dict:
        result["status"] = status
        result["halt_reason"] = reason
        return result

    # ── 1–2. Discovery → TopicCandidates ────────────────────────────────
    if candidates is not None:
        discovered = list(candidates)
    else:
        try:
            discovered = run_discovery(query=query, limit=limit)
        except Exception as exc:  # pylint: disable=broad-except
            log.warning("[Agent] Discovery failed (%s: %s) – halting.", type(exc).__name__, exc)
            discovered = []
    result["topic_candidates"] = discovered
    stages["discovery"] = "ok" if discovered else "empty"
    if not discovered:
        log.warning("[Agent] No topic candidates – aborting (status=discovery_empty).")
        return _halt("discovery_empty", "Discovery surfaced no topic candidates.")

    # ── 3–4. Selection → ResearchQuestion (memory-aware persistence) ─────
    selection_result = run_selection(
        query=query,
        limit=limit,
        candidates=discovered,
        check_researched=check_researched,
    )
    rq = selection_result["research_question"]
    result["selection"] = selection_result["selection"]
    result["research_question"] = rq
    result["research_question_id"] = selection_result.get("research_question_id")
    result["already_researched"] = bool(selection_result.get("already_researched"))
    stages["selection"] = "ok" if rq is not None else "empty"
    if rq is None:
        log.warning("[Agent] No research question framed – aborting (status=selection_empty).")
        return _halt("selection_empty", "No topic was selected / no research question was framed.")

    # ── 5. Multi-source Research (memory-aware, failure-tolerant) ────────
    try:
        research = run_research_or_reuse(
            rq,
            research_question_id=result["research_question_id"],
            use_llm=use_llm,
            reuse_researched=reuse_researched,
            skip_known_sources=skip_known_sources,
            max_queries=max_queries,
            max_sources_per_query=max_sources_per_query,
            max_sources=max_sources,
            min_score=min_score,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("[Agent] Research stage failed (%s: %s) – treating as empty.",
                    type(exc).__name__, exc)
        research = {
            "status": "empty",
            "research_sources": [],
            "research_source_ids": [],
            "research_question_id": result["research_question_id"],
            "reused_question_id": None,
            "duplicate_reason": None,
        }

    sources = research["research_sources"]
    result["research_sources"] = sources
    result["research_source_ids"] = research["research_source_ids"]
    result["research_question_id"] = research.get("research_question_id", result["research_question_id"])
    result["reused_question_id"] = research.get("reused_question_id")
    result["duplicate_reason"] = research.get("duplicate_reason")
    stages["research"] = research.get("status", "empty")

    if not sources:
        log.warning("[Agent] Research returned no usable sources (status=research_empty).")
        if not result["already_researched"] and result["research_question_id"]:
            try:
                delete_research_question(result["research_question_id"])
                log.info("[Agent] Cleaned up empty research session %s.",
                         result["research_question_id"])
            except Exception as exc:  # pylint: disable=broad-except
                log.warning("[Agent] Cleanup of empty research session failed (%s: %s).",
                            type(exc).__name__, exc)
            result["research_question_id"] = None
        return _halt("research_empty", "Research returned no usable sources.")

    # ── 6. Evidence / Claims ─────────────────────────────────────────────
    evidence_result = run_evidence(
        rq,
        sources,
        research_question_id=result["research_question_id"],
        use_llm=use_llm,
        max_claims_per_source=max_claims_per_source,
    )
    result["evidence"] = evidence_result["evidence"]
    stages["evidence"] = evidence_result["status"]
    if not evidence_result["evidence"]:
        log.warning("[Agent] No supported claims extracted (status=evidence_empty).")
        return _halt("evidence_empty", "No supported claims were extracted from the research sources.")

    # ── 7. Critical Analysis ─────────────────────────────────────────────
    analysis_result = run_critical_analysis(
        rq,
        result["evidence"],
        research_question_id=result["research_question_id"],
        use_llm=use_llm,
    )
    result["analysis"] = analysis_result["analysis"]
    stages["analysis"] = analysis_result["status"]
    if analysis_result["status"] != "ok":
        log.warning("[Agent] Critical analysis produced nothing usable (status=analysis_empty).")
        return _halt("analysis_empty", "Critical analysis produced no usable content.")

    # ── 8. Research Report ───────────────────────────────────────────────
    report_result = run_report(
        rq,
        sources,
        result["evidence"],
        result["analysis"],
        research_question_id=result["research_question_id"],
        use_llm=use_llm,
    )
    result["report"] = report_result["report"]
    stages["report"] = report_result["status"]
    if report_result["status"] != "ok":
        log.warning("[Agent] Report synthesis produced nothing usable (status=report_empty).")
        return _halt("report_empty", "Report synthesis produced no usable content.")

    # ── 9–10. LinkedIn/X drafts → PENDING storage → Telegram approval ────
    try:
        post = run_post(
            result["report"],
            research_question_id=result["research_question_id"],
            use_llm=use_llm,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("[Agent] Post/Draft stage failed (%s: %s) – halting.",
                    type(exc).__name__, exc)
        post = {
            "status": "empty",
            "post_id": None,
            "grounded": False,
            "issues": [f"{type(exc).__name__}: {exc}"],
        }
    result["post"] = post
    result["post_id"] = post.get("post_id")
    stages["post"] = post.get("status", "empty")
    if not post.get("post_id"):
        log.warning("[Agent] Draft could not be stored/altered (status=post_empty).")
        return _halt("post_empty", "The draft could not be stored or sent for approval.")

    if post.get("status") == "approval_alert_failed":
        result["status"] = "approval_alert_failed"
        result["halt_reason"] = (
            f"Draft {post['post_id']} stored as PENDING, but the Telegram approval "
            f"alert failed: {post.get('approval_error') or 'unknown error'}. "
            f"The PENDING draft is kept for recovery; nothing was published."
        )
        log.warning(
            "[Agent] Draft %s stored as PENDING, but the Telegram approval alert "
            "failed (status=approval_alert_failed). Nothing published.",
            post["post_id"],
        )
        return result

    log.info(
        "✅ Research Agent complete – draft %s stored, awaiting Telegram approval "
        "(status=posted). Investigated %d source(s), %d finding(s).",
        post["post_id"], len(sources), len(result["report"].findings) if result["report"] else 0,
    )
    result["status"] = "posted"
    return result


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
        result = run_research_or_reuse(rq)
        sys.exit(0)

    if args and args[0] == "--evidence":
        rq = ResearchQuestion(
            topic=args[1] if len(args) > 1 else "",
            question=args[1] if len(args) > 1 else "",
            aspects=[],
        )
        research = run_research_or_reuse(rq)
        question_id = research["research_question_id"]
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
        research = run_research_or_reuse(rq)
        question_id = research["research_question_id"]
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
        research = run_research_or_reuse(rq)
        question_id = research["research_question_id"]
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

    if args and args[0] == "--post":
        rq = ResearchQuestion(
            topic=args[1] if len(args) > 1 else "",
            question=args[1] if len(args) > 1 else "",
            aspects=[],
        )
        research = run_research_or_reuse(rq)
        question_id = research["research_question_id"]
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
        report_result = run_report(
            research["research_question"],
            research["research_sources"],
            evidence_result["evidence"],
            analysis_result["analysis"],
            research_question_id=question_id,
        )
        result = run_post(
            report_result["report"],
            research_question_id=question_id,
        )
        sys.exit(0)

    if args and args[0] == "--run":
        result = run_research_agent(
            query=args[1] if len(args) > 1 else "",
        )
        sys.exit(0 if result.get("status") == "posted" else 1)

    query_arg = args[0] if args else ""
    final_state = run_pipeline(query=query_arg)
    sys.exit(0 if not final_state.get("pipeline_halted") else 1)
