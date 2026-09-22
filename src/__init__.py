"""
PersonaPulse – Package Init
============================
Expose the primary pipeline entry point at the package level.

Usage:
    from personapulse.src import run_pipeline
    run_pipeline("latest AI news")
"""

from src.agent import (
    run_critical_analysis,
    run_evidence,
    run_pipeline,
    run_post,
    run_report,
    run_research,
    run_selection,
)
from src.models import (
    Claim,
    ClaimAnalysis,
    Counterargument,
    CriticalAnalysis,
    Evidence,
    ResearchQuestion,
    ResearchReport,
    ResearchSource,
    TopicCandidate,
    TopicSelection,
)
from src.post import draft_report_post
from src.report import synthesize_report

__all__ = [
    "run_pipeline",
    "run_selection",
    "run_research",
    "run_evidence",
    "run_critical_analysis",
    "run_report",
    "run_post",
    "synthesize_report",
    "draft_report_post",
    "TopicCandidate",
    "TopicSelection",
    "ResearchQuestion",
    "ResearchSource",
    "Evidence",
    "Claim",
    "ClaimAnalysis",
    "Counterargument",
    "CriticalAnalysis",
    "ResearchReport",
]
__version__ = "1.0.0"
