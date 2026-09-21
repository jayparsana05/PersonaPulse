"""
PersonaPulse – Package Init
============================
Expose the primary pipeline entry point at the package level.

Usage:
    from personapulse.src import run_pipeline
    run_pipeline("latest AI news")
"""

from src.agent import run_evidence, run_pipeline, run_research, run_selection
from src.models import (
    Claim,
    Evidence,
    ResearchQuestion,
    ResearchReport,
    ResearchSource,
    TopicCandidate,
    TopicSelection,
)

__all__ = [
    "run_pipeline",
    "run_selection",
    "run_research",
    "run_evidence",
    "TopicCandidate",
    "TopicSelection",
    "ResearchQuestion",
    "ResearchSource",
    "Evidence",
    "Claim",
    "ResearchReport",
]
__version__ = "1.0.0"
