"""
PersonaPulse – Package Init
============================
Expose the primary pipeline entry point at the package level.

Usage:
    from personapulse.src import run_pipeline
    run_pipeline("latest AI news")
"""

from src.agent import run_pipeline
from src.models import (
    Claim,
    Evidence,
    ResearchQuestion,
    ResearchReport,
    ResearchSource,
    TopicCandidate,
)

__all__ = [
    "run_pipeline",
    "TopicCandidate",
    "ResearchQuestion",
    "ResearchSource",
    "Evidence",
    "Claim",
    "ResearchReport",
]
__version__ = "1.0.0"
