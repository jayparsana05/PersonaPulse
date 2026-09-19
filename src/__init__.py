"""
PersonaPulse – Package Init
============================
Expose the primary pipeline entry point at the package level.

Usage:
    from personapulse.src import run_pipeline
    run_pipeline("latest AI news")
"""

from src.agent import run_pipeline

__all__ = ["run_pipeline"]
__version__ = "1.0.0"
