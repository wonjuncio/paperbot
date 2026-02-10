"""PaperBot - Journal RSS to md DB pipeline.

A tool for collecting academic papers from RSS feeds,
enriching metadata via Crossref, and pushing to md DB.
"""

__version__ = "3.0.0"
__author__ = "wonjunchoii"

from paperbot.config import Settings
from paperbot.models.paper import Paper

__all__ = ["Paper", "Settings", "__version__"]
