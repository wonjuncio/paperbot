"""PaperBot - Journal RSS to Zotero pipeline.

A tool for collecting academic papers from RSS feeds,
enriching metadata via Crossref, and pushing to Zotero.
"""

__version__ = "1.0.0"
__author__ = "PaperBot Contributors"

from paperbot.config import Settings
from paperbot.models.paper import Paper

__all__ = ["Paper", "Settings", "__version__"]
