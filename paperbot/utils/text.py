"""Text processing utilities for DOI extraction and title cleaning."""

import re
from typing import Any, Optional

# DOI regex pattern: 10.XXXX/... format
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)


def normalize_doi(doi: str) -> str:
    """Normalize DOI by removing URL prefixes and converting to lowercase."""
    doi = doi.strip()
    doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
    doi = doi.replace("https://dx.doi.org/", "").replace("http://dx.doi.org/", "")
    return doi.strip().lower()


def extract_doi(entry: dict[str, Any]) -> Optional[str]:
    """Extract DOI from an RSS feed entry.

    Searches multiple fields in the entry for DOI patterns.

    Args:
        entry: Parsed RSS feed entry dictionary

    Returns:
        Normalized DOI string if found, None otherwise
    """
    # Check dedicated DOI fields
    for key in ["doi", "prism_doi", "dc_identifier", "id", "guid"]:
        val = entry.get(key)
        if isinstance(val, str):
            match = DOI_RE.search(val)
            if match:
                return normalize_doi(match.group(0))

    # Check common text fields
    for field in ["link", "summary", "title"]:
        val = entry.get(field)
        if isinstance(val, str):
            match = DOI_RE.search(val)
            if match:
                return normalize_doi(match.group(0))

    # Check content array (Atom feeds)
    content = entry.get("content")
    if isinstance(content, list):
        for item in content:
            val = item.get("value")
            if isinstance(val, str):
                match = DOI_RE.search(val)
                if match:
                    return normalize_doi(match.group(0))

    return None


def clean_title(text: str) -> str:
    """Clean title by removing MathML/HTML tags and normalizing whitespace.

    Handles messy titles from RSS feeds (e.g., ScienceDirect) that may contain
    embedded MathML, HTML tags, and HTML entities.

    Args:
        text: Raw title string

    Returns:
        Cleaned title string, or "(no title)" if empty
    """
    if not text or not isinstance(text, str):
        return text or "(no title)"

    # Remove <math ...>...</math> blocks (including attributes, multiline)
    text = re.sub(r"<math[\s>].*?</math>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    # Remove self-closing <math ... />
    text = re.sub(r"<math[\s\S]*?/>", " ", text, flags=re.IGNORECASE)
    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")

    # Normalize whitespace
    text = " ".join(text.split()).strip()

    return text or "(no title)"


def parse_published(entry: dict[str, Any]) -> Optional[str]:
    """Parse publication date from RSS entry.

    Args:
        entry: Parsed RSS feed entry dictionary

    Returns:
        ISO format date string (YYYY-MM-DD) if found, None otherwise
    """
    from dateutil import parser as dtparser

    for field in ["published", "updated"]:
        val = entry.get(field)
        if val:
            try:
                dt = dtparser.parse(val)
                return dt.date().isoformat()
            except Exception:
                pass

    return None
