"""Crossref API service for DOI lookup and metadata enrichment."""

import re
from datetime import datetime
from typing import Any, Optional

import requests

from paperbot.utils.text import normalize_doi

CROSSREF_API_BASE = "https://api.crossref.org/works"


class CrossrefService:
    """Service for interacting with the Crossref API."""

    def __init__(self, contact_email: Optional[str] = None):
        """Initialize Crossref service.

        Args:
            contact_email: Email for polite pool access (recommended by Crossref)
        """
        self.contact_email = contact_email
        self._headers = self._build_headers()

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers with user agent."""
        if self.contact_email:
            return {"User-Agent": f"paperbot/1.0 (mailto:{self.contact_email})"}
        return {}

    def lookup(self, doi: str, timeout: int = 20) -> dict[str, Any]:
        """Look up paper metadata by DOI.

        Args:
            doi: DOI to look up
            timeout: Request timeout in seconds

        Returns:
            Crossref work metadata dictionary

        Raises:
            requests.RequestException: On API errors
        """
        url = f"{CROSSREF_API_BASE}/{requests.utils.quote(doi)}"
        response = requests.get(url, headers=self._headers, timeout=timeout)
        response.raise_for_status()
        return response.json().get("message", {})

    def search(
        self,
        title: str,
        container_title: Optional[str] = None,
        year: Optional[str] = None,
        issn: Optional[str] = None,
        timeout: int = 20,
    ) -> Optional[str]:
        """Search for a DOI by title and optional filters.

        Used when RSS entries don't contain a DOI directly.

        Args:
            title: Paper title to search for
            container_title: Journal name (optional)
            year: Publication year (optional)
            issn: Journal ISSN for precise filtering (optional)
            timeout: Request timeout in seconds

        Returns:
            Normalized DOI if found, None otherwise
        """
        params: dict[str, Any] = {
            "query.title": title,
            "rows": 1,
            "sort": "relevance",
        }

        if container_title:
            params["query.container-title"] = container_title

        # Build filters
        filters = []
        if issn:
            issn_clean = issn.replace(" ", "").strip()
            if issn_clean:
                filters.append(f"issn:{issn_clean}")

        if year:
            try:
                y = int(year)
                filters.append(f"from-pub-date:{y}-01-01")
                filters.append(f"until-pub-date:{y}-12-31")
            except ValueError:
                pass

        if filters:
            params["filter"] = ",".join(filters)

        if self.contact_email:
            params["mailto"] = self.contact_email

        response = requests.get(
            CROSSREF_API_BASE,
            params=params,
            headers=self._headers,
            timeout=timeout,
        )
        response.raise_for_status()

        data = response.json()
        items = data.get("message", {}).get("items")

        if not items:
            return None

        doi = items[0].get("DOI")
        return normalize_doi(doi) if doi else None

    @staticmethod
    def extract_metadata(
        meta: dict[str, Any],
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Extract best metadata from Crossref response.

        Args:
            meta: Crossref work metadata dictionary

        Returns:
            Tuple of (authors, journal, published_date, abstract)
        """
        # Authors
        authors = None
        author_list = meta.get("author")
        if isinstance(author_list, list):
            names = []
            for author in author_list[:20]:  # Limit to avoid huge lists
                given = author.get("given", "")
                family = author.get("family", "")
                full = " ".join([given, family]).strip()
                if full:
                    names.append(full)
            if names:
                authors = ", ".join(names)

        # Journal/container
        journal = None
        container = meta.get("container-title")
        if isinstance(container, list) and container:
            journal = container[0]
        elif isinstance(container, str):
            journal = container

        # Published date
        published = None
        for key in ["published-print", "published-online", "created", "issued"]:
            val = meta.get(key)
            if isinstance(val, dict):
                parts = val.get("date-parts")
                if (
                    isinstance(parts, list)
                    and parts
                    and isinstance(parts[0], list)
                    and parts[0]
                ):
                    try:
                        year = parts[0][0]
                        month = parts[0][1] if len(parts[0]) > 1 else 1
                        day = parts[0][2] if len(parts[0]) > 2 else 1
                        published = datetime(year, month, day).date().isoformat()
                        break
                    except Exception:
                        pass

        # Abstract (may contain JATS XML)
        abstract = None
        # if isinstance(meta.get("abstract"), str):
        #   Strip JATS/XML tags
        #   abstract = re.sub(r"<[^>]+>", "", meta["abstract"]).strip()

        return authors, journal, published, abstract
