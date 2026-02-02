"""RSS feed parsing service."""

import time
from pathlib import Path
from typing import Any, Iterator, Optional

import feedparser

from paperbot.config import load_feeds
from paperbot.models.paper import Paper
from paperbot.services.crossref_service import CrossrefService
from paperbot.utils.text import clean_title, extract_doi, parse_published


class FeedService:
    """Service for fetching and parsing RSS feeds."""

    def __init__(
        self,
        feeds_path: Path,
        crossref: CrossrefService,
        polite_delay: float = 0.2,
    ):
        """Initialize feed service.

        Args:
            feeds_path: Path to feeds.yaml configuration
            crossref: CrossrefService instance for metadata enrichment
            polite_delay: Delay between API calls (seconds)
        """
        self.feeds_path = feeds_path
        self.crossref = crossref
        self.polite_delay = polite_delay

    def fetch_all(self, max_entries_per_feed: int = 200) -> Iterator[Paper]:
        """Fetch papers from all configured feeds.

        Args:
            max_entries_per_feed: Maximum entries to process per feed

        Yields:
            Paper objects with enriched metadata
        """
        feeds = load_feeds(self.feeds_path)

        for feed_config in feeds:
            name = feed_config["name"]
            url = feed_config["url"]
            issn_hint = feed_config.get("issn")

            yield from self._process_feed(name, url, issn_hint, max_entries_per_feed)

    def _process_feed(
        self,
        name: str,
        url: str,
        issn_hint: Optional[str],
        max_entries: int,
    ) -> Iterator[Paper]:
        """Process a single RSS feed.

        Args:
            name: Feed name for logging/source
            url: Feed URL
            issn_hint: Optional ISSN for Crossref filtering
            max_entries: Maximum entries to process

        Yields:
            Paper objects
        """
        parsed = feedparser.parse(url)

        for entry in parsed.entries[:max_entries]:
            paper = self._parse_entry(entry, name, issn_hint)
            if paper:
                yield paper

    def _parse_entry(
        self,
        entry: dict[str, Any],
        source: str,
        issn_hint: Optional[str],
    ) -> Optional[Paper]:
        """Parse a single RSS entry into a Paper.

        Args:
            entry: Parsed feed entry
            source: Source feed name
            issn_hint: Optional ISSN for Crossref filtering

        Returns:
            Paper object or None if parsing fails
        """
        title_raw = entry.get("title", "").strip() or "(no title)"
        title = clean_title(title_raw)
        link = entry.get("link", "").strip() or ""
        doi = extract_doi(entry)
        published = parse_published(entry)

        authors = None
        journal = None
        abstract = None
        raw: dict[str, Any] = {"feed": source, "entry": dict(entry)}

        # If no DOI in RSS, search Crossref by title
        if not doi:
            doi = self._search_doi(title, source, published, issn_hint, raw)

        # Enrich metadata via Crossref lookup
        if doi:
            authors, journal, published_cr, abstract = self._enrich_metadata(doi, raw)
            published = published_cr or published

        return Paper(
            source=source,
            title=title,
            link=link,
            doi=doi,
            published=published,
            authors=authors,
            journal=journal,
            abstract=abstract,
            raw=raw,
        )

    def _search_doi(
        self,
        title: str,
        source: str,
        published: Optional[str],
        issn_hint: Optional[str],
        raw: dict[str, Any],
    ) -> Optional[str]:
        """Search for DOI via Crossref when not in RSS.

        Args:
            title: Paper title
            source: Source name (used as container hint)
            published: Publication date (for year filter)
            issn_hint: Optional ISSN
            raw: Raw data dict to update with search info

        Returns:
            Found DOI or None
        """
        container_hint = source.replace(" - Latest Articles", "").strip()
        year_hint = published[:4] if published else None

        try:
            found_doi = self.crossref.search(
                title=title,
                container_title=container_hint,
                year=year_hint,
                issn=issn_hint,
            )
            if found_doi:
                raw["doi_from_search"] = True
            time.sleep(self.polite_delay)
            return found_doi
        except Exception as e:
            raw["crossref_search_error"] = str(e)
            return None

    def _enrich_metadata(
        self,
        doi: str,
        raw: dict[str, Any],
    ) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
        """Enrich paper metadata via Crossref lookup.

        Args:
            doi: DOI to look up
            raw: Raw data dict to update with Crossref data

        Returns:
            Tuple of (authors, journal, published, abstract)
        """
        try:
            meta = self.crossref.lookup(doi)
            raw["crossref"] = meta
            time.sleep(self.polite_delay)
            return CrossrefService.extract_metadata(meta)
        except Exception as e:
            raw["crossref_error"] = str(e)
            return None, None, None, None

    def get_feed_info(self) -> list[dict[str, Any]]:
        """Get information about configured feeds.

        Returns:
            List of feed configuration dictionaries
        """
        return load_feeds(self.feeds_path)

    @staticmethod
    def check_feed(url: str) -> tuple[bool, Optional[str]]:
        """Check if a feed URL is valid and parseable.

        Args:
            url: Feed URL to check

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            parsed = feedparser.parse(url)
            if parsed.bozo:
                return False, str(parsed.bozo_exception)
            if not parsed.entries:
                return False, "No entries found"
            return True, None
        except Exception as e:
            return False, str(e)
