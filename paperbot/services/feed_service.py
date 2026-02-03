"""RSS feed parsing service."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import feedparser

from paperbot.config import load_feeds
from paperbot.models.paper import Paper
from paperbot.services.crossref_service import CrossrefService
from paperbot.utils.text import clean_title, extract_doi, parse_published


@dataclass
class RawEntry:
    """RSS-only entry before Crossref enrichment."""

    source: str
    title: str
    link: str
    doi: Optional[str]
    published: Optional[str]
    issn_hint: Optional[str]
    entry: dict[str, Any]


class FeedService:
    """Service for fetching and parsing RSS feeds."""

    def __init__(
        self,
        feeds_path: Path,
        crossref: CrossrefService,
        polite_delay: float = 0.1,
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

    def collect_raw_entries(
        self, max_entries_per_feed: int = 200
    ) -> list[RawEntry]:
        """Parse all RSS feeds and return raw entries (no Crossref). Fast."""
        feeds = load_feeds(self.feeds_path)
        raw_entries: list[RawEntry] = []

        for feed_config in feeds:
            name = feed_config["name"]
            url = feed_config["url"]
            issn_hint = feed_config.get("issn")
            parsed = feedparser.parse(url)
            for entry in parsed.entries[:max_entries_per_feed]:
                raw = self._entry_to_raw(entry, name, issn_hint)
                if raw:
                    raw_entries.append(raw)
        return raw_entries

    def _entry_to_raw(
        self,
        entry: dict[str, Any],
        source: str,
        issn_hint: Optional[str],
    ) -> Optional[RawEntry]:
        """Convert RSS entry to RawEntry (no API calls)."""
        title_raw = entry.get("title", "").strip() or "(no title)"
        title = clean_title(title_raw)
        link = entry.get("link", "").strip() or ""
        doi = extract_doi(entry)
        published = parse_published(entry)
        return RawEntry(
            source=source,
            title=title,
            link=link,
            doi=doi,
            published=published,
            issn_hint=issn_hint,
            entry=dict(entry),
        )

    def enrich_entry(self, raw: RawEntry) -> Paper:
        """Enrich a single raw entry with Crossref (search + lookup)."""
        authors = None
        journal = None
        abstract = None
        published = raw.published
        doi = raw.doi
        raw_dict: dict[str, Any] = {"feed": raw.source, "entry": raw.entry}

        if not doi:
            doi = self._search_doi(
                raw.title, raw.source, raw.published, raw.issn_hint, raw_dict
            )
        if doi:
            authors, journal, published_cr, abstract = self._enrich_metadata(
                doi, raw_dict
            )
            published = published_cr or published

        return Paper(
            source=raw.source,
            title=raw.title,
            link=raw.link,
            doi=doi,
            published=published,
            authors=authors,
            journal=journal,
            abstract=abstract,
            raw=raw_dict,
        )

    def fetch_all(
        self,
        max_entries_per_feed: int = 200,
        max_workers: int = 8,
    ) -> Iterator[Paper]:
        """Fetch papers from all feeds with parallel Crossref enrichment.

        RSS is parsed first (fast), then Crossref calls run in parallel.

        Args:
            max_entries_per_feed: Maximum entries per feed
            max_workers: Number of parallel workers for Crossref (default 8)

        Yields:
            Paper objects with enriched metadata
        """
        raw_entries = self.collect_raw_entries(max_entries_per_feed)
        if not raw_entries:
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.enrich_entry, raw): raw
                for raw in raw_entries
            }
            for future in as_completed(futures):
                try:
                    paper = future.result()
                    yield paper
                except Exception:
                    pass  # skip failed entries

    def _process_feed(
        self,
        name: str,
        url: str,
        issn_hint: Optional[str],
        max_entries: int,
    ) -> Iterator[Paper]:
        """Process a single RSS feed (legacy sequential path)."""
        parsed = feedparser.parse(url)
        for entry in parsed.entries[:max_entries]:
            raw = self._entry_to_raw(entry, name, issn_hint)
            if raw:
                yield self.enrich_entry(raw)

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
