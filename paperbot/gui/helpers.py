"""Shared helper functions for paper filtering, sorting, and parsing."""

from typing import Optional

from paperbot.models.paper import Paper


def filter_by_keywords(
    papers: list[Paper],
    keywords: list[str],
    mode: str = "or",
) -> list[Paper]:
    """Filter papers by keywords in title only.

    Uses substring matching (e.g., "oxygen" matches "deoxygenation").

    Args:
        papers: List of papers to filter
        keywords: List of keyword strings
        mode: 'or' (any keyword matches) or 'and' (all keywords must match)

    Returns:
        Filtered list of papers
    """
    if not keywords:
        return papers

    keywords_lower = [k.lower() for k in keywords]

    def matches_paper(paper: Paper) -> bool:
        title = (paper.title or "").lower()
        if mode == "and":
            return all(kw in title for kw in keywords_lower)
        else:  # or
            return any(kw in title for kw in keywords_lower)

    return [p for p in papers if matches_paper(p)]


def filter_by_date(
    papers: list[Paper],
    date_from: Optional[str],
    date_to: Optional[str],
    date_field: str = "published",
) -> list[Paper]:
    """Filter papers by date range.

    Args:
        papers: List of papers to filter
        date_from: Start date (YYYY-MM-DD), inclusive
        date_to: End date (YYYY-MM-DD), inclusive
        date_field: 'published' or 'created_at'

    Returns:
        Filtered list of papers
    """
    if not date_from and not date_to:
        return papers

    def get_date(paper: Paper) -> Optional[str]:
        if date_field == "created_at":
            return paper.created_at[:10] if paper.created_at else None
        else:
            return paper.published[:10] if paper.published else None

    def in_range(paper: Paper) -> bool:
        paper_date = get_date(paper)
        if not paper_date:
            return False
        if date_from and paper_date < date_from:
            return False
        if date_to and paper_date > date_to:
            return False
        return True

    return [p for p in papers if in_range(p)]


def sort_papers(
    papers: list[Paper],
    sort_by: str,
    order: str,
    scores: dict | None = None,
) -> list[Paper]:
    """Sort papers by the specified criteria (in-place).

    Args:
        papers: List of papers to sort
        sort_by: 'published', 'created_at', 'title', 'journal', or 'score'
        order: 'asc' or 'desc'
        scores: AI match scores dict (paper_id -> score), used when sort_by='score'

    Returns:
        The same list, sorted in-place
    """
    reverse = order == "desc"

    if sort_by == "score":
        if scores:
            papers.sort(key=lambda p: scores.get(p.id, -1), reverse=reverse)
        else:
            # Fallback to published when scores not available
            papers.sort(key=lambda p: p.published or "", reverse=reverse)
    elif sort_by == "created_at":
        papers.sort(key=lambda p: p.created_at or "", reverse=reverse)
    elif sort_by == "title":
        papers.sort(key=lambda p: (p.title or "").lower(), reverse=reverse)
    elif sort_by == "journal":
        papers.sort(key=lambda p: (p.journal or "").lower(), reverse=reverse)
    else:  # 'published' or any unrecognized value
        papers.sort(key=lambda p: p.published or "", reverse=reverse)

    return papers


def parse_authors(authors_str: Optional[str]) -> list[str]:
    """Parse authors string into list (comma / semicolon / ' and ')."""
    if not authors_str or not authors_str.strip():
        return []
    s = authors_str.replace(" and ", ", ").replace(";", ",")
    return [a.strip() for a in s.split(",") if a.strip()]
