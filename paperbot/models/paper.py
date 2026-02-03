"""Paper data model."""

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class Paper:
    """Represents a research paper with metadata."""

    source: str
    title: str
    link: str
    doi: Optional[str] = None
    published: Optional[str] = None
    authors: Optional[str] = None
    journal: Optional[str] = None
    abstract: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    # Database fields (set after persistence)
    id: Optional[int] = None
    status: Literal["new", "archived", "read"] = "new"
    zotero_key: Optional[str] = None
