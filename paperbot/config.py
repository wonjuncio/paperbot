"""Configuration management."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv


@dataclass
class Settings:
    """Application settings loaded from environment and config files."""

    contact_email: Optional[str]
    zotero_api_key: Optional[str]
    zotero_library_type: str
    zotero_library_id: Optional[str]
    zotero_collection_key: Optional[str]
    db_path: Path
    feeds_path: Path

    @classmethod
    def load(cls, base_dir: Optional[Path] = None) -> "Settings":
        """Load settings from environment variables and default paths."""
        load_dotenv()

        if base_dir is None:
            base_dir = Path(__file__).parent.parent

        return cls(
            contact_email=os.getenv("CONTACT_EMAIL"),
            zotero_api_key=os.getenv("ZOTERO_API_KEY"),
            zotero_library_type=os.getenv("ZOTERO_LIBRARY_TYPE", "user"),
            zotero_library_id=os.getenv("ZOTERO_LIBRARY_ID"),
            zotero_collection_key=os.getenv("ZOTERO_COLLECTION_KEY") or None,
            db_path=base_dir / "papers.db",
            feeds_path=base_dir / "feeds.yaml",
        )


def load_feeds(feeds_path: Path) -> list[dict[str, Any]]:
    """Load feed configurations from YAML file."""
    with open(feeds_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("feeds", [])
