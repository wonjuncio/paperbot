"""Zotero API service for pushing papers to library."""

import time
from typing import Any, Optional

from paperbot.models.paper import Paper

try:
    from pyzotero import zotero
except ImportError:
    zotero = None  # type: ignore


class ZoteroService:
    """Service for interacting with Zotero API."""

    def __init__(
        self,
        api_key: str,
        library_id: str,
        library_type: str = "user",
        collection_key: Optional[str] = None,
        polite_delay: float = 0.2,
    ):
        """Initialize Zotero service.

        Args:
            api_key: Zotero API key
            library_id: Library ID (user ID or group ID)
            library_type: 'user' or 'group'
            collection_key: Optional collection to add items to
            polite_delay: Delay between API calls (seconds)

        Raises:
            RuntimeError: If pyzotero is not installed
        """
        if zotero is None:
            raise RuntimeError(
                "pyzotero is not installed. Install with: pip install pyzotero"
            )

        self.collection_key = collection_key
        self.polite_delay = polite_delay
        self._client = zotero.Zotero(library_id, library_type, api_key)

    def push_paper(self, paper: Paper) -> Optional[str]:
        """Push a single paper to Zotero.

        Args:
            paper: Paper to push

        Returns:
            Zotero item key if successful, None otherwise
        """
        item = self._create_item(paper)
        template = self._client.item_template("journalArticle")
        template.update(item)

        created = self._client.create_items([template])

        # Extract key from response
        key = None
        try:
            key = list(created["successful"].values())[0]["key"]
        except (KeyError, IndexError, TypeError):
            pass

        # Add to collection if specified
        if key and self.collection_key:
            try:
                self._client.addto_collection(self.collection_key, [key])
            except Exception:
                pass

        time.sleep(self.polite_delay)
        return key

    def _create_item(self, paper: Paper) -> dict[str, Any]:
        """Create Zotero item dictionary from Paper.

        Args:
            paper: Paper object

        Returns:
            Zotero item dictionary
        """
        item: dict[str, Any] = {
            "itemType": "journalArticle",
            "title": paper.title,
            "url": paper.link,
            "DOI": paper.doi or "",
            "abstractNote": paper.abstract or "",
            "publicationTitle": paper.journal or "",
            "date": paper.published or "",
            "creators": [],
        }

        # Parse authors
        if paper.authors:
            for name in paper.authors.split(",")[:20]:  # Limit authors
                name = name.strip()
                if not name:
                    continue
                parts = name.split()
                if len(parts) == 1:
                    item["creators"].append({
                        "creatorType": "author",
                        "name": parts[0],
                    })
                else:
                    item["creators"].append({
                        "creatorType": "author",
                        "firstName": " ".join(parts[:-1]),
                        "lastName": parts[-1],
                    })

        return item

    def test_connection(self) -> bool:
        """Test connection to Zotero API.

        Returns:
            True if connection is successful
        """
        try:
            self._client.key_info()
            return True
        except Exception:
            return False


def create_zotero_service(
    api_key: Optional[str],
    library_id: Optional[str],
    library_type: str = "user",
    collection_key: Optional[str] = None,
) -> ZoteroService:
    """Factory function to create ZoteroService with validation.

    Args:
        api_key: Zotero API key
        library_id: Library ID
        library_type: 'user' or 'group'
        collection_key: Optional collection key

    Returns:
        Configured ZoteroService

    Raises:
        RuntimeError: If required credentials are missing
    """
    if not api_key or not library_id:
        raise RuntimeError(
            "Missing Zotero credentials. Set ZOTERO_API_KEY and ZOTERO_LIBRARY_ID in .env"
        )

    return ZoteroService(
        api_key=api_key,
        library_id=library_id,
        library_type=library_type,
        collection_key=collection_key,
    )
