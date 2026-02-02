"""Paper repository for database operations."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from paperbot.models.paper import Paper


class PaperRepository:
    """Repository for paper CRUD operations using SQLite."""

    def __init__(self, db_path: Path):
        """Initialize repository with database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    link TEXT NOT NULL,
                    doi TEXT,
                    published TEXT,
                    authors TEXT,
                    journal TEXT,
                    abstract TEXT,
                    status TEXT NOT NULL DEFAULT 'new',
                    zotero_key TEXT,
                    UNIQUE(doi),
                    UNIQUE(link)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON papers(status);")
            conn.commit()

    def upsert(self, paper: Paper) -> bool:
        """Insert a paper if it doesn't exist (by DOI or link).

        Args:
            paper: Paper object to insert

        Returns:
            True if paper was inserted, False if it already existed
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO papers
                (created_at, source, title, link, doi, published, authors, journal, abstract, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')
                """,
                (
                    now,
                    paper.source,
                    paper.title,
                    paper.link,
                    paper.doi,
                    paper.published,
                    paper.authors,
                    paper.journal,
                    paper.abstract,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def find_by_status(
        self,
        status: str,
        limit: int = 50,
        sort_by: str = "id",
    ) -> list[Paper]:
        """Find papers by status.

        Args:
            status: Paper status ('new', 'picked', 'pushed')
            limit: Maximum number of papers to return
            sort_by: Sort key - 'id', 'date', or 'title' (default: id)

        Returns:
            List of Paper objects
        """
        order_clauses = {
            "id": "id ASC",  # 1번이 맨 위
            "date": "COALESCE(published, created_at) DESC",
            "title": "title ASC",
        }
        order = order_clauses.get(sort_by, order_clauses["id"])

        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, source, title, link, doi, published, authors, journal, abstract, status, zotero_key
                FROM papers
                WHERE status = ?
                ORDER BY {order}
                LIMIT ?
                """,
                (status, limit),
            )
            rows = cursor.fetchall()

        papers = []
        for row in rows:
            paper = Paper(
                source=row["source"],
                title=row["title"],
                link=row["link"],
                doi=row["doi"],
                published=row["published"],
                authors=row["authors"],
                journal=row["journal"],
                abstract=row["abstract"],
                id=row["id"],
                status=row["status"],
                zotero_key=row["zotero_key"],
            )
            papers.append(paper)

        return papers

    def find_picked_without_zotero(self, limit: int = 30) -> list[Paper]:
        """Find picked papers that haven't been pushed to Zotero.

        Args:
            limit: Maximum number of papers to return

        Returns:
            List of Paper objects
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, source, title, link, doi, published, authors, journal, abstract, status, zotero_key
                FROM papers
                WHERE status = 'picked' AND (zotero_key IS NULL OR zotero_key = '')
                ORDER BY COALESCE(published, created_at) DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()

        papers = []
        for row in rows:
            paper = Paper(
                source=row["source"],
                title=row["title"],
                link=row["link"],
                doi=row["doi"],
                published=row["published"],
                authors=row["authors"],
                journal=row["journal"],
                abstract=row["abstract"],
                id=row["id"],
                status=row["status"],
                zotero_key=row["zotero_key"],
            )
            papers.append(paper)

        return papers

    def update_status(self, ids: list[int], status: str) -> int:
        """Update status for multiple papers.

        Args:
            ids: List of paper IDs to update
            status: New status value

        Returns:
            Number of papers updated
        """
        if not ids:
            return 0

        placeholders = ",".join(["?"] * len(ids))
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE papers SET status = ? WHERE id IN ({placeholders})",
                [status] + ids,
            )
            conn.commit()
            return cursor.rowcount

    def mark_pushed(self, paper_id: int, zotero_key: str) -> None:
        """Mark a paper as pushed to Zotero.

        Args:
            paper_id: Paper ID to update
            zotero_key: Zotero item key
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE papers SET status = 'pushed', zotero_key = ? WHERE id = ?",
                (zotero_key, paper_id),
            )
            conn.commit()
