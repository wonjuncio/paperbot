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
                    is_picked INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(doi),
                    UNIQUE(link)
                );
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON papers(status);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_picked ON papers(is_picked);")
            
            # Migrate existing data if is_picked doesn't exist
            cursor.execute("PRAGMA table_info(papers)")
            columns = [row[1] for row in cursor.fetchall()]
            if "is_picked" not in columns:
                cursor.execute("ALTER TABLE papers ADD COLUMN is_picked INTEGER NOT NULL DEFAULT 0")
                # Migrate: status='picked' → is_picked=1, status='new'
                cursor.execute("UPDATE papers SET is_picked = 1, status = 'new' WHERE status = 'picked'")
                conn.commit()
            
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
                (created_at, source, title, link, doi, published, authors, journal, abstract, status, is_picked)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'new', 0)
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

    def archive_old_new(self) -> int:
        """Archive all papers with status='new' to 'archived'.
        
        Called at the start of fetch to move old 'new' papers out of view.
        
        Returns:
            Number of papers archived
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE papers SET status = 'archived' WHERE status = 'new'")
            conn.commit()
            return cursor.rowcount

    def find_by_status(
        self,
        status: str,
        limit: int = 50,
        sort_by: str = "id",
        order: str = "desc",
        journal: Optional[str] = None,
    ) -> list[Paper]:
        """Find papers by status (or pseudo-status 'picked' for is_picked=1).

        Args:
            status: 'new', 'archived', 'read', or 'picked' (is_picked=1)
            limit: Maximum number of papers to return
            sort_by: Sort key - 'id', 'date', or 'title' (default: id)
            order: 'asc' or 'desc' for sort direction
            journal: If set, filter by this journal name only.

        Returns:
            List of Paper objects
        """
        direction = "DESC" if order.lower() == "desc" else "ASC"
        order_clauses = {
            "id": f"id {direction}",
            "date": f"COALESCE(published, created_at) {direction}",
            "title": f"title {direction}",
            "created_at": f"created_at {direction}",
        }
        order_sql = order_clauses.get(sort_by, order_clauses["id"])

        # Special case: status='picked' → is_picked=1
        if status == "picked":
            where_clause = "is_picked = 1"
        else:
            where_clause = "status = ?"
        if journal is not None:
            where_clause += " AND journal = ?"

        with self._connection() as conn:
            cursor = conn.cursor()
            if status == "picked":
                params: tuple = (limit,) if journal is None else (journal, limit)
                cursor.execute(
                    f"""
                    SELECT id, source, title, link, doi, published, authors, journal, abstract, status, is_picked, created_at
                    FROM papers
                    WHERE {where_clause}
                    ORDER BY {order_sql}
                    LIMIT ?
                    """,
                    params,
                )
            else:
                params = (status, limit) if journal is None else (status, journal, limit)
                cursor.execute(
                    f"""
                    SELECT id, source, title, link, doi, published, authors, journal, abstract, status, is_picked, created_at
                    FROM papers
                    WHERE {where_clause}
                    ORDER BY {order_sql}
                    LIMIT ?
                    """,
                    params,
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
                is_picked=row["is_picked"],
                created_at=row["created_at"],
            )
            papers.append(paper)

        return papers

    def find_picked(self, limit: int = 100, order: str = "desc") -> list[Paper]:
        """Find picked papers for export (is_picked=1).

        Args:
            limit: Maximum number of papers to return
            order: 'asc' or 'desc' for date sort direction

        Returns:
            List of Paper objects
        """
        direction = "DESC" if order.lower() == "desc" else "ASC"
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, source, title, link, doi, published, authors, journal, abstract, status, is_picked, created_at
                FROM papers
                WHERE is_picked = 1
                ORDER BY COALESCE(published, created_at) {direction}
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
                is_picked=row["is_picked"],
                created_at=row["created_at"],
            )
            papers.append(paper)

        return papers

    def get_status_counts(self) -> dict[str, int]:
        """Return counts per status (new, archived, read) and picked count.

        Returns:
            Dict with keys 'new', 'archived', 'read', 'picked' and counts.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT status, COUNT(*) AS cnt FROM papers GROUP BY status
                """
            )
            by_status = {row["status"]: row["cnt"] for row in cursor.fetchall()}
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM papers WHERE is_picked = 1"
            )
            picked = cursor.fetchone()["cnt"]
        return {
            "new": by_status.get("new", 0),
            "archived": by_status.get("archived", 0),
            "read": by_status.get("read", 0),
            "picked": picked,
        }

    def get_distinct_journals(self) -> list[str]:
        """Return distinct non-null journal names from the DB, sorted.

        Returns:
            Sorted list of journal name strings.
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT journal FROM papers
                WHERE journal IS NOT NULL AND journal != ''
                ORDER BY journal ASC
                """
            )
            return [row["journal"] for row in cursor.fetchall()]

    def find_by_id(self, paper_id: int) -> Optional[Paper]:
        """Find a single paper by ID.

        Args:
            paper_id: Paper ID to find

        Returns:
            Paper object if found, None otherwise
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, source, title, link, doi, published, authors, journal, abstract, status, is_picked, created_at
                FROM papers
                WHERE id = ?
                """,
                (paper_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        return Paper(
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
            is_picked=row["is_picked"],
            created_at=row["created_at"],
        )

    def find_all(self, limit: int = 500, sort_by: str = "id", order: str = "desc", journal: Optional[str] = None) -> list[Paper]:
        """Find papers from all statuses (for archive view).

        Args:
            limit: Maximum number of papers to return
            sort_by: 'id', 'date', or 'title'
            order: 'asc' or 'desc' for sort direction
            journal: If set, filter by this journal name only.

        Returns:
            List of Paper objects
        """
        direction = "DESC" if order.lower() == "desc" else "ASC"
        order_clauses = {
            "id": f"id {direction}",
            "date": f"COALESCE(published, created_at) {direction}",
            "title": f"title {direction}",
            "created_at": f"created_at {direction}",
        }
        order_sql = order_clauses.get(sort_by, f"id {direction}")
        where_clause = "" if journal is None else "WHERE journal = ?"
        params: tuple = (limit,) if journal is None else (journal, limit)
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT id, source, title, link, doi, published, authors, journal, abstract, status, is_picked, created_at
                FROM papers
                {where_clause}
                ORDER BY {order_sql}
                LIMIT ?
                """,
                params,
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
                is_picked=row["is_picked"],
                created_at=row["created_at"],
            )
            papers.append(paper)
        return papers

    def pick(self, ids: list[int]) -> int:
        """Set is_picked=1 for given IDs.
        
        Args:
            ids: Paper IDs to pick
            
        Returns:
            Number of papers picked
        """
        if not ids:
            return 0

        placeholders = ",".join(["?"] * len(ids))
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE papers SET is_picked = 1 WHERE id IN ({placeholders})",
                ids,
            )
            conn.commit()
            return cursor.rowcount

    def unpick(self, ids: list[int]) -> list[int]:
        """Set is_picked=0 only for papers that are currently picked.

        Args:
            ids: Paper IDs to unmark

        Returns:
            List of IDs that were actually picked and were unpicked
        """
        if not ids:
            return []

        placeholders = ",".join(["?"] * len(ids))
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT id FROM papers WHERE id IN ({placeholders}) AND is_picked = 1",
                ids,
            )
            picked_ids = [row["id"] for row in cursor.fetchall()]

            if not picked_ids:
                return []

            ph = ",".join(["?"] * len(picked_ids))
            cursor.execute(
                f"UPDATE papers SET is_picked = 0 WHERE id IN ({ph})",
                picked_ids,
            )
            conn.commit()
            return picked_ids

    def mark_exported(self, paper_ids: list[int]) -> None:
        """Mark papers as exported (status='read', is_picked=0).

        Args:
            paper_ids: List of paper IDs to mark as read
        """
        if not paper_ids:
            return

        placeholders = ",".join(["?"] * len(paper_ids))
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE papers SET status = 'read', is_picked = 0 WHERE id IN ({placeholders})",
                paper_ids,
            )
            conn.commit()

    def reset_all_picked(self) -> int:
        """Set is_picked=0 for all papers.

        Intended to be called when the program or DB session closes,
        so that picked state is not persisted across runs.

        Returns:
            Number of papers that were reset (had is_picked=1).
        """
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE papers SET is_picked = 0 WHERE is_picked = 1")
            conn.commit()
            return cursor.rowcount
