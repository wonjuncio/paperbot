"""Command-line interface handlers."""

import argparse
from typing import Optional

from paperbot.config import Settings, load_feeds
from paperbot.console import ConsoleUI
from paperbot.database.repository import PaperRepository
from paperbot.services.crossref_service import CrossrefService
from paperbot.services.feed_service import FeedService
from paperbot.services.zotero_service import create_zotero_service


class PaperBotCLI:
    """CLI application for PaperBot."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize CLI with settings.

        Args:
            settings: Application settings (loads from env if not provided)
        """
        self.settings = settings or Settings.load()
        self.ui = ConsoleUI()
        self.repo = PaperRepository(self.settings.db_path)

    def cmd_fetch(self) -> None:
        """Fetch papers from all configured feeds."""
        crossref = CrossrefService(self.settings.contact_email)
        feed_service = FeedService(
            feeds_path=self.settings.feeds_path,
            crossref=crossref,
        )

        feeds = load_feeds(self.settings.feeds_path)
        total_new = 0

        for feed_config in feeds:
            name = feed_config["name"]
            self.ui.fetching(name)

        # Process all papers
        for paper in feed_service.fetch_all():
            if self.repo.upsert(paper):
                total_new += 1

        self.ui.fetch_complete(total_new)

    def cmd_list(
        self,
        status: str = "new",
        limit: int = 50,
        sort_by: str = "id",
    ) -> None:
        """List papers by status.

        Args:
            status: Filter by status ('new', 'picked', 'pushed')
            limit: Maximum papers to display
            sort_by: Sort by 'id', 'date', or 'title' (default: id)
        """
        papers = self.repo.find_by_status(status, limit, sort_by)
        self.ui.display_papers(papers, status)

    def cmd_pick(self, ids: list[int]) -> None:
        """Mark papers as picked.

        Args:
            ids: Paper IDs to mark as picked
        """
        self.repo.update_status(ids, "picked")
        self.ui.picked(ids)

    def cmd_push_zotero(self, limit: int = 30) -> None:
        """Push picked papers to Zotero.

        Args:
            limit: Maximum papers to push
        """
        zotero = create_zotero_service(
            api_key=self.settings.zotero_api_key,
            library_id=self.settings.zotero_library_id,
            library_type=self.settings.zotero_library_type,
            collection_key=self.settings.zotero_collection_key,
        )

        papers = self.repo.find_picked_without_zotero(limit)
        pushed = 0

        for paper in papers:
            key = zotero.push_paper(paper)
            if paper.id is not None:
                self.repo.mark_pushed(paper.id, key or "")
            pushed += 1

        self.ui.pushed_zotero(pushed)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="paperbot",
        description="Journal RSS → Crossref → SQLite → Zotero",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch command
    subparsers.add_parser("fetch", help="Fetch feeds and store new papers")

    # list command
    list_parser = subparsers.add_parser("list", help="List papers by status")
    list_parser.add_argument(
        "--status",
        default="new",
        choices=["new", "picked", "pushed"],
        help="Filter by status (default: new)",
    )
    list_parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum papers to display (default: 50)",
    )
    list_parser.add_argument(
        "--sort",
        default="id",
        choices=["id", "date", "title"],
        dest="sort_by",
        help="Sort by id, date, or title (default: id)",
    )

    # pick command
    pick_parser = subparsers.add_parser("pick", help="Mark paper IDs as picked")
    pick_parser.add_argument(
        "ids",
        nargs="+",
        type=int,
        help="Paper IDs to mark as picked",
    )

    # push-zotero command
    push_parser = subparsers.add_parser(
        "push-zotero",
        help="Push picked papers to Zotero",
    )
    push_parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Maximum papers to push (default: 30)",
    )

    return parser


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    cli = PaperBotCLI()

    if args.command == "fetch":
        cli.cmd_fetch()
    elif args.command == "list":
        cli.cmd_list(args.status, args.limit, args.sort_by)
    elif args.command == "pick":
        cli.cmd_pick(args.ids)
    elif args.command == "push-zotero":
        cli.cmd_push_zotero(args.limit)
