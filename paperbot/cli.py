"""Command-line interface handlers."""

import argparse
from typing import Optional

from rich.progress import Progress, SpinnerColumn, TextColumn

from paperbot.config import Settings, load_feeds
from paperbot.console import ConsoleUI
from paperbot.database.repository import PaperRepository
from paperbot.services.crossref_service import CrossrefService
from paperbot.services.export_service import MarkdownExporter
from paperbot.services.feed_service import FeedService


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

    def cmd_fetch(self, workers: int = 8) -> None:
        """Fetch papers from all configured feeds (Crossref calls run in parallel)."""
        crossref = CrossrefService(self.settings.contact_email)
        feed_service = FeedService(
            feeds_path=self.settings.feeds_path,
            crossref=crossref,
        )

        feeds = load_feeds(self.settings.feeds_path)
        total_new = 0
        total_processed = 0

        for feed_config in feeds:
            name = feed_config["name"]
            self.ui.fetching(name)

        # Process all papers with progress (parallel Crossref enrichment)
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            console=self.ui.console,
        ) as progress:
            task = progress.add_task("Fetching papers...", total=None)
            for paper in feed_service.fetch_all(max_workers=workers):
                if self.repo.upsert(paper):
                    total_new += 1
                total_processed += 1
                progress.update(
                    task,
                    description=f"Processed {total_processed} papers, {total_new} new",
                )

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

    def cmd_unpick(self, ids: list[int]) -> None:
        """Unmark papers (set status back to new).

        Only papers that are currently 'picked' are changed. If none are picked,
        shows a message instead of silently doing nothing.

        Args:
            ids: Paper IDs to unmark
        """
        unpicked_ids = self.repo.unpick(ids)
        if unpicked_ids:
            self.ui.unpicked(unpicked_ids)
        else:
            self.ui.no_papers_to_unpick(ids)

    def cmd_export(self) -> None:
        """Export picked papers to markdown file."""
        papers = self.repo.find_picked()
        
        if not papers:
            self.ui.no_papers_to_export()
            return

        exporter = MarkdownExporter(self.settings.export_dir)
        filepath = exporter.export(papers)
        
        # Mark papers as read
        paper_ids = [p.id for p in papers if p.id is not None]
        self.repo.mark_exported(paper_ids)
        
        self.ui.exported(len(papers), filepath)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="paperbot",
        description="Journal RSS → Crossref → SQLite → Markdown",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch feeds and store new papers")
    fetch_parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel workers for Crossref API (default: 8)",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List papers by status")
    list_parser.add_argument(
        "--status",
        default="new",
        choices=["new", "picked", "read"],
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

    # unpick command
    unpick_parser = subparsers.add_parser("unpick", help="Unmark paper IDs (set back to new)")
    unpick_parser.add_argument(
        "ids",
        nargs="+",
        type=int,
        help="Paper IDs to unmark",
    )

    # export command
    subparsers.add_parser("export", help="Export picked papers to markdown")

    return parser


def main() -> None:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    cli = PaperBotCLI()

    if args.command == "fetch":
        cli.cmd_fetch(workers=args.workers)
    elif args.command == "list":
        cli.cmd_list(args.status, args.limit, args.sort_by)
    elif args.command == "pick":
        cli.cmd_pick(args.ids)
    elif args.command == "unpick":
        cli.cmd_unpick(args.ids)
    elif args.command == "export":
        cli.cmd_export()
