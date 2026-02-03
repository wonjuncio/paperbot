"""Console UI for terminal output using Rich."""

from typing import Optional

from rich.console import Console
from rich.table import Table

from paperbot.models.paper import Paper


class ConsoleUI:
    """Rich-based console UI for paper display and notifications."""

    def __init__(self):
        """Initialize console."""
        self._console = Console()

    def info(self, message: str) -> None:
        """Print an info message."""
        self._console.print(message)

    def success(self, message: str) -> None:
        """Print a success message in green."""
        self._console.print(f"[green]{message}[/green]")

    def warning(self, message: str) -> None:
        """Print a warning message in yellow."""
        self._console.print(f"[yellow]Warning:[/yellow] {message}")

    def error(self, message: str) -> None:
        """Print an error message in red."""
        self._console.print(f"[red]Error:[/red] {message}")

    def fetching(self, feed_name: str) -> None:
        """Print fetching status for a feed."""
        self._console.print(f"[bold]Fetching:[/bold] {feed_name}")

    def fetch_complete(self, new_count: int) -> None:
        """Print fetch completion summary."""
        self._console.print(
            f"\n[green]Done.[/green] New papers added: [bold]{new_count}[/bold]"
        )

    def picked(self, ids: list[int]) -> None:
        """Print picked confirmation."""
        self._console.print(f"[green]Picked[/green]: {ids}")

    def unpicked(self, ids: list[int]) -> None:
        """Print unpicked confirmation."""
        self._console.print(f"[green]Unpicked[/green]: {ids}")

    def no_papers_to_unpick(self, ids: list[int]) -> None:
        """Print message when none of the given IDs are in picked status."""
        self._console.print(
            f"[yellow]None of the given IDs are in picked status:[/yellow] {ids}"
        )

    def pushed_zotero(self, count: int) -> None:
        """Print Zotero push summary."""
        self._console.print(f"[green]Pushed to Zotero[/green]: {count} items")

    def display_papers(
        self,
        papers: list[Paper],
        status: str,
        show_tip: bool = True,
    ) -> None:
        """Display papers in a formatted table.

        Args:
            papers: List of papers to display
            status: Status filter used (for title)
            show_tip: Whether to show the pick tip
        """
        table = Table(title=f"Papers (status={status})")
        table.add_column("ID", justify="right")
        table.add_column("Date", width=10)
        table.add_column("Journal", overflow="fold")
        table.add_column("Title", overflow="fold")
        table.add_column("DOI", overflow="fold")

        for paper in papers:
            table.add_row(
                str(paper.id) if paper.id else "-",
                paper.published or "-",
                paper.journal or "-",
                paper.title,
                paper.doi or "-",
            )

        self._console.print(table)

        if papers and show_tip:
            self._console.print(
                "Tip: `paperbot pick 12 15 18` 처럼 번호로 선택 가능"
            )
        elif not papers:
            self._console.print("No papers found.")
