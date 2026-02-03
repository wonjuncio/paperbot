"""Markdown export service."""

from datetime import date, datetime
from pathlib import Path

from paperbot.models.paper import Paper


class MarkdownExporter:
    """Service for exporting papers to Markdown format."""

    def __init__(self, export_dir: Path):
        """Initialize exporter.

        Args:
            export_dir: Directory to save exported markdown files
        """
        self.export_dir = export_dir
        self.export_dir.mkdir(exist_ok=True)

    def export(self, papers: list[Paper]) -> Path:
        """Export papers to markdown file named by today's date.

        Args:
            papers: List of papers to export

        Returns:
            Path to the created markdown file
        """
        today = date.today().isoformat()
        filename = f"{today}.md"
        filepath = self.export_dir / filename

        # Group papers by published date
        date_groups: dict[str, list[Paper]] = {}
        for paper in papers:
            paper_date = paper.published or today
            if paper_date not in date_groups:
                date_groups[paper_date] = []
            date_groups[paper_date].append(paper)

        # Generate markdown content
        lines = []
        for paper_date in sorted(date_groups.keys(), reverse=True):
            lines.append(f"## {paper_date}\n")
            for paper in date_groups[paper_date]:
                lines.append(f"### {paper.title}")
                if paper.journal and paper.published:
                    lines.append(f"- Journal: {paper.journal} ({paper.published[:4]})")
                elif paper.journal:
                    lines.append(f"- Journal: {paper.journal}")
                if paper.doi:
                    lines.append(f"- DOI: {paper.doi}")
                    lines.append(f"- Link: https://doi.org/{paper.doi}")
                elif paper.link:
                    lines.append(f"- Link: {paper.link}")
                lines.append("")

        # Write to file (overwrites if exists)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return filepath
