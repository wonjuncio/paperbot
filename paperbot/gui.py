"""Streamlit GUI for PaperBot. Launched when no CLI args are given."""

import os

import pandas as pd
import streamlit as st

from paperbot.config import Settings
from paperbot.database.repository import PaperRepository
from paperbot.models.paper import Paper
from paperbot.services.crossref_service import CrossrefService
from paperbot.services.export_service import MarkdownExporter
from paperbot.services.feed_service import FeedService


def run_gui() -> None:
    """Start the PaperBot Streamlit GUI."""
    settings = Settings.load()
    repo = PaperRepository(settings.db_path)
    crossref = CrossrefService(settings.contact_email)
    feed_service = FeedService(
        feeds_path=settings.feeds_path,
        crossref=crossref,
    )
    exporter = MarkdownExporter(settings.export_dir)

    st.set_page_config(page_title="PaperBot GUI", layout="wide")
    st.title("PaperBot Dashboard")

    with st.sidebar:
        st.header("Statistics")
        stats = repo.get_status_counts()
        for status, count in stats.items():
            st.metric(label=status.upper(), value=count)
        st.divider()
        if st.button("Reset All Views", width="stretch"):
            st.rerun()

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("Fetch New Papers", type="primary", width="stretch"):
            with st.spinner("RSS 피드 수집 중..."):
                archived = repo.archive_old_new()
                if archived > 0:
                    st.caption(f"Archived {archived} old 'new' papers.")
                total_new = 0
                total_processed = 0
                workers = min(8, (os.cpu_count() or 2) - 1)
                workers = max(1, workers)
                for paper in feed_service.fetch_all(max_workers=workers):
                    if repo.upsert(paper):
                        total_new += 1
                    total_processed += 1
                st.success(f"{total_new} new, {total_processed} processed.")
                st.rerun()

    with col2:
        if st.button("Export Picked", width="stretch"):
            picked_papers = repo.find_picked()
            if picked_papers:
                filepath = exporter.export(picked_papers)
                paper_ids = [p.id for p in picked_papers if p.id is not None]
                repo.mark_exported(paper_ids)
                st.success(f"Exported to {filepath}")
                st.rerun()
            else:
                st.warning("No picked papers.")

    tab_new, tab_picked, tab_archive = st.tabs(["New", "Picked", "Archive"])

    journals = repo.get_distinct_journals()
    journal_options = ["All"] + (journals if journals else [])

    with tab_new:
        st.subheader("New papers")
        selected_journal_new = st.selectbox(
            "Filter by journal",
            options=journal_options,
            key="new_journal_filter",
        )
        journal_filter_new = None if selected_journal_new == "All" else selected_journal_new
        papers = repo.find_by_status("new", limit=200, sort_by="id", journal=journal_filter_new)
        if not papers:
            st.info("No new papers. Click Fetch New Papers." if journal_filter_new is None else f"No new papers for journal: {journal_filter_new}")
        else:
            _display_paper_table(papers, repo)

    with tab_picked:
        st.subheader("Picked (ready to export)")
        picked = repo.find_picked()
        if not picked:
            st.write("Pick papers in the New tab.")
        else:
            for p in picked:
                with st.container(border=True):
                    c1, c2 = st.columns([7, 1])
                    with c1:
                        st.markdown(f"**{p.title}**")
                        st.caption(f"{p.journal or ''} | {p.published or ''}")
                    with c2:
                        if p.id is not None and st.button("Unpick", key=f"un_{p.id}"):
                            repo.unpick([p.id])
                            st.rerun()

    with tab_archive:
        st.subheader("All papers")
        selected_journal_archive = st.selectbox(
            "Filter by journal",
            options=journal_options,
            key="archive_journal_filter",
        )
        journal_filter_archive = None if selected_journal_archive == "All" else selected_journal_archive
        all_papers = repo.find_all(limit=500, sort_by="date", journal=journal_filter_archive)
        if not all_papers:
            st.info("No papers yet." if journal_filter_archive is None else f"No papers for journal: {journal_filter_archive}")
        else:
            data = []
            for p in all_papers:
                data.append({
                    "ID": p.id,
                    "Title": p.title,
                    "Journal": p.journal,
                    "Published": p.published,
                    "Status": p.status,
                    "DOI": p.doi,
                })
            df = pd.DataFrame(data)
            st.dataframe(df, width="stretch", hide_index=True)


def _display_paper_table(papers: list[Paper], repo: PaperRepository) -> None:
    """Show paper list with Pick checkbox; apply pick/unpick on button."""
    data = []
    for p in papers:
        data.append({
            "ID": p.id,
            "Pick": bool(p.is_picked),
            "Title": p.title,
            "Journal": p.journal,
            "Published": p.published,
            "DOI": p.doi,
        })
    df = pd.DataFrame(data)
    edited_df = st.data_editor(
        df,
        column_config={
            "Pick": st.column_config.CheckboxColumn(help="Pick for export"),
        },
        disabled=["ID", "Title", "Journal", "Published"],
        hide_index=True,
        width="stretch",
    )

    if st.button("Apply Selection"):
        to_pick = []
        to_unpick = []
        for _, row in edited_df.iterrows():
            pid = row["ID"]
            if pid is None:
                continue
            was_picked = next((p.is_picked for p in papers if p.id == pid), 0)
            now_picked = bool(row["Pick"])
            if now_picked and not was_picked:
                to_pick.append(pid)
            elif not now_picked and was_picked:
                to_unpick.append(pid)
        if to_pick:
            repo.pick(to_pick)
        if to_unpick:
            repo.unpick(to_unpick)
        st.toast("Saved.")
        st.rerun()


if __name__ == "__main__":
    run_gui()
