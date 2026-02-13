"""FastAPI + HTMX GUI for PaperBot — application entry point."""

import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI

from paperbot.config import Settings
from paperbot.database.repository import PaperRepository
from paperbot.services.crossref_service import CrossrefService
from paperbot.services.export_service import MarkdownExporter
from paperbot.services.feed_service import FeedService
from paperbot.services.ranking_service import RankingService
from paperbot.services.semantic_map_service import SemanticMapService

from paperbot.gui.state import preload_models, state
from paperbot.gui.routers import actions, common, papers, semantic


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    state.settings = Settings.load()
    state.repo = PaperRepository(state.settings.db_path)
    state.crossref = CrossrefService(state.settings.contact_email)
    state.feed_service = FeedService(
        feeds_path=state.settings.feeds_path,
        crossref=state.crossref,
    )
    state.exporter = MarkdownExporter(state.settings.export_dir)
    state.fetch_status = {"running": False, "message": "", "complete": False}
    # AI ranking service
    state.ranker = RankingService(repo=state.repo)
    state._ranking_scores = {}
    state._ranking_top_ids = set()
    state._ranking_gold_ids = set()
    state._ranking_blue_ids = set()
    state._ranking_computed = False
    state._ranking_computing = False
    state.ranking_status = {"phase": "idle", "message": ""}
    # Semantic map service
    state.semantic_map_service = SemanticMapService(
        ranking_service=state.ranker, repo=state.repo,
    )
    state._smap_cache = None
    state._smap_cache_status = None
    state._smap_computing = False
    state.smap_status = {"phase": "idle", "message": ""}
    # Pre-load AI models into RAM in a background thread (non-blocking)
    threading.Thread(target=preload_models, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)

# Register routers
app.include_router(common.router)
app.include_router(papers.router)
app.include_router(actions.router)
app.include_router(semantic.router)
