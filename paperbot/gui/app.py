"""FastAPI + HTMX GUI for PaperBot."""

import html
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, Form, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from paperbot.config import Settings
from paperbot.database.repository import PaperRepository
from paperbot.models.paper import Paper
from paperbot.services.crossref_service import CrossrefService
from paperbot.services.export_service import MarkdownExporter
from paperbot.services.feed_service import FeedService
from paperbot.services.openalex_service import get_paper_info as openalex_get_paper_info
from paperbot.services.ranking_service import RankingService


# Global state for services
class AppState:
    settings: Settings
    repo: PaperRepository
    crossref: CrossrefService
    feed_service: FeedService
    exporter: MarkdownExporter
    fetch_status: dict = {"running": False, "message": "", "complete": False}
    # AI ranking
    ranker: Optional[RankingService] = None
    _ranking_scores: dict = {}       # paper_id → match %
    _ranking_top_ids: set = set()    # top-5 paper IDs
    _ranking_computed: bool = False
    _ranking_computing: bool = False
    # Ranking progress toast state
    ranking_status: dict = {"phase": "idle", "message": ""}


state = AppState()


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
    # AI ranking service (models loaded lazily on first use)
    state.ranker = RankingService()
    state._ranking_scores = {}
    state._ranking_top_ids = set()
    state._ranking_computed = False
    state._ranking_computing = False
    state.ranking_status = {"phase": "idle", "message": ""}
    yield


# Setup templates
base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))
# 초록 등에 들어온 이스케이프된 HTML(&lt;, &gt;) 복원 후 렌더링용
templates.env.filters["unescape_html"] = lambda s: html.unescape(s) if s else ""


def format_read_date(date_str: str) -> str:
    """Format date string to 'Feb 11, 2026' style."""
    if not date_str:
        return ""
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00").split("+")[0])
        return dt.strftime("%b %d, %Y")
    except Exception:
        return date_str[:10] if date_str else ""


def get_date_key(date_str: str) -> str:
    """Extract date part (YYYY-MM-DD) from datetime string."""
    if not date_str:
        return ""
    return date_str[:10]


templates.env.filters["format_read_date"] = format_read_date
templates.env.filters["date_key"] = get_date_key

app = FastAPI(lifespan=lifespan)


# ============================================================================
# AI Ranking Helpers
# ============================================================================


def _compute_rankings() -> None:
    """Compute AI match scores for NEW papers against the READ library.

    Heavy (model inference); always call from a background thread.
    Updates ``state.ranking_status`` at each phase so the frontend can
    show progress toasts via polling.
    """
    from paperbot.services.ranking_service import BI_ENCODER_MODEL, CROSS_ENCODER_MODEL

    try:
        new_papers = state.repo.find_by_status("new", limit=200, sort_by="date", order="desc")
        read_papers = state.repo.find_by_status("read", limit=500)

        if not read_papers or not new_papers:
            state._ranking_scores = {}
            state._ranking_top_ids = set()
            state._ranking_computed = True
            state.ranking_status = {"phase": "idle", "message": ""}
            return

        # Phase: model download / loading
        missing = state.ranker.needs_download()
        if missing:
            for name in missing:
                state.ranking_status = {
                    "phase": "downloading",
                    "message": f'"{name}" 모델 다운로드 중…',
                }
                # Accessing the property triggers the download
                if "reranker" in name:
                    _ = state.ranker.cross_encoder
                else:
                    _ = state.ranker.bi_encoder
        else:
            # Models cached — still need to load into RAM on first call
            if state.ranker._bi_encoder is None or state.ranker._cross_encoder is None:
                state.ranking_status = {
                    "phase": "loading",
                    "message": "AI 모델 로딩 중…",
                }
                _ = state.ranker.bi_encoder
                _ = state.ranker.cross_encoder

        # Phase: scoring
        state.ranking_status = {
            "phase": "scoring",
            "message": f"{len(new_papers)}개 논문 AI 매칭 점수 계산 중…",
        }

        ranked = state.ranker.rank(new_papers, read_papers, top_k_bi=100)
        state._ranking_scores = {r.paper.id: r.score for r in ranked}
        top5 = ranked[:5]
        state._ranking_top_ids = {r.paper.id for r in top5}
        state._ranking_computed = True

        # Phase: done (keep for a few seconds so frontend can poll it)
        state.ranking_status = {
            "phase": "done",
            "message": f"{len(ranked)}개 논문 AI 매칭 완료",
        }
    except Exception as e:
        print(f"[PaperBot] AI ranking failed: {e}")
        state.ranking_status = {"phase": "error", "message": f"AI 랭킹 오류: {e}"}
    finally:
        state._ranking_computing = False


def _start_ranking_bg() -> None:
    """Kick off ranking computation in a daemon thread (non-blocking)."""
    if state._ranking_computing:
        return
    state._ranking_computing = True
    threading.Thread(target=_compute_rankings, daemon=True).start()


def _invalidate_rankings() -> None:
    """Clear cached rankings (call when library or new-paper set changes)."""
    state._ranking_scores = {}
    state._ranking_top_ids = set()
    state._ranking_computed = False
    if state.ranker:
        state.ranker.invalidate_cache()


# ============================================================================
# Main Pages
# ============================================================================


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    stats = state.repo.get_status_counts()
    journals = state.repo.get_distinct_journals()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "stats": stats,
            "journals": journals,
            "active_tab": "new",
        },
    )


# ============================================================================
# Paper List Endpoints (HTMX partials)
# ============================================================================


def _filter_by_keywords(
    papers: list[Paper],
    keywords: list[str],
    mode: str = "or",
) -> list[Paper]:
    """Filter papers by keywords in title only.
    
    Uses substring matching (e.g., "oxygen" matches "deoxygenation").
    
    Args:
        papers: List of papers to filter
        keywords: List of keyword strings
        mode: 'or' (any keyword matches) or 'and' (all keywords must match)
    
    Returns:
        Filtered list of papers
    """
    if not keywords:
        return papers
    
    keywords_lower = [k.lower() for k in keywords]
    
    def matches_paper(paper: Paper) -> bool:
        title = (paper.title or "").lower()
        if mode == "and":
            return all(kw in title for kw in keywords_lower)
        else:  # or
            return any(kw in title for kw in keywords_lower)
    
    return [p for p in papers if matches_paper(p)]


def _filter_by_date(
    papers: list[Paper],
    date_from: Optional[str],
    date_to: Optional[str],
    date_field: str = "published",
) -> list[Paper]:
    """Filter papers by date range.
    
    Args:
        papers: List of papers to filter
        date_from: Start date (YYYY-MM-DD), inclusive
        date_to: End date (YYYY-MM-DD), inclusive
        date_field: 'published' or 'created_at'
    
    Returns:
        Filtered list of papers
    """
    if not date_from and not date_to:
        return papers
    
    def get_date(paper: Paper) -> Optional[str]:
        if date_field == "created_at":
            return paper.created_at[:10] if paper.created_at else None
        else:
            return paper.published[:10] if paper.published else None
    
    def in_range(paper: Paper) -> bool:
        paper_date = get_date(paper)
        if not paper_date:
            return False
        if date_from and paper_date < date_from:
            return False
        if date_to and paper_date > date_to:
            return False
        return True
    
    return [p for p in papers if in_range(p)]


@app.get("/papers/new", response_class=HTMLResponse)
async def papers_new(
    request: Request,
    q: str = Query("", description="Search query"),
    journal: str = Query("", description="Journal filter"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
):
    """Get new papers list (partial for HTMX)."""
    journal_filter = journal if journal else None
    papers = state.repo.find_by_status("new", limit=200, sort_by="published", order=order, journal=journal_filter)
    
    # Filter by search query if provided
    if q:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.journal or "").lower()
            or q_lower in (p.authors or "").lower()
        ]
    
    # Filter by keywords
    if keywords:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        papers = _filter_by_keywords(papers, kw_list, keyword_mode)
    
    # Filter by date range
    if date_from or date_to:
        papers = _filter_by_date(papers, date_from or None, date_to or None, "published")
    
    # --- AI Ranking (badges only, no re-sort) ---
    scores: dict = {}
    top_ids: set = set()
    if state._ranking_computed and state._ranking_scores:
        scores = state._ranking_scores
        top_ids = state._ranking_top_ids
    elif not state._ranking_computing:
        # Kick off background computation so next reload has scores
        _start_ranking_bg()
    
    response = templates.TemplateResponse(
        "partials/paper_list.html",
        {
            "request": request,
            "papers": papers,
            "tab": "new",
            "empty_message": "새로운 논문이 없습니다. Fetch New 버튼을 클릭하세요.",
            "scores": scores,
            "top_ids": top_ids,
        },
    )
    response.headers["X-Paper-Count"] = str(len(papers))
    return response


@app.get("/papers/picked", response_class=HTMLResponse)
async def papers_picked(
    request: Request,
    q: str = Query(""),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
):
    """Get picked papers list (partial for HTMX)."""
    papers = state.repo.find_picked(limit=100, order=order)
    
    if q:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.journal or "").lower()
        ]
    
    # Filter by keywords
    if keywords:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        papers = _filter_by_keywords(papers, kw_list, keyword_mode)
    
    # Filter by date range
    if date_from or date_to:
        papers = _filter_by_date(papers, date_from or None, date_to or None, "published")
    
    response = templates.TemplateResponse(
        "partials/paper_list.html",
        {"request": request, "papers": papers, "tab": "picked", "empty_message": "선택된 논문이 없습니다."},
    )
    response.headers["X-Paper-Count"] = str(len(papers))
    return response


@app.get("/papers/archive", response_class=HTMLResponse)
async def papers_archive(
    request: Request,
    q: str = Query(""),
    journal: str = Query(""),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
):
    """Get archived papers list (partial for HTMX)."""
    journal_filter = journal if journal else None
    papers = state.repo.find_by_status("archived", limit=500, sort_by="date", order=order, journal=journal_filter)
    
    if q:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.journal or "").lower()
        ]
    
    # Filter by keywords
    if keywords:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        papers = _filter_by_keywords(papers, kw_list, keyword_mode)
    
    # Filter by date range
    if date_from or date_to:
        papers = _filter_by_date(papers, date_from or None, date_to or None, "published")
    
    response = templates.TemplateResponse(
        "partials/paper_list.html",
        {"request": request, "papers": papers, "tab": "archive", "empty_message": "아카이브된 논문이 없습니다."},
    )
    response.headers["X-Paper-Count"] = str(len(papers))
    return response


@app.get("/papers/read", response_class=HTMLResponse)
async def papers_read(
    request: Request,
    q: str = Query(""),
    journal: str = Query(""),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
):
    """Get read papers list (partial for HTMX). Sorted by created_at (read date)."""
    journal_filter = journal if journal else None
    papers = state.repo.find_by_status("read", limit=500, sort_by="created_at", order=order, journal=journal_filter)
    
    if q:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.journal or "").lower()
            or q_lower in (p.authors or "").lower()
        ]
    
    # Filter by keywords
    if keywords:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        papers = _filter_by_keywords(papers, kw_list, keyword_mode)
    
    # Filter by date range (using created_at for Read tab)
    if date_from or date_to:
        papers = _filter_by_date(papers, date_from or None, date_to or None, "created_at")
    
    response = templates.TemplateResponse(
        "partials/paper_list.html",
        {"request": request, "papers": papers, "tab": "read", "empty_message": "읽은 논문이 없습니다."},
    )
    response.headers["X-Paper-Count"] = str(len(papers))
    return response


@app.get("/papers/all", response_class=HTMLResponse)
async def papers_all(
    request: Request,
    q: str = Query(""),
    journal: str = Query(""),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
):
    """Get all papers in DB (partial for HTMX)."""
    journal_filter = journal if journal else None
    papers = state.repo.find_all(limit=500, sort_by="date", order=order, journal=journal_filter)
    
    if q:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.journal or "").lower()
            or q_lower in (p.authors or "").lower()
        ]
    
    # Filter by keywords
    if keywords:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        papers = _filter_by_keywords(papers, kw_list, keyword_mode)
    
    # Filter by date range
    if date_from or date_to:
        papers = _filter_by_date(papers, date_from or None, date_to or None, "published")
    
    response = templates.TemplateResponse(
        "partials/paper_list.html",
        {"request": request, "papers": papers, "tab": "all", "empty_message": "논문이 없습니다."},
    )
    response.headers["X-Paper-Count"] = str(len(papers))
    return response


# ============================================================================
# Paper Detail
# ============================================================================


def _parse_authors(authors_str: Optional[str]) -> list[str]:
    """Parse authors string into list (comma / semicolon / ' and ')."""
    if not authors_str or not authors_str.strip():
        return []
    s = authors_str.replace(" and ", ", ").replace(";", ",")
    return [a.strip() for a in s.split(",") if a.strip()]


@app.get("/papers/{paper_id}", response_class=HTMLResponse)
async def paper_detail(request: Request, paper_id: int):
    """Get paper detail view (partial for HTMX)."""
    paper = state.repo.find_by_id(paper_id)
    
    if not paper:
        return HTMLResponse("<div class='p-8 text-center text-content-muted'>논문을 찾을 수 없습니다.</div>")
    
    authors_list = _parse_authors(paper.authors)

    # AI match score (may be None if not yet computed)
    ai_score = state._ranking_scores.get(paper_id) if state._ranking_computed else None
    ai_is_top = paper_id in state._ranking_top_ids if state._ranking_computed else False

    return templates.TemplateResponse(
        "partials/detail.html",
        {
            "request": request,
            "paper": paper,
            "authors_list": authors_list,
            "ai_score": ai_score,
            "ai_is_top": ai_is_top,
        },
    )


@app.get("/papers/{paper_id}/enrich", response_class=HTMLResponse)
async def paper_detail_enrich(request: Request, paper_id: int):
    """Fetch OpenAlex metadata by DOI and return HTML fragment. Called when detail card is shown."""
    paper = state.repo.find_by_id(paper_id)
    if not paper or not paper.doi:
        return HTMLResponse(
            "<div class='mb-6 text-sm text-content-muted'>DOI가 없어 OpenAlex 정보를 불러올 수 없습니다.</div>"
        )

    data = await openalex_get_paper_info(paper.doi)
    if "error" in data:
        return templates.TemplateResponse(
            "partials/detail_enrich.html",
            {
                "request": request,
                "error": data["error"],
                "authors": [],
                "journal": "",
                "abstract": "",
            },
        )

    return templates.TemplateResponse(
        "partials/detail_enrich.html",
        {
            "request": request,
            "error": None,
            "authors": data.get("authors", []),
            "journal": data.get("journal", ""),
            "abstract": data.get("abstract", ""),
        },
    )


# ============================================================================
# AI Ranking Status (polling endpoint for toast)
# ============================================================================


@app.get("/actions/ranking-status")
async def ranking_status():
    """Return current ranking progress as JSON for frontend polling."""
    from fastapi.responses import JSONResponse

    return JSONResponse(state.ranking_status)


# ============================================================================
# Stats Endpoint
# ============================================================================


@app.get("/stats", response_class=HTMLResponse)
async def get_stats(request: Request):
    """Get stats partial for sidebar."""
    stats = state.repo.get_status_counts()
    return templates.TemplateResponse(
        "partials/stats.html",
        {"request": request, "stats": stats},
    )


@app.get("/stats/badges", response_class=HTMLResponse)
async def get_badges(request: Request):
    """Get badge counts as JSON for tab updates."""
    from fastapi.responses import JSONResponse
    stats = state.repo.get_status_counts()
    return JSONResponse({"new": stats.get("new", 0), "picked": stats.get("picked", 0)})


@app.get("/date-range")
async def get_date_range(
    tab: str = Query("new", description="Tab name: new, picked, archive, read, all"),
):
    """Get min/max date range for papers in the given tab."""
    from fastapi.responses import JSONResponse
    
    # Determine status and date field based on tab
    if tab == "read":
        date_field = "created_at"
        status = "read"
    elif tab == "new":
        date_field = "published"
        status = "new"
    elif tab == "archive":
        date_field = "published"
        status = "archived"
    elif tab == "picked":
        # Picked papers can have any status, get range from all picked
        date_field = "published"
        status = None  # Will need special handling
    else:  # all
        date_field = "published"
        status = None
    
    date_range = state.repo.get_date_range(status=status, date_field=date_field)
    return JSONResponse({
        "min_date": date_range["min_date"],
        "max_date": date_range["max_date"],
        "date_field": "collected" if tab == "read" else "published",
    })


# ============================================================================
# Actions: Fetch, Export, Pick/Unpick
# ============================================================================


def _do_fetch():
    """Background task to fetch papers."""
    state.fetch_status = {"running": True, "message": "RSS 피드 수집 중...", "complete": False}
    
    try:
        archived_ids = state.repo.archive_old_new()
        total_new = 0
        total_processed = 0
        workers = min(8, (os.cpu_count() or 2) - 1)
        workers = max(1, workers)
        
        for paper in state.feed_service.fetch_all(max_workers=workers):
            if state.repo.upsert(paper):
                total_new += 1
            total_processed += 1
        
        # If no new papers found, restore previously archived papers
        if total_new == 0 and archived_ids:
            restored = state.repo.restore_to_new(archived_ids)
            state.fetch_status = {
                "running": False,
                "message": f"신규 논문 없음 ({restored}개 복원됨)",
                "complete": True,
            }
        else:
            archived_count = len(archived_ids)

            # Compute AI match scores before signalling completion
            _invalidate_rankings()
            state.fetch_status = {"running": True, "message": "AI 매칭 점수 계산 중...", "complete": False}
            state._ranking_computing = True
            _compute_rankings()

            state.fetch_status = {
                "running": False,
                "message": f"완료: {total_new}개 신규, {total_processed}개 처리됨" + (f" ({archived_count}개 아카이브됨)" if archived_count > 0 else ""),
                "complete": True,
            }
    except Exception as e:
        state.fetch_status = {"running": False, "message": f"오류: {str(e)}", "complete": True}


@app.post("/actions/fetch", response_class=HTMLResponse)
async def fetch_papers(request: Request, background_tasks: BackgroundTasks):
    """Start fetching new papers."""
    if state.fetch_status.get("running"):
        return templates.TemplateResponse(
            "partials/fetch_status.html",
            {"request": request, "status": state.fetch_status},
        )
    
    background_tasks.add_task(_do_fetch)
    state.fetch_status = {"running": True, "message": "시작 중...", "complete": False}
    
    return templates.TemplateResponse(
        "partials/fetch_status.html",
        {"request": request, "status": state.fetch_status},
    )


@app.get("/actions/fetch-status", response_class=HTMLResponse)
async def fetch_status(request: Request):
    """Get current fetch status."""
    response = templates.TemplateResponse(
        "partials/fetch_status.html",
        {"request": request, "status": state.fetch_status},
    )
    # Trigger events when fetch completes
    if state.fetch_status.get("complete"):
        response.headers["HX-Trigger"] = "fetchComplete, statsUpdated"
    return response


@app.post("/actions/export", response_class=HTMLResponse)
async def export_picked(request: Request):
    """Export picked papers to markdown."""
    picked_papers = state.repo.find_picked()
    
    if not picked_papers:
        return HTMLResponse(
            """<div class="toast toast-warning toast-auto">
                <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path>
                </svg>
                <span>선택된 논문이 없습니다.</span>
            </div>"""
        )
    
    try:
        filepath = state.exporter.export(picked_papers)
        paper_ids = [p.id for p in picked_papers if p.id is not None]
        state.repo.mark_exported(paper_ids)
        _invalidate_rankings()  # library changed → recompute next time
        
        return HTMLResponse(
            f"""<div class="toast toast-success toast-auto">
                <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                </svg>
                <span>{len(picked_papers)}개 논문 내보내기 완료!</span>
            </div>"""
        )
    except Exception as e:
        return HTMLResponse(
            f"""<div class="toast toast-error toast-auto">
                <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                </svg>
                <span>오류: {str(e)}</span>
            </div>"""
        )


@app.post("/actions/pick/{paper_id}", response_class=HTMLResponse)
async def pick_paper(request: Request, paper_id: int):
    """Toggle pick status for a paper."""
    from fastapi.responses import Response
    
    # Check current status
    paper = state.repo.find_by_id(paper_id)
    
    if paper:
        if paper.is_picked:
            state.repo.unpick([paper_id])
        else:
            state.repo.pick([paper_id])
    
    new_picked = not paper.is_picked if paper else False
    
    # Distinguish list (checkbox) vs detail panel (button) via custom header
    pick_context = request.headers.get("X-Pick-Context", "")
    from_detail = pick_context.strip().lower() == "detail"
    
    if from_detail:
        # Return updated button for detail view (must include hx-headers so next click still sends context)
        if new_picked:
            html = f"""<button hx-post="/actions/pick/{paper_id}"
                    hx-headers='{{"X-Pick-Context": "detail"}}'
                    hx-swap="outerHTML"
                    class="flex-shrink-0 px-4 py-2 rounded-lg font-semibold text-sm transition-all
                           bg-accent-warning-subtle text-accent-warning hover:opacity-80">
                <span class="flex items-center gap-1.5">
                    <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 17.27L18.18 21l-1.64-7.03L22 9.24l-7.19-.61L12 2 9.19 8.63 2 9.24l5.46 4.73L5.82 21z"/>
                    </svg>
                    Picked
                </span>
            </button>"""
        else:
            html = f"""<button hx-post="/actions/pick/{paper_id}"
                    hx-headers='{{"X-Pick-Context": "detail"}}'
                    hx-swap="outerHTML"
                    class="flex-shrink-0 px-4 py-2 rounded-lg font-semibold text-sm transition-all
                           bg-accent-muted-subtle text-content-secondary hover:bg-accent-primary-subtle hover:text-accent-primary">
                <span class="flex items-center gap-1.5">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"/>
                    </svg>
                    Pick
                </span>
            </button>"""
    else:
        # Return updated checkbox for list view (keep hx-target so response never hits detail-view)
        html = f"""<input type="checkbox" 
            class="w-4 h-4 rounded border-border-base text-accent-primary focus:ring-accent-primary cursor-pointer" 
            hx-post="/actions/pick/{paper_id}"
            hx-target="this"
            hx-swap="outerHTML"
            hx-trigger="click"
            hx-headers='{{"X-Pick-Context": "list"}}'
            {"checked" if new_picked else ""}>"""
    
    # Return with HX-Trigger header to update stats
    response = Response(content=html, media_type="text/html")
    response.headers["HX-Trigger"] = "statsUpdated"
    return response


@app.post("/actions/undo-read/{paper_id}", response_class=HTMLResponse)
async def undo_read(request: Request, paper_id: int):
    """Undo read: move paper to archived + picked, show toast with revert."""
    from fastapi.responses import Response

    paper = state.repo.find_by_id(paper_id)
    if not paper:
        return HTMLResponse('<div class="toast toast-error toast-auto"><span>논문을 찾을 수 없습니다.</span></div>')

    state.repo.undo_read(paper_id)
    title_short = (paper.title or "")[:40]
    if len(paper.title or "") > 40:
        title_short += "…"

    toast_html = f"""<div class="toast toast-info" id="undo-read-toast-{paper_id}" style="animation: toastSlideIn 0.3s ease-out;">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
        </svg>
        <span class="flex-1 text-sm">논문을 'Archive'로 옮겼습니다.</span>
        <button onclick="revertUndoRead({paper_id}, this)"
                class="ml-2 px-2 py-0.5 rounded font-semibold text-xs underline underline-offset-2 hover:opacity-80 transition-opacity flex-shrink-0">
            되돌리기
        </button>
    </div>"""

    response = Response(content=toast_html, media_type="text/html")
    response.headers["HX-Trigger"] = "statsUpdated"
    return response


@app.post("/actions/revert-undo-read/{paper_id}", response_class=HTMLResponse)
async def revert_undo_read(request: Request, paper_id: int):
    """Revert undo-read: move paper back to read, is_picked=0."""
    from fastapi.responses import Response

    state.repo.revert_undo_read(paper_id)

    toast_html = f"""<div class="toast toast-success toast-auto">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 14l-4-4m0 0l4-4m-4 4h11.586a2 2 0 012 2v2"></path>
        </svg>
        <span class="text-sm">논문을 'Read'로 되돌렸습니다.</span>
    </div>"""

    response = Response(content=toast_html, media_type="text/html")
    response.headers["HX-Trigger"] = "statsUpdated"
    return response


@app.post("/actions/pick-all", response_class=HTMLResponse)
async def pick_all(request: Request, ids: str = Form(...)):
    """Pick multiple papers by IDs."""
    try:
        paper_ids = [int(x) for x in ids.split(",") if x.strip()]
        if paper_ids:
            state.repo.pick(paper_ids)
        return HTMLResponse('<span class="text-green-600 text-sm">선택됨!</span>')
    except Exception as e:
        return HTMLResponse(f'<span class="text-red-600 text-sm">오류: {str(e)}</span>')


@app.post("/actions/unpick-all", response_class=HTMLResponse)
async def unpick_all(request: Request, ids: str = Form(...)):
    """Unpick multiple papers by IDs."""
    try:
        paper_ids = [int(x) for x in ids.split(",") if x.strip()]
        if paper_ids:
            state.repo.unpick(paper_ids)
        return HTMLResponse('<span class="text-green-600 text-sm">선택 해제됨!</span>')
    except Exception as e:
        return HTMLResponse(f'<span class="text-red-600 text-sm">오류: {str(e)}</span>')


# ============================================================================
# AI Ranking Trigger
# ============================================================================


@app.post("/actions/rank", response_class=HTMLResponse)
async def trigger_ranking(request: Request):
    """Manually trigger AI ranking computation."""
    if state._ranking_computing:
        return HTMLResponse(
            """<div class="toast toast-info toast-auto">
                <span>AI 매칭 점수를 이미 계산 중입니다…</span>
            </div>"""
        )
    _start_ranking_bg()
    return HTMLResponse(
        """<div class="toast toast-info toast-auto">
            <span>AI 매칭 점수 계산을 시작했습니다. 잠시 후 새로고침하세요.</span>
        </div>"""
    )


# ============================================================================
# Journal Filter
# ============================================================================


@app.get("/journals", response_class=HTMLResponse)
async def get_journals(request: Request):
    """Get journal options for filter dropdown."""
    journals = state.repo.get_distinct_journals()
    options = ['<option value="">전체 저널</option>']
    for j in journals:
        options.append(f'<option value="{j}">{j}</option>')
    return HTMLResponse("\n".join(options))