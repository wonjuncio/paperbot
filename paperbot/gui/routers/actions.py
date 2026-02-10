"""Action routes: fetch, export, pick/unpick, undo-read, AI ranking trigger."""

import os

from fastapi import APIRouter, BackgroundTasks, Form, Request
from fastapi.responses import HTMLResponse, Response

from paperbot.gui.state import (
    compute_rankings,
    invalidate_rankings,
    start_ranking_bg,
    state,
    templates,
)

router = APIRouter()


# ============================================================================
# Fetch
# ============================================================================


def _do_fetch():
    """Background task to fetch papers.

    Order: fetch first -> if new papers found, archive old ones -> upsert new.
    If no new papers, do nothing (old papers stay as-is).
    """
    state.fetch_status = {"running": True, "message": "RSS 피드 수집 중...", "complete": False}

    try:
        # Step 1: Fetch papers into a temporary list (don't touch DB yet)
        workers = min(8, (os.cpu_count() or 2) - 1)
        workers = max(1, workers)

        fetched_papers: list = []
        for paper in state.feed_service.fetch_all(max_workers=workers):
            fetched_papers.append(paper)

        # Step 2: Snapshot current 'new' paper IDs before upserting
        old_new_ids = state.repo.get_new_paper_ids()

        # Step 3: Upsert fetched papers — count genuinely new ones
        total_new = 0
        for paper in fetched_papers:
            if state.repo.upsert(paper):
                total_new += 1

        # Step 4: If no new papers, do nothing — keep existing New tab as-is
        if total_new == 0:
            state.fetch_status = {
                "running": False,
                "message": "신규 논문 없음 (기존 목록 유지)",
                "complete": True,
            }
            return

        # Step 5: New papers found -> archive old 'new' papers
        archived_count = state.repo.archive_by_ids(old_new_ids)

        # Step 6: Compute AI match scores
        invalidate_rankings()
        state.fetch_status = {"running": True, "message": "AI 매칭 점수 계산 중...", "complete": False}
        state._ranking_computing = True
        compute_rankings()

        state.fetch_status = {
            "running": False,
            "message": f"완료: {total_new}개 신규, {len(fetched_papers)}개 처리됨" + (f" ({archived_count}개 아카이브됨)" if archived_count > 0 else ""),
            "complete": True,
        }
    except Exception as e:
        state.fetch_status = {"running": False, "message": f"오류: {str(e)}", "complete": True}


@router.post("/actions/fetch", response_class=HTMLResponse)
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


@router.get("/actions/fetch-status", response_class=HTMLResponse)
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


# ============================================================================
# Export
# ============================================================================


@router.post("/actions/export", response_class=HTMLResponse)
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
        invalidate_rankings()  # library changed -> recompute next time

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


# ============================================================================
# Pick / Unpick
# ============================================================================


@router.post("/actions/pick/{paper_id}", response_class=HTMLResponse)
async def pick_paper(request: Request, paper_id: int):
    """Toggle pick status for a paper."""
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
        # Return updated button for detail view
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
        # Return updated checkbox for list view
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


# ============================================================================
# Undo Read
# ============================================================================


@router.post("/actions/undo-read/{paper_id}", response_class=HTMLResponse)
async def undo_read(request: Request, paper_id: int):
    """Undo read: move paper to archived + picked, show toast with revert."""
    paper = state.repo.find_by_id(paper_id)
    if not paper:
        return HTMLResponse('<div class="toast toast-error toast-auto"><span>논문을 찾을 수 없습니다.</span></div>')

    state.repo.undo_read(paper_id)
    title_short = (paper.title or "")[:40]
    if len(paper.title or "") > 40:
        title_short += "…"

    toast_html = f"""<div class="toast toast-info" id="undo-read-toast-{paper_id}" style="animation: toastSlideIn 0.5s cubic-bezier(0.22, 1, 0.36, 1);">
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


@router.post("/actions/revert-undo-read/{paper_id}", response_class=HTMLResponse)
async def revert_undo_read(request: Request, paper_id: int):
    """Revert undo-read: move paper back to read, is_picked=0."""
    state.repo.revert_undo_read(paper_id)

    toast_html = """<div class="toast toast-success toast-auto">
        <svg class="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 14l-4-4m0 0l4-4m-4 4h11.586a2 2 0 012 2v2"></path>
        </svg>
        <span class="text-sm">논문을 'Read'로 되돌렸습니다.</span>
    </div>"""

    response = Response(content=toast_html, media_type="text/html")
    response.headers["HX-Trigger"] = "statsUpdated"
    return response


@router.post("/actions/pick-all", response_class=HTMLResponse)
async def pick_all(request: Request, ids: str = Form(...)):
    """Pick multiple papers by IDs."""
    try:
        paper_ids = [int(x) for x in ids.split(",") if x.strip()]
        if paper_ids:
            state.repo.pick(paper_ids)
        return HTMLResponse('<span class="text-green-600 text-sm">선택됨!</span>')
    except Exception as e:
        return HTMLResponse(f'<span class="text-red-600 text-sm">오류: {str(e)}</span>')


@router.post("/actions/unpick-all", response_class=HTMLResponse)
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


@router.post("/actions/rank", response_class=HTMLResponse)
async def trigger_ranking(request: Request):
    """Manually trigger AI ranking computation."""
    if state._ranking_computing:
        return HTMLResponse(
            """<div class="toast toast-info toast-auto">
                <span>AI 매칭 점수를 이미 계산 중입니다…</span>
            </div>"""
        )
    start_ranking_bg()
    return HTMLResponse(
        """<div class="toast toast-info toast-auto">
            <span>AI 매칭 점수 계산을 시작했습니다. 잠시 후 새로고침하세요.</span>
        </div>"""
    )
