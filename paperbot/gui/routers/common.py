"""Common routes: index page, stats, badges, date-range, journals."""

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from paperbot import __version__
from paperbot.gui.state import state, templates

router = APIRouter()


# ============================================================================
# Main Page
# ============================================================================


@router.get("/", response_class=HTMLResponse)
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
            "version": __version__,
        },
    )


# ============================================================================
# Stats
# ============================================================================


@router.get("/stats", response_class=HTMLResponse)
async def get_stats(request: Request):
    """Get stats partial for sidebar."""
    stats = state.repo.get_status_counts()
    return templates.TemplateResponse(
        "partials/stats.html",
        {"request": request, "stats": stats},
    )


@router.get("/stats/badges", response_class=HTMLResponse)
async def get_badges(request: Request):
    """Get badge counts as JSON for tab updates."""
    stats = state.repo.get_status_counts()
    return JSONResponse({"new": stats.get("new", 0), "picked": stats.get("picked", 0)})


@router.get("/date-range")
async def get_date_range(
    tab: str = Query("new", description="Tab name: new, picked, archive, read, all"),
):
    """Get min/max date range for papers in the given tab."""
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
# AI Ranking Status (polling endpoint for toast)
# ============================================================================


@router.get("/actions/ranking-status")
async def ranking_status():
    """Return current ranking progress as JSON for frontend polling."""
    return JSONResponse(state.ranking_status)


# ============================================================================
# Journal Filter
# ============================================================================


@router.get("/journals", response_class=HTMLResponse)
async def get_journals(request: Request):
    """Get journal options for filter dropdown."""
    journals = state.repo.get_distinct_journals()
    options = ['<option value="">전체 저널</option>']
    for j in journals:
        options.append(f'<option value="{j}">{j}</option>')
    return HTMLResponse("\n".join(options))
