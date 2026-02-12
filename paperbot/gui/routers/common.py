"""Common routes: index page, stats, badges, date-range, journals, LLM models/profiles."""

from dataclasses import asdict

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from paperbot import __version__
from paperbot.config import LLMProfile, load_llm_models, save_email, save_llm_profiles
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


@router.get("/preferences", response_class=HTMLResponse)
async def preferences(request: Request):
    """Preferences modal fragment (loaded via HTMX)."""
    return templates.TemplateResponse(
        "preferences.html",
        {"request": request},
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
    """Get badge counts as JSON for tab updates and export dropdown."""
    stats = state.repo.get_status_counts()
    return JSONResponse({
        "new": stats.get("new", 0),
        "picked": stats.get("picked", 0),
        "read": stats.get("read", 0),
    })


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


# ============================================================================
# Email
# ============================================================================


class EmailPayload(BaseModel):
    """Request body for updating the contact email."""
    email: str


@router.get("/api/email")
async def get_email():
    """Return the current contact email."""
    return JSONResponse({"email": state.settings.contact_email or ""})


@router.put("/api/email")
async def update_email(body: EmailPayload):
    """Update the contact email and persist to ``email.yaml``."""
    email = body.email.strip() or None
    state.settings.contact_email = email
    # Crossref 서비스에도 반영
    if hasattr(state, "crossref") and state.crossref is not None:
        state.crossref.contact_email = email
    save_email(state.settings.metadata_dir / "email.yaml", email)
    return JSONResponse({"email": email or ""})


# ============================================================================
# LLM Model Registry
# ============================================================================


@router.get("/api/llm-models")
async def get_llm_models():
    """Return the built-in LLM model registry as JSON.

    Used by the preferences UI to dynamically populate the model
    ``<select>`` dropdown.
    """
    models = load_llm_models()
    return JSONResponse([asdict(m) for m in models])


# ============================================================================
# LLM Profile CRUD  (/api/llm-profiles)
# ============================================================================


class ProfilePayload(BaseModel):
    """Request body for creating / updating an LLM profile."""
    name: str
    model: str
    api_key: str


class SetActivePayload(BaseModel):
    """Request body for setting the active profile."""
    id: str | None = None


def _profiles_path():
    return state.settings.metadata_dir / "llm_profiles.yaml"


def _sync_to_disk() -> None:
    """Write current in-memory profiles to ``llm_profiles.yaml``."""
    save_llm_profiles(
        _profiles_path(),
        state.settings.llm_profiles,
        state.settings.active_llm_id,
    )


@router.get("/api/llm-profiles")
async def list_profiles():
    """Return all LLM profiles and the active profile id."""
    s = state.settings
    return JSONResponse({
        "active": s.active_llm_id,
        "profiles": [
            {"id": p.id, "name": p.name, "model": p.model, "api_key": p.api_key}
            for p in s.llm_profiles
        ],
    })


@router.post("/api/llm-profiles")
async def create_profile(body: ProfilePayload):
    """Create a new LLM profile.  Returns the created profile."""
    import uuid
    pid = str(uuid.uuid4())[:8]
    profile = LLMProfile(id=pid, name=body.name, model=body.model, api_key=body.api_key)
    s = state.settings
    s.llm_profiles.append(profile)
    # 첫 프로필이면 자동으로 active 설정
    if s.active_llm_id is None:
        s.active_llm_id = pid
    _sync_to_disk()
    return JSONResponse(
        {"id": profile.id, "name": profile.name, "model": profile.model, "api_key": profile.api_key},
        status_code=201,
    )


# NOTE: /active 는 /{profile_id} 보다 먼저 등록해야 "active" 가 path param 으로 캡처되지 않음
@router.put("/api/llm-profiles/active")
async def set_active_profile(body: SetActivePayload):
    """Set which profile is currently active."""
    s = state.settings
    if body.id is not None:
        exists = any(p.id == body.id for p in s.llm_profiles)
        if not exists:
            return JSONResponse({"error": "Profile not found"}, status_code=404)
    s.active_llm_id = body.id
    _sync_to_disk()
    return JSONResponse({"active": s.active_llm_id})


@router.put("/api/llm-profiles/{profile_id}")
async def update_profile(profile_id: str, body: ProfilePayload):
    """Update an existing LLM profile."""
    s = state.settings
    target = next((p for p in s.llm_profiles if p.id == profile_id), None)
    if target is None:
        return JSONResponse({"error": "Profile not found"}, status_code=404)
    target.name = body.name
    target.model = body.model
    target.api_key = body.api_key
    _sync_to_disk()
    return JSONResponse({"id": target.id, "name": target.name, "model": target.model, "api_key": target.api_key})


@router.delete("/api/llm-profiles/{profile_id}")
async def delete_profile(profile_id: str):
    """Delete an LLM profile."""
    s = state.settings
    before = len(s.llm_profiles)
    s.llm_profiles = [p for p in s.llm_profiles if p.id != profile_id]
    if len(s.llm_profiles) == before:
        return JSONResponse({"error": "Profile not found"}, status_code=404)
    if s.active_llm_id == profile_id:
        s.active_llm_id = s.llm_profiles[0].id if s.llm_profiles else None
    _sync_to_disk()
    return JSONResponse({"ok": True})
