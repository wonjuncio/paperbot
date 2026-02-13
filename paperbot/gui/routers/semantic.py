"""Semantic Map API endpoints."""

from __future__ import annotations

import threading

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse

from paperbot.gui.state import state

router = APIRouter(prefix="/api", tags=["semantic"])


# ============================================================================
# Semantic Map Data
# ============================================================================


@router.get("/semantic-map")
async def semantic_map(
    status: str = Query("all", description="Paper status filter: new, read, archived, all"),
):
    """Return full semantic map data (2D/3D coords, clusters, top-3 edges).

    Heavy computation runs in a background thread; this endpoint returns
    the current status or cached result.
    """
    smap = state.semantic_map_service
    if smap is None:
        return JSONResponse(
            {"status": "error", "message": "Semantic map service not initialised"},
            status_code=503,
        )

    # Check if computation is in progress
    if getattr(state, "_smap_computing", False):
        return JSONResponse({
            "status": "computing",
            "message": state.smap_status.get("message", "계산 중…"),
        })

    # Return cached result if available
    cached = getattr(state, "_smap_cache", None)
    if cached is not None:
        cached_status = getattr(state, "_smap_cache_status", None)
        if cached_status == status:
            return JSONResponse({"status": "ready", "data": cached})

    # Start background computation
    _start_smap_bg(status)
    return JSONResponse({
        "status": "computing",
        "message": "Semantic map 계산을 시작합니다…",
    })


@router.get("/semantic-map/status")
async def semantic_map_status():
    """Poll computation progress."""
    if getattr(state, "_smap_computing", False):
        return JSONResponse({
            "status": "computing",
            "message": state.smap_status.get("message", "계산 중…"),
        })

    cached = getattr(state, "_smap_cache", None)
    if cached is not None:
        return JSONResponse({"status": "ready", "data": cached})

    return JSONResponse({"status": "idle"})


# ============================================================================
# Background computation
# ============================================================================


def _start_smap_bg(status: str) -> None:
    """Kick off semantic map computation in a background thread."""
    if getattr(state, "_smap_computing", False):
        return
    state._smap_computing = True
    state.smap_status = {"phase": "starting", "message": "Semantic map 준비 중…"}
    threading.Thread(
        target=_compute_smap, args=(status,), daemon=True,
    ).start()


def _compute_smap(status: str) -> None:
    """Run the full semantic map pipeline (background thread)."""
    try:
        state.smap_status = {"phase": "loading", "message": "논문 데이터 로딩 중…"}

        # Gather papers based on status filter
        if status == "all":
            papers = state.repo.find_all(limit=2000)
        elif status == "read":
            papers = state.repo.find_by_status("read", limit=2000)
        elif status == "new":
            papers = state.repo.find_by_status("new", limit=2000)
        elif status == "archived":
            papers = state.repo.find_by_status("archived", limit=2000)
        else:
            papers = state.repo.find_all(limit=2000)

        if not papers:
            state._smap_cache = {"points": [], "n_papers": 0}
            state._smap_cache_status = status
            state.smap_status = {"phase": "done", "message": "논문이 없습니다."}
            return

        n = len(papers)
        state.smap_status = {
            "phase": "embedding",
            "message": f"{n}개 논문 임베딩 처리 중…",
        }

        # Ensure model is loaded
        _ = state.ranker.bi_encoder

        state.smap_status = {
            "phase": "computing",
            "message": f"{n}개 논문 계산 중…",
        }

        result = state.semantic_map_service.generate(papers)

        # Serialise to dict for JSON response
        data = {
            "points": [
                {
                    "id": pt.id,
                    "title": pt.title,
                    "journal": pt.journal,
                    "published": pt.published,
                    "status": pt.status,
                    "x2": pt.x2, "y2": pt.y2,
                    "x3": pt.x3, "y3": pt.y3, "z3": pt.z3,
                    "clusters": pt.clusters,
                    "top3": pt.top3,
                }
                for pt in result.points
            ],
            "n_papers": result.n_papers,
        }

        state._smap_cache = data
        state._smap_cache_status = status
        state.smap_status = {
            "phase": "done",
            "message": f"{n}개 논문 semantic map 완료",
        }

    except Exception as e:
        state.smap_status = {
            "phase": "error",
            "message": f"Semantic map 오류: {e}",
        }
    finally:
        state._smap_computing = False
