"""Application state, templates, and AI ranking helpers."""

import html
import os
import threading
from typing import Optional

from fastapi.templating import Jinja2Templates

from paperbot import __version__
from paperbot.config import Settings
from paperbot.database.repository import PaperRepository
from paperbot.services.crossref_service import CrossrefService
from paperbot.services.export_service import MarkdownExporter
from paperbot.services.feed_service import FeedService
from paperbot.services.ranking_service import RankingService


# ============================================================================
# Global State
# ============================================================================


class AppState:
    """Mutable singleton holding all runtime services and AI ranking state."""

    settings: Settings
    repo: PaperRepository
    crossref: CrossrefService
    feed_service: FeedService
    exporter: MarkdownExporter
    fetch_status: dict = {"running": False, "message": "", "complete": False}
    # AI ranking
    ranker: Optional[RankingService] = None
    _ranking_scores: dict = {}       # paper_id → match %
    _ranking_top_ids: set = set()    # gold shimmer (top 5)
    _ranking_gold_ids: set = set()   # gold (ranks 6–15)
    _ranking_blue_ids: set = set()   # blue (ranks 16–30)
    _ranking_computed: bool = False
    _ranking_computing: bool = False
    # Ranking progress toast state
    ranking_status: dict = {"phase": "idle", "message": ""}


state = AppState()


# ============================================================================
# Templates & Filters
# ============================================================================

base_dir = os.path.dirname(__file__)
templates = Jinja2Templates(directory=os.path.join(base_dir, "templates"))

# Unescape HTML entities in abstracts
templates.env.filters["unescape_html"] = lambda s: html.unescape(s) if s else ""

# Clean abstract: remove "Abstract" prefix + LaTeX → plain text
from paperbot.utils.text import clean_abstract  # noqa: E402

templates.env.filters["clean_abstract"] = clean_abstract


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


# ============================================================================
# AI Ranking Helpers
# ============================================================================


def preload_models() -> None:
    """Pre-load AI models into RAM at startup (background thread).

    Only loads the Bi-Encoder (used for ranking).  Cross-Encoder is
    loaded lazily when the user first views the AI Insight panel.
    """
    try:
        state.ranking_status = {
            "phase": "loading",
            "message": "AI 모델 로딩 중…",
        }
        _ = state.ranker.bi_encoder
        state.ranking_status = {"phase": "idle", "message": ""}
    except Exception as e:
        print(f"[PaperBot] Model pre-load failed: {e}")
        state.ranking_status = {"phase": "idle", "message": ""}


def compute_rankings() -> None:
    """Compute AI match scores for NEW papers against the READ library.

    Score = library-distribution percentile of each new paper's best
    cosine similarity to any read paper.  No Cross-Encoder inference.

    Uses a persistent DB cache keyed by ``library_hash`` so that:

    * **Restart** → scores loaded from cache; no model inference.
    * **New papers added (library unchanged)** → only uncached scored.
    * **Library changed** → hash mismatch forces full recomputation.

    Always call from a background thread.
    """
    try:
        new_papers = state.repo.find_by_status("new", limit=500, sort_by="date", order="desc")
        read_papers = state.repo.find_by_status("read", limit=5000)

        if not read_papers or not new_papers:
            state._ranking_scores = {}
            state._ranking_top_ids = set()
            state._ranking_gold_ids = set()
            state._ranking_blue_ids = set()
            state._ranking_computed = True
            state.ranking_status = {"phase": "idle", "message": ""}
            return

        # ── Check persistent cache ────────────────────────────────────
        library_hash = f"{__version__}:{state.repo.get_library_hash()}"
        cached = state.repo.load_ranking_cache(library_hash)

        new_paper_ids = {p.id for p in new_papers}
        cached_hits = {
            pid: score_stage
            for pid, score_stage in cached.items()
            if pid in new_paper_ids
        }
        uncached_papers = [p for p in new_papers if p.id not in cached_hits]

        if not uncached_papers:
            # Full cache hit
            all_scores: dict[int, float] = {
                pid: score for pid, (score, _stage) in cached_hits.items()
            }
            state._ranking_scores = all_scores
            _set_top_ids(all_scores)
            state._ranking_computed = True
            state.ranking_status = {
                "phase": "done",
                "message": f"캐시에서 {len(cached_hits)}개 논문 점수 로드 완료",
            }
            return

        # ── Load Bi-Encoder (CE not needed for ranking) ───────────────
        missing = state.ranker.needs_download()
        bi_missing = [m for m in missing if "reranker" not in m]
        if bi_missing:
            for name in bi_missing:
                state.ranking_status = {
                    "phase": "downloading",
                    "message": f'"{name}" 모델 다운로드 중…',
                }
                _ = state.ranker.bi_encoder
        elif state.ranker._bi_encoder is None:
            state.ranking_status = {
                "phase": "loading",
                "message": "AI 모델 로딩 중…",
            }
            _ = state.ranker.bi_encoder

        # ── Incremental scoring (only uncached papers) ────────────────
        n_cached = len(cached_hits)
        n_uncached = len(uncached_papers)
        msg_prefix = (
            f"{n_uncached}개 신규 논문"
            if n_cached
            else f"{n_uncached}개 논문"
        )
        state.ranking_status = {
            "phase": "scoring",
            "message": f"{msg_prefix} AI 매칭 점수 계산 중…",
        }

        # Library distribution may need recomputation when library
        # changes; invalidate so rank() recomputes from current embs.
        state.ranker._lib_dist = None

        ranked = state.ranker.rank(uncached_papers, read_papers)

        # Persist newly computed scores
        new_entries = [
            (r.paper.id, r.score, r.stage) for r in ranked if r.paper.id is not None
        ]
        state.repo.save_ranking_cache(new_entries, library_hash)

        # ── Merge cached + newly computed ─────────────────────────────
        all_scores = {
            pid: score for pid, (score, _stage) in cached_hits.items()
        }
        for r in ranked:
            if r.paper.id is not None:
                all_scores[r.paper.id] = r.score

        state._ranking_scores = all_scores
        _set_top_ids(all_scores)
        state._ranking_computed = True

        state.ranking_status = {
            "phase": "done",
            "message": (
                f"{n_uncached}개 논문 AI 매칭 완료"
                + (f" (캐시 {n_cached}개 재사용)" if n_cached else "")
            ),
        }
    except Exception as e:
        print(f"[PaperBot] AI ranking failed: {e}")
        state.ranking_status = {"phase": "error", "message": f"AI 랭킹 오류: {e}"}
    finally:
        state._ranking_computing = False


def _set_top_ids(scores: dict[int, float]) -> None:
    """Compute badge tier sets from *scores* (priority system).

    Only papers with **score >= 60** are eligible for badges.
    Among eligible papers (sorted by score descending):

    * **Gold shimmer** — top 3
    * **Gold**         — ranks 4–5, OR any paper > 85 (except top 3)
    * **Blue**         — ranks 6–10 (only if score <= 85)
    * **None**         — score < 60, or rank > 10 with score <= 85
    """
    eligible = [(pid, sc) for pid, sc in scores.items() if sc >= 60.0]
    eligible.sort(key=lambda x: x[1], reverse=True)

    shimmer: set[int] = set()
    gold: set[int] = set()
    blue: set[int] = set()

    for rank, (pid, sc) in enumerate(eligible, 1):
        if rank <= 3:
            shimmer.add(pid)
        elif sc > 85.0:
            gold.add(pid)          # 85 초과면 순위 무관 Gold
        elif rank <= 5:
            gold.add(pid)
        elif rank <= 10:
            blue.add(pid)
        # else: no badge

    state._ranking_top_ids = shimmer
    state._ranking_gold_ids = gold
    state._ranking_blue_ids = blue


def start_ranking_bg() -> None:
    """Kick off ranking computation in a daemon thread (non-blocking).

    Sets ``ranking_status`` to ``"starting"`` **before** spawning the
    thread so the frontend polling sees a non-idle phase immediately
    and keeps the spinner toast alive.
    """
    if state._ranking_computing:
        return
    state._ranking_computing = True
    state.ranking_status = {
        "phase": "starting",
        "message": "AI 매칭 준비 중…",
    }
    threading.Thread(target=compute_rankings, daemon=True).start()


def invalidate_rankings() -> None:
    """Clear in-memory ranking state.

    The persistent DB cache (``ranking_cache`` table) is **preserved**.
    Its validity is checked at computation time via ``library_hash``:
    if the READ library changed, the hash won't match and the cache is
    naturally bypassed.  This means a restart or invalidation does NOT
    require wiping the on-disk cache.
    """
    state._ranking_scores = {}
    state._ranking_top_ids = set()
    state._ranking_gold_ids = set()
    state._ranking_blue_ids = set()
    state._ranking_computed = False
    # NOTE: Do NOT call ranker.invalidate_cache() here.
    # Library changes are handled incrementally by _get_lib_embeddings();
    # invalidate_cache() would wipe all persisted embeddings needlessly.
