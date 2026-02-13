"""Paper list endpoints (HTMX partials) and paper detail views."""

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse

from paperbot.gui.helpers import (
    filter_by_date,
    filter_by_keywords,
    parse_authors,
    sort_papers,
)
from paperbot.gui.state import start_ranking_bg, state, templates
from paperbot.services.openalex_service import get_paper_info as openalex_get_paper_info

router = APIRouter()

# Page size for lazy-loaded list (same for initial and append)
PAGE_SIZE = 100


# ============================================================================
# Paper List Endpoints
# ============================================================================


def _paper_list_response(
    request: Request,
    papers: list,
    tab: str,
    empty_message: str,
    offset: int,
    append: bool,
    has_more: bool,
    **template_ctx,
) -> HTMLResponse:
    """Build list or cards-only response and set pagination headers."""
    if append:
        r = templates.TemplateResponse(
            "partials/paper_list_cards.html",
            {"request": request, "papers": papers, "tab": tab, **template_ctx},
        )
    else:
        r = templates.TemplateResponse(
            "partials/paper_list.html",
            {
                "request": request,
                "papers": papers,
                "tab": tab,
                "empty_message": empty_message,
                **template_ctx,
            },
        )
    r.headers["X-Paper-Count"] = str(len(papers))
    r.headers["X-Has-More"] = "true" if has_more else "false"
    r.headers["X-Next-Offset"] = str(offset + PAGE_SIZE)
    return r


@router.get("/papers/new", response_class=HTMLResponse)
async def papers_new(
    request: Request,
    q: str = Query("", description="Search query"),
    journal: str = Query("", description="Journal filter"),
    sort_by: str = Query("score", description="Sort criteria"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
    offset: int = Query(0, description="Pagination offset for lazy load"),
):
    """Get new papers list (partial for HTMX)."""
    journal_filter = journal if journal else None
    papers = state.repo.find_by_status(
        "new", limit=PAGE_SIZE, offset=offset, sort_by="published", order="desc", journal=journal_filter
    )
    raw_len = len(papers)

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
        papers = filter_by_keywords(papers, kw_list, keyword_mode)

    # Filter by date range
    if date_from or date_to:
        papers = filter_by_date(papers, date_from or None, date_to or None, "published")

    # --- AI Ranking ---
    scores: dict = {}
    top_ids: set = set()
    gold_ids: set = set()
    blue_ids: set = set()
    if state._ranking_computed and state._ranking_scores:
        scores = state._ranking_scores
        top_ids = state._ranking_top_ids
        gold_ids = state._ranking_gold_ids
        blue_ids = state._ranking_blue_ids
    elif not state._ranking_computing:
        start_ranking_bg()

    # Apply sorting
    sort_papers(papers, sort_by, order, scores)

    return _paper_list_response(
        request,
        papers,
        "new",
        "새로운 논문이 없습니다. Fetch New 버튼을 클릭하세요.",
        offset,
        append=(offset > 0),
        has_more=(raw_len == PAGE_SIZE),
        scores=scores,
        top_ids=top_ids,
        gold_ids=gold_ids,
        blue_ids=blue_ids,
    )


@router.get("/papers/picked", response_class=HTMLResponse)
async def papers_picked(
    request: Request,
    q: str = Query(""),
    sort_by: str = Query("published", description="Sort criteria"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
    offset: int = Query(0, description="Pagination offset for lazy load"),
):
    """Get picked papers list (partial for HTMX)."""
    papers = state.repo.find_picked(limit=PAGE_SIZE, offset=offset, order="desc")
    raw_len = len(papers)

    if q:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.journal or "").lower()
        ]

    if keywords:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        papers = filter_by_keywords(papers, kw_list, keyword_mode)

    if date_from or date_to:
        papers = filter_by_date(papers, date_from or None, date_to or None, "published")

    sort_papers(papers, sort_by, order)

    return _paper_list_response(
        request,
        papers,
        "picked",
        "선택된 논문이 없습니다.",
        offset,
        append=(offset > 0),
        has_more=(raw_len == PAGE_SIZE),
    )


@router.get("/papers/archive", response_class=HTMLResponse)
async def papers_archive(
    request: Request,
    q: str = Query(""),
    journal: str = Query(""),
    sort_by: str = Query("published", description="Sort criteria"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
    offset: int = Query(0, description="Pagination offset for lazy load"),
):
    """Get archived papers list (partial for HTMX)."""
    journal_filter = journal if journal else None
    papers = state.repo.find_by_status(
        "archived", limit=PAGE_SIZE, offset=offset, sort_by="date", order="desc", journal=journal_filter
    )
    raw_len = len(papers)

    if q:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.journal or "").lower()
        ]

    if keywords:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        papers = filter_by_keywords(papers, kw_list, keyword_mode)

    if date_from or date_to:
        papers = filter_by_date(papers, date_from or None, date_to or None, "published")

    sort_papers(papers, sort_by, order)

    return _paper_list_response(
        request,
        papers,
        "archive",
        "아카이브된 논문이 없습니다.",
        offset,
        append=(offset > 0),
        has_more=(raw_len == PAGE_SIZE),
    )


@router.get("/papers/read", response_class=HTMLResponse)
async def papers_read(
    request: Request,
    q: str = Query(""),
    journal: str = Query(""),
    sort_by: str = Query("created_at", description="Sort criteria"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
    offset: int = Query(0, description="Pagination offset for lazy load"),
):
    """Get read papers list (partial for HTMX). Sorted by created_at (read date)."""
    journal_filter = journal if journal else None
    papers = state.repo.find_by_status(
        "read", limit=PAGE_SIZE, offset=offset, sort_by="created_at", order="desc", journal=journal_filter
    )
    raw_len = len(papers)

    if q:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.journal or "").lower()
            or q_lower in (p.authors or "").lower()
        ]

    if keywords:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        papers = filter_by_keywords(papers, kw_list, keyword_mode)

    if date_from or date_to:
        papers = filter_by_date(papers, date_from or None, date_to or None, "created_at")

    sort_papers(papers, sort_by, order)

    return _paper_list_response(
        request,
        papers,
        "read",
        "읽은 논문이 없습니다.",
        offset,
        append=(offset > 0),
        has_more=(raw_len == PAGE_SIZE),
    )


@router.get("/papers/all", response_class=HTMLResponse)
async def papers_all(
    request: Request,
    q: str = Query(""),
    journal: str = Query(""),
    sort_by: str = Query("published", description="Sort criteria"),
    order: str = Query("desc", description="Sort order: asc or desc"),
    keywords: str = Query("", description="Comma-separated keywords"),
    keyword_mode: str = Query("or", description="Keyword match mode: or/and"),
    date_from: str = Query("", description="Start date (YYYY-MM-DD)"),
    date_to: str = Query("", description="End date (YYYY-MM-DD)"),
    offset: int = Query(0, description="Pagination offset for lazy load"),
):
    """Get all papers in DB (partial for HTMX)."""
    journal_filter = journal if journal else None
    papers = state.repo.find_all(
        limit=PAGE_SIZE, offset=offset, sort_by="date", order="desc", journal=journal_filter
    )
    raw_len = len(papers)

    if q:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in (p.title or "").lower()
            or q_lower in (p.journal or "").lower()
            or q_lower in (p.authors or "").lower()
        ]

    if keywords:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        papers = filter_by_keywords(papers, kw_list, keyword_mode)

    if date_from or date_to:
        papers = filter_by_date(papers, date_from or None, date_to or None, "published")

    sort_papers(papers, sort_by, order)

    return _paper_list_response(
        request,
        papers,
        "all",
        "논문이 없습니다.",
        offset,
        append=(offset > 0),
        has_more=(raw_len == PAGE_SIZE),
    )


# ============================================================================
# Paper Detail
# ============================================================================


@router.get("/papers/{paper_id}", response_class=HTMLResponse)
async def paper_detail(request: Request, paper_id: int):
    """Get paper detail view (partial for HTMX)."""
    paper = state.repo.find_by_id(paper_id)

    if not paper:
        return HTMLResponse("<div class='p-8 text-center text-content-muted'>논문을 찾을 수 없습니다.</div>")

    authors_list = parse_authors(paper.authors)

    # AI match score (may be None if not yet computed)
    ai_score = state._ranking_scores.get(paper_id) if state._ranking_computed else None
    ai_is_top = paper_id in state._ranking_top_ids if state._ranking_computed else False
    ai_is_gold = paper_id in state._ranking_gold_ids if state._ranking_computed else False
    ai_is_blue = paper_id in state._ranking_blue_ids if state._ranking_computed else False

    return templates.TemplateResponse(
        "partials/detail.html",
        {
            "request": request,
            "paper": paper,
            "authors_list": authors_list,
            "ai_score": ai_score,
            "ai_is_top": ai_is_top,
            "ai_is_gold": ai_is_gold,
            "ai_is_blue": ai_is_blue,
        },
    )


@router.get("/papers/{paper_id}/enrich", response_class=HTMLResponse)
async def paper_detail_enrich(request: Request, paper_id: int):
    """Fetch OpenAlex metadata by DOI and return HTML fragment."""
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
# AI Semantic Insight (lazy-loaded in detail view)
# ============================================================================


@router.get("/papers/{paper_id}/ai-insight", response_class=HTMLResponse)
async def paper_ai_insight(request: Request, paper_id: int):
    """Return AI Semantic Insight HTML fragment for a single paper.

    Uses Bi-Encoder cosine similarity to find the most relevant READ
    papers.  Displayed scores use library-distribution percentile
    (same scale as badges).  Lazy-loaded via HTMX.
    """
    paper = state.repo.find_by_id(paper_id)
    if not paper:
        return HTMLResponse("")

    read_papers = state.repo.find_by_status("read", limit=5000)
    if not read_papers:
        return HTMLResponse("")

    # If ranker models aren't loaded yet, show a brief waiting message
    if state.ranker is None or state.ranker._bi_encoder is None:
        return HTMLResponse(
            '<div class="text-sm text-content-muted py-2">AI 모델 로딩 중… 잠시 후 다시 열어주세요.</div>'
        )

    try:
        similar = state.ranker.find_similar(paper, read_papers, top_k=3)
    except Exception:
        return HTMLResponse("")

    if not similar:
        return HTMLResponse("")

    return templates.TemplateResponse(
        "partials/detail_ai_insight.html",
        {
            "request": request,
            "paper": paper,
            "similar_papers": similar,  # list[tuple[Paper, float]]
        },
    )
