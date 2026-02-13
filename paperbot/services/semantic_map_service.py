"""Semantic Map service — clustering + dimensionality reduction for paper visualization.

Computes:
  1) 2D + 3D t-SNE coordinates from SPECTER2 embeddings
  2) KMeans clusters for k=2..10 (pre-computed for instant client-side switching)
  3) Top-3 cosine-similarity neighbours per paper (for edge connections)

All heavy computation happens once; the client receives a single JSON payload
and handles filtering, K-switching, and 2D/3D toggling with zero latency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from paperbot.database.repository import PaperRepository
    from paperbot.services.ranking_service import RankingService

from paperbot.models.paper import Paper

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------

@dataclass
class SemanticMapPoint:
    """A single paper projected onto 2D/3D space with cluster assignments."""

    id: int
    title: str
    journal: Optional[str]
    published: Optional[str]
    status: str
    # 2D coordinates (t-SNE)
    x2: float
    y2: float
    # 3D coordinates (t-SNE)
    x3: float
    y3: float
    z3: float
    # Cluster assignments for each k (k=2..10)
    clusters: dict[int, int] = field(default_factory=dict)
    # Top-3 most similar paper IDs
    top3: list[int] = field(default_factory=list)


@dataclass
class SemanticMapResult:
    """Complete semantic map data ready for JSON serialisation."""

    points: list[SemanticMapPoint]
    n_papers: int


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class SemanticMapService:
    """Generates semantic map data from paper embeddings.

    Reuses the RankingService's SPECTER2 encoder and cached embeddings
    so no redundant model loading or encoding occurs.
    """

    K_MIN = 2
    K_MAX = 10

    def __init__(
        self,
        ranking_service: "RankingService",
        repo: "PaperRepository",
    ) -> None:
        self._ranking = ranking_service
        self._repo = repo
        # In-memory cache: cleared when library changes
        self._cache: Optional[SemanticMapResult] = None
        self._cache_ids: Optional[frozenset[int]] = None

    # -- public API ----------------------------------------------------------

    def generate(self, papers: list[Paper]) -> SemanticMapResult:
        """Generate the full semantic map for the given papers.

        Returns cached result if the paper set hasn't changed.
        """
        if not papers:
            return SemanticMapResult(points=[], n_papers=0)

        current_ids = frozenset(p.id for p in papers if p.id is not None)

        # Return cache if paper set unchanged
        if self._cache is not None and self._cache_ids == current_ids:
            return self._cache

        embeddings = self._get_embeddings(papers)
        n = len(papers)

        if n < 2:
            # Edge case: single paper — no clustering/t-SNE possible
            pt = SemanticMapPoint(
                id=papers[0].id,
                title=papers[0].title or "",
                journal=papers[0].journal,
                published=papers[0].published,
                status=papers[0].status,
                x2=0.0, y2=0.0,
                x3=0.0, y3=0.0, z3=0.0,
                clusters={k: 0 for k in range(self.K_MIN, self.K_MAX + 1)},
                top3=[],
            )
            result = SemanticMapResult(points=[pt], n_papers=1)
            self._cache = result
            self._cache_ids = current_ids
            return result

        # ── 1) Dimensionality reduction ────────────────────────────────
        coords_2d, coords_3d = self._reduce_dimensions(embeddings, n)

        # ── 2) Clustering (k=2..10) ───────────────────────────────────
        all_clusters = self._compute_clusters(embeddings, n)

        # ── 3) Top-3 neighbours ───────────────────────────────────────
        top3_map = self._compute_top3(embeddings, papers)

        # ── 4) Assemble points ────────────────────────────────────────
        points: list[SemanticMapPoint] = []
        for i, paper in enumerate(papers):
            clusters = {k: int(labels[i]) for k, labels in all_clusters.items()}
            points.append(SemanticMapPoint(
                id=paper.id,
                title=paper.title or "",
                journal=paper.journal,
                published=paper.published,
                status=paper.status,
                x2=round(float(coords_2d[i, 0]), 4),
                y2=round(float(coords_2d[i, 1]), 4),
                x3=round(float(coords_3d[i, 0]), 4),
                y3=round(float(coords_3d[i, 1]), 4),
                z3=round(float(coords_3d[i, 2]), 4),
                clusters=clusters,
                top3=top3_map.get(paper.id, []),
            ))

        result = SemanticMapResult(points=points, n_papers=n)
        self._cache = result
        self._cache_ids = current_ids
        return result

    def invalidate_cache(self) -> None:
        """Clear cached map data (call when library changes)."""
        self._cache = None
        self._cache_ids = None

    # -- internal helpers ----------------------------------------------------

    def _get_embeddings(self, papers: list[Paper]) -> np.ndarray:
        """Get embeddings for papers, reusing RankingService's cache."""
        self._ranking._ensure_emb_loaded()

        to_encode: list[Paper] = []
        for p in papers:
            if p.id not in self._ranking._emb_store:
                to_encode.append(p)

        if to_encode:
            texts = [(p.title or "").strip() or "(no content)" for p in to_encode]
            new_embs = self._ranking._encode(texts)
            new_entries: dict[int, np.ndarray] = {}
            for paper, emb in zip(to_encode, new_embs):
                self._ranking._emb_store[paper.id] = emb
                new_entries[paper.id] = emb
            self._ranking._save_new_embeddings(new_entries)

        return np.stack([self._ranking._emb_store[p.id] for p in papers])

    @staticmethod
    def _reduce_dimensions(
        embeddings: np.ndarray, n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute 2D and 3D t-SNE projections."""
        from sklearn.manifold import TSNE

        # Adjust perplexity for small datasets
        perplexity = min(30, max(2, n // 3))

        coords_2d = TSNE(
            n_components=2, random_state=42, perplexity=perplexity,
        ).fit_transform(embeddings)

        coords_3d = TSNE(
            n_components=3, random_state=42, perplexity=perplexity,
        ).fit_transform(embeddings)

        return coords_2d, coords_3d

    def _compute_clusters(
        self, embeddings: np.ndarray, n: int,
    ) -> dict[int, np.ndarray]:
        """Run KMeans for k=2..10 (capped at n)."""
        from sklearn.cluster import KMeans

        results: dict[int, np.ndarray] = {}
        for k in range(self.K_MIN, self.K_MAX + 1):
            if k > n:
                break
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            results[k] = km.fit_predict(embeddings)
        return results

    @staticmethod
    def _compute_top3(
        embeddings: np.ndarray, papers: list[Paper],
    ) -> dict[int, list[int]]:
        """Compute top-3 most similar papers for each paper (cosine)."""
        sim = embeddings @ embeddings.T  # (N, N) — already L2-normalised
        np.fill_diagonal(sim, -1.0)

        top3_map: dict[int, list[int]] = {}
        n = len(papers)
        k = min(3, n - 1)

        if k <= 0:
            return top3_map

        for i in range(n):
            top_idx = np.argsort(sim[i])[::-1][:k]
            top3_map[papers[i].id] = [papers[int(j)].id for j in top_idx]

        return top3_map
