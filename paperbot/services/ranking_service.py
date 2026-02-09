"""AI-powered paper ranking service.

Uses a two-stage retrieve-then-rerank pipeline to match new papers
against the user's READ library:

  Stage 1 – Bi-Encoder  (BAAI/bge-small-en-v1.5)
      Encodes papers into dense vectors and computes cosine similarity
      against the library's semantic centroid.  → fast candidate retrieval

  Stage 2 – Cross-Encoder (BAAI/bge-reranker-v2-m3)
      Takes (query, document) pairs and produces fine-grained relevance
      scores via cross-attention.  → precise match percentage
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np

from paperbot.models.paper import Paper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BI_ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"

# Local cache directory (project-root/.models/) — avoids re-downloading
from pathlib import Path as _Path

_MODEL_CACHE_DIR = str(_Path(__file__).resolve().parent.parent.parent / ".models")

# Number of representative library papers used to build the Cross-Encoder query
_N_REPRESENTATIVE = 5

# Max characters fed to each side of the Cross-Encoder
# bge-reranker-v2-m3 supports up to 8192 tokens; 1024 chars ≈ 256 tokens
_CE_MAX_LEN = 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _paper_text(paper: Paper) -> str:
    """Build searchable text from a paper's title and abstract."""
    parts: list[str] = []
    if paper.title:
        parts.append(paper.title)
    if paper.abstract:
        parts.append(paper.abstract)
    return " ".join(parts).strip() or "(no content)"


@dataclass
class RankedPaper:
    """A paper annotated with its AI match score."""

    paper: Paper
    score: float          # 0–100  match percentage
    stage: str = "bi"     # "bi" = Bi-Encoder only, "ce" = Cross-Encoder reranked


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class RankingService:
    """Two-stage paper ranker: Bi-Encoder retrieval → Cross-Encoder reranking.

    Models are loaded lazily on first use and kept in memory for subsequent
    calls.  The library profile (centroid embedding) is cached and
    auto-invalidated when the set of READ paper IDs changes.
    """

    def __init__(self) -> None:
        self._bi_encoder = None
        self._cross_encoder = None

        # Cached library profile
        self._lib_centroid: Optional[np.ndarray] = None
        self._lib_embeddings: Optional[np.ndarray] = None
        self._lib_ids: Optional[frozenset[int]] = None

    # -- lazy model loading --------------------------------------------------

    @staticmethod
    @contextmanager
    def _quiet_load():
        """Suppress all noisy output during model loading.

        Covers Python loggers (transformers, sentence_transformers) **and**
        bare ``print()`` / ``warnings.warn()`` calls from safetensors etc.
        """
        import io
        import logging as _logging
        import sys
        import warnings

        # 1) Raise logger levels
        loggers = []
        for name in (
            "transformers",
            "transformers.modeling_utils",
            "sentence_transformers",
            "sentence_transformers.models",
            "sentence_transformers.SentenceTransformer",
            "sentence_transformers.cross_encoder",
        ):
            lg = _logging.getLogger(name)
            loggers.append((lg, lg.level))
            lg.setLevel(_logging.ERROR)

        # 2) Redirect stdout/stderr to swallow print() noise
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                yield
            finally:
                sys.stdout, sys.stderr = old_out, old_err
                for lg, lvl in loggers:
                    lg.setLevel(lvl)

    @staticmethod
    def is_model_cached(model_name: str) -> bool:
        """Check if a model is already saved in the local cache directory."""
        cache = _Path(_MODEL_CACHE_DIR) / model_name.replace("/", "_")
        return cache.exists() and any(cache.iterdir())

    def needs_download(self) -> list[str]:
        """Return list of model names that need downloading (not cached yet)."""
        missing = []
        if not self.is_model_cached(BI_ENCODER_MODEL):
            missing.append(BI_ENCODER_MODEL)
        if not self.is_model_cached(CROSS_ENCODER_MODEL):
            missing.append(CROSS_ENCODER_MODEL)
        return missing

    @property
    def bi_encoder(self):
        """Lazy-load the Bi-Encoder (BGE-small-en-v1.5, ~33 M params)."""
        if self._bi_encoder is None:
            from sentence_transformers import SentenceTransformer

            local_path = _Path(_MODEL_CACHE_DIR) / BI_ENCODER_MODEL.replace("/", "_")
            with self._quiet_load():
                if local_path.exists():
                    # Load from local cache — no network call
                    self._bi_encoder = SentenceTransformer(str(local_path))
                else:
                    # First run: download → save to local cache
                    self._bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
                    self._bi_encoder.save(str(local_path))
        return self._bi_encoder

    @property
    def cross_encoder(self):
        """Lazy-load the Cross-Encoder / Reranker (BGE-reranker-v2-m3)."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            local_path = _Path(_MODEL_CACHE_DIR) / CROSS_ENCODER_MODEL.replace("/", "_")
            with self._quiet_load():
                if local_path.exists():
                    self._cross_encoder = CrossEncoder(str(local_path))
                else:
                    self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
                    self._cross_encoder.save(str(local_path))
        return self._cross_encoder

    # -- encoding helpers ----------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalised dense vectors."""
        return self.bi_encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )

    # -- library profile (cached) -------------------------------------------

    def _build_library_profile(
        self, read_papers: list[Paper]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute and cache (centroid, embeddings) for the READ library.

        The cache is keyed on the *set* of paper IDs; any change in
        library composition triggers a recomputation.
        """
        current_ids = frozenset(p.id for p in read_papers)
        if (
            self._lib_centroid is not None
            and self._lib_ids == current_ids
        ):
            assert self._lib_embeddings is not None
            return self._lib_centroid, self._lib_embeddings

        texts = [_paper_text(p) for p in read_papers]
        embeddings = self._encode(texts)

        centroid: np.ndarray = embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-10

        self._lib_embeddings = embeddings
        self._lib_centroid = centroid
        self._lib_ids = current_ids
        return centroid, embeddings

    def _representative_papers(
        self,
        read_papers: list[Paper],
        centroid: np.ndarray,
        embeddings: np.ndarray,
        n: int = _N_REPRESENTATIVE,
    ) -> list[Paper]:
        """Select the *n* papers closest to the centroid as library representatives."""
        sims: np.ndarray = embeddings @ centroid
        top_idx = np.argsort(sims)[::-1][:n]
        return [read_papers[int(i)] for i in top_idx]

    # -- pairwise similarity (optimised) ------------------------------------

    @staticmethod
    def _pairwise_max_sim(
        query_embs: np.ndarray,
        lib_embs: np.ndarray,
        chunk_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute max cosine similarity of each query against all lib papers.

        Uses chunked matrix multiply to bound peak memory at
        ``chunk_size * N_lib * sizeof(float32)`` instead of
        ``N_query * N_lib * sizeof(float32)``.

        Returns
        -------
        max_sims : ndarray, shape (N_query,)
            Best cosine similarity for each query paper.
        best_idx : ndarray, shape (N_query,)
            Index (into *lib_embs*) of the best-matching library paper.
        """
        n = query_embs.shape[0]
        max_sims = np.empty(n, dtype=np.float32)
        best_idx = np.empty(n, dtype=np.intp)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = query_embs[start:end] @ lib_embs.T   # (chunk, N_lib)
            max_sims[start:end] = chunk.max(axis=1)
            best_idx[start:end] = chunk.argmax(axis=1)

        return max_sims, best_idx

    # -- public API ----------------------------------------------------------

    def rank(
        self,
        new_papers: list[Paper],
        read_papers: list[Paper],
        *,
        top_k_bi: int = 100,
    ) -> list[RankedPaper]:
        """Rank *new_papers* by relevance to the user's *read_papers* library.

        Pipeline
        --------
        Stage 1 — Bi-Encoder (pairwise max):
            For each new paper, compute cosine similarity against **every**
            library paper and keep the **maximum** (best individual match).
            This replaces the centroid-based approach for better
            discriminative power.

        Stage 2 — Cross-Encoder (enriched query):
            For the top-K Bi-Encoder candidates, run the Cross-Encoder
            with a rich query built from the library's 5 representative
            papers.  Each representative is allocated ``_CE_MAX_LEN //
            _N_REPRESENTATIVE`` characters (~200 chars at _CE_MAX_LEN=1024).

        Score calibration:
            Cross-Encoder logits are passed through sigmoid then **min-max
            rescaled** within the batch when scores are too clustered,
            guaranteeing meaningful differentiation.

        Parameters
        ----------
        new_papers:
            Candidate papers to rank (typically ``status='new'``).
        read_papers:
            The user's library (typically ``status='read'``).
        top_k_bi:
            Number of candidates to keep after the Bi-Encoder stage.

        Returns
        -------
        list[RankedPaper]
            **All** *new_papers*, sorted by match score descending.
        """
        if not read_papers:
            return [RankedPaper(paper=p, score=0.0) for p in new_papers]

        if not new_papers:
            return []

        # ── Stage 1: Bi-Encoder — pairwise max similarity ─────────────
        centroid, lib_embs = self._build_library_profile(read_papers)

        new_texts = [_paper_text(p) for p in new_papers]
        new_embs = self._encode(new_texts)

        # Best individual match per new paper (chunked for memory)
        bi_scores, _best_lib = self._pairwise_max_sim(new_embs, lib_embs)

        # O(N) partial sort — only need top-K, not full sort
        k = min(top_k_bi, len(new_papers))
        if k < len(new_papers):
            top_idx = np.argpartition(bi_scores, -k)[-k:]
            # Sort only the top-K for stable ordering
            top_idx = top_idx[np.argsort(bi_scores[top_idx])[::-1]]
        else:
            top_idx = np.argsort(bi_scores)[::-1]
        top_set = set(top_idx.tolist())

        # ── Stage 2: Cross-Encoder — enriched representative query ────

        reps = self._representative_papers(
            read_papers, centroid, lib_embs, n=_N_REPRESENTATIVE,
        )
        # Each representative gets a generous text allocation
        per_rep_len = _CE_MAX_LEN // _N_REPRESENTATIVE
        query = " [SEP] ".join(
            _paper_text(p)[:per_rep_len] for p in reps
        )
        query = query[:_CE_MAX_LEN]

        pairs = [
            [query, _paper_text(new_papers[int(i)])[:_CE_MAX_LEN]]
            for i in top_idx
        ]
        raw_scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        ce_scores = np.asarray(raw_scores, dtype=np.float64)

        ce_pct = self._logits_to_pct(ce_scores)

        # ── Assemble final results ──────────────────────────────────────
        results: list[RankedPaper] = []

        for rank_pos, orig_idx in enumerate(top_idx):
            results.append(
                RankedPaper(
                    paper=new_papers[int(orig_idx)],
                    score=round(float(ce_pct[rank_pos]), 1),
                    stage="ce",
                )
            )

        # Remaining papers — Bi-Encoder pairwise-max scores
        for idx in range(len(new_papers)):
            if idx not in top_set:
                pct = float(np.clip(bi_scores[idx], 0.0, 1.0)) * 100.0
                results.append(
                    RankedPaper(
                        paper=new_papers[idx],
                        score=round(pct, 1),
                        stage="bi",
                    )
                )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # -- score helpers -------------------------------------------------------

    @staticmethod
    def _logits_to_pct(logits: np.ndarray) -> np.ndarray:
        """Convert Cross-Encoder raw logits → match percentage [0, 100].

        1. Sigmoid maps logits to (0, 1) for absolute calibration.
        2. When sigmoid scores are clustered within a narrow band
           (< 15 pp spread), min-max rescaling stretches them around
           the batch mean for meaningful differentiation.
        """
        if logits.size == 0:
            return logits

        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
        pct = probs * 100.0

        if logits.size <= 1:
            return pct

        lo, hi = float(pct.min()), float(pct.max())
        spread = hi - lo

        if spread >= 15.0:
            # Good natural differentiation — use sigmoid as-is
            return pct

        # Scores too clustered → rescale around batch center
        center = float(pct.mean())
        half_band = 15.0  # guarantee ≥30 pp total range
        target_lo = max(center - half_band, 0.0)
        target_hi = min(center + half_band, 100.0)

        if spread < 1e-10:
            return np.full_like(pct, center)

        return target_lo + (pct - lo) / spread * (target_hi - target_lo)

    # -- single-paper similarity ---------------------------------------------

    def find_similar(
        self,
        paper: Paper,
        library: list[Paper],
        top_k: int = 3,
    ) -> list[tuple[Paper, float]]:
        """Find the most similar papers in *library* to a single *paper*.

        Uses Bi-Encoder cosine similarity (fast, no Cross-Encoder).

        Parameters
        ----------
        paper:
            The target paper to compare against the library.
        library:
            Reference papers (typically ``status='read'``).
        top_k:
            Number of most-similar papers to return.

        Returns
        -------
        list[tuple[Paper, float]]
            Up to *top_k* ``(paper, similarity_pct)`` pairs sorted by
            similarity descending.  Similarity is in [0, 100].
            The target paper itself is excluded if present in the library.
        """
        if not library:
            return []

        # Reuse cached library embeddings
        _centroid, lib_embs = self._build_library_profile(library)

        # Encode the target paper
        target_emb = self._encode([_paper_text(paper)])  # shape (1, dim)

        # Pairwise cosine similarity (both sides L2-normalised)
        sims: np.ndarray = (target_emb @ lib_embs.T).flatten()  # shape (N,)

        # Exclude self (same paper.id)
        for idx, lib_paper in enumerate(library):
            if lib_paper.id is not None and lib_paper.id == paper.id:
                sims[idx] = -1.0

        # Top-K
        k = min(top_k, len(library))
        top_idx = np.argsort(sims)[::-1][:k]

        results: list[tuple[Paper, float]] = []
        for idx in top_idx:
            pct = float(np.clip(sims[idx], 0.0, 1.0)) * 100.0
            if pct > 0:
                results.append((library[int(idx)], round(pct, 1)))

        return results

    # -- cache management ----------------------------------------------------

    def invalidate_cache(self) -> None:
        """Clear cached library embeddings.

        Call this when the user's READ library changes (e.g. after export).
        """
        self._lib_centroid = None
        self._lib_embeddings = None
        self._lib_ids = None
