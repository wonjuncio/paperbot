"""AI-powered paper ranking service.

Scores each NEW paper by how well it matches the user's READ library
using **library-distribution percentile scoring**:

  1) Bi-Encoder (allenai/specter2_base)
       Encodes paper **titles** → dense vectors.

  2) Per-paper hybrid scoring
       For each (new paper, read paper) pair, compute::

           h_i = W_TOPK × cosine(new, read_i)
               + W_CENT × cosine(new, library centroid)

       Badge = max(h_1 … h_k) = hybrid score of the best match.
       The centroid term anchors scores to the library's overall topic
       direction, reducing false positives from related-but-off-topic
       papers (e.g. "ML for drug discovery" vs "ML for materials").

  3) Library distribution
       Compute the same hybrid among READ papers themselves.
       This defines "normal similarity in my field".

  4) Percentile mapping
       ``score = percentile_in_library_distribution(new_hybrid)``
       → 0–100.  A score of 80 means "this paper matches your library
       better than 80 % of your own papers match each other".

Library embeddings are updated **incrementally** and **persisted**
to disk so that restarts are instant.
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
BI_ENCODER_MODEL = "allenai/specter2_base"
SIM_TOPK = 3    # top-k neighbours averaged for similarity (both lib & new)
W_TOPK = 0.7    # weight for top-k mean cosine in hybrid score
W_CENT = 0.3    # weight for centroid cosine in hybrid score

# Local cache directory (project-root/.models/) — avoids re-downloading
from pathlib import Path as _Path

_MODEL_CACHE_DIR = str(_Path(__file__).resolve().parent.parent.parent / ".models")





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _paper_text(paper: Paper) -> str:
    """Build embedding text from a paper's title.

    Uses title only (not abstract) so that scoring reflects *topic*
    relevance rather than writing style or verbose detail.
    """
    return (paper.title or "").strip() or "(no content)"


@dataclass
class RankedPaper:
    """A paper annotated with its AI match score."""

    paper: Paper
    score: float          # 0–100  library-percentile match score


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class RankingService:
    """Library-distribution percentile ranker with incremental embeddings.

    Key design:

    * **Score = percentile** — a new paper's per-paper hybrid
      (best-match cosine + centroid) is compared to the library's
      own internal distribution.  Score 80 = "matches better than
      80 % of your own library papers match each other".
    * **Incremental encoding** — only newly-added read papers are encoded;
      existing embeddings are reused from an in-memory dict.
    * **Disk persistence** — embeddings + library distribution are saved
      so restarts skip all encoding.
    """

    def __init__(self) -> None:
        self._bi_encoder = None

        # Incremental per-paper embedding store:  paper_id → vector
        self._emb_store: dict[int, np.ndarray] = {}
        self._emb_loaded: bool = False
        # Include model name in path → auto-invalidate if model changes
        _safe_model = BI_ENCODER_MODEL.replace("/", "_")
        self._emb_path: _Path = _Path(_MODEL_CACHE_DIR) / f"lib_emb_{_safe_model}.npz"

        # Library similarity distribution (sorted array of hybrid cosine
        # values among read papers).  Computed once per library state.
        self._lib_dist: Optional[np.ndarray] = None
        self._centroid: Optional[np.ndarray] = None

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
        return missing

    @property
    def bi_encoder(self):
        """Lazy-load the Bi-Encoder (SPECTER2, ~110 M params)."""
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

    # -- encoding helpers ----------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts into L2-normalised dense vectors."""
        return self.bi_encoder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=64,
        )

    # -- incremental library embeddings --------------------------------------

    def _ensure_emb_loaded(self) -> None:
        """Load persisted library embeddings from disk (first access only)."""
        if self._emb_loaded:
            return
        self._emb_loaded = True
        if self._emb_path.exists():
            try:
                data = np.load(str(self._emb_path), allow_pickle=False)
                for pid, emb in zip(
                    data["ids"].tolist(), data["embeddings"]
                ):
                    self._emb_store[int(pid)] = emb
            except Exception:
                self._emb_store.clear()

    def _save_embeddings(self) -> None:
        """Persist the embedding store to a compressed ``.npz`` file."""
        if not self._emb_store:
            if self._emb_path.exists():
                self._emb_path.unlink()
            return
        self._emb_path.parent.mkdir(parents=True, exist_ok=True)
        ids = np.array(list(self._emb_store.keys()), dtype=np.int64)
        embs = np.stack(list(self._emb_store.values()))
        np.savez_compressed(str(self._emb_path), ids=ids, embeddings=embs)

    def _get_lib_embeddings(
        self, read_papers: list[Paper],
    ) -> np.ndarray:
        """Return the library embedding matrix, encoding only **new** papers.

        Complexity: ``O(Δ)`` where ``Δ`` = number of newly-added read
        papers since the last call, instead of ``O(N)`` for the full
        library.  Removed papers are evicted from the store.

        Returns
        -------
        np.ndarray, shape ``(len(read_papers), dim)``
            Rows ordered identically to *read_papers*.
        """
        self._ensure_emb_loaded()

        current_ids = {p.id for p in read_papers}
        cached_ids = set(self._emb_store.keys())

        # Evict papers no longer in the library
        removed = cached_ids - current_ids
        for pid in removed:
            del self._emb_store[pid]

        # Encode only brand-new papers
        to_encode = [p for p in read_papers if p.id not in self._emb_store]
        if to_encode:
            texts = [_paper_text(p) for p in to_encode]
            new_embs = self._encode(texts)
            for paper, emb in zip(to_encode, new_embs):
                self._emb_store[paper.id] = emb

        # Persist when the store changed
        if to_encode or removed:
            self._save_embeddings()

        # Build matrix in read_papers order (index-consistent)
        return np.stack([self._emb_store[p.id] for p in read_papers])

    # -- centroid ------------------------------------------------------------

    @staticmethod
    def _compute_centroid(embs: np.ndarray) -> np.ndarray:
        """Compute L2-normalised centroid of an embedding matrix."""
        centroid = embs.mean(axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm > 0 else centroid

    def _get_centroid(self, lib_embs: np.ndarray) -> np.ndarray:
        """Return (or compute + cache) the library centroid."""
        if self._centroid is None:
            self._centroid = self._compute_centroid(lib_embs)
        return self._centroid

    # -- pairwise similarity (optimised) ------------------------------------

    @staticmethod
    def _pairwise_topk_sim(
        query_embs: np.ndarray,
        lib_embs: np.ndarray,
        k: int = SIM_TOPK,
        chunk_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-query top-k **mean** cosine similarity against the library.

        Uses chunked matrix multiply to bound peak memory and
        ``argpartition`` for O(N_lib) top-k selection per row.

        Averaging top-k (instead of taking the max) reduces score
        inflation from a single near-duplicate in the library.

        Returns
        -------
        topk_mean : ndarray, shape (N_query,)
            Mean of top-k cosine similarities for each query paper.
        topk_idx : ndarray, shape (N_query, k')
            Column indices of the *k'* best-matching library papers per
            query, sorted descending.  ``k' = min(k, N_lib)``.
        """
        n_q = query_embs.shape[0]
        n_lib = lib_embs.shape[0]
        actual_k = min(k, n_lib)

        topk_mean = np.empty(n_q, dtype=np.float32)
        topk_idx = np.empty((n_q, actual_k), dtype=np.intp)

        for start in range(0, n_q, chunk_size):
            end = min(start + chunk_size, n_q)
            sim = query_embs[start:end] @ lib_embs.T  # (chunk, N_lib)

            if actual_k < n_lib:
                # O(N_lib) partial sort per row — faster than full sort
                part = np.argpartition(sim, -actual_k, axis=1)[:, -actual_k:]
                # Gather top-k values and average
                topk_vals = np.take_along_axis(sim, part, axis=1)
                topk_mean[start:end] = topk_vals.mean(axis=1)
                for i in range(end - start):
                    row_topk = part[i]
                    order = np.argsort(sim[i, row_topk])[::-1]
                    topk_idx[start + i] = row_topk[order]
            else:
                topk_idx[start:end] = np.argsort(sim, axis=1)[:, ::-1]
                topk_mean[start:end] = sim.mean(axis=1)

        return topk_mean, topk_idx

    # -- library distribution ------------------------------------------------

    @staticmethod
    def _compute_lib_distribution(
        lib_embs: np.ndarray,
        centroid: np.ndarray,
    ) -> np.ndarray:
        """Compute the library's internal hybrid similarity distribution.

        For each read paper *r*, compute::

            s_match = max cosine(r, other reads)
            s_cent  = cosine(r, centroid)
            s_hybrid = W_TOPK * s_match + W_CENT * s_cent

        The sorted array of these hybrid values defines "normal
        similarity in my field".  The centroid term anchors scores to
        the library's overall topic direction, reducing false positives
        from related-but-off-topic papers.

        Returns
        -------
        np.ndarray
            Sorted 1-D array of hybrid values (ascending).
            If the library has < 2 papers, returns ``[0.0]`` as a
            conservative fallback so any positive similarity scores > 0.
        """
        n = lib_embs.shape[0]
        if n < 2:
            return np.array([0.0], dtype=np.float32)

        # ── Best-match cosine (excluding self) ────────────────────────
        sim = lib_embs @ lib_embs.T            # (N, N)
        np.fill_diagonal(sim, -np.inf)          # exclude self
        max_cos = sim.max(axis=1)               # (N,)

        # ── Centroid cosine per read paper ─────────────────────────────
        cent_cos = (lib_embs @ centroid).flatten()  # (N,)

        # ── Hybrid ─────────────────────────────────────────────────────
        hybrid = W_TOPK * max_cos + W_CENT * cent_cos
        return np.sort(hybrid)

    def _get_lib_distribution(self, lib_embs: np.ndarray) -> np.ndarray:
        """Return (or compute + cache) the library distribution."""
        if self._lib_dist is None:
            centroid = self._get_centroid(lib_embs)
            self._lib_dist = self._compute_lib_distribution(lib_embs, centroid)
        return self._lib_dist

    @staticmethod
    def _cosine_to_score(
        cosines: np.ndarray,
        lib_dist: np.ndarray,
    ) -> np.ndarray:
        """Map cosine values → 0–100 percentile scores.

        ``score[i] = 100 × (# lib values ≤ cosines[i]) / len(lib_dist)``

        A score of 80 means "this paper matches your library better than
        80 % of your own papers match each other".
        """
        # searchsorted on a sorted array gives the insertion index,
        # which equals the count of elements ≤ value (with side='right').
        positions = np.searchsorted(lib_dist, cosines, side="right")
        return positions / len(lib_dist) * 100.0

    # -- public API ----------------------------------------------------------

    def rank(
        self,
        new_papers: list[Paper],
        read_papers: list[Paper],
    ) -> list[RankedPaper]:
        """Rank *new_papers* by relevance to the user's *read_papers* library.

        Pipeline
        --------
        1. **Encode** — Bi-Encoder embeds paper titles (incremental
           for the library).
        2. **Best-match cosine** — max cosine to any read paper.
        3. **Centroid cosine** — cosine to the library centroid.
        4. **Per-paper hybrid** — ``h_i = W_TOPK × cos_i + W_CENT × cent``.
           Badge = max(h_1 … h_k) = hybrid of best match.
        5. **Library distribution** — same hybrid among read papers.
        6. **Percentile mapping** — ``score = percentile(hybrid,
           lib_distribution) × 100``.

        Complexity: ``O(N_new × N_read)`` for cosine (chunked matmul),
        ``O(N_read²)`` one-time for library distribution (cached).

        Returns
        -------
        list[RankedPaper]
            **All** *new_papers*, sorted by match score descending.
        """
        if not read_papers:
            return [RankedPaper(paper=p, score=0.0) for p in new_papers]

        if not new_papers:
            return []

        # ── Step 1: Embeddings ─────────────────────────────────────────
        lib_embs = self._get_lib_embeddings(read_papers)

        new_texts = [_paper_text(p) for p in new_papers]
        new_embs = self._encode(new_texts)

        # ── Step 2: Best-match cosine per new paper ──────────────────
        max_cos, _ = self._pairwise_topk_sim(new_embs, lib_embs, k=1)

        # ── Step 3: Centroid cosine per new paper ─────────────────────
        centroid = self._get_centroid(lib_embs)
        cent_cos = (new_embs @ centroid).flatten()  # (N_new,)

        # ── Step 4: Per-paper hybrid (badge = max = best match hybrid)─
        hybrid = W_TOPK * max_cos + W_CENT * cent_cos

        # ── Step 5: Library distribution (also hybrid) ────────────────
        lib_dist = self._get_lib_distribution(lib_embs)

        # ── Step 6: Percentile scoring ────────────────────────────────
        scores = self._cosine_to_score(hybrid, lib_dist)

        # ── Assemble results ──────────────────────────────────────────
        results: list[RankedPaper] = []
        for idx, paper in enumerate(new_papers):
            results.append(
                RankedPaper(
                    paper=paper,
                    score=round(float(scores[idx]), 1),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # -- single-paper similarity (AI Insight) ---------------------------------

    def find_similar(
        self,
        paper: Paper,
        library: list[Paper],
        top_k: int = 3,
    ) -> list[tuple[Paper, float]]:
        """Find the most similar READ papers to a single NEW paper.

        Used by the AI Insight panel.  Each similar paper receives a
        **per-paper hybrid score**::

            h_i = W_TOPK × cos(new, read_i) + W_CENT × cos(new, centroid)

        mapped through the hybrid library distribution → percentile.
        The highest insight score (insight #1) equals the badge score.

        Parameters
        ----------
        paper:
            The target new paper.
        library:
            Read papers (the user's library).
        top_k:
            Number of most-similar papers to return.

        Returns
        -------
        list[tuple[Paper, float]]
            Up to *top_k* ``(paper, percentile_score)`` pairs sorted by
            hybrid score descending.  Score is in [0, 100].
        """
        if not library:
            return []

        # Reuse cached incremental library embeddings
        lib_embs = self._get_lib_embeddings(library)

        # Library distribution (hybrid, reuse cache from rank())
        lib_dist = self._get_lib_distribution(lib_embs)

        # Encode the target paper
        target_emb = self._encode([_paper_text(paper)])  # shape (1, dim)

        # Pairwise cosine similarity (both sides L2-normalised)
        sims: np.ndarray = (target_emb @ lib_embs.T).flatten()  # (N,)

        # Exclude self (same paper.id)
        for idx, lib_paper in enumerate(library):
            if lib_paper.id is not None and lib_paper.id == paper.id:
                sims[idx] = -1.0

        # Top-K by cosine (centroid is constant, so cosine order = hybrid order)
        k = min(top_k, len(library))
        top_idx = np.argsort(sims)[::-1][:k]

        # Per-paper hybrid score: W_TOPK × cos_i + W_CENT × cos(new, centroid)
        centroid = self._get_centroid(lib_embs)
        cent_cos = float((target_emb @ centroid).flatten()[0])
        top_cosines = np.clip(sims[top_idx], 0.0, 1.0)
        top_hybrid = W_TOPK * top_cosines + W_CENT * cent_cos

        # Map hybrid → percentile
        top_pcts = self._cosine_to_score(top_hybrid, lib_dist)

        results: list[tuple[Paper, float]] = []
        for i, idx in enumerate(top_idx):
            pct = round(float(top_pcts[i]), 1)
            if pct > 0:
                results.append((library[int(idx)], pct))

        return results

    # -- cache management ----------------------------------------------------

    def invalidate_cache(self) -> None:
        """Clear all cached library embeddings and distribution.

        Call this when the user's READ library changes (e.g. after export)
        or when the model version changes.
        """
        self._emb_store.clear()
        self._emb_loaded = False
        self._lib_dist = None
        self._centroid = None
        if self._emb_path.exists():
            try:
                self._emb_path.unlink()
            except OSError:
                pass
