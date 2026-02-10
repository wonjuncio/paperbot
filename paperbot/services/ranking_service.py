"""AI-powered paper ranking service.

Scores each NEW paper by how well it matches the user's READ library
using **library-distribution percentile scoring**:

  1) Bi-Encoder (BAAI/bge-small-en-v1.5)
       Encodes papers → dense vectors.  For each new paper, compute
       ``max cosine(new, read_i)`` = its best match in the library.

  2) Library distribution
       Compute the same ``max cosine`` among READ papers themselves
       (each read paper's nearest neighbour in the library).  This
       defines "normal similarity in my field".

  3) Percentile mapping
       ``score = percentile_in_library_distribution(new_paper_max_cos)``
       → 0–100.  A score of 80 means "this paper matches your library
       better than 80 % of your own papers match each other".

  Cross-Encoder (BAAI/bge-reranker-v2-m3) is used **only** for the
  AI Insight panel — reranking the top-K candidates to show the user
  *which* read papers are most similar (evidence / explanation).

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
BI_ENCODER_MODEL = "BAAI/bge-small-en-v1.5"
CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"

# Local cache directory (project-root/.models/) — avoids re-downloading
from pathlib import Path as _Path

_MODEL_CACHE_DIR = str(_Path(__file__).resolve().parent.parent.parent / ".models")

# Max characters fed to each side of the Cross-Encoder (insight only).
_CE_MAX_LEN = 512

# Per-paper top-K read papers for CE reranking (insight only).
_TOP_K_READ = 5


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
    score: float          # 0–100  library-percentile match score
    stage: str = "bi"     # always "bi" (kept for cache compat)


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class RankingService:
    """Library-distribution percentile ranker with incremental embeddings.

    Key design:

    * **Score = percentile** — a new paper's max cosine similarity to
      the library is compared to the library's own internal similarity
      distribution.  Score 80 = "matches better than 80 % of your own
      library papers match each other".
    * **Incremental encoding** — only newly-added read papers are encoded;
      existing embeddings are reused from an in-memory dict.
    * **Disk persistence** — embeddings + library distribution are saved
      so restarts skip all encoding.
    * **CE for insight only** — Cross-Encoder runs on-demand when the
      user views a paper's AI Insight panel, not during batch ranking.
    """

    def __init__(self) -> None:
        self._bi_encoder = None
        self._cross_encoder = None

        # Incremental per-paper embedding store:  paper_id → vector
        self._emb_store: dict[int, np.ndarray] = {}
        self._emb_loaded: bool = False
        # Include model name in path → auto-invalidate if model changes
        _safe_model = BI_ENCODER_MODEL.replace("/", "_")
        self._emb_path: _Path = _Path(_MODEL_CACHE_DIR) / f"lib_emb_{_safe_model}.npz"

        # Library similarity distribution (sorted array of max-cosine
        # values among read papers).  Computed once per library state.
        self._lib_dist: Optional[np.ndarray] = None

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
                    self._cross_encoder = CrossEncoder(
                        str(local_path), max_length=512,
                    )
                else:
                    self._cross_encoder = CrossEncoder(
                        CROSS_ENCODER_MODEL, max_length=512,
                    )
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

    # -- pairwise similarity (optimised) ------------------------------------

    @staticmethod
    def _pairwise_topk_sim(
        query_embs: np.ndarray,
        lib_embs: np.ndarray,
        k: int = _TOP_K_READ,
        chunk_size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-query top-k cosine similarities against the library.

        Uses chunked matrix multiply to bound peak memory and
        ``argpartition`` for O(N_lib) top-k selection per row.

        Returns
        -------
        max_sims : ndarray, shape (N_query,)
            Best cosine similarity for each query paper.
        topk_idx : ndarray, shape (N_query, k')
            Column indices of the *k'* best-matching library papers per
            query, sorted descending.  ``k' = min(k, N_lib)``.
        """
        n_q = query_embs.shape[0]
        n_lib = lib_embs.shape[0]
        actual_k = min(k, n_lib)

        max_sims = np.empty(n_q, dtype=np.float32)
        topk_idx = np.empty((n_q, actual_k), dtype=np.intp)

        for start in range(0, n_q, chunk_size):
            end = min(start + chunk_size, n_q)
            sim = query_embs[start:end] @ lib_embs.T  # (chunk, N_lib)

            max_sims[start:end] = sim.max(axis=1)

            if actual_k < n_lib:
                # O(N_lib) partial sort per row — faster than full sort
                part = np.argpartition(sim, -actual_k, axis=1)[:, -actual_k:]
                for i in range(end - start):
                    row_topk = part[i]
                    order = np.argsort(sim[i, row_topk])[::-1]
                    topk_idx[start + i] = row_topk[order]
            else:
                topk_idx[start:end] = np.argsort(sim, axis=1)[:, ::-1]

        return max_sims, topk_idx

    # -- library distribution ------------------------------------------------

    @staticmethod
    def _compute_lib_distribution(lib_embs: np.ndarray) -> np.ndarray:
        """Compute the library's internal similarity distribution.

        For each read paper, find its **max cosine** to any *other*
        read paper.  The sorted array of these values defines "normal
        similarity in this field".

        Returns
        -------
        np.ndarray
            Sorted 1-D array of max-cosine values (ascending).
            If the library has < 2 papers, returns ``[0.0]`` as a
            conservative fallback so any positive similarity scores > 0.
        """
        n = lib_embs.shape[0]
        if n < 2:
            return np.array([0.0], dtype=np.float32)

        # Pairwise cosine (embeddings already L2-normalised)
        sim = lib_embs @ lib_embs.T            # (N, N)
        np.fill_diagonal(sim, -np.inf)          # exclude self
        max_sims = sim.max(axis=1)              # (N,)
        return np.sort(max_sims)

    def _get_lib_distribution(self, lib_embs: np.ndarray) -> np.ndarray:
        """Return (or compute + cache) the library distribution."""
        if self._lib_dist is None:
            self._lib_dist = self._compute_lib_distribution(lib_embs)
        return self._lib_dist

    @staticmethod
    def _cosine_to_score(
        cosines: np.ndarray,
        lib_dist: np.ndarray,
    ) -> np.ndarray:
        """Map max-cosine values → 0–100 percentile scores.

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
        1. **Encode** — Bi-Encoder embeds all papers (incremental for
           the library).
        2. **Pairwise max cosine** — for each new paper, find its best
           cosine similarity to any read paper.
        3. **Library distribution** — compute ``max cosine`` among read
           papers themselves (each paper's nearest neighbour).
        4. **Percentile mapping** — ``score = percentile(new_max_cos,
           lib_distribution) × 100``.

        No Cross-Encoder inference — the entire process is a single
        matrix multiply + percentile lookup.

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

        # ── Step 2: Max cosine per new paper ───────────────────────────
        bi_max, _ = self._pairwise_topk_sim(new_embs, lib_embs, k=1)

        # ── Step 3: Library distribution ───────────────────────────────
        lib_dist = self._get_lib_distribution(lib_embs)

        # ── Step 4: Percentile scoring ─────────────────────────────────
        scores = self._cosine_to_score(bi_max, lib_dist)

        # ── Assemble results ───────────────────────────────────────────
        results: list[RankedPaper] = []
        for idx, paper in enumerate(new_papers):
            results.append(
                RankedPaper(
                    paper=paper,
                    score=round(float(scores[idx]), 1),
                    stage="bi",
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

        Used by the AI Insight panel.  Pipeline:

        1. **Bi-Encoder retrieval** — cosine similarity selects initial
           ``top_k × 3`` candidates from the library (fast).
        2. **Cross-Encoder reranking** — rescores the candidates to pick
           the final ``top_k`` most relevant papers (accurate ordering).
        3. **Display score** — each result shows its **BI cosine
           similarity × 100** (genuine vector similarity, same scale as
           the main percentile reference).

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
            Up to *top_k* ``(paper, cosine_pct)`` pairs sorted by CE
            relevance.  ``cosine_pct`` is BI cosine × 100 ∈ [0, 100].
        """
        if not library:
            return []

        # Reuse cached incremental library embeddings
        lib_embs = self._get_lib_embeddings(library)

        # Encode the target paper
        target_emb = self._encode([_paper_text(paper)])  # shape (1, dim)

        # Pairwise cosine similarity (both sides L2-normalised)
        sims: np.ndarray = (target_emb @ lib_embs.T).flatten()  # (N,)

        # Exclude self (same paper.id)
        for idx, lib_paper in enumerate(library):
            if lib_paper.id is not None and lib_paper.id == paper.id:
                sims[idx] = -1.0

        # Retrieve more candidates for CE to rerank
        retrieve_k = min(top_k * 3, len(library))
        top_idx = np.argsort(sims)[::-1][:retrieve_k]

        if len(top_idx) > 0:
            # Cross-Encoder reranking for accurate ordering
            paper_text = _paper_text(paper)[:_CE_MAX_LEN]
            pairs = [
                [_paper_text(library[int(i)])[:_CE_MAX_LEN], paper_text]
                for i in top_idx
            ]
            try:
                raw = self.cross_encoder.predict(
                    pairs, show_progress_bar=False,
                )
                ce_logits = np.asarray(raw, dtype=np.float64)
                # Reorder by CE relevance (descending)
                order = np.argsort(ce_logits)[::-1][:top_k]
            except Exception:
                # Fallback: keep BI order
                order = np.arange(min(top_k, len(top_idx)))

            results: list[tuple[Paper, float]] = []
            for rank_pos in order:
                orig_idx = int(top_idx[rank_pos])
                cos_pct = round(float(np.clip(sims[orig_idx], 0.0, 1.0)) * 100.0, 1)
                if cos_pct > 0:
                    results.append((library[orig_idx], cos_pct))
            return results

        return []

    # -- cache management ----------------------------------------------------

    def invalidate_cache(self) -> None:
        """Clear all cached library embeddings and distribution.

        Call this when the user's READ library changes (e.g. after export)
        or when the model version changes.
        """
        self._emb_store.clear()
        self._emb_loaded = False
        self._lib_dist = None
        if self._emb_path.exists():
            try:
                self._emb_path.unlink()
            except OSError:
                pass
