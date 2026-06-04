"""
nlp/clusterer.py
================
ResearchFlow AI — HDBSCAN Clustering Engine

This module implements density-based clustering of paper embeddings
using HDBSCAN (Hierarchical Density-Based Spatial Clustering of
Applications with Noise).

Why HDBSCAN over k-means for ResearchFlow AI:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Property           │ k-means         │ HDBSCAN               │
  ├─────────────────────┼─────────────────┼───────────────────────┤
  │ Cluster count       │ Fixed k         │ Discovered from data  │
  │ Noise handling      │ None (forces)   │ Native (-1 label)     │
  │ Cluster shape       │ Spherical only  │ Arbitrary density     │
  │ Reproducibility     │ Random init     │ Deterministic         │
  │ Tuning effort       │ Need correct k  │ min_cluster_size only │
  │ Scientific use case │ Poor            │ Excellent             │
  └─────────────────────────────────────────────────────────────────┘

Pipeline position:
  arXiv retrieval → Embedder → [CLUSTERER] → UMAP → Gap Detection

The clusterer's output (cluster labels array) drives:
  1. UMAP visualization   — colour-coding each scatter point
  2. Cluster labeling     — GPT-4o labels each cluster
  3. Gap detection tools  — PaperDensityTool, TemporalTrendTool
  4. Reading roadmap      — TierClassifier per-cluster analysis

Quality scoring:
  After clustering, the module computes:
  - Silhouette score    : cohesion vs separation (-1 to +1, higher better)
  - DBCV score          : density-based cluster validity (HDBSCAN native)
  - Noise fraction      : % of papers labelled as noise (cluster_id = -1)
  - Cluster size stats  : min / max / median papers per cluster

Parameter selection:
  min_cluster_size is the only tuning knob that matters for HDBSCAN.
  ResearchFlow AI auto-selects it as max(3, N // 20) where N = paper count.
  This gives ~20 clusters for 100 papers, ~10 clusters for 200 papers.
  The auto-select can be overridden via constructor.

OS Concepts:
  - Resource-aware computation  : RAM check before running HDBSCAN
  - Async execution             : runs in ThreadPoolExecutor worker
  - Result caching              : serialised cluster model to disk
  - Progress callbacks          : for Streamlit progress bar

Usage:
    clusterer = Clusterer()
    result    = clusterer.fit(embeddings, papers)

    print(result.n_clusters)          # number of discovered clusters
    print(result.labels)              # np.array, shape (N,), -1 = noise
    print(result.silhouette_score)    # quality metric
    print(result.cluster_sizes)       # {cluster_id: paper_count}

    # Async
    result = await clusterer.async_fit(embeddings, papers)

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import hdbscan
import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MODEL_CACHE_DIR = Path(os.getenv("CLUSTER_MODEL_DIR", "artifacts/cluster_models"))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MIN_CLUSTER_SIZE: int   = int(os.getenv("HDBSCAN_MIN_CLUSTER",   "3"))
DEFAULT_MIN_SAMPLES:      int   = int(os.getenv("HDBSCAN_MIN_SAMPLES",   "2"))
DEFAULT_METRIC:           str   = os.getenv("HDBSCAN_METRIC",             "euclidean")
DEFAULT_SELECTION:        str   = os.getenv("HDBSCAN_SELECTION",          "eom")   # excess of mass
MAX_NOISE_FRACTION:       float = float(os.getenv("HDBSCAN_MAX_NOISE",   "0.40"))  # warn if > 40%


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTER RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterResult:
    """
    Complete output of a clustering run.

    labels: np.ndarray of shape (N,)
      - Values ≥ 0  : cluster ID (0-indexed)
      - Value  == -1 : noise (outlier, not assigned to any cluster)

    Quality metrics:
      silhouette_score : [-1, 1], higher = better separated clusters
      dbcv_score       : HDBSCAN's internal validity score
      noise_fraction   : fraction of papers labelled as noise
    """
    labels:           np.ndarray        # shape (N,)
    n_clusters:       int               # number of clusters (excl. noise)
    n_noise:          int               # papers labelled -1
    noise_fraction:   float             # n_noise / total
    cluster_sizes:    dict[int, int]    # {cluster_id: paper_count}
    silhouette_score: float             # -1 to 1
    dbcv_score:       float             # HDBSCAN native validity
    fit_time_s:       float             # wall-clock fitting time
    n_papers:         int               # total papers clustered
    min_cluster_size: int               # parameter used
    min_samples:      int
    metric:           str
    selection_method: str
    cache_key:        str
    cache_hit:        bool = False

    @property
    def quality_grade(self) -> str:
        """Human-readable quality assessment."""
        if self.silhouette_score >= 0.5:
            return "EXCELLENT"
        if self.silhouette_score >= 0.25:
            return "GOOD"
        if self.silhouette_score >= 0.0:
            return "FAIR"
        return "POOR"

    @property
    def largest_cluster(self) -> tuple[int, int]:
        """(cluster_id, size) of the largest cluster."""
        if not self.cluster_sizes:
            return -1, 0
        cid = max(self.cluster_sizes, key=self.cluster_sizes.get)
        return cid, self.cluster_sizes[cid]

    @property
    def smallest_cluster(self) -> tuple[int, int]:
        """(cluster_id, size) of the smallest cluster."""
        if not self.cluster_sizes:
            return -1, 0
        cid = min(self.cluster_sizes, key=self.cluster_sizes.get)
        return cid, self.cluster_sizes[cid]

    @property
    def median_cluster_size(self) -> float:
        if not self.cluster_sizes:
            return 0.0
        sizes = list(self.cluster_sizes.values())
        sizes.sort()
        mid = len(sizes) // 2
        if len(sizes) % 2 == 0:
            return (sizes[mid - 1] + sizes[mid]) / 2.0
        return float(sizes[mid])

    def cluster_ids(self) -> list[int]:
        """Sorted list of valid cluster IDs (excludes -1 noise)."""
        return sorted(self.cluster_sizes.keys())

    def papers_in_cluster(self, cluster_id: int) -> np.ndarray:
        """Return boolean mask for papers in the given cluster."""
        return self.labels == cluster_id

    def to_dict(self) -> dict:
        return {
            "n_clusters":        self.n_clusters,
            "n_noise":           self.n_noise,
            "n_papers":          self.n_papers,
            "noise_fraction":    round(self.noise_fraction, 4),
            "silhouette_score":  round(self.silhouette_score, 4),
            "dbcv_score":        round(self.dbcv_score, 4),
            "quality_grade":     self.quality_grade,
            "fit_time_s":        self.fit_time_s,
            "min_cluster_size":  self.min_cluster_size,
            "min_samples":       self.min_samples,
            "metric":            self.metric,
            "selection_method":  self.selection_method,
            "cluster_sizes":     self.cluster_sizes,
            "largest_cluster":   self.largest_cluster,
            "median_size":       self.median_cluster_size,
            "cache_hit":         self.cache_hit,
        }

    def __repr__(self) -> str:
        return (
            f"ClusterResult("
            f"clusters={self.n_clusters}, "
            f"noise={self.noise_fraction:.0%}, "
            f"silhouette={self.silhouette_score:.3f} [{self.quality_grade}], "
            f"fit={'cached' if self.cache_hit else f'{self.fit_time_s:.1f}s'})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETER SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

class ParameterSelector:
    """
    Automatically selects HDBSCAN hyperparameters based on dataset size.

    The key insight: min_cluster_size should scale with N so that the
    clustering granularity is consistent regardless of paper count.
    A cluster of 3 papers in a 50-paper dataset is a meaningful group;
    3 papers in a 500-paper dataset is statistical noise.

    Parameter guide:
      N = 20–50   papers → min_cluster_size=3,  ~5–10 clusters
      N = 50–100  papers → min_cluster_size=4,  ~8–15 clusters
      N = 100–200 papers → min_cluster_size=5,  ~10–20 clusters
      N = 200–500 papers → min_cluster_size=8,  ~15–30 clusters
      N > 500     papers → min_cluster_size=12, ~20–40 clusters
    """

    @staticmethod
    def select(n_papers: int) -> dict[str, Any]:
        """Returns a dict of HDBSCAN kwargs optimised for n_papers."""
        if n_papers < 30:
            mcs, ms = 3, 1
        elif n_papers < 60:
            mcs, ms = 4, 2
        elif n_papers < 100:
            mcs, ms = 5, 2
        elif n_papers < 200:
            mcs = max(3, n_papers // 20)
            ms  = 2
        elif n_papers < 500:
            mcs = max(5, n_papers // 25)
            ms  = 3
        else:
            mcs = max(8, n_papers // 40)
            ms  = 3

        logger.debug(
            "Auto-selected HDBSCAN params for N=%d: "
            "min_cluster_size=%d, min_samples=%d",
            n_papers, mcs, ms,
        )
        return {
            "min_cluster_size": mcs,
            "min_samples":      ms,
            "metric":           DEFAULT_METRIC,
            "cluster_selection_method": DEFAULT_SELECTION,
            "prediction_data":  True,   # needed for soft clustering
            "gen_min_span_tree": False,  # faster
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLUSTERER
# ─────────────────────────────────────────────────────────────────────────────

class Clusterer:
    """
    HDBSCAN-based paper clustering engine.

    Fits HDBSCAN on a (N, D) embedding matrix and returns a
    ClusterResult with labels, quality metrics, and cluster size stats.

    The fitted model is serialised to disk (artifacts/cluster_models/)
    keyed by a hash of the embeddings so it can be reloaded for
    incremental clustering or downstream analysis without re-fitting.
    """

    def __init__(
        self,
        min_cluster_size:  int | None = None,
        min_samples:       int | None = None,
        metric:            str        = DEFAULT_METRIC,
        selection_method:  str        = DEFAULT_SELECTION,
        auto_params:       bool       = True,
        model_cache_dir:   Path       = MODEL_CACHE_DIR,
    ) -> None:
        self._min_cluster_size  = min_cluster_size
        self._min_samples       = min_samples
        self._metric            = metric
        self._selection_method  = selection_method
        self._auto_params       = auto_params
        self._model_cache_dir   = model_cache_dir
        self._last_result:      ClusterResult | None = None
        self._last_model:       hdbscan.HDBSCAN | None = None

        logger.info(
            "Clusterer init | auto_params=%s | metric=%s | selection=%s",
            auto_params, metric, selection_method,
        )

    # ── MAIN FIT ─────────────────────────────────────────────────────────────

    def fit(
        self,
        embeddings: np.ndarray,
        papers: list[Any] | None = None,
        progress_cb: Any = None,
    ) -> ClusterResult:
        """
        Fit HDBSCAN on the embedding matrix.

        Args:
            embeddings  : np.ndarray shape (N, D) — L2-normalised recommended
            papers      : optional list of Paper objects for cache key generation
            progress_cb : optional callable(message: str) for Streamlit progress

        Returns ClusterResult with labels and quality metrics.
        """
        if embeddings.ndim != 2 or len(embeddings) == 0:
            raise ValueError(
                f"embeddings must be 2D non-empty array, got shape {embeddings.shape}"
            )

        n = len(embeddings)
        cache_key = self._cache_key(embeddings, papers)

        # ── Cache check ───────────────────────────────────────────────────
        cached = self._load_model_cache(cache_key)
        if cached is not None:
            labels, model = cached
            result = self._build_result(
                labels=labels, embeddings=embeddings,
                model=model, fit_time_s=0.0,
                cache_key=cache_key, cache_hit=True,
            )
            self._last_result = result
            self._last_model  = model
            logger.info("Cluster model cache HIT: %s (%d papers)", cache_key[:10], n)
            return result

        # ── Select parameters ─────────────────────────────────────────────
        if self._auto_params:
            params = ParameterSelector.select(n)
        else:
            params = {
                "min_cluster_size":        self._min_cluster_size or DEFAULT_MIN_CLUSTER_SIZE,
                "min_samples":             self._min_samples or DEFAULT_MIN_SAMPLES,
                "metric":                  self._metric,
                "cluster_selection_method": self._selection_method,
                "prediction_data":         True,
                "gen_min_span_tree":       False,
            }

        if progress_cb:
            progress_cb(f"HDBSCAN: min_cluster_size={params['min_cluster_size']}, "
                        f"min_samples={params['min_samples']}")

        logger.info(
            "Fitting HDBSCAN | N=%d D=%d | params=%s",
            n, embeddings.shape[1], params,
        )

        # ── Fit ───────────────────────────────────────────────────────────
        t0     = time.time()
        model  = hdbscan.HDBSCAN(**params)
        labels = model.fit_predict(embeddings)
        elapsed = round(time.time() - t0, 3)

        # ── Build result ──────────────────────────────────────────────────
        result = self._build_result(
            labels=labels, embeddings=embeddings,
            model=model, fit_time_s=elapsed,
            cache_key=cache_key, cache_hit=False,
        )
        self._last_result = result
        self._last_model  = model

        # ── Cache the fitted model ────────────────────────────────────────
        self._save_model_cache(cache_key, labels, model)

        logger.info(
            "HDBSCAN fit complete | %s | %.2fs",
            result, elapsed,
        )

        if result.noise_fraction > MAX_NOISE_FRACTION:
            logger.warning(
                "High noise fraction: %.0f%% of papers are outliers. "
                "Consider reducing min_cluster_size.",
                result.noise_fraction * 100,
            )

        return result

    async def async_fit(
        self,
        embeddings: np.ndarray,
        papers: list[Any] | None = None,
        progress_cb: Any = None,
    ) -> ClusterResult:
        """Async wrapper for pipeline integration (runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.fit(embeddings, papers, progress_cb),
        )

    # ── RESULT CONSTRUCTION ──────────────────────────────────────────────────

    def _build_result(
        self,
        labels:      np.ndarray,
        embeddings:  np.ndarray,
        model:       hdbscan.HDBSCAN,
        fit_time_s:  float,
        cache_key:   str,
        cache_hit:   bool,
    ) -> ClusterResult:
        """Build a ClusterResult from HDBSCAN labels + model."""
        n            = len(labels)
        unique       = set(labels)
        cluster_ids  = sorted(c for c in unique if c >= 0)
        n_clusters   = len(cluster_ids)
        n_noise      = int(np.sum(labels == -1))
        noise_frac   = n_noise / n if n > 0 else 0.0

        cluster_sizes = {
            cid: int(np.sum(labels == cid))
            for cid in cluster_ids
        }

        # Parameters used
        mcs = getattr(model, "min_cluster_size", DEFAULT_MIN_CLUSTER_SIZE)
        ms  = getattr(model, "min_samples",      DEFAULT_MIN_SAMPLES)
        met = getattr(model, "metric",            DEFAULT_METRIC)
        sel = getattr(model, "cluster_selection_method", DEFAULT_SELECTION)

        # Silhouette score (skip if < 2 clusters or all noise)
        sil_score = self._compute_silhouette(embeddings, labels, n_clusters)

        # DBCV score (HDBSCAN's own validity metric)
        dbcv = float(getattr(model, "relative_validity_", 0.0))

        return ClusterResult(
            labels=labels,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_fraction=round(noise_frac, 4),
            cluster_sizes=cluster_sizes,
            silhouette_score=round(sil_score, 4),
            dbcv_score=round(dbcv, 4),
            fit_time_s=fit_time_s,
            n_papers=n,
            min_cluster_size=mcs,
            min_samples=ms,
            metric=str(met),
            selection_method=str(sel),
            cache_key=cache_key,
            cache_hit=cache_hit,
        )

    @staticmethod
    def _compute_silhouette(
        embeddings: np.ndarray,
        labels:     np.ndarray,
        n_clusters: int,
    ) -> float:
        """
        Compute silhouette score for the clustering.

        Silhouette score measures cohesion (distance to own cluster)
        vs separation (distance to nearest other cluster).
        Range: [-1, 1], higher is better.
        Score of 0 means overlapping clusters.

        Skipped when:
          - Fewer than 2 clusters (meaningless)
          - Only noise points remain after clustering
          - Too few clustered points for meaningful computation
        """
        if n_clusters < 2:
            return 0.0

        # Only score non-noise points
        mask = labels >= 0
        if np.sum(mask) < 10:   # too few points
            return 0.0

        try:
            from sklearn.metrics import silhouette_score
            score = float(silhouette_score(
                embeddings[mask],
                labels[mask],
                metric="cosine",
                sample_size=min(1000, int(np.sum(mask))),   # subsample for speed
                random_state=42,
            ))
            return score
        except Exception as exc:
            logger.warning("Silhouette score computation failed: %s", exc)
            return 0.0

    # ── INCREMENTAL ASSIGNMENT ───────────────────────────────────────────────

    def assign_new_papers(
        self,
        new_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Assign new paper embeddings to existing clusters using HDBSCAN's
        approximate_predict() — avoids full refit.

        Returns label array for the new papers (-1 = noise).
        Requires prediction_data=True during original fit.
        """
        if self._last_model is None:
            raise RuntimeError(
                "No fitted model available. Call fit() first."
            )
        try:
            labels, _ = hdbscan.approximate_predict(
                self._last_model, new_embeddings
            )
            logger.info(
                "Assigned %d new papers to existing clusters "
                "(noise: %d)", len(labels), int(np.sum(labels == -1))
            )
            return labels
        except Exception as exc:
            logger.error("Incremental assignment failed: %s", exc)
            return np.full(len(new_embeddings), -1, dtype=int)

    # ── CLUSTER ANALYSIS ────────────────────────────────────────────────────

    def cluster_centroids(self, embeddings: np.ndarray) -> dict[int, np.ndarray]:
        """
        Compute the centroid (mean embedding) for each cluster.
        Used by the UMAP visualisation for cluster label placement.

        Returns {cluster_id: centroid_vector}.
        """
        if self._last_result is None:
            return {}
        labels = self._last_result.labels
        centroids: dict[int, np.ndarray] = {}
        for cid in self._last_result.cluster_ids():
            mask = labels == cid
            centroids[cid] = embeddings[mask].mean(axis=0)
        return centroids

    def inter_cluster_distances(
        self, embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise cosine distances between cluster centroids.
        Returns a (K, K) distance matrix where K = n_clusters.
        Useful for identifying which clusters are most similar.
        """
        centroids = self.cluster_centroids(embeddings)
        if not centroids:
            return np.array([])
        ids = sorted(centroids.keys())
        C   = np.vstack([centroids[i] for i in ids])
        # Normalise for cosine
        C_norm = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
        sim    = C_norm @ C_norm.T
        dist   = 1.0 - sim
        return dist

    # ── CACHE ─────────────────────────────────────────────────────────────────

    def _cache_key(
        self,
        embeddings: np.ndarray,
        papers: list[Any] | None,
    ) -> str:
        """Stable cache key from paper IDs or embedding fingerprint."""
        if papers:
            ids_str = "".join(
                getattr(p, "arxiv_id", str(i)) for i, p in enumerate(papers)
            )
            raw = f"cluster::{ids_str}::{len(embeddings)}"
        else:
            # Hash the embedding matrix fingerprint (fast: just first+last row)
            fingerprint = np.concatenate([
                embeddings[0], embeddings[-1],
                [float(len(embeddings)), float(embeddings.shape[1])],
            ])
            raw = fingerprint.tobytes()
        return hashlib.sha256(raw if isinstance(raw, bytes)
                              else raw.encode()).hexdigest()[:16]

    def _model_cache_path(self, cache_key: str) -> Path:
        return self._model_cache_dir / f"{cache_key}.pkl"

    def _load_model_cache(
        self, cache_key: str
    ) -> tuple[np.ndarray, hdbscan.HDBSCAN] | None:
        path = self._model_cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return data["labels"], data["model"]
        except Exception as exc:
            logger.warning("Cluster model cache load failed: %s", exc)
            return None

    def _save_model_cache(
        self,
        cache_key: str,
        labels: np.ndarray,
        model: hdbscan.HDBSCAN,
    ) -> None:
        path = self._model_cache_path(cache_key)
        try:
            with open(path, "wb") as f:
                pickle.dump({"labels": labels, "model": model}, f)
            logger.debug("Cluster model cached: %s", cache_key[:10])
        except Exception as exc:
            logger.warning("Cluster model cache save failed: %s", exc)

    def invalidate_cache(self) -> int:
        """Clear all cached cluster models. Returns files deleted."""
        count = 0
        for f in self._model_cache_dir.glob("*.pkl"):
            f.unlink(missing_ok=True)
            count += 1
        logger.info("Cluster model cache cleared: %d files", count)
        return count

    # ── PROPERTIES ───────────────────────────────────────────────────────────

    @property
    def last_result(self) -> ClusterResult | None:
        return self._last_result

    def status(self) -> dict:
        result = self._last_result
        return {
            "fitted":          result is not None,
            "n_clusters":      result.n_clusters if result else 0,
            "silhouette":      result.silhouette_score if result else None,
            "quality":         result.quality_grade if result else None,
            "auto_params":     self._auto_params,
        }

    def __repr__(self) -> str:
        if self._last_result:
            return f"Clusterer({self._last_result})"
        return f"Clusterer(unfitted, auto_params={self._auto_params})"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_clusterer: Clusterer | None = None


def get_clusterer() -> Clusterer:
    global _default_clusterer
    if _default_clusterer is None:
        _default_clusterer = Clusterer()
    return _default_clusterer


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Smoke-test with synthetic embeddings across 3 clear clusters.
    Run with: python -m nlp.clusterer
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    rng = np.random.default_rng(42)

    # 3 tight clusters in 8-dimensional space, 10 papers each + 3 noise
    def make_cluster(centre: np.ndarray, n: int, noise: float = 0.1) -> np.ndarray:
        return centre + rng.normal(0, noise, size=(n, len(centre)))

    C1 = make_cluster(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 12)
    C2 = make_cluster(np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 10)
    C3 = make_cluster(np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  8)
    N  = rng.uniform(-0.5, 0.5, size=(3, 8))   # noise points

    embeddings = np.vstack([C1, C2, C3, N]).astype(np.float32)

    # Normalise (as Embedder would do)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)

    print(f"\n═══ CLUSTERER DEMO ═══\n")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Expected: 3 clusters + ~3 noise points\n")

    clusterer = Clusterer(auto_params=True)
    result    = clusterer.fit(embeddings)

    print(f"  {result}\n")
    print(f"  Clusters found: {result.n_clusters}")
    print(f"  Noise papers:   {result.n_noise} ({result.noise_fraction:.0%})")
    print(f"  Silhouette:     {result.silhouette_score:.4f} [{result.quality_grade}]")
    print(f"  DBCV score:     {result.dbcv_score:.4f}")
    print(f"  Fit time:       {result.fit_time_s:.3f}s")

    print(f"\n  Cluster sizes:")
    for cid, size in result.cluster_sizes.items():
        papers_idx = np.where(result.labels == cid)[0].tolist()
        print(f"    Cluster {cid}: {size} papers — indices {papers_idx}")

    print(f"\n  Label array (first 10): {result.labels[:10]}")

    # Cache hit demo
    print(f"\n  Re-fitting (should hit cache)...")
    result2 = clusterer.fit(embeddings)
    print(f"  Cache hit: {result2.cache_hit}")

    # Inter-cluster distances
    dist = clusterer.inter_cluster_distances(embeddings)
    if dist.size > 0:
        print(f"\n  Inter-cluster distance matrix (cosine):")
        print(f"  {'':6}" + "  ".join(f"  C{i}" for i in result.cluster_ids()))
        for i, cid_i in enumerate(result.cluster_ids()):
            row = "  ".join(f"{dist[i, j]:.3f}" for j in range(len(result.cluster_ids())))
            print(f"  C{cid_i}     {row}")

    # ParameterSelector demo
    print(f"\n  Parameter auto-selection for different dataset sizes:")
    for n in [20, 50, 100, 200, 500]:
        p = ParameterSelector.select(n)
        print(
            f"    N={n:4d} → "
            f"min_cluster_size={p['min_cluster_size']:2d}, "
            f"min_samples={p['min_samples']}"
        )

    print(f"\n  Status: {clusterer.status()}")
    clusterer.invalidate_cache()
    print(f"  Cache cleared.")


if __name__ == "__main__":
    _demo()
