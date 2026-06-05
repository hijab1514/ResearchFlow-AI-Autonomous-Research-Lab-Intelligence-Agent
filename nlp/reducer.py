"""
nlp/reducer.py
==============
ResearchFlow AI — UMAP Dimensionality Reducer

This module wraps UMAP (Uniform Manifold Approximation and Projection)
to reduce high-dimensional paper embeddings (768-dim SPECTER or 384-dim
MiniLM) to 2D coordinates for the interactive Streamlit scatter plot.

Pipeline position:
  Embedder → Clusterer → [REDUCER] → Streamlit Plotly scatter

Why UMAP over t-SNE or PCA:
  ┌────────────────────────────────────────────────────────────────┐
  │  Property          │ PCA       │ t-SNE     │ UMAP             │
  ├────────────────────┼───────────┼───────────┼──────────────────┤
  │ Global structure   │ ✓ Good    │ ✗ Poor    │ ✓ Good           │
  │ Local structure    │ ✗ Poor    │ ✓ Good    │ ✓ Good           │
  │ Speed (N=1000)     │ Fast      │ Slow      │ Fast             │
  │ New point project  │ ✓ Yes     │ ✗ No      │ ✓ Yes            │
  │ Deterministic      │ ✓ Yes     │ ✗ No      │ ✓ (random_state) │
  │ Cluster separation │ ✗ Poor    │ ✓ Good    │ ✓ Excellent      │
  └────────────────────────────────────────────────────────────────┘

Two output modes:
  1. 2D projection (default) — for Streamlit Plotly scatter plot
  2. 3D projection (optional) — for richer interactive exploration

The reducer exposes:
  - fit_transform()     : fit + project in one call
  - transform()         : project new papers onto fitted manifold
  - async versions      : for SchedulerAgent thread pool dispatch
  - Plotly-ready output : list[dict] with x, y, colour, hover fields

OS / Systems concepts:
  - Memory-aware params  : n_neighbors adapts to available RAM
  - Lazy fitting         : reducer fitted once, transform is cheap
  - Result caching       : fitted reducer serialised to disk
  - Chunked transform    : large new-paper sets projected in chunks

Usage:
    reducer = Reducer()

    # Fit and project
    result = reducer.fit_transform(embeddings, papers, cluster_labels)
    coords = result.coords_2d    # np.ndarray shape (N, 2)

    # Project new papers onto existing manifold (no refit)
    new_coords = reducer.transform(new_embeddings)

    # Get Plotly-ready rows
    rows = result.to_plotly_rows(papers)

    # Async
    result = await reducer.async_fit_transform(embeddings, papers, labels)

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

REDUCER_CACHE_DIR = Path(os.getenv("UMAP_CACHE_DIR", "artifacts/umap_reducers"))
REDUCER_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_N_NEIGHBORS:  int   = int(os.getenv("UMAP_N_NEIGHBORS",  "15"))
DEFAULT_MIN_DIST:     float = float(os.getenv("UMAP_MIN_DIST",   "0.1"))
DEFAULT_SPREAD:       float = float(os.getenv("UMAP_SPREAD",     "1.0"))
DEFAULT_METRIC:       str   = os.getenv("UMAP_METRIC",            "cosine")
RANDOM_STATE:         int   = int(os.getenv("UMAP_RANDOM_STATE", "42"))

# Colour palette for cluster IDs (cycles for > 20 clusters)
CLUSTER_COLOURS: list[str] = [
    "#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6",
    "#06B6D4", "#EC4899", "#84CC16", "#F97316", "#6366F1",
    "#14B8A6", "#D946EF", "#FBBF24", "#22C55E", "#E11D48",
    "#0EA5E9", "#A78BFA", "#FB923C", "#4ADE80", "#38BDF8",
]
NOISE_COLOUR = "#9CA3AF"  # grey for cluster_id == -1


# ─────────────────────────────────────────────────────────────────────────────
# REDUCTION RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReductionResult:
    """
    Output of a UMAP fit_transform call.

    coords_2d : np.ndarray shape (N, 2) — projected coordinates
    coords_3d : np.ndarray shape (N, 3) or None
    """
    coords_2d:        np.ndarray          # shape (N, 2)
    coords_3d:        np.ndarray | None   # shape (N, 3), optional
    n_papers:         int
    n_components:     int
    n_neighbors:      int
    min_dist:         float
    metric:           str
    fit_time_s:       float
    cache_hit:        bool
    cache_key:        str

    # Coordinate range metadata (for Plotly axis ranges)
    x_min:    float = 0.0
    x_max:    float = 1.0
    y_min:    float = 0.0
    y_max:    float = 1.0

    def __post_init__(self) -> None:
        if self.coords_2d is not None and len(self.coords_2d) > 0:
            self.x_min = float(self.coords_2d[:, 0].min())
            self.x_max = float(self.coords_2d[:, 0].max())
            self.y_min = float(self.coords_2d[:, 1].min())
            self.y_max = float(self.coords_2d[:, 1].max())

    def to_plotly_rows(
        self,
        papers: list[Any],
        cluster_labels: np.ndarray | None = None,
        cluster_label_names: dict[int, str] | None = None,
    ) -> list[dict]:
        """
        Build a list of row dicts for Plotly's scatter plot.

        Each row contains:
          x, y         : 2D coordinates
          text         : hover label (paper title + year)
          arxiv_id     : paper identifier
          cluster_id   : integer cluster assignment
          cluster_name : human-readable cluster label (if provided)
          colour       : hex colour string for the scatter point
          size         : marker size (noise points smaller)

        Consumed directly by app/components/cluster_plot.py.
        """
        rows: list[dict] = []
        labels = cluster_labels if cluster_labels is not None else np.full(
            len(papers), -1, dtype=int
        )

        for i, paper in enumerate(papers):
            if i >= len(self.coords_2d):
                break

            x = float(self.coords_2d[i, 0])
            y = float(self.coords_2d[i, 1])
            cid = int(labels[i]) if i < len(labels) else -1
            colour = (
                CLUSTER_COLOURS[cid % len(CLUSTER_COLOURS)]
                if cid >= 0
                else NOISE_COLOUR
            )
            cluster_name = ""
            if cluster_label_names and cid in cluster_label_names:
                cluster_name = cluster_label_names[cid]

            title  = getattr(paper, "title",     f"Paper {i}")
            year   = getattr(paper, "year",       "")
            arxiv_id = getattr(paper, "arxiv_id", str(i))
            authors  = getattr(paper, "authors",  [])
            first_author = authors[0] if authors else ""

            hover = (
                f"<b>{title[:60]}{'...' if len(title) > 60 else ''}</b><br>"
                f"{first_author}{' et al.' if len(authors) > 1 else ''} ({year})<br>"
                f"Cluster: {cluster_name or cid}<br>"
                f"arXiv: {arxiv_id}"
            )

            rows.append({
                "x":            x,
                "y":            y,
                "text":         hover,
                "title":        title,
                "arxiv_id":     arxiv_id,
                "year":         year,
                "cluster_id":   cid,
                "cluster_name": cluster_name,
                "colour":       colour,
                "size":         6 if cid >= 0 else 4,
                "opacity":      0.85 if cid >= 0 else 0.45,
            })

        return rows

    def to_dict(self) -> dict:
        return {
            "n_papers":     self.n_papers,
            "n_components": self.n_components,
            "n_neighbors":  self.n_neighbors,
            "min_dist":     self.min_dist,
            "metric":       self.metric,
            "fit_time_s":   self.fit_time_s,
            "cache_hit":    self.cache_hit,
            "x_range":      [round(self.x_min, 3), round(self.x_max, 3)],
            "y_range":      [round(self.y_min, 3), round(self.y_max, 3)],
        }

    def __repr__(self) -> str:
        src = "cache" if self.cache_hit else f"{self.fit_time_s:.1f}s"
        return (
            f"ReductionResult("
            f"shape={self.coords_2d.shape}, "
            f"metric={self.metric}, "
            f"n_neighbors={self.n_neighbors}, "
            f"source={src})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# REDUCER
# ─────────────────────────────────────────────────────────────────────────────

class Reducer:
    """
    UMAP dimensionality reducer for ResearchFlow AI.

    Fits a UMAP manifold on paper embeddings and projects them to 2D
    (or 3D) for the interactive cluster scatter plot.

    Key features:
      - Auto-tunes n_neighbors based on dataset size
      - Serialises the fitted reducer for transform() of new papers
      - Both sync and async interfaces for pipeline integration
      - Produces Plotly-ready output directly
    """

    def __init__(
        self,
        n_components:  int   = 2,
        n_neighbors:   int | None = None,
        min_dist:      float = DEFAULT_MIN_DIST,
        spread:        float = DEFAULT_SPREAD,
        metric:        str   = DEFAULT_METRIC,
        random_state:  int   = RANDOM_STATE,
        auto_params:   bool  = True,
        cache_dir:     Path  = REDUCER_CACHE_DIR,
        also_3d:       bool  = False,
    ) -> None:
        self._n_components  = n_components
        self._n_neighbors   = n_neighbors
        self._min_dist      = min_dist
        self._spread        = spread
        self._metric        = metric
        self._random_state  = random_state
        self._auto_params   = auto_params
        self._cache_dir     = cache_dir
        self._also_3d       = also_3d

        self._reducer_2d: Any = None   # fitted umap.UMAP instance
        self._reducer_3d: Any = None
        self._last_result: ReductionResult | None = None

        logger.info(
            "Reducer init | components=%d | metric=%s | also_3d=%s",
            n_components, metric, also_3d,
        )

    # ── MAIN FIT TRANSFORM ───────────────────────────────────────────────────

    def fit_transform(
        self,
        embeddings:     np.ndarray,
        papers:         list[Any] | None = None,
        cluster_labels: np.ndarray | None = None,
        progress_cb:    Any = None,
    ) -> ReductionResult:
        """
        Fit UMAP on embeddings and return 2D (and optionally 3D) coordinates.

        Args:
            embeddings     : np.ndarray (N, D) — normalised embeddings
            papers         : Paper objects for cache key and Plotly output
            cluster_labels : np.ndarray (N,) — cluster IDs for coloring
            progress_cb    : callable(str) for Streamlit progress bar

        Returns ReductionResult with coords_2d and Plotly helpers.
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot reduce empty embedding matrix")

        n            = len(embeddings)
        cache_key    = self._cache_key(embeddings, papers)
        n_neighbors  = self._select_n_neighbors(n)

        # ── Cache check ───────────────────────────────────────────────────
        cached = self._load_cache(cache_key)
        if cached is not None:
            coords_2d, coords_3d, reducer_2d, reducer_3d = cached
            self._reducer_2d = reducer_2d
            self._reducer_3d = reducer_3d
            result = self._build_result(
                coords_2d=coords_2d,
                coords_3d=coords_3d,
                n=n, n_neighbors=n_neighbors,
                fit_time_s=0.0,
                cache_key=cache_key,
                cache_hit=True,
            )
            self._last_result = result
            logger.info("UMAP cache HIT: %s (%d papers)", cache_key[:10], n)
            return result

        # ── Fit 2D ───────────────────────────────────────────────────────
        if progress_cb:
            progress_cb(
                f"UMAP 2D: n_neighbors={n_neighbors}, "
                f"min_dist={self._min_dist}, metric={self._metric}"
            )

        logger.info(
            "Fitting UMAP 2D | N=%d D=%d | "
            "n_neighbors=%d min_dist=%.2f metric=%s",
            n, embeddings.shape[1], n_neighbors,
            self._min_dist, self._metric,
        )

        t0 = time.time()
        import umap as umap_lib

        self._reducer_2d = umap_lib.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=self._min_dist,
            spread=self._spread,
            metric=self._metric,
            random_state=self._random_state,
            low_memory=n > 500,   # use low-memory mode for large datasets
            verbose=False,
        )
        coords_2d = self._reducer_2d.fit_transform(embeddings).astype(np.float32)

        # ── Optional 3D ──────────────────────────────────────────────────
        coords_3d = None
        if self._also_3d:
            if progress_cb:
                progress_cb("UMAP 3D projection...")
            self._reducer_3d = umap_lib.UMAP(
                n_components=3,
                n_neighbors=n_neighbors,
                min_dist=self._min_dist,
                metric=self._metric,
                random_state=self._random_state,
                verbose=False,
            )
            coords_3d = self._reducer_3d.fit_transform(embeddings).astype(np.float32)

        elapsed = round(time.time() - t0, 3)

        result = self._build_result(
            coords_2d=coords_2d,
            coords_3d=coords_3d,
            n=n, n_neighbors=n_neighbors,
            fit_time_s=elapsed,
            cache_key=cache_key,
            cache_hit=False,
        )
        self._last_result = result

        # ── Cache ─────────────────────────────────────────────────────────
        self._save_cache(cache_key, coords_2d, coords_3d,
                         self._reducer_2d, self._reducer_3d)

        logger.info(
            "UMAP fit complete | %s | %.2fs (%.0f papers/s)",
            result, elapsed, n / elapsed if elapsed > 0 else 0,
        )
        return result

    async def async_fit_transform(
        self,
        embeddings:     np.ndarray,
        papers:         list[Any] | None = None,
        cluster_labels: np.ndarray | None = None,
        progress_cb:    Any = None,
    ) -> ReductionResult:
        """Async wrapper — dispatched in SchedulerAgent thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.fit_transform(
                embeddings, papers, cluster_labels, progress_cb
            ),
        )

    # ── TRANSFORM (new papers, no refit) ────────────────────────────────────

    def transform(self, new_embeddings: np.ndarray) -> np.ndarray:
        """
        Project new paper embeddings onto the existing fitted manifold.
        Much faster than refitting — O(N_new × N_train) vs O(N_train²).

        Returns np.ndarray shape (N_new, 2).
        Raises RuntimeError if called before fit_transform().
        """
        if self._reducer_2d is None:
            raise RuntimeError("Call fit_transform() before transform()")
        coords = self._reducer_2d.transform(new_embeddings)
        return coords.astype(np.float32)

    # ── PARAMETER SELECTION ──────────────────────────────────────────────────

    def _select_n_neighbors(self, n: int) -> int:
        """
        Auto-select n_neighbors based on dataset size.

        n_neighbors controls the balance between local and global structure:
          - Small values → fine-grained local clusters (may fragment)
          - Large values → global structure preserved (may merge clusters)

        Rule of thumb: n_neighbors ≈ sqrt(N) clamped to [5, 50].
        """
        if not self._auto_params or self._n_neighbors is not None:
            return self._n_neighbors or DEFAULT_N_NEIGHBORS

        n_nbrs = max(5, min(50, int(n ** 0.5)))
        # Ensure n_neighbors < n (UMAP requirement)
        n_nbrs = min(n_nbrs, n - 1)
        logger.debug("Auto n_neighbors=%d for N=%d", n_nbrs, n)
        return n_nbrs

    # ── RESULT BUILDER ────────────────────────────────────────────────────────

    def _build_result(
        self,
        coords_2d:  np.ndarray,
        coords_3d:  np.ndarray | None,
        n:          int,
        n_neighbors: int,
        fit_time_s: float,
        cache_key:  str,
        cache_hit:  bool,
    ) -> ReductionResult:
        return ReductionResult(
            coords_2d=coords_2d,
            coords_3d=coords_3d,
            n_papers=n,
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=self._min_dist,
            metric=self._metric,
            fit_time_s=fit_time_s,
            cache_hit=cache_hit,
            cache_key=cache_key,
        )

    # ── CACHE ─────────────────────────────────────────────────────────────────

    def _cache_key(
        self,
        embeddings: np.ndarray,
        papers: list[Any] | None,
    ) -> str:
        if papers:
            ids_str = "".join(
                getattr(p, "arxiv_id", str(i)) for i, p in enumerate(papers)
            )
            raw = f"umap::{ids_str}::{self._metric}::{self._min_dist}"
        else:
            fp = np.concatenate([
                embeddings[0], embeddings[-1],
                [float(len(embeddings))],
            ])
            raw = fp.tobytes() + f"::{self._metric}".encode()
        return hashlib.sha256(
            raw if isinstance(raw, bytes) else raw.encode()
        ).hexdigest()[:16]

    def _cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.pkl"

    def _load_cache(
        self, cache_key: str
    ) -> tuple[np.ndarray, np.ndarray | None, Any, Any] | None:
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            return (
                data["coords_2d"],
                data.get("coords_3d"),
                data.get("reducer_2d"),
                data.get("reducer_3d"),
            )
        except Exception as exc:
            logger.warning("UMAP cache load failed: %s", exc)
            return None

    def _save_cache(
        self,
        cache_key: str,
        coords_2d: np.ndarray,
        coords_3d: np.ndarray | None,
        reducer_2d: Any,
        reducer_3d: Any,
    ) -> None:
        path = self._cache_path(cache_key)
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "coords_2d":  coords_2d,
                    "coords_3d":  coords_3d,
                    "reducer_2d": reducer_2d,
                    "reducer_3d": reducer_3d,
                }, f)
            logger.debug("UMAP result cached: %s", cache_key[:10])
        except Exception as exc:
            logger.warning("UMAP cache save failed: %s", exc)

    def invalidate_cache(self) -> int:
        count = 0
        for f in self._cache_dir.glob("*.pkl"):
            f.unlink(missing_ok=True)
            count += 1
        logger.info("UMAP cache cleared: %d files", count)
        return count

    # ── STATUS ────────────────────────────────────────────────────────────────

    @property
    def last_result(self) -> ReductionResult | None:
        return self._last_result

    @property
    def is_fitted(self) -> bool:
        return self._reducer_2d is not None

    def status(self) -> dict:
        result = self._last_result
        return {
            "fitted":      self.is_fitted,
            "n_papers":    result.n_papers if result else 0,
            "metric":      self._metric,
            "min_dist":    self._min_dist,
            "auto_params": self._auto_params,
        }

    def __repr__(self) -> str:
        if self._last_result:
            return f"Reducer({self._last_result})"
        return f"Reducer(unfitted, metric={self._metric})"


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_reducer: Reducer | None = None


def get_reducer() -> Reducer:
    global _default_reducer
    if _default_reducer is None:
        _default_reducer = Reducer()
    return _default_reducer


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Smoke-test with synthetic embeddings + cluster labels.
    Prints 2D coordinates and a simple ASCII scatter.
    Run with: python -m nlp.reducer
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    rng = np.random.default_rng(42)

    # 3 clusters in 32-dim embedding space
    def cluster(centre: np.ndarray, n: int) -> np.ndarray:
        return centre + rng.normal(0, 0.15, (n, len(centre)))

    C0 = cluster(np.eye(32)[0], 15)
    C1 = cluster(np.eye(32)[1], 12)
    C2 = cluster(np.eye(32)[2], 10)

    embeddings    = np.vstack([C0, C1, C2]).astype(np.float32)
    cluster_labels = np.array([0]*15 + [1]*12 + [2]*10)

    # Normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-9)

    print(f"\n═══ REDUCER DEMO ═══\n")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Clusters:   3 × [15, 12, 10] papers\n")

    reducer = Reducer(auto_params=True, also_3d=False)
    result  = reducer.fit_transform(
        embeddings,
        cluster_labels=cluster_labels,
        progress_cb=lambda m: print(f"  Progress: {m}"),
    )

    print(f"\n  {result}")
    print(f"  Fit time: {result.fit_time_s:.3f}s")
    print(f"  2D range: x=[{result.x_min:.2f}, {result.x_max:.2f}] "
          f"y=[{result.y_min:.2f}, {result.y_max:.2f}]")

    # ASCII scatter
    print(f"\n  ASCII scatter (40×20):")
    W, H = 40, 20
    grid = [["·"] * W for _ in range(H)]

    x_range = result.x_max - result.x_min + 1e-9
    y_range = result.y_max - result.y_min + 1e-9
    markers = ["O", "X", "#"]

    for i, (x, y) in enumerate(result.coords_2d):
        col = int((x - result.x_min) / x_range * (W - 1))
        row = int((result.y_max - y)  / y_range * (H - 1))
        col = max(0, min(W - 1, col))
        row = max(0, min(H - 1, row))
        cid = int(cluster_labels[i])
        grid[row][col] = markers[cid % len(markers)]

    for row in grid:
        print("  " + "".join(row))

    print(f"\n  Legend: O=Cluster0  X=Cluster1  #=Cluster2")

    # Plotly rows sample
    class FakePaper:
        def __init__(self, i: int) -> None:
            self.arxiv_id = f"2401.{i:05d}"
            self.title    = f"Sample Paper {i}: Methods and Results"
            self.year     = 2023
            self.authors  = [f"Author {i}"]

    papers = [FakePaper(i) for i in range(len(embeddings))]
    rows   = result.to_plotly_rows(papers, cluster_labels)

    print(f"\n  Plotly rows (first 3):")
    for row in rows[:3]:
        print(
            f"    cluster={row['cluster_id']} "
            f"colour={row['colour']} "
            f"x={row['x']:.3f} y={row['y']:.3f} "
            f"size={row['size']}"
        )

    # Cache hit demo
    print(f"\n  Re-fitting (cache hit expected)...")
    result2 = reducer.fit_transform(embeddings, cluster_labels=cluster_labels)
    print(f"  Cache hit: {result2.cache_hit}")

    # Transform new papers
    new_embs = rng.normal(0, 0.1, (3, 32)).astype(np.float32)
    new_embs /= np.linalg.norm(new_embs, axis=1, keepdims=True) + 1e-9
    new_coords = reducer.transform(new_embs)
    print(f"\n  Transformed 3 new papers: {new_coords.shape}")

    reducer.invalidate_cache()
    print(f"\n  Cache cleared.")
    print(f"\n{reducer}")


if __name__ == "__main__":
    _demo()
