"""
vector_store/vector_store.py
=============================
ResearchFlow AI — FAISS Vector Store, Paper Indexer & Semantic Search

Merges:
  - faiss_store.py    : FAISS index build, save, load, search
  - paper_indexer.py  : maps paper ID → FAISS vector position
  - semantic_search.py: k-NN query interface for gap detection tools

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)

INDEX_DIR      = Path(os.getenv("FAISS_INDEX_DIR", "artifacts/faiss_indexes"))
INDEX_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_INDEX_NAME = "researchflow"
TOP_K_DEFAULT      = int(os.getenv("FAISS_TOP_K", "10"))


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    """Single result from a k-NN search."""
    rank:       int
    arxiv_id:   str
    score:      float          # cosine similarity (0–1, higher = more similar)
    paper:      Any | None     # Paper object if available
    metadata:   dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rank":     self.rank,
            "arxiv_id": self.arxiv_id,
            "score":    round(self.score, 4),
            "title":    getattr(self.paper, "title", "") if self.paper else "",
            "year":     getattr(self.paper, "year",  "") if self.paper else "",
        }

    def __repr__(self) -> str:
        title = getattr(self.paper, "title", self.arxiv_id)
        return f"SearchResult(rank={self.rank}, score={self.score:.3f}, '{title[:50]}')"


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    FAISS-backed vector store for ResearchFlow AI paper embeddings.

    Combines index management, paper ID mapping, and semantic search
    into a single clean interface consumed by:
      - SemanticSimilarityTool  (ReAct gap detection agent)
      - ResearchAgent           (find_similar_papers)
      - RoadmapAgent            (gap-adjacent paper discovery)

    Index type:
      Uses IndexFlatIP (inner product) on L2-normalised vectors,
      which gives exact cosine similarity. For N > 10,000 papers,
      auto-upgrades to IVFFlat for faster approximate search.

    Persistence:
      Index + ID mapping saved to artifacts/faiss_indexes/<name>.faiss
      and artifacts/faiss_indexes/<name>.pkl respectively.
    """

    def __init__(self, name: str = DEFAULT_INDEX_NAME) -> None:
        self._name      = name
        self._index:    faiss.Index | None = None
        self._id_map:   list[str] = []        # position → arxiv_id
        self._paper_map: dict[str, Any] = {}  # arxiv_id → Paper object
        self._dim:      int = 0

        logger.info("VectorStore init | name=%s", name)

    # ── BUILD ────────────────────────────────────────────────────────────────

    def build(
        self,
        embeddings: np.ndarray,
        papers:     list[Any],
    ) -> None:
        """
        Build the FAISS index from a (N, D) embedding matrix.

        Embeddings must be L2-normalised (as produced by Embedder).
        Uses IndexFlatIP for exact cosine similarity (inner product
        on normalised vectors = cosine similarity).

        Automatically upgrades to IVFFlat for N > 10,000.
        """
        n, d = embeddings.shape
        self._dim = d

        # Build ID map
        self._id_map = [
            getattr(p, "arxiv_id", str(i)) for i, p in enumerate(papers)
        ]
        self._paper_map = {
            getattr(p, "arxiv_id", str(i)): p
            for i, p in enumerate(papers)
        }

        # Ensure float32
        vecs = embeddings.astype(np.float32)

        # Choose index type based on size
        if n > 10_000:
            # IVFFlat: faster approximate search for large corpora
            nlist = min(256, int(n ** 0.5))
            quantiser = faiss.IndexFlatIP(d)
            self._index = faiss.IndexIVFFlat(quantiser, d, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(vecs)
            logger.info("Using IVFFlat index (nlist=%d) for N=%d", nlist, n)
        else:
            # Exact search for typical ResearchFlow paper counts (50–500)
            self._index = faiss.IndexFlatIP(d)

        self._index.add(vecs)
        logger.info(
            "FAISS index built | N=%d D=%d | index_type=%s",
            n, d, type(self._index).__name__,
        )

    # ── SEARCH ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k:           int = TOP_K_DEFAULT,
        exclude_ids:     set[str] | None = None,
    ) -> list[SearchResult]:
        """
        k-NN semantic search against the index.

        Args:
            query_embedding : np.ndarray shape (D,) or (1, D) — L2-normalised
            top_k           : number of results to return
            exclude_ids     : set of arxiv_ids to exclude from results

        Returns list of SearchResult sorted descending by cosine similarity.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("VectorStore.search() called on empty index")
            return []

        vec = query_embedding.astype(np.float32)
        if vec.ndim == 1:
            vec = vec.reshape(1, -1)

        # Fetch extra results to account for exclusions
        fetch_k = min(top_k + len(exclude_ids or set()) + 5, self._index.ntotal)
        scores, indices = self._index.search(vec, fetch_k)

        results: list[SearchResult] = []
        rank = 1
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._id_map):
                continue
            arxiv_id = self._id_map[idx]
            if exclude_ids and arxiv_id in exclude_ids:
                continue
            results.append(SearchResult(
                rank=rank,
                arxiv_id=arxiv_id,
                score=float(score),
                paper=self._paper_map.get(arxiv_id),
            ))
            rank += 1
            if len(results) >= top_k:
                break

        return results

    def search_by_text(
        self,
        query_embedding: np.ndarray,
        top_k:           int = TOP_K_DEFAULT,
        min_score:       float = 0.0,
    ) -> list[SearchResult]:
        """
        Search with an optional minimum similarity threshold.
        Used by SemanticSimilarityTool in the ReAct agent.
        """
        results = self.search(query_embedding, top_k=top_k * 2)
        return [r for r in results if r.score >= min_score][:top_k]

    def find_similar_to_paper(
        self,
        arxiv_id: str,
        top_k:    int = 5,
    ) -> list[SearchResult]:
        """
        Find papers most similar to a given paper (by arxiv_id).
        Excludes the query paper itself from results.
        """
        if arxiv_id not in self._paper_map:
            logger.warning("Paper %s not found in index", arxiv_id)
            return []

        try:
            pos = self._id_map.index(arxiv_id)
        except ValueError:
            return []

        # Reconstruct embedding from index (for IndexFlatIP only)
        try:
            vec = faiss.rev_swig_ptr(
                self._index.get_xb(), self._index.ntotal * self._dim
            ).reshape(self._index.ntotal, self._dim)
            query_vec = vec[pos]
            return self.search(query_vec, top_k=top_k + 1,
                               exclude_ids={arxiv_id})
        except Exception:
            logger.warning("Cannot reconstruct vector for %s — using zero query", arxiv_id)
            return []

    # ── PERSISTENCE ──────────────────────────────────────────────────────────

    def save(self, name: str | None = None) -> Path:
        """Save FAISS index and ID mapping to disk."""
        n = name or self._name
        index_path = INDEX_DIR / f"{n}.faiss"
        meta_path  = INDEX_DIR / f"{n}.pkl"

        if self._index is None:
            raise RuntimeError("No index to save — call build() first")

        faiss.write_index(self._index, str(index_path))
        with open(meta_path, "wb") as f:
            pickle.dump({
                "id_map":    self._id_map,
                "paper_map": self._paper_map,
                "dim":       self._dim,
                "name":      n,
            }, f)

        logger.info(
            "VectorStore saved | %d vectors | %s",
            self._index.ntotal, index_path,
        )
        return index_path

    def load(self, name: str | None = None) -> bool:
        """
        Load FAISS index and ID mapping from disk.
        Returns True on success, False if files not found.
        """
        n = name or self._name
        index_path = INDEX_DIR / f"{n}.faiss"
        meta_path  = INDEX_DIR / f"{n}.pkl"

        if not index_path.exists() or not meta_path.exists():
            logger.debug("VectorStore files not found: %s", n)
            return False

        try:
            self._index = faiss.read_index(str(index_path))
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self._id_map    = meta["id_map"]
            self._paper_map = meta["paper_map"]
            self._dim       = meta["dim"]
            self._name      = meta.get("name", n)
            logger.info(
                "VectorStore loaded | %d vectors | dim=%d",
                self._index.ntotal, self._dim,
            )
            return True
        except Exception as exc:
            logger.error("VectorStore load failed: %s", exc)
            return False

    # ── INDEX MANAGEMENT ─────────────────────────────────────────────────────

    def add(
        self,
        embeddings: np.ndarray,
        papers:     list[Any],
    ) -> int:
        """
        Add new papers to an existing index without rebuilding.
        Returns the number of vectors added.
        """
        if self._index is None:
            self.build(embeddings, papers)
            return len(papers)

        vecs = embeddings.astype(np.float32)
        self._index.add(vecs)

        for i, paper in enumerate(papers):
            arxiv_id = getattr(paper, "arxiv_id", str(self._index.ntotal - len(papers) + i))
            self._id_map.append(arxiv_id)
            self._paper_map[arxiv_id] = paper

        logger.info("Added %d vectors | total=%d", len(papers), self._index.ntotal)
        return len(papers)

    def reset(self) -> None:
        """Clear the index and all mappings."""
        self._index     = None
        self._id_map    = []
        self._paper_map = {}
        self._dim       = 0
        logger.info("VectorStore reset")

    # ── PROPERTIES ───────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index else 0

    @property
    def is_built(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    @property
    def dim(self) -> int:
        return self._dim

    def get_paper(self, arxiv_id: str) -> Any | None:
        return self._paper_map.get(arxiv_id)

    def all_ids(self) -> list[str]:
        return list(self._id_map)

    def status(self) -> dict:
        return {
            "name":     self._name,
            "size":     self.size,
            "dim":      self._dim,
            "is_built": self.is_built,
            "index_dir": str(INDEX_DIR),
        }

    def __repr__(self) -> str:
        return (
            f"VectorStore(name={self._name}, "
            f"size={self.size}, dim={self._dim})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    import numpy as np

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    class FakePaper:
        def __init__(self, i: int, cluster: int) -> None:
            self.arxiv_id = f"2401.{i:05d}"
            self.title    = f"Paper {i} (cluster {cluster})"
            self.year     = 2023
            self.cluster_id = cluster

    rng    = np.random.default_rng(42)
    n, d   = 30, 64
    papers = [FakePaper(i, i % 3) for i in range(n)]

    # Create 3 tight clusters in embedding space
    embeddings = np.zeros((n, d), dtype=np.float32)
    for i, p in enumerate(papers):
        base = np.zeros(d)
        base[p.cluster_id * (d // 3)] = 1.0
        embeddings[i] = base + rng.normal(0, 0.1, d)

    # Normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= norms + 1e-9

    print("\n═══ VECTOR STORE DEMO ═══\n")

    store = VectorStore(name="demo")
    store.build(embeddings, papers)
    print(f"  Built: {store}")

    # Search with a cluster-0 query
    query = embeddings[0]
    results = store.search(query, top_k=5)

    print(f"\n  Top-5 similar to Paper 0 (cluster 0):")
    for r in results:
        print(
            f"    Rank {r.rank} | score={r.score:.3f} | "
            f"{r.arxiv_id} | cluster={getattr(r.paper, 'cluster_id', '?')}"
        )

    # Min-score filtered search
    filtered = store.search_by_text(query, top_k=5, min_score=0.9)
    print(f"\n  Search with min_score=0.9: {len(filtered)} results")

    # Save / load round-trip
    path = store.save()
    print(f"\n  Saved to: {path}")

    store2 = VectorStore(name="demo")
    loaded = store2.load()
    print(f"  Loaded: {store2} | success={loaded}")

    results2 = store2.search(query, top_k=3)
    print(f"  Search after load — top result: {results2[0]}")

    print(f"\n  Status: {store.status()}")

    # Cleanup
    (INDEX_DIR / "demo.faiss").unlink(missing_ok=True)
    (INDEX_DIR / "demo.pkl").unlink(missing_ok=True)
    print("  Demo files cleaned up.")


if __name__ == "__main__":
    _demo()
