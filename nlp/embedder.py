"""
nlp/embedder.py
===============
ResearchFlow AI — Scientific Text Embedding Engine

This module wraps the Sentence Transformers library to generate
dense vector embeddings for academic paper abstracts. It is the
first computation-heavy stage of the ResearchFlow AI NLP pipeline:

  arXiv retrieval → [EMBEDDER] → HDBSCAN clustering → UMAP → gaps

Two embedding models are supported with automatic fallback:

  Primary   : allenai-specter
    - Domain-optimised for scientific text (Allen AI)
    - Trained on citation relationships between papers
    - 768-dimensional embeddings
    - Significantly outperforms general-purpose models on:
        SciFact, SciDocs, TREC-COVID, NFCorpus

  Fallback  : all-MiniLM-L6-v2
    - General-purpose, fast, lightweight (384-dim)
    - Used when SPECTER download fails or GPU memory is limited
    - Still produces good clustering results on short abstracts

Pipeline integration:
  - The SchedulerAgent submits embedding as a HIGH-RAM, HIGH-CPU task
  - The ResourcePolicy gates dispatch on available RAM > 25%
  - Results are cached as .npy arrays (content-addressed by paper IDs)
  - The Embedder exposes both sync and async interfaces so the
    SchedulerAgent can run it in a ThreadPoolExecutor worker thread

OS Concepts:
  - Memory-aware batching    : batch size adapts to available RAM
  - Lazy loading             : model downloaded only on first encode()
  - Cache-aside pattern      : check disk cache before computing
  - Chunked processing       : large paper sets split to avoid OOM

Usage:
    embedder = Embedder()

    # Encode a list of Paper objects
    embeddings = embedder.encode(papers)           # np.ndarray (N, D)

    # Async version (for use in async pipeline stages)
    embeddings = await embedder.async_encode(papers)

    # Check cache without computing
    cached = embedder.load_from_cache(papers)
    if cached is None:
        embeddings = embedder.encode(papers)

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PRIMARY_MODEL   = os.getenv("EMBED_MODEL",    "allenai-specter")
FALLBACK_MODEL  = os.getenv("EMBED_FALLBACK", "all-MiniLM-L6-v2")
CACHE_DIR       = Path(os.getenv("EMBED_CACHE_DIR", "cache/embeddings"))
DEFAULT_BATCH   = int(os.getenv("EMBED_BATCH_SIZE",  "32"))
CACHE_TTL_HOURS = float(os.getenv("EMBED_CACHE_TTL", "48"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model output dimensions (used for validation)
MODEL_DIMS: dict[str, int] = {
    "allenai-specter":   768,
    "all-MiniLM-L6-v2": 384,
    "all-mpnet-base-v2": 768,
}


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EmbeddingResult:
    """
    Wraps the raw numpy array with provenance metadata.
    Stored alongside the .npy cache file as a sidecar JSON.
    """
    embeddings:    np.ndarray    # shape (N, D)
    model_name:    str
    n_papers:      int
    embedding_dim: int
    encode_time_s: float
    cache_hit:     bool
    cache_key:     str

    @property
    def shape(self) -> tuple[int, int]:
        return self.embeddings.shape  # type: ignore[return-value]

    def __repr__(self) -> str:
        src = "cache" if self.cache_hit else f"{self.encode_time_s:.1f}s"
        return (
            f"EmbeddingResult(shape={self.shape}, "
            f"model={self.model_name}, source={src})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDER
# ─────────────────────────────────────────────────────────────────────────────

class Embedder:
    """
    Scientific text embedding engine for ResearchFlow AI.

    Responsibilities:
      1. Lazy-load the sentence transformer model (download on first use)
      2. Encode paper abstracts in memory-aware batches
      3. Cache results as .npy files addressed by paper ID hash
      4. Provide both sync and async interfaces for pipeline integration
      5. Gracefully fall back to a lighter model on OOM or download failure

    Memory management:
      batch_size is computed dynamically from available RAM when
      auto_batch=True (default). This prevents OOM errors when encoding
      large paper sets on memory-constrained machines.

      Batch size heuristic:
        available_ram_gb < 4  → batch = 8
        available_ram_gb < 8  → batch = 16
        available_ram_gb < 16 → batch = 32  (default)
        available_ram_gb ≥ 16 → batch = 64
    """

    def __init__(
        self,
        model_name:  str   = PRIMARY_MODEL,
        fallback:    str   = FALLBACK_MODEL,
        batch_size:  int   = DEFAULT_BATCH,
        auto_batch:  bool  = True,
        normalize:   bool  = True,
        cache_dir:   Path  = CACHE_DIR,
        device:      str | None = None,
    ) -> None:
        self._model_name = model_name
        self._fallback   = fallback
        self._batch_size = batch_size
        self._auto_batch = auto_batch
        self._normalize  = normalize
        self._cache_dir  = cache_dir
        self._device     = device  # None = auto-detect (CUDA if available)
        self._model: Any = None    # loaded lazily

        logger.info(
            "Embedder init | model=%s | fallback=%s | batch=%d | normalize=%s",
            model_name, fallback, batch_size, normalize,
        )

    # ── MODEL LOADING ────────────────────────────────────────────────────────

    @property
    def model(self) -> Any:
        """
        Lazy-loads the sentence transformer.
        Attempts primary model first; falls back on any error.
        """
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        for model_id in [self._model_name, self._fallback]:
            try:
                logger.info("Loading sentence transformer: %s", model_id)
                t0 = time.time()
                self._model = SentenceTransformer(model_id, device=self._device)
                self._model_name = model_id   # update to what actually loaded
                logger.info(
                    "Model loaded: %s in %.1fs | dim=%d",
                    model_id, time.time() - t0,
                    self._model.get_sentence_embedding_dimension(),
                )
                return self._model
            except Exception as exc:
                logger.warning("Failed to load %s: %s — trying fallback", model_id, exc)

        raise RuntimeError(
            f"Could not load any embedding model. "
            f"Tried: {self._model_name}, {self._fallback}"
        )

    @property
    def embedding_dim(self) -> int:
        """Embedding dimensionality of the loaded model."""
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            return MODEL_DIMS.get(self._model_name, 384)

    @property
    def model_name(self) -> str:
        return self._model_name

    # ── ENCODING ─────────────────────────────────────────────────────────────

    def encode(
        self,
        papers: list[Any],
        show_progress: bool = False,
    ) -> EmbeddingResult:
        """
        Encode paper abstracts into dense vectors.

        Accepts a list of Paper objects (uses .abstract field) or
        plain strings. Returns an EmbeddingResult with the embedding
        matrix and metadata.

        Cache-aside: checks disk cache before encoding. If a valid
        cached .npy exists for this paper set, returns it immediately.
        """
        texts, cache_key = self._prepare(papers)

        # ── Cache check ───────────────────────────────────────────────────
        cached = self._load_cache(cache_key)
        if cached is not None:
            logger.info(
                "Embedding cache HIT: %s (%d papers, dim=%d)",
                cache_key[:10], len(texts), cached.shape[1],
            )
            return EmbeddingResult(
                embeddings=cached,
                model_name=self._model_name,
                n_papers=len(texts),
                embedding_dim=cached.shape[1],
                encode_time_s=0.0,
                cache_hit=True,
                cache_key=cache_key,
            )

        # ── Compute ───────────────────────────────────────────────────────
        batch_size = self._compute_batch_size() if self._auto_batch else self._batch_size
        logger.info(
            "Encoding %d abstracts | model=%s | batch=%d",
            len(texts), self._model_name, batch_size,
        )

        t0 = time.time()
        embeddings = self._encode_batched(texts, batch_size, show_progress)
        elapsed = time.time() - t0

        logger.info(
            "Encoding complete: shape=%s in %.1fs (%.0f papers/s)",
            embeddings.shape, elapsed, len(texts) / elapsed if elapsed > 0 else 0,
        )

        # ── Cache write ───────────────────────────────────────────────────
        self._save_cache(cache_key, embeddings)

        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self._model_name,
            n_papers=len(texts),
            embedding_dim=embeddings.shape[1],
            encode_time_s=round(elapsed, 3),
            cache_hit=False,
            cache_key=cache_key,
        )

    async def async_encode(
        self,
        papers: list[Any],
        show_progress: bool = False,
    ) -> EmbeddingResult:
        """
        Async wrapper — runs encode() in a ThreadPoolExecutor so it
        doesn't block the event loop. Used by ResearchPipeline stage runner.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.encode(papers, show_progress),
        )

    def encode_texts(
        self,
        texts: list[str],
        cache_key: str | None = None,
    ) -> np.ndarray:
        """
        Encode a list of raw strings (no Paper objects needed).
        Used by the SemanticSimilarityTool for query embedding.
        Returns raw numpy array.
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        key = cache_key or self._hash_texts(texts)
        cached = self._load_cache(key)
        if cached is not None:
            return cached

        batch_size = self._compute_batch_size() if self._auto_batch else self._batch_size
        embeddings = self._encode_batched(texts, batch_size, show_progress=False)
        self._save_cache(key, embeddings)
        return embeddings

    # ── INTERNAL ENCODING ────────────────────────────────────────────────────

    def _encode_batched(
        self,
        texts: list[str],
        batch_size: int,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts in batches to control peak memory usage.

        Batching is the key memory management strategy:
        - A single 80-paper batch at batch_size=80 uses ~2GB RAM
        - The same papers at batch_size=16 use ~400MB peak RAM
        - Results are identical (batch size doesn't affect quality)

        This is analogous to OS virtual memory chunking — processing
        large datasets in smaller working-set windows to avoid
        exhausting physical memory.
        """
        all_embeddings: list[np.ndarray] = []
        n_batches = -(-len(texts) // batch_size)  # ceil division

        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            batch_num = i // batch_size + 1

            logger.debug(
                "Encoding batch %d/%d (%d texts)",
                batch_num, n_batches, len(batch),
            )

            try:
                batch_emb = self.model.encode(
                    batch,
                    batch_size=len(batch),
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
                all_embeddings.append(batch_emb.astype(np.float32))

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    # OOM: halve batch size and retry this batch
                    new_batch = max(4, batch_size // 2)
                    logger.warning(
                        "OOM on batch %d — retrying with batch_size=%d",
                        batch_num, new_batch,
                    )
                    sub_embs: list[np.ndarray] = []
                    for j in range(0, len(batch), new_batch):
                        sub = batch[j: j + new_batch]
                        sub_emb = self.model.encode(
                            sub,
                            normalize_embeddings=self._normalize,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                        sub_embs.append(sub_emb.astype(np.float32))
                    all_embeddings.append(np.vstack(sub_embs))
                else:
                    raise

        return np.vstack(all_embeddings) if all_embeddings else np.empty(
            (0, self.embedding_dim), dtype=np.float32
        )

    def _prepare(
        self,
        papers: list[Any],
    ) -> tuple[list[str], str]:
        """
        Extract text strings from Paper objects or raw strings,
        and compute the cache key from paper IDs.
        """
        if not papers:
            return [], ""

        # Accept Paper objects or plain strings
        if isinstance(papers[0], str):
            texts = papers
            cache_key = self._hash_texts(texts)
        else:
            # Paper objects — use arxiv_id for stable cache key
            texts = [
                getattr(p, "abstract", "") or getattr(p, "text", "") or str(p)
                for p in papers
            ]
            ids_str = "".join(
                getattr(p, "arxiv_id", str(i)) for i, p in enumerate(papers)
            )
            cache_key = hashlib.sha256(
                f"{self._model_name}::{ids_str}".encode()
            ).hexdigest()[:16]

        return texts, cache_key

    # ── MEMORY-AWARE BATCH SIZING ────────────────────────────────────────────

    def _compute_batch_size(self) -> int:
        """
        Compute a safe batch size based on available system RAM.

        This implements memory-aware admission control at the batch level:
        we probe available RAM and choose a batch size that stays within
        a safe working set, preventing OOM kills during encoding.
        """
        try:
            import psutil
            available_gb = psutil.virtual_memory().available / (1024 ** 3)
            if available_gb < 4:
                batch = 8
            elif available_gb < 8:
                batch = 16
            elif available_gb < 16:
                batch = 32
            else:
                batch = 64
            logger.debug(
                "Auto batch size: %d (available RAM: %.1fGB)", batch, available_gb
            )
            return batch
        except Exception:
            return self._batch_size

    # ── CACHE MANAGEMENT ─────────────────────────────────────────────────────

    def _cache_path(self, cache_key: str) -> Path:
        return self._cache_dir / f"{cache_key}.npy"

    def _load_cache(self, cache_key: str) -> np.ndarray | None:
        """Load cached embeddings if they exist and are within TTL."""
        path = self._cache_path(cache_key)
        if not path.exists():
            return None

        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > CACHE_TTL_HOURS:
            logger.debug("Embedding cache expired: %s (%.1fh)", cache_key[:10], age_hours)
            return None

        try:
            emb = np.load(path)
            logger.debug(
                "Embedding cache loaded: %s shape=%s age=%.1fh",
                cache_key[:10], emb.shape, age_hours,
            )
            return emb
        except Exception as exc:
            logger.warning("Cache load failed for %s: %s", cache_key[:10], exc)
            return None

    def _save_cache(self, cache_key: str, embeddings: np.ndarray) -> None:
        """Save embeddings to disk cache."""
        path = self._cache_path(cache_key)
        try:
            np.save(path, embeddings)
            logger.debug(
                "Embedding cache saved: %s shape=%s size=%.1fMB",
                cache_key[:10], embeddings.shape,
                path.stat().st_size / (1024 ** 2),
            )
        except Exception as exc:
            logger.warning("Cache save failed: %s", exc)

    def load_from_cache(self, papers: list[Any]) -> np.ndarray | None:
        """
        Public cache-check method. Returns cached embeddings or None.
        Used by the pipeline to skip encoding when a cache hit is available.
        """
        _, cache_key = self._prepare(papers)
        return self._load_cache(cache_key) if cache_key else None

    def invalidate_cache(self, papers: list[Any] | None = None) -> int:
        """
        Invalidate cache entries.
        If papers=None, clears entire cache directory.
        Returns number of files deleted.
        """
        if papers is None:
            count = 0
            for f in self._cache_dir.glob("*.npy"):
                f.unlink(missing_ok=True)
                count += 1
            logger.info("Embedding cache cleared: %d files removed", count)
            return count

        _, cache_key = self._prepare(papers)
        path = self._cache_path(cache_key)
        if path.exists():
            path.unlink()
            logger.info("Cache entry invalidated: %s", cache_key[:10])
            return 1
        return 0

    @staticmethod
    def _hash_texts(texts: list[str]) -> str:
        joined = "".join(t[:100] for t in texts)
        return hashlib.sha256(joined.encode()).hexdigest()[:16]

    # ── SIMILARITY ───────────────────────────────────────────────────────────

    def cosine_similarity(
        self,
        a: np.ndarray,
        b: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarity between two embedding matrices.
        If embeddings are L2-normalised (normalize=True), this reduces
        to a simple dot product: similarity = A @ B.T
        """
        if self._normalize:
            return a @ b.T
        # Manually normalise
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
        return a_norm @ b_norm.T

    def most_similar(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Find the top-k most similar embeddings in a corpus.
        Returns [(index, score)] sorted descending by similarity.
        """
        scores = self.cosine_similarity(
            query_embedding.reshape(1, -1), corpus_embeddings
        ).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    # ── STATUS ───────────────────────────────────────────────────────────────

    def cache_stats(self) -> dict:
        """Count and size of cached embedding files."""
        files = list(self._cache_dir.glob("*.npy"))
        total_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)
        return {
            "cached_files":  len(files),
            "total_size_mb": round(total_mb, 2),
            "cache_dir":     str(self._cache_dir),
            "model":         self._model_name,
            "model_loaded":  self._model is not None,
        }

    def status(self) -> dict:
        return {
            "model_name":    self._model_name,
            "embedding_dim": MODEL_DIMS.get(self._model_name, "?"),
            "model_loaded":  self._model is not None,
            "normalize":     self._normalize,
            "batch_size":    self._batch_size,
            "auto_batch":    self._auto_batch,
            **self.cache_stats(),
        }

    def __repr__(self) -> str:
        loaded = "loaded" if self._model is not None else "lazy"
        return (
            f"Embedder(model={self._model_name}, "
            f"dim={MODEL_DIMS.get(self._model_name, '?')}, "
            f"state={loaded})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Return the module-level singleton Embedder."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = Embedder()
    return _default_embedder


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Smoke-test with synthetic paper abstracts.
    Run with: python -m nlp.embedder
    """
    import time as t

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # Synthetic abstracts covering two distinct topics
    abstracts = [
        "We propose a novel vision-language model for medical image diagnosis using cross-modal attention.",
        "Our approach combines BERT with contrastive learning for clinical NLP tasks.",
        "A transformer-based architecture for chest X-ray report generation with zero-shot capabilities.",
        "Federated learning for edge devices with heterogeneous data distributions.",
        "Communication-efficient federated optimization with gradient compression.",
        "Privacy-preserving machine learning on IoT sensor data using federated averaging.",
        "HDBSCAN outperforms k-means on high-dimensional text clustering benchmarks.",
        "Density-based clustering for scientific literature with noise robustness.",
    ]

    class FakePaper:
        def __init__(self, i: int, abstract: str) -> None:
            self.arxiv_id = f"2401.{i:05d}"
            self.abstract = abstract

    papers = [FakePaper(i, a) for i, a in enumerate(abstracts)]

    print("\n═══ EMBEDDER DEMO ═══\n")

    # Use fallback model for faster demo
    embedder = Embedder(model_name=FALLBACK_MODEL, auto_batch=True)
    print(f"  Model: {embedder._model_name}")
    print(f"  Papers: {len(papers)}")

    # First encode (no cache)
    print("\n  First encode (computing)...")
    result = embedder.encode(papers)
    print(f"  Result: {result}")
    print(f"  Shape:  {result.shape}")
    print(f"  Time:   {result.encode_time_s:.2f}s")

    # Second encode (cache hit)
    print("\n  Second encode (from cache)...")
    result2 = embedder.encode(papers)
    print(f"  Cache hit: {result2.cache_hit}")

    # Similarity test
    print("\n  Cosine similarity matrix (8×8):")
    sim = embedder.cosine_similarity(result.embeddings, result.embeddings)
    print("  " + "        ".join(f"P{i}" for i in range(len(abstracts))))
    for i in range(len(abstracts)):
        row = "  ".join(f"{sim[i, j]:.2f}" for j in range(len(abstracts)))
        print(f"  P{i}  {row}")

    # Most similar to P0 (medical VLM)
    print("\n  Most similar to P0 (medical VLM):")
    for idx, score in embedder.most_similar(result.embeddings[0], result.embeddings, top_k=3):
        print(f"    P{idx} ({score:.3f}): {abstracts[idx][:60]}...")

    # Cache stats
    print(f"\n  Cache stats: {embedder.cache_stats()}")
    print(f"\n{embedder}")

    # Invalidate
    embedder.invalidate_cache(papers)
    print(f"\n  Cache invalidated.")


if __name__ == "__main__":
    _demo()
