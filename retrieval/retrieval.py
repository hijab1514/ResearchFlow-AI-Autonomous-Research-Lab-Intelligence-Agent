"""
retrieval/retrieval.py
======================
ResearchFlow AI — arXiv Client, Paper Parser & Cache Manager

Merges:
  - arxiv_client.py  : arXiv API wrapper with pagination + dedup
  - paper_parser.py  : metadata extraction → Paper dataclass
  - cache_manager.py : local JSON cache with TTL & eviction

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import arxiv

logger = logging.getLogger(__name__)

CACHE_DIR       = Path(os.getenv("ARXIV_CACHE_DIR", "cache/papers"))
CACHE_TTL_HOURS = float(os.getenv("CACHE_TTL_HOURS", "24"))
MAX_RESULTS     = int(os.getenv("ARXIV_MAX_RESULTS", "80"))
MAX_CACHE_FILES = int(os.getenv("ARXIV_MAX_CACHE_FILES", "200"))

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAPER DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    """
    Canonical representation of a single arXiv paper.
    Shared across the entire ResearchFlow AI pipeline.
    """
    arxiv_id:   str
    title:      str
    abstract:   str
    authors:    list[str]
    published:  str           # ISO date "YYYY-MM-DD"
    year:       int
    categories: list[str]
    url:        str
    cluster_id: int = -1      # assigned by Clusterer (-1 = noise)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Paper:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return f"Paper(id={self.arxiv_id}, year={self.year}, '{self.title[:50]}')"


# ─────────────────────────────────────────────────────────────────────────────
# CACHE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class CacheManager:
    """
    Local JSON cache for arXiv API responses.

    Cache-aside pattern:
      1. Check cache (O(1) file existence check)
      2. If HIT and within TTL → return cached papers
      3. If MISS or EXPIRED → fetch from API, write to cache

    Eviction:
      When cache exceeds MAX_CACHE_FILES, the oldest files
      are deleted (LRU-style based on file mtime).

    Cache key: SHA-256 of sorted query strings → 16-char hex.
    """

    def __init__(
        self,
        cache_dir:  Path  = CACHE_DIR,
        ttl_hours:  float = CACHE_TTL_HOURS,
        max_files:  int   = MAX_CACHE_FILES,
    ) -> None:
        self._dir      = cache_dir
        self._ttl      = ttl_hours
        self._max      = max_files

    def key(self, queries: list[str]) -> str:
        raw = "|".join(sorted(q.lower().strip() for q in queries))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def load(self, queries: list[str]) -> list[Paper] | None:
        """Return cached papers if valid, else None."""
        path = self._path(self.key(queries))
        if not path.exists():
            return None
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h > self._ttl:
            logger.debug("Cache expired (%.1fh): %s", age_h, path.name)
            return None
        try:
            raw = json.loads(path.read_text())
            papers = [Paper.from_dict(p) for p in raw]
            logger.info(
                "Cache HIT: %s — %d papers (%.1fh old)",
                path.name[:12], len(papers), age_h,
            )
            return papers
        except Exception as exc:
            logger.warning("Cache load error: %s", exc)
            return None

    def save(self, queries: list[str], papers: list[Paper]) -> None:
        """Write papers to cache and evict old entries if needed."""
        path = self._path(self.key(queries))
        try:
            path.write_text(json.dumps([p.to_dict() for p in papers], indent=2))
            logger.debug("Cache saved: %s (%d papers)", path.name[:12], len(papers))
        except Exception as exc:
            logger.warning("Cache save error: %s", exc)
        self._evict()

    def _evict(self) -> None:
        """Delete oldest files when cache exceeds max size."""
        files = sorted(self._dir.glob("*.json"), key=lambda f: f.stat().st_mtime)
        while len(files) > self._max:
            files.pop(0).unlink(missing_ok=True)

    def invalidate(self, queries: list[str] | None = None) -> int:
        """Invalidate specific or all cache entries."""
        if queries:
            path = self._path(self.key(queries))
            if path.exists():
                path.unlink()
                return 1
            return 0
        count = 0
        for f in self._dir.glob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        logger.info("Cache cleared: %d files removed", count)
        return count

    def stats(self) -> dict:
        files = list(self._dir.glob("*.json"))
        total_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)
        return {
            "files":      len(files),
            "size_mb":    round(total_mb, 2),
            "ttl_hours":  self._ttl,
            "cache_dir":  str(self._dir),
        }


# ─────────────────────────────────────────────────────────────────────────────
# PAPER PARSER
# ─────────────────────────────────────────────────────────────────────────────

class PaperParser:
    """
    Converts a raw arxiv.Result object into a clean Paper dataclass.

    Handles:
      - ID extraction from entry_id URL
      - Author list truncation (store first 5 only)
      - Abstract normalisation (collapse newlines)
      - Date parsing to ISO string + integer year
    """

    @staticmethod
    def parse(result: arxiv.Result) -> Paper:
        arxiv_id = result.entry_id.split("/abs/")[-1].split("v")[0]
        published_str = result.published.strftime("%Y-%m-%d")
        abstract = result.summary.strip().replace("\n", " ")
        authors  = [str(a) for a in result.authors[:5]]

        return Paper(
            arxiv_id=arxiv_id,
            title=result.title.strip(),
            abstract=abstract,
            authors=authors,
            published=published_str,
            year=result.published.year,
            categories=list(result.categories),
            url=result.entry_id,
        )

    @staticmethod
    def parse_many(results: list[arxiv.Result]) -> list[Paper]:
        papers: list[Paper] = []
        seen: set[str] = set()
        for r in results:
            try:
                p = PaperParser.parse(r)
                if p.arxiv_id not in seen:
                    seen.add(p.arxiv_id)
                    papers.append(p)
            except Exception as exc:
                logger.warning("Paper parse error: %s", exc)
        return papers


# ─────────────────────────────────────────────────────────────────────────────
# ARXIV CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class ArxivClient:
    """
    arXiv API wrapper with pagination, deduplication, and caching.

    For each query in the expanded query list, fetches up to
    max_results_per_query papers sorted by relevance. Results are
    deduplicated by arxiv_id across all queries.

    The cache-aside check happens before any API call:
      1. Check CacheManager for this query set
      2. HIT → return immediately (no API call)
      3. MISS → fetch from API, parse, cache, return
    """

    def __init__(
        self,
        max_results:  int   = MAX_RESULTS,
        cache:        CacheManager | None = None,
        parser:       PaperParser | None  = None,
    ) -> None:
        self._max     = max_results
        self._cache   = cache  or CacheManager()
        self._parser  = parser or PaperParser()
        self._client  = arxiv.Client(
            page_size=50,
            delay_seconds=1.0,
            num_retries=3,
        )
        logger.info("ArxivClient init | max_results=%d", max_results)

    # ── MAIN FETCH ───────────────────────────────────────────────────────────

    async def fetch(self, queries: list[str]) -> list[Paper]:
        """
        Fetch papers for a list of expanded queries.
        Returns deduplicated list of Paper objects.
        Uses cache if available.
        """
        # Cache check
        cached = self._cache.load(queries)
        if cached is not None:
            return cached

        # Fetch from API (run sync client in thread pool)
        loop = asyncio.get_event_loop()
        papers = await loop.run_in_executor(
            None, lambda: self._fetch_sync(queries)
        )

        # Cache results
        if papers:
            self._cache.save(queries, papers)

        return papers

    def fetch_sync(self, queries: list[str]) -> list[Paper]:
        """Synchronous version for non-async callers."""
        cached = self._cache.load(queries)
        if cached is not None:
            return cached
        papers = self._fetch_sync(queries)
        if papers:
            self._cache.save(queries, papers)
        return papers

    # ── INTERNAL ─────────────────────────────────────────────────────────────

    def _fetch_sync(self, queries: list[str]) -> list[Paper]:
        """Fetch from arXiv API synchronously with deduplication."""
        per_query = max(20, self._max // max(len(queries), 1))
        seen: set[str] = set()
        all_papers: list[Paper] = []

        for query in queries:
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=per_query,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                results = list(self._client.results(search))
                batch   = self._parser.parse_many(results)

                for paper in batch:
                    if paper.arxiv_id not in seen:
                        seen.add(paper.arxiv_id)
                        all_papers.append(paper)

                logger.info(
                    "arXiv query '%s': %d results (%d new)",
                    query[:40], len(batch),
                    sum(1 for p in batch if p.arxiv_id not in seen | {p.arxiv_id}),
                )

            except Exception as exc:
                logger.error("arXiv API error for '%s': %s", query, exc)

        logger.info(
            "arXiv fetch complete: %d unique papers from %d queries",
            len(all_papers), len(queries),
        )
        return all_papers

    def _iter_results(
        self, query: str, max_results: int
    ) -> Iterator[arxiv.Result]:
        """Iterator over arXiv results with error handling."""
        try:
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )
            yield from self._client.results(search)
        except Exception as exc:
            logger.error("arXiv iterator error: %s", exc)
            return

    # ── DATE FILTER ──────────────────────────────────────────────────────────

    async def fetch_recent(
        self,
        queries:    list[str],
        since_year: int = 2022,
    ) -> list[Paper]:
        """
        Fetch only papers published on or after since_year.
        Useful for the TIER_4_RECENT roadmap tier.
        """
        papers = await self.fetch(queries)
        return [p for p in papers if p.year >= since_year]

    # ── STATUS ────────────────────────────────────────────────────────────────

    def cache_stats(self) -> dict:
        return self._cache.stats()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETONS
# ─────────────────────────────────────────────────────────────────────────────

_default_cache:  CacheManager | None = None
_default_client: ArxivClient  | None = None


def get_cache() -> CacheManager:
    global _default_cache
    if _default_cache is None:
        _default_cache = CacheManager()
    return _default_cache


def get_client() -> ArxivClient:
    global _default_client
    if _default_client is None:
        _default_client = ArxivClient()
    return _default_client


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

async def _demo() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    client = ArxivClient(max_results=20)
    queries = [
        "multimodal large language models medical diagnosis",
        "vision language models radiology",
    ]

    print("\n═══ RETRIEVAL DEMO ═══\n")
    print(f"  Queries: {queries}")
    print(f"  Fetching...")

    papers = await client.fetch(queries)
    print(f"\n  Retrieved: {len(papers)} unique papers\n")

    for p in papers[:5]:
        print(f"  [{p.arxiv_id}] {p.title[:60]}")
        print(f"           {', '.join(p.authors[:2])} ({p.year}) | {p.categories[:2]}")

    # Cache hit demo
    print(f"\n  Re-fetching (cache hit expected)...")
    t0     = time.time()
    papers2 = await client.fetch(queries)
    elapsed = time.time() - t0
    print(f"  Cache hit: {len(papers2)} papers in {elapsed:.3f}s")

    # Cache stats
    print(f"\n  Cache stats: {client.cache_stats()}")

    # Cleanup demo cache
    cache = CacheManager()
    cache.invalidate(queries)
    print("  Demo cache entry invalidated.")


if __name__ == "__main__":
    asyncio.run(_demo())
