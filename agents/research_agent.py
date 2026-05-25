"""
agents/research_agent.py
========================
ResearchFlow AI — Core Research Intelligence Agent

A LangChain ReAct (Reasoning + Acting) agent that:
  1. Reformulates natural language research queries into structured arXiv syntax
  2. Retrieves and caches papers from the arXiv API
  3. Generates dense SPECTER/MiniLM embeddings for all abstracts
  4. Clusters papers with HDBSCAN and projects them with UMAP
  5. Labels each cluster via GPT-4o summarization
  6. Runs a ReAct loop over 5 custom tools to identify research gaps
  7. Scores reproducibility and generates ranked thesis hypotheses
  8. Returns a fully structured ResearchReport dataclass

Usage:
    agent = ResearchAgent()
    report = await agent.run("Multimodal LLMs for medical diagnosis")

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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import arxiv
import hdbscan
import numpy as np
import umap
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & PATHS
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("cache/papers")
EMBED_CACHE_DIR = Path("cache/embeddings")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "claude-sonnet-4-20250514"  # swap to gpt-4o if preferred
EMBED_MODEL = "allenai-specter"             # domain-optimised for scientific text
EMBED_FALLBACK = "all-MiniLM-L6-v2"        # lightweight fallback

MAX_PAPERS = int(os.getenv("ARXIV_MAX_RESULTS", "80"))
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Paper:
    """Represents a single arXiv paper with all extracted metadata."""
    arxiv_id: str
    title: str
    abstract: str
    authors: list[str]
    published: str          # ISO date string
    year: int
    categories: list[str]
    url: str
    cluster_id: int = -1    # assigned after HDBSCAN; -1 = noise/outlier


@dataclass
class Cluster:
    """Represents a thematic cluster of papers."""
    cluster_id: int
    label: str              # GPT-4o generated thematic label
    paper_ids: list[str]
    paper_count: int
    year_range: tuple[int, int]
    dominant_category: str
    reproducibility_score: float    # 0.0–1.0 heuristic
    temporal_trend: str             # "growing" | "stagnating" | "declining"


@dataclass
class ResearchGap:
    """A single identified research gap with supporting evidence."""
    gap_id: int
    title: str
    description: str
    supporting_clusters: list[int]
    evidence: dict[str, Any]        # paper_count, recency, repro_score, etc.
    confidence: float               # 0.0–1.0
    suggested_methodology: str


@dataclass
class ThesisHypothesis:
    """A generated thesis hypothesis with novelty ranking."""
    hypothesis_id: int
    research_question: str
    hypothesis_statement: str
    methodology_type: str           # e.g. "benchmark study", "architecture proposal"
    novelty_score: float            # 0.0–1.0
    related_gap_ids: list[int]
    suggested_datasets: list[str]


@dataclass
class ResearchReport:
    """Complete output from a ResearchAgent run."""
    session_id: str
    query: str
    expanded_queries: list[str]
    papers: list[Paper]
    clusters: list[Cluster]
    gaps: list[ResearchGap]
    hypotheses: list[ThesisHypothesis]
    reading_roadmap: list[str]      # ordered list of arxiv IDs
    umap_coords: list[tuple[float, float]]
    generated_at: str
    execution_time_s: float


# ─────────────────────────────────────────────────────────────────────────────
# LANGCHAIN TOOL INPUT SCHEMAS
# ─────────────────────────────────────────────────────────────────────────────

class ClusterSummaryInput(BaseModel):
    cluster_id: int = Field(description="The integer cluster ID to summarise")


class PaperDensityInput(BaseModel):
    cluster_id: int = Field(description="Cluster ID to get paper count for")


class TemporalTrendInput(BaseModel):
    cluster_id: int = Field(description="Cluster ID to analyse publication trend")


class ReproducibilityInput(BaseModel):
    cluster_id: int = Field(description="Cluster ID to score for reproducibility")


class SemanticSimilarityInput(BaseModel):
    query_text: str = Field(description="Free-text query to find semantically similar papers")
    top_k: int = Field(default=5, description="Number of similar papers to return")


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM LANGCHAIN TOOLS
# ─────────────────────────────────────────────────────────────────────────────

class ClusterSummaryTool(BaseTool):
    """
    Returns the GPT-4o generated thematic label and a brief description
    of papers in the given cluster. Used by the ReAct agent to understand
    what research territory each cluster covers.
    """

    name: str = "cluster_summary"
    description: str = (
        "Use this to get a summary and thematic label of a research cluster. "
        "Input: cluster_id (integer). "
        "Output: cluster label, paper count, dominant category, year range."
    )
    args_schema: type[BaseModel] = ClusterSummaryInput
    clusters: list[Cluster] = Field(default_factory=list)

    def _run(self, cluster_id: int) -> str:
        cluster = next((c for c in self.clusters if c.cluster_id == cluster_id), None)
        if cluster is None:
            return f"No cluster found with ID {cluster_id}."
        return (
            f"Cluster {cluster_id}: '{cluster.label}'\n"
            f"  Papers: {cluster.paper_count}\n"
            f"  Year range: {cluster.year_range[0]}–{cluster.year_range[1]}\n"
            f"  Dominant category: {cluster.dominant_category}\n"
            f"  Temporal trend: {cluster.temporal_trend}\n"
            f"  Reproducibility score: {cluster.reproducibility_score:.2f}"
        )

    async def _arun(self, cluster_id: int) -> str:
        return self._run(cluster_id)


class PaperDensityTool(BaseTool):
    """
    Returns paper counts per cluster, making sparse clusters immediately
    visible as potentially underexplored research areas.
    """

    name: str = "paper_density"
    description: str = (
        "Use this to get the number of papers in each cluster. "
        "Sparse clusters (few papers) indicate potentially underexplored areas. "
        "Input: cluster_id (integer). Output: paper count and density assessment."
    )
    args_schema: type[BaseModel] = PaperDensityInput
    clusters: list[Cluster] = Field(default_factory=list)

    def _run(self, cluster_id: int) -> str:
        cluster = next((c for c in self.clusters if c.cluster_id == cluster_id), None)
        if cluster is None:
            return f"No cluster found with ID {cluster_id}."

        all_counts = [c.paper_count for c in self.clusters]
        median_count = float(np.median(all_counts)) if all_counts else 1.0
        density = "sparse" if cluster.paper_count < median_count * 0.4 else (
            "dense" if cluster.paper_count > median_count * 1.8 else "moderate"
        )
        return (
            f"Cluster {cluster_id} paper density: {cluster.paper_count} papers ({density})\n"
            f"  Median across all clusters: {median_count:.0f} papers\n"
            f"  Interpretation: {'Potentially underexplored — low paper count.' if density == 'sparse' else 'Well-covered research area.' if density == 'dense' else 'Moderately explored.'}"
        )

    async def _arun(self, cluster_id: int) -> str:
        return self._run(cluster_id)


class TemporalTrendTool(BaseTool):
    """
    Analyses publication year distribution within a cluster to detect
    stagnating vs. actively growing research areas.
    """

    name: str = "temporal_trend"
    description: str = (
        "Analyse the publication year trend within a cluster. "
        "Use this to determine if a research area is growing, stagnating, or declining. "
        "Input: cluster_id (integer). Output: year distribution and trend assessment."
    )
    args_schema: type[BaseModel] = TemporalTrendInput
    clusters: list[Cluster] = Field(default_factory=list)
    papers: list[Paper] = Field(default_factory=list)

    def _run(self, cluster_id: int) -> str:
        cluster = next((c for c in self.clusters if c.cluster_id == cluster_id), None)
        if cluster is None:
            return f"No cluster found with ID {cluster_id}."

        cluster_papers = [p for p in self.papers if p.cluster_id == cluster_id]
        if not cluster_papers:
            return f"No papers found for cluster {cluster_id}."

        year_counts: dict[int, int] = {}
        for paper in cluster_papers:
            year_counts[paper.year] = year_counts.get(paper.year, 0) + 1

        sorted_years = sorted(year_counts.items())
        year_str = "  ".join(f"{yr}: {cnt} papers" for yr, cnt in sorted_years[-5:])

        # Simple trend: compare first half vs second half paper counts
        years = [p.year for p in cluster_papers]
        if len(years) >= 4:
            midpoint = sorted(years)[len(years) // 2]
            older = sum(1 for y in years if y < midpoint)
            newer = sum(1 for y in years if y >= midpoint)
            if newer > older * 1.5:
                trend = "GROWING — significant increase in recent publications"
            elif newer < older * 0.6:
                trend = "DECLINING — fewer recent publications than older ones"
            else:
                trend = "STAGNATING — publication rate has plateaued"
        else:
            trend = "INSUFFICIENT DATA — too few papers to assess trend"

        return (
            f"Cluster {cluster_id} temporal trend: {cluster.temporal_trend}\n"
            f"  Recent year distribution (last 5 years):\n  {year_str}\n"
            f"  Trend assessment: {trend}"
        )

    async def _arun(self, cluster_id: int) -> str:
        return self._run(cluster_id)


class ReproducibilityTool(BaseTool):
    """
    Scores a cluster's reproducibility based on keyword signals in abstracts
    (code availability, dataset release, benchmarks). Acts as a proxy for
    how reproducible the research in that area tends to be.
    """

    name: str = "reproducibility_score"
    description: str = (
        "Score the reproducibility of research in a cluster using keyword signals. "
        "Checks for mentions of 'code available', 'open source', 'dataset', 'benchmark', etc. "
        "Input: cluster_id (integer). Output: reproducibility score (0–1) with breakdown."
    )
    args_schema: type[BaseModel] = ReproducibilityInput
    clusters: list[Cluster] = Field(default_factory=list)
    papers: list[Paper] = Field(default_factory=list)

    # Keywords indicating good reproducibility practices
    REPRO_KEYWORDS: list[str] = [
        "code available", "open source", "open-source", "github",
        "dataset released", "publicly available", "benchmark",
        "reproducible", "replication", "our code", "we release",
        "available at", "hugging face", "open dataset",
    ]

    def _run(self, cluster_id: int) -> str:
        cluster = next((c for c in self.clusters if c.cluster_id == cluster_id), None)
        if cluster is None:
            return f"No cluster found with ID {cluster_id}."

        cluster_papers = [p for p in self.papers if p.cluster_id == cluster_id]
        if not cluster_papers:
            return f"No papers found for cluster {cluster_id}."

        repro_count = 0
        keyword_hits: dict[str, int] = {}

        for paper in cluster_papers:
            abstract_lower = paper.abstract.lower()
            for kw in self.REPRO_KEYWORDS:
                if kw in abstract_lower:
                    repro_count += 1
                    keyword_hits[kw] = keyword_hits.get(kw, 0) + 1
                    break   # count each paper once

        score = repro_count / len(cluster_papers) if cluster_papers else 0.0
        top_kws = sorted(keyword_hits.items(), key=lambda x: -x[1])[:3]
        kw_str = ", ".join(f"'{k}' ({v})" for k, v in top_kws) or "none detected"

        interpretation = (
            "HIGH — most papers provide code/data" if score >= 0.6 else
            "MEDIUM — some papers share resources" if score >= 0.3 else
            "LOW — few papers share code or data (reproducibility gap)"
        )

        return (
            f"Cluster {cluster_id} reproducibility score: {score:.2f} ({interpretation})\n"
            f"  Papers with reproducibility signals: {repro_count}/{len(cluster_papers)}\n"
            f"  Top keywords found: {kw_str}"
        )

    async def _arun(self, cluster_id: int) -> str:
        return self._run(cluster_id)


class SemanticSimilarityTool(BaseTool):
    """
    Finds papers semantically similar to a free-text query using the
    FAISS vector index. Enables the agent to probe specific research
    directions and check whether they are already well-covered.
    """

    name: str = "semantic_similarity"
    description: str = (
        "Find papers most semantically similar to a free-text research query. "
        "Use this to check if a potential gap direction is already well-covered. "
        "Input: query_text (string), top_k (int, default 5). "
        "Output: top-k most similar papers with titles and abstracts."
    )
    args_schema: type[BaseModel] = SemanticSimilarityInput
    papers: list[Paper] = Field(default_factory=list)
    embeddings: Any = Field(default=None)          # np.ndarray shape (N, D)
    embedder: Any = Field(default=None)            # SentenceTransformer

    def _run(self, query_text: str, top_k: int = 5) -> str:
        if self.embedder is None or self.embeddings is None:
            return "Semantic similarity unavailable — embeddings not initialised."

        query_vec = self.embedder.encode([query_text], normalize_embeddings=True)
        emb_norm = self.embeddings / (
            np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-9
        )
        scores = (emb_norm @ query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            paper = self.papers[idx]
            results.append(
                f"{rank}. [{paper.arxiv_id}] {paper.title} ({paper.year})\n"
                f"   Similarity: {scores[idx]:.3f} | Cluster: {paper.cluster_id}\n"
                f"   Abstract excerpt: {paper.abstract[:200]}..."
            )

        return "\n".join(results) if results else "No similar papers found."

    async def _arun(self, query_text: str, top_k: int = 5) -> str:
        return self._run(query_text, top_k)


# ─────────────────────────────────────────────────────────────────────────────
# REACT AGENT PROMPT
# ─────────────────────────────────────────────────────────────────────────────

REACT_PROMPT_TEMPLATE = """You are ResearchFlow AI's gap detection agent — a rigorous academic research analyst.

Your task is to analyse a set of research paper clusters and identify the most significant, underexplored research gaps in the field: "{topic}".

You have access to the following tools:
{tools}

Tool names available: {tool_names}

You MUST use a systematic approach:
1. First, call cluster_summary for EACH cluster to understand the research landscape.
2. Use paper_density to identify sparse clusters (potentially underexplored areas).
3. Use temporal_trend to find stagnating sub-fields that may warrant new work.
4. Use reproducibility_score to flag areas with low open-science practices.
5. Use semantic_similarity to probe specific gap hypotheses.
6. Synthesise your findings into 3–5 well-evidenced research gaps.

For each gap you identify, provide:
- A concise title (≤10 words)
- A 2–3 sentence description with specific evidence from your tool calls
- Which clusters it relates to
- Confidence level (0.0–1.0)
- Suggested methodology type (benchmark study / architecture proposal / dataset curation / survey / empirical analysis)

Use the following format STRICTLY:

Thought: [your reasoning about what to do next]
Action: [tool name — must be one of {tool_names}]
Action Input: [the input to the tool, as a JSON object]
Observation: [the result of the tool call]
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now have enough evidence to synthesise the research gaps.
Final Answer: [your structured gap analysis as described above]

Begin your analysis now.

{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate(
    input_variables=["topic", "tools", "tool_names", "agent_scratchpad"],
    template=REACT_PROMPT_TEMPLATE,
)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RESEARCH AGENT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class ResearchAgent:
    """
    Autonomous research intelligence agent.

    Orchestrates the full pipeline:
        arXiv retrieval → embedding → clustering → UMAP → gap detection → hypotheses

    All heavy operations (embedding, clustering) are designed to be dispatched
    via the SchedulerAgent's priority queue — this class exposes each stage
    as an independent async method so the scheduler can manage resource-aware
    dispatch.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        embed_model: str = EMBED_MODEL,
        max_papers: int = MAX_PAPERS,
        temperature: float = 0.0,
        verbose: bool = True,
    ) -> None:
        self.model_name = model_name
        self.embed_model_name = embed_model
        self.max_papers = max_papers
        self.temperature = temperature
        self.verbose = verbose

        # Initialise LLM (lazy — no API call until needed)
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            streaming=True,
        )

        # Sentence transformer (lazy load on first use)
        self._embedder: SentenceTransformer | None = None

        # State populated progressively through the pipeline
        self._papers: list[Paper] = []
        self._embeddings: np.ndarray | None = None
        self._clusters: list[Cluster] = []

        logger.info("ResearchAgent initialised | model=%s embed=%s", model_name, embed_model)

    # ── PROPERTIES ──────────────────────────────────────────────────────────

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazy-loads the sentence transformer (downloads on first call)."""
        if self._embedder is None:
            logger.info("Loading embedding model: %s", self.embed_model_name)
            try:
                self._embedder = SentenceTransformer(self.embed_model_name)
                logger.info("Loaded %s successfully.", self.embed_model_name)
            except Exception:
                logger.warning(
                    "Failed to load %s — falling back to %s",
                    self.embed_model_name, EMBED_FALLBACK,
                )
                self._embedder = SentenceTransformer(EMBED_FALLBACK)
        return self._embedder

    # ── STAGE 1: QUERY REFORMULATION ────────────────────────────────────────

    async def reformulate_query(self, raw_query: str) -> list[str]:
        """
        Expands a natural language topic into multiple structured arXiv queries.
        Uses GPT-4o to add domain synonyms, abbreviation expansion, and sub-queries.

        Returns a list of query strings (primary + expanded variants).
        """
        logger.info("Reformulating query: '%s'", raw_query)

        prompt = (
            "You are an expert academic search assistant. Given a research topic, "
            "generate 3–4 distinct arXiv API search queries that maximise coverage.\n\n"
            "Rules:\n"
            "- Expand abbreviations (e.g. LLM → large language model)\n"
            "- Add domain synonyms and related sub-topics\n"
            "- Each query should be 3–7 words, suitable for arXiv full-text search\n"
            "- Return ONLY a JSON array of query strings, nothing else\n\n"
            f"Research topic: {raw_query}\n\n"
            "JSON array:"
        )

        response = await self.llm.ainvoke(prompt)
        content = response.content.strip()

        # Strip markdown fences if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        try:
            queries: list[str] = json.loads(content)
            logger.info("Expanded to %d queries: %s", len(queries), queries)
            return [raw_query, *queries]
        except json.JSONDecodeError:
            logger.warning("Query expansion JSON parse failed — using raw query only")
            return [raw_query]

    # ── STAGE 2: ARXIV RETRIEVAL ─────────────────────────────────────────────

    async def retrieve_papers(self, queries: list[str]) -> list[Paper]:
        """
        Retrieves papers from arXiv for all expanded queries.
        Deduplicates by arXiv ID. Caches results as JSON (TTL: CACHE_TTL_HOURS).

        Returns a deduplicated list of Paper objects.
        """
        cache_key = hashlib.md5("|".join(sorted(queries)).encode()).hexdigest()[:12]
        cache_file = CACHE_DIR / f"{cache_key}.json"

        # Check cache
        if cache_file.exists():
            age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
            if age_hours < CACHE_TTL_HOURS:
                logger.info("Cache hit — loading %s (age: %.1fh)", cache_key, age_hours)
                raw = json.loads(cache_file.read_text())
                return [Paper(**p) for p in raw]

        logger.info("Retrieving papers for %d queries (max %d per query)...", len(queries), self.max_papers // len(queries))

        seen_ids: set[str] = set()
        papers: list[Paper] = []
        papers_per_query = max(20, self.max_papers // len(queries))

        client = arxiv.Client()

        for query in queries:
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=papers_per_query,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                for result in client.results(search):
                    arxiv_id = result.entry_id.split("/abs/")[-1]
                    if arxiv_id in seen_ids:
                        continue
                    seen_ids.add(arxiv_id)

                    published_str = result.published.strftime("%Y-%m-%d")
                    year = result.published.year

                    papers.append(Paper(
                        arxiv_id=arxiv_id,
                        title=result.title.strip(),
                        abstract=result.summary.strip().replace("\n", " "),
                        authors=[str(a) for a in result.authors[:5]],
                        published=published_str,
                        year=year,
                        categories=result.categories,
                        url=result.entry_id,
                    ))
            except Exception as exc:
                logger.error("arXiv query failed for '%s': %s", query, exc)

        logger.info("Retrieved %d unique papers", len(papers))

        # Persist to cache
        cache_file.write_text(
            json.dumps([p.__dict__ for p in papers], indent=2)
        )

        self._papers = papers
        return papers

    # ── STAGE 3: EMBEDDING ──────────────────────────────────────────────────

    async def embed_papers(self, papers: list[Paper]) -> np.ndarray:
        """
        Generates SPECTER dense embeddings for all paper abstracts.
        Runs in batches to avoid OOM on large paper sets.
        Caches resulting numpy array to disk.

        Returns embeddings array of shape (N, embedding_dim).
        """
        cache_key = hashlib.md5(
            "".join(p.arxiv_id for p in papers).encode()
        ).hexdigest()[:12]
        cache_file = EMBED_CACHE_DIR / f"{cache_key}.npy"

        if cache_file.exists():
            logger.info("Embedding cache hit — loading %s", cache_key)
            embeddings = np.load(cache_file)
            self._embeddings = embeddings
            return embeddings

        logger.info("Generating embeddings for %d papers...", len(papers))

        abstracts = [p.abstract for p in papers]
        batch_size = 32

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(abstracts), batch_size):
            batch = abstracts[i : i + batch_size]
            batch_embs = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda b=batch: self.embedder.encode(
                    b,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
            )
            all_embeddings.append(batch_embs)
            logger.debug("Embedded batch %d/%d", i // batch_size + 1, -(-len(abstracts) // batch_size))

        embeddings = np.vstack(all_embeddings)
        np.save(cache_file, embeddings)
        logger.info("Embeddings shape: %s — saved to cache", embeddings.shape)

        self._embeddings = embeddings
        return embeddings

    # ── STAGE 4: CLUSTERING ──────────────────────────────────────────────────

    async def cluster_papers(
        self,
        papers: list[Paper],
        embeddings: np.ndarray,
    ) -> tuple[list[Paper], np.ndarray]:
        """
        Clusters paper embeddings with HDBSCAN (no fixed cluster count).
        Assigns cluster_id to each Paper object in-place.
        Also computes UMAP 2D coordinates for visualization.

        Returns (papers_with_clusters, umap_2d_coords).
        """
        logger.info("Clustering %d papers with HDBSCAN...", len(papers))

        # HDBSCAN on the full embedding space
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(3, len(papers) // 20),
            min_samples=2,
            metric="euclidean",
            cluster_selection_method="eom",
        )

        labels = await asyncio.get_event_loop().run_in_executor(
            None, clusterer.fit_predict, embeddings
        )

        for paper, label in zip(papers, labels):
            paper.cluster_id = int(label)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = sum(1 for l in labels if l == -1)
        logger.info("HDBSCAN: %d clusters, %d noise points", n_clusters, n_noise)

        # UMAP 2D projection (for Streamlit scatter plot)
        logger.info("Computing UMAP 2D projection...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, len(papers) - 1),
            min_dist=0.1,
            random_state=42,
        )
        coords_2d = await asyncio.get_event_loop().run_in_executor(
            None, reducer.fit_transform, embeddings
        )

        self._papers = papers
        return papers, coords_2d

    # ── STAGE 5: CLUSTER LABELING ────────────────────────────────────────────

    async def label_clusters(
        self,
        papers: list[Paper],
        embeddings: np.ndarray,
    ) -> list[Cluster]:
        """
        Generates a GPT-4o thematic label for each cluster by sampling
        3–5 representative abstracts. Also computes cluster metadata:
        year range, dominant category, temporal trend, reproducibility score.

        Returns a list of Cluster objects.
        """
        cluster_ids = sorted(set(p.cluster_id for p in papers if p.cluster_id >= 0))
        clusters: list[Cluster] = []

        logger.info("Labeling %d clusters...", len(cluster_ids))

        for cid in cluster_ids:
            cluster_papers = [p for p in papers if p.cluster_id == cid]

            # Sample up to 5 representative abstracts
            sample = cluster_papers[:5]
            abstracts_text = "\n\n".join(
                f"Paper {i+1}: {p.title}\n{p.abstract[:400]}"
                for i, p in enumerate(sample)
            )

            prompt = (
                "You are an academic research classifier. Read these paper abstracts "
                "and generate a single concise thematic label (5–8 words) that best "
                "describes the research sub-field they represent.\n\n"
                "Return ONLY the label, nothing else.\n\n"
                f"Abstracts:\n{abstracts_text}\n\nThematic label:"
            )

            try:
                response = await self.llm.ainvoke(prompt)
                label = response.content.strip().strip('"').strip("'")
            except Exception as exc:
                logger.warning("Cluster labeling failed for cluster %d: %s", cid, exc)
                label = f"Cluster {cid}"

            # Compute metadata
            years = [p.year for p in cluster_papers]
            all_categories = [cat for p in cluster_papers for cat in p.categories]
            dominant_cat = max(set(all_categories), key=all_categories.count) if all_categories else "cs.AI"

            # Temporal trend
            if len(years) >= 4:
                mid = sorted(years)[len(years) // 2]
                newer = sum(1 for y in years if y >= mid)
                older = sum(1 for y in years if y < mid)
                trend = "growing" if newer > older * 1.5 else (
                    "declining" if newer < older * 0.6 else "stagnating"
                )
            else:
                trend = "insufficient data"

            # Reproducibility heuristic
            repro_kws = ["code available", "open source", "github", "dataset released",
                         "benchmark", "publicly available", "we release"]
            repro_hits = sum(
                1 for p in cluster_papers
                if any(kw in p.abstract.lower() for kw in repro_kws)
            )
            repro_score = repro_hits / len(cluster_papers) if cluster_papers else 0.0

            cluster = Cluster(
                cluster_id=cid,
                label=label,
                paper_ids=[p.arxiv_id for p in cluster_papers],
                paper_count=len(cluster_papers),
                year_range=(min(years), max(years)) if years else (0, 0),
                dominant_category=dominant_cat,
                reproducibility_score=round(repro_score, 3),
                temporal_trend=trend,
            )
            clusters.append(cluster)
            logger.debug("Cluster %d labeled: '%s' (%d papers, %s)", cid, label, len(cluster_papers), trend)

        self._clusters = clusters
        return clusters

    # ── STAGE 6: GAP DETECTION (REACT AGENT) ────────────────────────────────

    async def detect_gaps(
        self,
        topic: str,
        papers: list[Paper],
        clusters: list[Cluster],
        embeddings: np.ndarray,
    ) -> list[ResearchGap]:
        """
        Runs the LangChain ReAct agent over the 5 custom tools to identify
        underexplored research gaps. This is the core GenAI innovation:
        a reasoning loop that forms hypotheses, calls tools for evidence,
        and synthesises a structured gap report.

        Returns a list of ResearchGap objects.
        """
        logger.info("Running ReAct gap detection agent for topic: '%s'", topic)

        # Wire tools with current state
        tools = [
            ClusterSummaryTool(clusters=clusters),
            PaperDensityTool(clusters=clusters),
            TemporalTrendTool(clusters=clusters, papers=papers),
            ReproducibilityTool(clusters=clusters, papers=papers),
            SemanticSimilarityTool(
                papers=papers,
                embeddings=embeddings,
                embedder=self.embedder,
            ),
        ]

        # Build ReAct agent
        agent = create_react_agent(
            llm=self.llm,
            tools=tools,
            prompt=REACT_PROMPT,
        )
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.verbose,
            handle_parsing_errors=True,
            max_iterations=20,
            return_intermediate_steps=True,
        )

        try:
            result = await executor.ainvoke({"topic": topic})
            raw_output: str = result.get("output", "")
        except Exception as exc:
            logger.error("ReAct agent execution failed: %s", exc)
            raw_output = f"Agent failed: {exc}"

        # Parse the free-text gap analysis into structured ResearchGap objects
        gaps = await self._parse_gaps_from_output(raw_output, clusters)
        logger.info("Identified %d research gaps", len(gaps))
        return gaps

    async def _parse_gaps_from_output(
        self,
        raw_output: str,
        clusters: list[Cluster],
    ) -> list[ResearchGap]:
        """
        Uses a secondary GPT-4o call to parse the ReAct agent's free-text
        output into structured ResearchGap objects as JSON.
        """
        cluster_ids = [c.cluster_id for c in clusters]

        parse_prompt = (
            "Parse the following research gap analysis into structured JSON.\n\n"
            f"Available cluster IDs: {cluster_ids}\n\n"
            "Return a JSON array where each gap has these fields:\n"
            "  title (string, ≤10 words)\n"
            "  description (string, 2–3 sentences)\n"
            "  supporting_clusters (array of cluster IDs as integers)\n"
            "  confidence (float 0.0–1.0)\n"
            "  suggested_methodology (one of: 'benchmark study', 'architecture proposal', "
            "'dataset curation', 'survey', 'empirical analysis')\n\n"
            "Return ONLY valid JSON array, no markdown fences, no preamble.\n\n"
            f"Gap analysis text:\n{raw_output}"
        )

        try:
            response = await self.llm.ainvoke(parse_prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            gap_dicts: list[dict] = json.loads(content)
        except Exception as exc:
            logger.error("Gap parsing failed: %s", exc)
            # Return a single placeholder gap
            return [ResearchGap(
                gap_id=0,
                title="Gap detection parsing failed",
                description=raw_output[:300],
                supporting_clusters=[],
                evidence={},
                confidence=0.0,
                suggested_methodology="empirical analysis",
            )]

        gaps = []
        for i, g in enumerate(gap_dicts[:6]):   # cap at 6 gaps
            gaps.append(ResearchGap(
                gap_id=i,
                title=g.get("title", f"Gap {i}"),
                description=g.get("description", ""),
                supporting_clusters=g.get("supporting_clusters", []),
                evidence={
                    "cluster_labels": [
                        c.label for c in clusters
                        if c.cluster_id in g.get("supporting_clusters", [])
                    ],
                    "confidence": g.get("confidence", 0.5),
                },
                confidence=float(g.get("confidence", 0.5)),
                suggested_methodology=g.get("suggested_methodology", "empirical analysis"),
            ))

        return gaps

    # ── STAGE 7: HYPOTHESIS GENERATION ──────────────────────────────────────

    async def generate_hypotheses(
        self,
        topic: str,
        gaps: list[ResearchGap],
        clusters: list[Cluster],
    ) -> list[ThesisHypothesis]:
        """
        Generates 3–5 actionable thesis hypothesis statements from the
        identified gaps. Each hypothesis includes a research question,
        methodology type, novelty score, and suggested datasets.

        Returns a ranked list of ThesisHypothesis objects.
        """
        logger.info("Generating thesis hypotheses for %d gaps...", len(gaps))

        gap_descriptions = "\n".join(
            f"Gap {g.gap_id}: {g.title}\n  {g.description}\n  "
            f"Methodology: {g.suggested_methodology} | Confidence: {g.confidence:.2f}"
            for g in gaps
        )

        prompt = (
            f"You are a research advisor helping a PhD student in the field of: {topic}\n\n"
            "Based on the following identified research gaps, generate 3–5 thesis hypothesis "
            "statements. Each should be specific, testable, and novel.\n\n"
            "For each hypothesis return a JSON object with:\n"
            "  research_question (string — the 'can we...' or 'does X improve Y when...' question)\n"
            "  hypothesis_statement (string — a falsifiable hypothesis statement)\n"
            "  methodology_type (string — specific methodology)\n"
            "  novelty_score (float 0.0–1.0 — how novel vs existing work)\n"
            "  related_gap_ids (array of gap IDs as integers)\n"
            "  suggested_datasets (array of 2–3 dataset names relevant to the hypothesis)\n\n"
            "Return a JSON array, ranked by novelty_score descending.\n"
            "Return ONLY valid JSON, no markdown fences.\n\n"
            f"Research gaps:\n{gap_descriptions}"
        )

        try:
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            hyp_dicts: list[dict] = json.loads(content)
        except Exception as exc:
            logger.error("Hypothesis generation failed: %s", exc)
            return []

        hypotheses = []
        for i, h in enumerate(hyp_dicts[:5]):
            hypotheses.append(ThesisHypothesis(
                hypothesis_id=i,
                research_question=h.get("research_question", ""),
                hypothesis_statement=h.get("hypothesis_statement", ""),
                methodology_type=h.get("methodology_type", ""),
                novelty_score=float(h.get("novelty_score", 0.5)),
                related_gap_ids=h.get("related_gap_ids", []),
                suggested_datasets=h.get("suggested_datasets", []),
            ))

        # Sort by novelty descending
        hypotheses.sort(key=lambda h: h.novelty_score, reverse=True)
        logger.info("Generated %d hypotheses", len(hypotheses))
        return hypotheses

    # ── STAGE 8: READING ROADMAP ─────────────────────────────────────────────

    async def build_reading_roadmap(
        self,
        papers: list[Paper],
        clusters: list[Cluster],
        gaps: list[ResearchGap],
    ) -> list[str]:
        """
        Builds an ordered reading list of arXiv IDs structured as:
            Tier 1 — Foundational (oldest high-cited cluster papers)
            Tier 2 — Methodological (gap-adjacent cluster papers)
            Tier 3 — Recent results (most recent papers in growing clusters)

        Returns an ordered list of arXiv IDs (10–20 papers).
        """
        logger.info("Building reading roadmap...")

        # Tier 1: Oldest papers in largest clusters (foundational)
        large_clusters = sorted(clusters, key=lambda c: -c.paper_count)[:3]
        foundational: list[Paper] = []
        for cluster in large_clusters:
            cluster_papers = [p for p in papers if p.cluster_id == cluster.cluster_id]
            oldest = sorted(cluster_papers, key=lambda p: p.year)[:2]
            foundational.extend(oldest)

        # Tier 2: Papers in gap-related clusters
        gap_cluster_ids = set(
            cid for gap in gaps for cid in gap.supporting_clusters
        )
        gap_papers: list[Paper] = []
        for p in papers:
            if p.cluster_id in gap_cluster_ids:
                gap_papers.append(p)
        gap_papers = sorted(gap_papers, key=lambda p: -p.year)[:8]

        # Tier 3: Most recent papers in growing clusters
        growing = [c for c in clusters if c.temporal_trend == "growing"]
        recent: list[Paper] = []
        for cluster in growing[:2]:
            cluster_papers = [p for p in papers if p.cluster_id == cluster.cluster_id]
            newest = sorted(cluster_papers, key=lambda p: -p.year)[:2]
            recent.extend(newest)

        # Deduplicate while preserving tier order
        seen: set[str] = set()
        roadmap_ids: list[str] = []
        for p in foundational + gap_papers + recent:
            if p.arxiv_id not in seen:
                seen.add(p.arxiv_id)
                roadmap_ids.append(p.arxiv_id)
            if len(roadmap_ids) >= 20:
                break

        logger.info("Reading roadmap: %d papers", len(roadmap_ids))
        return roadmap_ids

    # ── MAIN ENTRYPOINT ──────────────────────────────────────────────────────

    async def run(self, query: str) -> ResearchReport:
        """
        Executes the full research intelligence pipeline end-to-end.

        Pipeline stages (designed to be scheduler-dispatched independently):
            1. Query reformulation (LLM)
            2. arXiv retrieval + caching
            3. Embedding generation (GPU/CPU intensive)
            4. HDBSCAN clustering + UMAP (CPU/RAM intensive)
            5. Cluster labeling (LLM)
            6. ReAct gap detection (LLM + tools)
            7. Hypothesis generation (LLM)
            8. Reading roadmap construction

        Returns a fully populated ResearchReport.
        """
        session_id = hashlib.md5(
            f"{query}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:10]

        logger.info("=" * 60)
        logger.info("ResearchAgent.run() | session=%s | query='%s'", session_id, query)
        logger.info("=" * 60)

        start_time = time.time()

        # Stage 1
        expanded_queries = await self.reformulate_query(query)

        # Stage 2
        papers = await self.retrieve_papers(expanded_queries)
        if not papers:
            raise ValueError(f"No papers retrieved for query: '{query}'")

        # Stage 3 — CPU/GPU intensive, scheduler should check resources before here
        embeddings = await self.embed_papers(papers)

        # Stage 4 — RAM intensive
        papers, umap_coords = await self.cluster_papers(papers, embeddings)

        # Stage 5 — LLM calls
        clusters = await self.label_clusters(papers, embeddings)

        # Stage 6 — Core ReAct agent
        gaps = await self.detect_gaps(query, papers, clusters, embeddings)

        # Stage 7
        hypotheses = await self.generate_hypotheses(query, gaps, clusters)

        # Stage 8
        roadmap = await self.build_reading_roadmap(papers, clusters, gaps)

        elapsed = time.time() - start_time

        report = ResearchReport(
            session_id=session_id,
            query=query,
            expanded_queries=expanded_queries,
            papers=papers,
            clusters=clusters,
            gaps=gaps,
            hypotheses=hypotheses,
            reading_roadmap=roadmap,
            umap_coords=[(float(x), float(y)) for x, y in umap_coords],
            generated_at=datetime.utcnow().isoformat(),
            execution_time_s=round(elapsed, 2),
        )

        logger.info(
            "Pipeline complete | %.1fs | %d papers | %d clusters | %d gaps | %d hypotheses",
            elapsed, len(papers), len(clusters), len(gaps), len(hypotheses),
        )

        return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI / QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

async def _demo() -> None:
    """Quick smoke-test — run with: python -m agents.research_agent"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    agent = ResearchAgent(verbose=True)
    report = await agent.run("Multimodal Large Language Models for Medical Diagnosis")

    print("\n" + "=" * 60)
    print(f"Session: {report.session_id}")
    print(f"Papers retrieved: {len(report.papers)}")
    print(f"Clusters found:   {len(report.clusters)}")
    print(f"Gaps identified:  {len(report.gaps)}")
    print(f"Hypotheses:       {len(report.hypotheses)}")
    print(f"Execution time:   {report.execution_time_s}s")
    print("=" * 60)

    print("\n── CLUSTERS ──")
    for cluster in report.clusters:
        print(f"  [{cluster.cluster_id}] {cluster.label} ({cluster.paper_count} papers, {cluster.temporal_trend})")

    print("\n── RESEARCH GAPS ──")
    for gap in report.gaps:
        print(f"  [{gap.gap_id}] {gap.title} (confidence: {gap.confidence:.2f})")
        print(f"       {gap.description[:120]}...")

    print("\n── TOP HYPOTHESIS ──")
    if report.hypotheses:
        h = report.hypotheses[0]
        print(f"  Q:  {h.research_question}")
        print(f"  H:  {h.hypothesis_statement}")
        print(f"  Method: {h.methodology_type} | Novelty: {h.novelty_score:.2f}")


if __name__ == "__main__":
    asyncio.run(_demo())
