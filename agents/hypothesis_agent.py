"""
agents/hypothesis_agent.py
==========================
ResearchFlow AI — Thesis Hypothesis Generation & Novelty Scoring Agent

This agent takes the structured ResearchGap objects produced by the
ResearchAgent's ReAct loop and transforms them into academically rigorous,
actionable thesis hypothesis statements.

Pipeline (within this agent):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  ResearchGaps + Clusters + Papers                                   │
  │         ↓                                                           │
  │  [1] GapEnricher      — pulls supporting evidence per gap          │
  │         ↓                                                           │
  │  [2] HypothesisWriter — GPT-4o generates hypothesis statements     │
  │         ↓                                                           │
  │  [3] NoveltyScorer    — multi-signal novelty score (0.0–1.0)       │
  │         ↓                                                           │
  │  [4] MethodologyAdvisor — suggests concrete experimental design    │
  │         ↓                                                           │
  │  [5] HypothesisRanker — final ranking by composite score           │
  │         ↓                                                           │
  │  List[ScoredHypothesis]  (ranked, evidence-backed, actionable)     │
  └─────────────────────────────────────────────────────────────────────┘

Key outputs per hypothesis:
  - Falsifiable hypothesis statement
  - Specific research question
  - Novelty score  (0.0–1.0, multi-signal)
  - Feasibility score  (0.0–1.0)
  - Impact score  (0.0–1.0)
  - Composite score  (weighted average)
  - Concrete methodology with dataset suggestions
  - Related prior work (with arxiv IDs)
  - Identified risks and mitigation strategies
  - Estimated timeline (months)

Usage:
    agent = HypothesisAgent()
    hypotheses = await agent.run(
        topic="Multimodal LLMs for Medical Diagnosis",
        gaps=gaps,               # List[ResearchGap] from ResearchAgent
        clusters=clusters,       # List[Cluster]
        papers=papers,           # List[Paper]
    )
    for h in hypotheses:
        print(h.rank, h.composite_score, h.research_question)

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_openai import ChatOpenAI

# Internal imports — adjust path if running outside package
try:
    from agents.research_agent import Cluster, Paper, ResearchGap
except ImportError:
    # Graceful fallback for standalone testing
    from research_agent import Cluster, Paper, ResearchGap  # type: ignore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS & CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

class MethodologyType(str, Enum):
    BENCHMARK_STUDY      = "benchmark_study"
    ARCHITECTURE_PROPOSAL = "architecture_proposal"
    DATASET_CURATION     = "dataset_curation"
    EMPIRICAL_ANALYSIS   = "empirical_analysis"
    SURVEY               = "survey"
    THEORETICAL_ANALYSIS = "theoretical_analysis"
    SYSTEM_DESIGN        = "system_design"
    HUMAN_STUDY          = "human_study"


class ImpactDomain(str, Enum):
    FUNDAMENTAL   = "fundamental"    # advances core theory
    APPLIED       = "applied"        # solves practical problems
    BENCHMARK     = "benchmark"      # improves evaluation infrastructure
    DATASET       = "dataset"        # releases new data resource
    TOOL          = "tool"           # produces reusable system/tool


# Composite score weights
NOVELTY_WEIGHT     = 0.40
FEASIBILITY_WEIGHT = 0.30
IMPACT_WEIGHT      = 0.30

DEFAULT_MODEL = "gpt-4o"
MAX_HYPOTHESES = 5


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NoveltySignals:
    """
    Decomposed signals used to compute the novelty score.
    Each signal is independently computed and contributes to the final score.
    Stored separately so the dashboard can show the scoring breakdown.
    """
    cluster_sparsity:      float = 0.0  # sparse cluster → more novel
    temporal_recency:      float = 0.0  # stagnating area → more novel
    reproducibility_gap:   float = 0.0  # low repro → unexplored infra gap
    semantic_distance:     float = 0.0  # distance from existing cluster centres
    cross_domain_overlap:  float = 0.0  # bridges multiple clusters → novel
    llm_novelty_estimate:  float = 0.0  # GPT-4o self-assessed novelty

    def composite(self) -> float:
        """Weighted average of all novelty signals."""
        weights = {
            "cluster_sparsity":     0.20,
            "temporal_recency":     0.20,
            "reproducibility_gap":  0.15,
            "semantic_distance":    0.15,
            "cross_domain_overlap": 0.15,
            "llm_novelty_estimate": 0.15,
        }
        total = 0.0
        for attr, w in weights.items():
            total += getattr(self, attr) * w
        return round(min(1.0, max(0.0, total)), 4)


@dataclass
class MethodologyPlan:
    """Concrete experimental design plan for a hypothesis."""
    methodology_type:     MethodologyType
    description:          str              # 2–3 sentence overview
    steps:                list[str]        # ordered experimental steps
    suggested_datasets:   list[str]        # named datasets to use
    suggested_baselines:  list[str]        # models/methods to compare against
    evaluation_metrics:   list[str]        # e.g. AUROC, F1, BLEU
    tools_and_frameworks: list[str]        # e.g. HuggingFace, PyTorch Lightning
    estimated_months:     int              # rough timeline
    compute_requirements: str             # e.g. "1x A100 for ~48h"
    risks:                list[str]        # potential failure modes
    mitigations:          list[str]        # mitigation strategies per risk


@dataclass
class ScoredHypothesis:
    """
    A fully scored, evidence-backed thesis hypothesis.
    This is the primary output of the HypothesisAgent.
    """
    hypothesis_id:        str
    rank:                 int             # 1 = best (by composite_score)

    # Core statements
    research_question:    str            # "Can X improve Y in setting Z?"
    hypothesis_statement: str            # falsifiable H1 statement
    null_hypothesis:      str            # H0 for statistical framing
    one_liner:            str            # ≤15-word tweet-length summary

    # Source
    source_gap_ids:       list[int]
    source_cluster_ids:   list[int]

    # Scores
    novelty_signals:      NoveltySignals
    novelty_score:        float          # 0.0–1.0
    feasibility_score:    float          # 0.0–1.0
    impact_score:         float          # 0.0–1.0
    composite_score:      float          # weighted average

    # Justification
    novelty_justification:    str
    feasibility_justification: str
    impact_justification:     str

    # Impact classification
    impact_domain:        ImpactDomain
    impact_domain_reason: str

    # Methodology
    methodology:          MethodologyPlan

    # Prior work
    related_paper_ids:    list[str]      # arXiv IDs of most relevant papers
    prior_work_summary:   str            # 2–3 sentence prior work description
    differentiation:      str            # how this H differs from prior work

    # Metadata
    generated_at:         str = field(default_factory=lambda: datetime.utcnow().isoformat())
    topic:                str = ""


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — GAP ENRICHER
# ─────────────────────────────────────────────────────────────────────────────

class GapEnricher:
    """
    Attaches quantitative evidence to each ResearchGap before hypothesis
    generation. Pulls cluster statistics and computes per-gap feature
    vectors used by the NoveltyScorer.
    """

    def enrich(
        self,
        gaps: list[ResearchGap],
        clusters: list[Cluster],
        papers: list[Paper],
    ) -> list[dict[str, Any]]:
        """
        Returns a list of enriched gap dicts, one per gap, containing:
          - original gap fields
          - cluster_metadata: stats for supporting clusters
          - supporting_papers: papers from those clusters
          - cross_cluster_flag: True if gap spans multiple clusters
        """
        cluster_map = {c.cluster_id: c for c in clusters}
        enriched = []

        for gap in gaps:
            supporting = [
                cluster_map[cid]
                for cid in gap.supporting_clusters
                if cid in cluster_map
            ]

            supporting_papers = [
                p for p in papers
                if p.cluster_id in gap.supporting_clusters
            ]

            cluster_meta = [
                {
                    "cluster_id":           c.cluster_id,
                    "label":                c.label,
                    "paper_count":          c.paper_count,
                    "year_range":           list(c.year_range),
                    "temporal_trend":       c.temporal_trend,
                    "reproducibility_score": c.reproducibility_score,
                    "dominant_category":    c.dominant_category,
                }
                for c in supporting
            ]

            enriched.append({
                "gap_id":             gap.gap_id,
                "title":              gap.title,
                "description":        gap.description,
                "confidence":         gap.confidence,
                "suggested_methodology": gap.suggested_methodology,
                "cluster_metadata":   cluster_meta,
                "supporting_papers":  [
                    {
                        "arxiv_id": p.arxiv_id,
                        "title":    p.title,
                        "year":     p.year,
                        "abstract": p.abstract[:300],
                    }
                    for p in sorted(supporting_papers, key=lambda p: -p.year)[:8]
                ],
                "cross_cluster_flag": len(gap.supporting_clusters) > 1,
                "avg_papers_per_cluster": (
                    sum(c.paper_count for c in supporting) / len(supporting)
                    if supporting else 0
                ),
                "min_repro_score": (
                    min(c.reproducibility_score for c in supporting)
                    if supporting else 0.0
                ),
                "stagnating_count": sum(
                    1 for c in supporting if c.temporal_trend == "stagnating"
                ),
            })

        logger.info("GapEnricher: enriched %d gaps", len(enriched))
        return enriched


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — HYPOTHESIS WRITER
# ─────────────────────────────────────────────────────────────────────────────

class HypothesisWriter:
    """
    Uses GPT-4o to generate multiple hypothesis candidates per gap.
    Produces structured JSON that feeds the NoveltyScorer and
    MethodologyAdvisor.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def write(
        self,
        topic: str,
        enriched_gap: dict,
        n_hypotheses: int = 2,
    ) -> list[dict]:
        """
        Generates n_hypotheses candidate hypotheses for a single enriched gap.
        Returns a list of raw hypothesis dicts (before scoring).
        """
        cluster_summary = "\n".join(
            f"  - Cluster '{m['label']}': {m['paper_count']} papers, "
            f"trend={m['temporal_trend']}, repro={m['reproducibility_score']:.2f}"
            for m in enriched_gap["cluster_metadata"]
        )

        recent_titles = "\n".join(
            f"  [{p['year']}] {p['title']}"
            for p in enriched_gap["supporting_papers"][:5]
        )

        prompt = f"""You are an expert research advisor helping a PhD student in: {topic}

A research gap has been identified with the following evidence:

GAP: {enriched_gap['title']}
DESCRIPTION: {enriched_gap['description']}
CONFIDENCE: {enriched_gap['confidence']:.2f}
CROSS-CLUSTER: {enriched_gap['cross_cluster_flag']}

Supporting clusters:
{cluster_summary}

Recent papers in this area:
{recent_titles}

Generate exactly {n_hypotheses} distinct, actionable thesis hypothesis candidates for this gap.
Each candidate must be DIFFERENT in approach (e.g. one architectural, one empirical).

For each hypothesis return a JSON object with these exact fields:
  "research_question"    : The specific "Can X improve Y when Z?" question (1 sentence)
  "hypothesis_statement" : A falsifiable H1 statement beginning with "We hypothesise that..." (2 sentences)
  "null_hypothesis"      : The corresponding H0 statement (1 sentence)
  "one_liner"            : A ≤15-word plain-English summary of the contribution
  "novelty_justification": Why this is novel vs prior work (2–3 sentences)
  "feasibility_score"    : Float 0.0–1.0 — how achievable within a 12-month PhD chapter
  "feasibility_justification": Why you scored it this way (1–2 sentences)
  "impact_domain"        : One of: fundamental | applied | benchmark | dataset | tool
  "impact_justification" : Why this has high impact (1–2 sentences)
  "impact_score"         : Float 0.0–1.0 — potential impact on the field
  "llm_novelty_estimate" : Float 0.0–1.0 — your honest novelty assessment
  "related_paper_titles" : List of 3–5 most relevant paper titles from the evidence above
  "differentiation"      : How this hypothesis differs from those related papers (2 sentences)

Return a JSON array of {n_hypotheses} objects. No markdown fences. No preamble. Valid JSON only."""

        try:
            response = await self._llm.ainvoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            candidates: list[dict] = json.loads(content)
            logger.debug(
                "HypothesisWriter: %d candidates for gap '%s'",
                len(candidates), enriched_gap["title"],
            )
            return candidates
        except Exception as exc:
            logger.error("HypothesisWriter failed for gap %s: %s", enriched_gap["gap_id"], exc)
            return []


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — NOVELTY SCORER
# ─────────────────────────────────────────────────────────────────────────────

class NoveltyScorer:
    """
    Computes a multi-signal novelty score for each hypothesis candidate.

    Signals:
      1. cluster_sparsity      — fewer papers in supporting clusters = more novel
      2. temporal_recency      — stagnating trend = underexplored = more novel
      3. reproducibility_gap   — low repro score = infrastructure gap = novel angle
      4. semantic_distance     — cross-cluster gaps are inherently more novel
      5. cross_domain_overlap  — bridging multiple clusters signals novelty
      6. llm_novelty_estimate  — GPT-4o's self-assessed novelty from HypothesisWriter

    Each signal is normalised to [0.0, 1.0] before weighting.
    """

    def __init__(self, all_clusters: list[Cluster]) -> None:
        self._clusters = all_clusters
        # Pre-compute global stats for normalisation
        all_counts = [c.paper_count for c in all_clusters] or [1]
        self._max_papers = max(all_counts)
        self._min_papers = min(all_counts)

    def score(
        self,
        candidate: dict,
        enriched_gap: dict,
    ) -> NoveltySignals:
        """Compute all novelty signals for a single hypothesis candidate."""

        signals = NoveltySignals()

        # ── Signal 1: Cluster sparsity ──────────────────────────────────────
        # Inverse of normalised average paper count in supporting clusters.
        # Sparse cluster (few papers) → high novelty signal.
        avg_papers = enriched_gap.get("avg_papers_per_cluster", self._max_papers)
        paper_range = self._max_papers - self._min_papers
        if paper_range > 0:
            normalised = (avg_papers - self._min_papers) / paper_range
            signals.cluster_sparsity = round(1.0 - normalised, 4)
        else:
            signals.cluster_sparsity = 0.5

        # ── Signal 2: Temporal recency / stagnation ──────────────────────────
        # Stagnating clusters = nobody has worked here recently = novel opportunity.
        stagnating = enriched_gap.get("stagnating_count", 0)
        total = len(enriched_gap.get("cluster_metadata", [])) or 1
        signals.temporal_recency = round(stagnating / total, 4)

        # ── Signal 3: Reproducibility gap ───────────────────────────────────
        # Low reproducibility score = open infra opportunity = novel contribution.
        min_repro = enriched_gap.get("min_repro_score", 0.5)
        signals.reproducibility_gap = round(1.0 - min_repro, 4)

        # ── Signal 4: Semantic distance (cross-cluster proxy) ───────────────
        # If gap spans multiple clusters, the hypothesis bridges semantic space.
        n_clusters = len(enriched_gap.get("cluster_metadata", []))
        # Normalise: 1 cluster = 0.0, 4+ clusters = 1.0
        signals.semantic_distance = round(min(1.0, (n_clusters - 1) / 3.0), 4)

        # ── Signal 5: Cross-domain overlap ───────────────────────────────────
        cross = enriched_gap.get("cross_cluster_flag", False)
        dominant_cats = list({
            m["dominant_category"]
            for m in enriched_gap.get("cluster_metadata", [])
        })
        # Multi-category gap scores higher
        signals.cross_domain_overlap = round(
            min(1.0, 0.4 * int(cross) + 0.2 * len(dominant_cats)), 4
        )

        # ── Signal 6: LLM novelty estimate ───────────────────────────────────
        signals.llm_novelty_estimate = float(
            candidate.get("llm_novelty_estimate", 0.5)
        )

        return signals


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — METHODOLOGY ADVISOR
# ─────────────────────────────────────────────────────────────────────────────

class MethodologyAdvisor:
    """
    Generates a concrete, step-by-step experimental methodology plan
    for each shortlisted hypothesis using GPT-4o.

    The plan includes datasets, baselines, metrics, compute requirements,
    risks, and a month-by-month timeline — giving the hypothesis enough
    substance for a real thesis proposal.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def advise(
        self,
        topic: str,
        hypothesis: dict,
        enriched_gap: dict,
    ) -> MethodologyPlan:
        """
        Generate a full MethodologyPlan for a hypothesis candidate.
        """
        prompt = f"""You are a research methodology expert advising a PhD student in: {topic}

HYPOTHESIS:
  Research question: {hypothesis.get('research_question', '')}
  Statement:         {hypothesis.get('hypothesis_statement', '')}
  Impact domain:     {hypothesis.get('impact_domain', 'applied')}
  Suggested approach from gap analysis: {enriched_gap.get('suggested_methodology', 'empirical_analysis')}

Design a detailed experimental methodology plan.

Return a single JSON object with these exact fields:
  "methodology_type"     : One of: benchmark_study | architecture_proposal | dataset_curation | empirical_analysis | survey | theoretical_analysis | system_design | human_study
  "description"          : 2–3 sentence methodology overview
  "steps"                : Ordered list of 5–8 concrete experimental steps (each ≤20 words)
  "suggested_datasets"   : List of 3–5 specific dataset names (real datasets, e.g. "MIMIC-CXR", "MS-COCO")
  "suggested_baselines"  : List of 3–5 specific model/method names to compare against
  "evaluation_metrics"   : List of 4–6 evaluation metrics (e.g. "AUROC", "macro-F1", "BERTScore")
  "tools_and_frameworks" : List of 3–5 tools/libraries (e.g. "HuggingFace Transformers", "PyTorch Lightning")
  "estimated_months"     : Integer — realistic months for a single PhD chapter (3–12)
  "compute_requirements" : String describing GPU/compute needs (e.g. "2x A100 40GB for ~72h training")
  "risks"                : List of 3–5 concrete risks that could sink this experiment
  "mitigations"          : List of mitigation strategies, one per risk (same length as risks)

No markdown fences. No preamble. Valid JSON only."""

        try:
            response = await self._llm.ainvoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            data: dict = json.loads(content)
        except Exception as exc:
            logger.error("MethodologyAdvisor failed: %s", exc)
            data = {}

        # Parse methodology_type safely
        raw_type = data.get("methodology_type", "empirical_analysis")
        try:
            mtype = MethodologyType(raw_type)
        except ValueError:
            mtype = MethodologyType.EMPIRICAL_ANALYSIS

        return MethodologyPlan(
            methodology_type=mtype,
            description=data.get("description", ""),
            steps=data.get("steps", []),
            suggested_datasets=data.get("suggested_datasets", []),
            suggested_baselines=data.get("suggested_baselines", []),
            evaluation_metrics=data.get("evaluation_metrics", []),
            tools_and_frameworks=data.get("tools_and_frameworks", []),
            estimated_months=int(data.get("estimated_months", 6)),
            compute_requirements=data.get("compute_requirements", "Unknown"),
            risks=data.get("risks", []),
            mitigations=data.get("mitigations", []),
        )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — HYPOTHESIS RANKER
# ─────────────────────────────────────────────────────────────────────────────

class HypothesisRanker:
    """
    Computes the composite score for each hypothesis and returns a ranked
    list. Applies a small diversity penalty to prevent all top hypotheses
    from the same methodology type.

    Composite score formula:
        score = (novelty × 0.40) + (feasibility × 0.30) + (impact × 0.30)

    Diversity adjustment: if 3+ hypotheses share the same methodology_type,
    reduce later ones by 0.05 to encourage methodological variety.
    """

    def rank(
        self,
        hypotheses: list[ScoredHypothesis],
        max_results: int = MAX_HYPOTHESES,
    ) -> list[ScoredHypothesis]:
        """Returns hypotheses sorted by composite_score descending."""

        # Recompute composite scores (novelty/feasibility/impact may have
        # been updated by the MethodologyAdvisor round)
        for h in hypotheses:
            h.composite_score = round(
                h.novelty_score * NOVELTY_WEIGHT
                + h.feasibility_score * FEASIBILITY_WEIGHT
                + h.impact_score * IMPACT_WEIGHT,
                4,
            )

        # Sort descending
        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)

        # Diversity penalty
        method_counts: dict[str, int] = {}
        for h in hypotheses:
            mtype = h.methodology.methodology_type.value
            method_counts[mtype] = method_counts.get(mtype, 0) + 1
            if method_counts[mtype] > 2:
                h.composite_score = max(0.0, h.composite_score - 0.05)

        # Re-sort after penalty
        hypotheses.sort(key=lambda h: h.composite_score, reverse=True)

        # Assign ranks
        for i, h in enumerate(hypotheses[:max_results], 1):
            h.rank = i

        logger.info(
            "HypothesisRanker: top %d | scores: %s",
            max_results,
            [round(h.composite_score, 3) for h in hypotheses[:max_results]],
        )

        return hypotheses[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HYPOTHESIS AGENT
# ─────────────────────────────────────────────────────────────────────────────

class HypothesisAgent:
    """
    Orchestrates the full hypothesis generation pipeline:
        GapEnricher → HypothesisWriter → NoveltyScorer
            → MethodologyAdvisor → HypothesisRanker

    Designed to be scheduler-dispatched as a single HIGH-priority task
    after the ResearchAgent's gap detection completes.

    The agent generates hypotheses_per_gap × len(gaps) candidates,
    scores and ranks them, and returns the top MAX_HYPOTHESES.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.3,
        hypotheses_per_gap: int = 2,
        max_output: int = MAX_HYPOTHESES,
    ) -> None:
        self._llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._hypotheses_per_gap = hypotheses_per_gap
        self._max_output = max_output

        self._enricher  = GapEnricher()
        self._writer    = HypothesisWriter(self._llm)
        self._advisor   = MethodologyAdvisor(self._llm)
        self._ranker    = HypothesisRanker()

        logger.info(
            "HypothesisAgent initialised | model=%s | per_gap=%d | max_out=%d",
            model_name, hypotheses_per_gap, max_output,
        )

    # ── MAIN ENTRYPOINT ──────────────────────────────────────────────────────

    async def run(
        self,
        topic: str,
        gaps: list[ResearchGap],
        clusters: list[Cluster],
        papers: list[Paper],
    ) -> list[ScoredHypothesis]:
        """
        Full hypothesis generation pipeline.

        Steps:
          1. Enrich gaps with cluster and paper evidence
          2. Generate hypothesis candidates for each gap (parallel)
          3. Score novelty for all candidates
          4. Shortlist top candidates (2× max_output)
          5. Generate methodology plans for shortlisted candidates (parallel)
          6. Final ranking and diversity adjustment

        Returns up to self._max_output ranked ScoredHypothesis objects.
        """
        start_t = time.time()
        logger.info("HypothesisAgent.run() | topic='%s' | %d gaps", topic, len(gaps))

        if not gaps:
            logger.warning("No gaps provided — returning empty hypothesis list")
            return []

        # ── Stage 1: Enrich gaps ─────────────────────────────────────────────
        enriched_gaps = self._enricher.enrich(gaps, clusters, papers)

        scorer = NoveltyScorer(clusters)

        # ── Stage 2 + 3: Write and score hypotheses (parallel per gap) ───────
        all_candidates: list[tuple[dict, dict, NoveltySignals]] = []

        write_tasks = [
            self._writer.write(topic, eg, self._hypotheses_per_gap)
            for eg in enriched_gaps
        ]
        gap_candidate_lists = await asyncio.gather(*write_tasks)

        for enriched_gap, candidates in zip(enriched_gaps, gap_candidate_lists):
            for candidate in candidates:
                signals = scorer.score(candidate, enriched_gap)
                all_candidates.append((candidate, enriched_gap, signals))

        logger.info(
            "Generated %d total candidates from %d gaps",
            len(all_candidates), len(gaps),
        )

        if not all_candidates:
            logger.error("No hypothesis candidates generated")
            return []

        # ── Stage 4: Shortlist (pre-rank before expensive methodology step) ──
        shortlist = self._shortlist(all_candidates, n=self._max_output * 2)
        logger.info("Shortlisted %d candidates for methodology generation", len(shortlist))

        # ── Stage 5: Generate methodology plans (parallel) ───────────────────
        methodology_tasks = [
            self._advisor.advise(topic, cand, egap)
            for cand, egap, _ in shortlist
        ]
        methodology_plans = await asyncio.gather(*methodology_tasks)

        # ── Stage 6: Assemble ScoredHypothesis objects ───────────────────────
        scored: list[ScoredHypothesis] = []
        gap_map = {g.gap_id: g for g in gaps}

        for i, ((candidate, enriched_gap, signals), methodology) in enumerate(
            zip(shortlist, methodology_plans)
        ):
            gap_id = enriched_gap["gap_id"]
            gap = gap_map.get(gap_id)

            novelty  = signals.composite()
            feasibility = float(candidate.get("feasibility_score", 0.5))
            impact   = float(candidate.get("impact_score", 0.5))
            composite = round(
                novelty * NOVELTY_WEIGHT
                + feasibility * FEASIBILITY_WEIGHT
                + impact * IMPACT_WEIGHT,
                4,
            )

            # Map related_paper_titles back to arxiv IDs (best-effort)
            related_titles = candidate.get("related_paper_titles", [])
            related_ids = self._match_paper_ids(related_titles, papers)

            # Impact domain
            raw_domain = candidate.get("impact_domain", "applied")
            try:
                impact_domain = ImpactDomain(raw_domain)
            except ValueError:
                impact_domain = ImpactDomain.APPLIED

            h = ScoredHypothesis(
                hypothesis_id=f"H{i+1:02d}_{gap_id}",
                rank=i + 1,   # temporary; overwritten by ranker
                research_question=candidate.get("research_question", ""),
                hypothesis_statement=candidate.get("hypothesis_statement", ""),
                null_hypothesis=candidate.get("null_hypothesis", ""),
                one_liner=candidate.get("one_liner", ""),
                source_gap_ids=[gap_id],
                source_cluster_ids=enriched_gap.get(
                    "supporting_clusters",
                    gap.supporting_clusters if gap else [],
                ),
                novelty_signals=signals,
                novelty_score=novelty,
                feasibility_score=feasibility,
                impact_score=impact,
                composite_score=composite,
                novelty_justification=candidate.get("novelty_justification", ""),
                feasibility_justification=candidate.get("feasibility_justification", ""),
                impact_justification=candidate.get("impact_justification", ""),
                impact_domain=impact_domain,
                impact_domain_reason=candidate.get("impact_justification", ""),
                methodology=methodology,
                related_paper_ids=related_ids,
                prior_work_summary="",
                differentiation=candidate.get("differentiation", ""),
                topic=topic,
            )
            scored.append(h)

        # ── Stage 7: Final ranking ───────────────────────────────────────────
        ranked = self._ranker.rank(scored, self._max_output)

        elapsed = time.time() - start_t
        logger.info(
            "HypothesisAgent complete | %.1fs | %d hypotheses | "
            "top composite scores: %s",
            elapsed,
            len(ranked),
            [round(h.composite_score, 3) for h in ranked],
        )

        return ranked

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _shortlist(
        self,
        candidates: list[tuple[dict, dict, NoveltySignals]],
        n: int,
    ) -> list[tuple[dict, dict, NoveltySignals]]:
        """
        Pre-rank candidates by novelty + feasibility + impact before the
        expensive methodology generation step (saves LLM calls).
        """
        def quick_score(item: tuple) -> float:
            candidate, _, signals = item
            nov  = signals.composite()
            feas = float(candidate.get("feasibility_score", 0.5))
            imp  = float(candidate.get("impact_score", 0.5))
            return (
                nov  * NOVELTY_WEIGHT
                + feas * FEASIBILITY_WEIGHT
                + imp  * IMPACT_WEIGHT
            )

        ranked = sorted(candidates, key=quick_score, reverse=True)

        # Ensure diversity: at most 2 per source gap
        gap_counts: dict[int, int] = {}
        shortlist = []
        for item in ranked:
            gap_id = item[1]["gap_id"]
            if gap_counts.get(gap_id, 0) < 2:
                shortlist.append(item)
                gap_counts[gap_id] = gap_counts.get(gap_id, 0) + 1
            if len(shortlist) >= n:
                break

        return shortlist

    def _match_paper_ids(
        self,
        titles: list[str],
        papers: list[Paper],
    ) -> list[str]:
        """
        Fuzzy-match hypothesis related titles back to arXiv IDs.
        Uses simple substring matching — good enough for this purpose.
        """
        matched: list[str] = []
        for title in titles:
            title_lower = title.lower()
            for paper in papers:
                if title_lower[:30] in paper.title.lower():
                    matched.append(paper.arxiv_id)
                    break
        return matched[:5]

    # ── FORMATTING HELPERS (for Streamlit display) ───────────────────────────

    @staticmethod
    def to_markdown(hypothesis: ScoredHypothesis) -> str:
        """
        Render a ScoredHypothesis as a Markdown card — used by
        app/pages/03_hypothesis_studio.py for display.
        """
        m = hypothesis.methodology
        sig = hypothesis.novelty_signals

        return f"""## {hypothesis.rank}. {hypothesis.one_liner}

**Hypothesis ID:** `{hypothesis.hypothesis_id}`
**Composite Score:** `{hypothesis.composite_score:.3f}`
&nbsp;&nbsp;| Novelty: `{hypothesis.novelty_score:.3f}`
&nbsp;&nbsp;| Feasibility: `{hypothesis.feasibility_score:.3f}`
&nbsp;&nbsp;| Impact: `{hypothesis.impact_score:.3f}`

---

### Research Question
{hypothesis.research_question}

### Hypothesis Statement (H₁)
{hypothesis.hypothesis_statement}

### Null Hypothesis (H₀)
{hypothesis.null_hypothesis}

---

### Novelty Score Breakdown
| Signal | Score |
|--------|-------|
| Cluster sparsity | `{sig.cluster_sparsity:.3f}` |
| Temporal recency | `{sig.temporal_recency:.3f}` |
| Reproducibility gap | `{sig.reproducibility_gap:.3f}` |
| Semantic distance | `{sig.semantic_distance:.3f}` |
| Cross-domain overlap | `{sig.cross_domain_overlap:.3f}` |
| LLM novelty estimate | `{sig.llm_novelty_estimate:.3f}` |
| **Composite novelty** | **`{hypothesis.novelty_score:.3f}`** |

{hypothesis.novelty_justification}

---

### Methodology: `{m.methodology_type.value}`

{m.description}

**Steps:**
{"".join(f'{chr(10)}{i+1}. {s}' for i, s in enumerate(m.steps))}

**Suggested datasets:** {", ".join(m.suggested_datasets) or "—"}
**Baselines:** {", ".join(m.suggested_baselines) or "—"}
**Metrics:** {", ".join(m.evaluation_metrics) or "—"}
**Tools:** {", ".join(m.tools_and_frameworks) or "—"}
**Timeline:** ~{m.estimated_months} months
**Compute:** {m.compute_requirements}

### Risks & Mitigations
{"".join(f'{chr(10)}- **Risk {i+1}:** {r}  {chr(10)}  *Mitigation:* {m.mitigations[i] if i < len(m.mitigations) else "N/A"}' for i, r in enumerate(m.risks))}

---

### Differentiation from Prior Work
{hypothesis.differentiation}

**Related arXiv papers:** {", ".join(f"`{pid}`" for pid in hypothesis.related_paper_ids) or "—"}
"""

    @staticmethod
    def to_summary_dict(h: ScoredHypothesis) -> dict:
        """Compact dict for Streamlit table / session history."""
        return {
            "rank":                h.rank,
            "hypothesis_id":       h.hypothesis_id,
            "one_liner":           h.one_liner,
            "novelty_score":       h.novelty_score,
            "feasibility_score":   h.feasibility_score,
            "impact_score":        h.impact_score,
            "composite_score":     h.composite_score,
            "methodology_type":    h.methodology.methodology_type.value,
            "estimated_months":    h.methodology.estimated_months,
            "impact_domain":       h.impact_domain.value,
            "n_datasets":          len(h.methodology.suggested_datasets),
            "n_related_papers":    len(h.related_paper_ids),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

async def _demo() -> None:
    """
    Smoke-test with synthetic gaps/clusters/papers.
    Run with: python -m agents.hypothesis_agent
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # ── Synthetic test data ──────────────────────────────────────────────────
    clusters = [
        Cluster(
            cluster_id=0, label="Zero-Shot Radiology Report Generation",
            paper_ids=["p1", "p2", "p3"], paper_count=3,
            year_range=(2020, 2022), dominant_category="cs.CV",
            reproducibility_score=0.15, temporal_trend="stagnating",
        ),
        Cluster(
            cluster_id=1, label="Multimodal Clinical Decision Support",
            paper_ids=["p4", "p5", "p6", "p7", "p8"],
            paper_count=5, year_range=(2021, 2024), dominant_category="cs.AI",
            reproducibility_score=0.42, temporal_trend="growing",
        ),
        Cluster(
            cluster_id=2, label="Low-Resource Medical NLP",
            paper_ids=["p9", "p10"], paper_count=2,
            year_range=(2019, 2021), dominant_category="cs.CL",
            reproducibility_score=0.08, temporal_trend="stagnating",
        ),
    ]

    papers = [
        Paper(
            arxiv_id=f"240{i}.0000{i}", title=f"Sample Paper {i}",
            abstract="We propose a multimodal approach to medical diagnosis "
                     "using vision-language models. Code available at github.",
            authors=["Author A"], published="2023-01-01", year=2023,
            categories=["cs.CV"], url=f"https://arxiv.org/abs/240{i}.0000{i}",
            cluster_id=i % 3,
        )
        for i in range(10)
    ]

    gaps = [
        ResearchGap(
            gap_id=0,
            title="Cross-modal alignment in low-resource clinical settings",
            description=(
                "Existing vision-language models for medical imaging assume large "
                "annotated datasets. Only 3 papers address the low-resource regime, "
                "all published before 2022, suggesting significant opportunity for "
                "cross-modal alignment techniques that work with <100 labelled cases."
            ),
            supporting_clusters=[0, 2],
            evidence={"paper_count": 3, "recency": "stagnating"},
            confidence=0.78,
            suggested_methodology="architecture_proposal",
        ),
        ResearchGap(
            gap_id=1,
            title="Reproducibility infrastructure for clinical NLP benchmarks",
            description=(
                "Only 8% of papers in the low-resource medical NLP cluster release "
                "code or data. No standardised benchmark exists for low-resource "
                "clinical NLP, making it impossible to compare methods fairly."
            ),
            supporting_clusters=[1, 2],
            evidence={"repro_score": 0.08},
            confidence=0.65,
            suggested_methodology="dataset_curation",
        ),
    ]

    # ── Run agent ────────────────────────────────────────────────────────────
    agent = HypothesisAgent(hypotheses_per_gap=2, max_output=3)
    hypotheses = await agent.run(
        topic="Multimodal Large Language Models for Medical Diagnosis",
        gaps=gaps,
        clusters=clusters,
        papers=papers,
    )

    # ── Print results ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  Generated {len(hypotheses)} ranked hypotheses")
    print("=" * 70)

    for h in hypotheses:
        print(f"\n  Rank {h.rank} | Score {h.composite_score:.3f} | {h.one_liner}")
        print(f"  Q:  {h.research_question}")
        print(f"  H1: {h.hypothesis_statement[:120]}...")
        print(f"  Method: {h.methodology.methodology_type.value} "
              f"| ~{h.methodology.estimated_months}mo "
              f"| {h.methodology.compute_requirements}")
        print(f"  Novelty signals: sparsity={h.novelty_signals.cluster_sparsity:.2f} "
              f"recency={h.novelty_signals.temporal_recency:.2f} "
              f"repro_gap={h.novelty_signals.reproducibility_gap:.2f} "
              f"llm={h.novelty_signals.llm_novelty_estimate:.2f}")
        print(f"  Datasets: {', '.join(h.methodology.suggested_datasets[:3])}")
        print(f"  Risks: {len(h.methodology.risks)} identified")

    print("\n" + "─" * 70)
    print("  Markdown card (first hypothesis):")
    print("─" * 70)
    if hypotheses:
        print(HypothesisAgent.to_markdown(hypotheses[0])[:800] + "\n  ...")


if __name__ == "__main__":
    asyncio.run(_demo())
