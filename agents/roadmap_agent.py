"""
agents/roadmap_agent.py
=======================
ResearchFlow AI — Intelligent Reading Roadmap Builder Agent

Transforms raw paper clusters, research gaps, and ranked hypotheses into
a structured, personalised reading roadmap — ordered from foundational
concepts through to the bleeding edge of the field.

Pipeline (within this agent):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Papers + Clusters + Gaps + Hypotheses                              │
  │         ↓                                                           │
  │  [1] TierClassifier   — assigns every paper to a reading tier      │
  │         ↓                                                           │
  │  [2] PaperSelector    — picks the best N papers per tier           │
  │         ↓                                                           │
  │  [3] AnnotationWriter — GPT-4o writes a reading note per paper     │
  │         ↓                                                           │
  │  [4] PathBuilder      — assembles the ordered roadmap with         │
  │                          milestone markers and time estimates       │
  │         ↓                                                           │
  │  ReadingRoadmap  (structured, annotated, exportable)               │
  └─────────────────────────────────────────────────────────────────────┘

Reading tiers:
  TIER_1_FOUNDATION   — Seminal papers; must-read before anything else
  TIER_2_METHODS      — Core methodology papers in dominant clusters
  TIER_3_GAP_ADJACENT — Papers closest to the identified research gaps
  TIER_4_RECENT       — Most recent work (last 18 months) in growing clusters
  TIER_5_EXTENDED     — Stretch reading: related fields, datasets, surveys

Output:
  ReadingRoadmap
    ├── milestones        — named checkpoints with paper groups
    ├── annotated_papers  — AnnotatedPaper per entry (note, why_read, time_est)
    ├── weekly_plan       — optional week-by-week schedule
    ├── total_papers      — count
    ├── total_hours_est   — estimated reading hours
    └── markdown / json export methods

Usage:
    agent = RoadmapAgent()
    roadmap = await agent.run(
        topic="Multimodal LLMs for Medical Diagnosis",
        papers=papers,
        clusters=clusters,
        gaps=gaps,
        hypotheses=hypotheses,
        target_papers=20,
        weeks_available=8,
    )
    print(roadmap.to_markdown())

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain_openai import ChatOpenAI

try:
    from agents.research_agent import Cluster, Paper, ResearchGap
    from agents.hypothesis_agent import ScoredHypothesis
except ImportError:
    from research_agent import Cluster, Paper, ResearchGap          # type: ignore
    from hypothesis_agent import ScoredHypothesis                   # type: ignore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL        = "gpt-4o"
DEFAULT_TARGET       = 20          # total papers in roadmap
DEFAULT_WEEKS        = 8           # weeks available to read
AVG_MINS_PER_PAPER   = 45          # estimated reading time per paper (minutes)

# Tier target proportions (sum to ~1.0)
TIER_PROPORTIONS = {
    "TIER_1_FOUNDATION":   0.20,   # ~4 of 20
    "TIER_2_METHODS":      0.25,   # ~5 of 20
    "TIER_3_GAP_ADJACENT": 0.25,   # ~5 of 20
    "TIER_4_RECENT":       0.20,   # ~4 of 20
    "TIER_5_EXTENDED":     0.10,   # ~2 of 20
}


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class ReadingTier(str, Enum):
    TIER_1_FOUNDATION   = "TIER_1_FOUNDATION"
    TIER_2_METHODS      = "TIER_2_METHODS"
    TIER_3_GAP_ADJACENT = "TIER_3_GAP_ADJACENT"
    TIER_4_RECENT       = "TIER_4_RECENT"
    TIER_5_EXTENDED     = "TIER_5_EXTENDED"


TIER_LABELS = {
    ReadingTier.TIER_1_FOUNDATION:   "📚 Tier 1 — Foundations",
    ReadingTier.TIER_2_METHODS:      "🔬 Tier 2 — Core Methods",
    ReadingTier.TIER_3_GAP_ADJACENT: "🎯 Tier 3 — Gap-Adjacent",
    ReadingTier.TIER_4_RECENT:       "🚀 Tier 4 — Recent Work",
    ReadingTier.TIER_5_EXTENDED:     "🌐 Tier 5 — Extended Reading",
}

TIER_DESCRIPTIONS = {
    ReadingTier.TIER_1_FOUNDATION: (
        "Seminal papers that define the field. Read these first to build "
        "conceptual vocabulary and understand the problem space."
    ),
    ReadingTier.TIER_2_METHODS: (
        "Core methodology papers from the dominant research clusters. "
        "These establish the technical toolkit you will build upon."
    ),
    ReadingTier.TIER_3_GAP_ADJACENT: (
        "Papers closest to the identified research gaps. Essential reading "
        "before designing experiments — know exactly what has been tried."
    ),
    ReadingTier.TIER_4_RECENT: (
        "Most recent publications in growing sub-fields. Track the frontier "
        "and avoid duplicating work published in the past 18 months."
    ),
    ReadingTier.TIER_5_EXTENDED: (
        "Stretch reading: related fields, benchmark datasets, survey papers. "
        "Read once core tiers are complete."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnotatedPaper:
    """
    A single paper entry in the roadmap, enriched with reading guidance.
    """
    arxiv_id:         str
    title:            str
    authors:          list[str]
    year:             int
    url:              str
    cluster_label:    str
    tier:             ReadingTier
    reading_order:    int               # 1-based global position in roadmap

    # GPT-4o generated annotations
    one_line_summary: str               # ≤20-word contribution summary
    why_read:         str               # why this paper belongs in the roadmap
    key_takeaway:     str               # the single most important insight
    reading_note:     str               # what to focus on while reading
    connects_to:      list[str]         # arxiv IDs of papers to read before/after

    # Time and effort
    estimated_mins:   int = AVG_MINS_PER_PAPER
    difficulty:       str = "medium"    # "easy" | "medium" | "hard"
    priority:         str = "high"      # "high" | "medium" | "low"


@dataclass
class Milestone:
    """
    A named checkpoint in the roadmap — groups related papers and marks
    progress stages. Analogous to a sprint boundary in a research plan.
    """
    milestone_id:     int
    name:             str               # e.g. "Understand the landscape"
    description:      str               # what you should know after this milestone
    tier:             ReadingTier
    paper_ids:        list[str]         # arxiv IDs in this milestone
    week_start:       int               # suggested start week (1-based)
    week_end:         int               # suggested end week (1-based)
    learning_goals:   list[str]         # 3–5 concrete goals
    self_check:       list[str]         # questions to answer before moving on


@dataclass
class WeeklyPlan:
    """One week in the optional weekly reading schedule."""
    week:             int
    milestone_name:   str
    papers:           list[str]         # arxiv IDs
    paper_titles:     list[str]
    estimated_hours:  float
    focus:            str               # what to focus on this week


@dataclass
class ReadingRoadmap:
    """
    The complete output of the RoadmapAgent.
    All fields are populated; the roadmap is immediately usable for
    display, export, or session persistence.
    """
    session_id:       str
    topic:            str
    generated_at:     str

    # Core content
    annotated_papers: list[AnnotatedPaper]
    milestones:       list[Milestone]
    weekly_plan:      list[WeeklyPlan]

    # Metadata
    total_papers:     int
    total_hours_est:  float
    weeks_available:  int
    hours_per_week:   float

    # Tier breakdown
    tier_counts:      dict[str, int]    # tier_name → paper count

    # Gap / hypothesis alignment
    gap_coverage:     dict[str, list[str]]  # gap_title → [arxiv_ids]
    hypothesis_links: dict[str, list[str]]  # hypothesis_id → [arxiv_ids]

    # Executive summary (GPT-4o generated)
    executive_summary: str
    field_overview:    str

    def to_markdown(self) -> str:
        """Render the full roadmap as a structured Markdown document."""
        return _render_markdown(self)

    def to_dict(self) -> dict:
        """Serialise to a JSON-safe dict for persistence."""
        return asdict(self)

    def paper_ids_ordered(self) -> list[str]:
        """Ordered list of arXiv IDs — the minimal roadmap representation."""
        return [p.arxiv_id for p in self.annotated_papers]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — TIER CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class TierClassifier:
    """
    Assigns each paper to a ReadingTier using deterministic rules
    (no LLM calls — fast and reproducible).

    Rules (evaluated in priority order):
      TIER_1: Oldest papers in the largest clusters (foundational)
      TIER_4: Published in last 18 months AND in a growing cluster (recent)
      TIER_3: In a cluster that overlaps with an identified research gap
      TIER_2: All other papers in named clusters (core methods)
      TIER_5: Noise/outlier papers (cluster_id == -1) or low-confidence
    """

    def __init__(
        self,
        clusters: list[Cluster],
        gaps: list[ResearchGap],
    ) -> None:
        self._cluster_map = {c.cluster_id: c for c in clusters}
        self._gap_cluster_ids: set[int] = set(
            cid for g in gaps for cid in g.supporting_clusters
        )

        # Foundational clusters: top-3 by paper count
        sorted_by_size = sorted(clusters, key=lambda c: -c.paper_count)
        self._foundational_cluster_ids: set[int] = {
            c.cluster_id for c in sorted_by_size[:3]
        }

        # Year threshold for "recent" (last 18 months from most recent paper)
        all_years = [
            p_year
            for c in clusters
            for p_year in range(c.year_range[0], c.year_range[1] + 1)
        ]
        self._recent_threshold = (max(all_years) - 1) if all_years else 2023

    def classify(self, paper: Paper) -> ReadingTier:
        cluster = self._cluster_map.get(paper.cluster_id)

        # Noise / outlier
        if paper.cluster_id == -1 or cluster is None:
            return ReadingTier.TIER_5_EXTENDED

        # Tier 1: foundational cluster AND old enough to be seminal
        if (
            paper.cluster_id in self._foundational_cluster_ids
            and paper.year <= self._recent_threshold - 2
        ):
            return ReadingTier.TIER_1_FOUNDATION

        # Tier 4: recent work in a growing cluster
        if (
            paper.year >= self._recent_threshold
            and cluster.temporal_trend == "growing"
        ):
            return ReadingTier.TIER_4_RECENT

        # Tier 3: gap-adjacent cluster
        if paper.cluster_id in self._gap_cluster_ids:
            return ReadingTier.TIER_3_GAP_ADJACENT

        # Tier 2: all other clustered papers
        return ReadingTier.TIER_2_METHODS

    def classify_all(self, papers: list[Paper]) -> dict[str, ReadingTier]:
        """Returns {arxiv_id: ReadingTier} for all papers."""
        return {p.arxiv_id: self.classify(p) for p in papers}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — PAPER SELECTOR
# ─────────────────────────────────────────────────────────────────────────────

class PaperSelector:
    """
    Selects the best papers per tier to hit the target_papers budget.

    Selection criteria within each tier:
      - Tier 1: oldest first (most seminal)
      - Tier 2: most papers in cluster first (broadest coverage)
      - Tier 3: closest to gap confidence (highest gap confidence clusters)
      - Tier 4: newest first (most recent)
      - Tier 5: highest reproducibility score first (most useful resources)

    Also ensures gap coverage: at least one paper per identified gap.
    """

    def __init__(
        self,
        clusters: list[Cluster],
        gaps: list[ResearchGap],
        target_papers: int = DEFAULT_TARGET,
    ) -> None:
        self._cluster_map = {c.cluster_id: c for c in clusters}
        self._gaps = gaps
        self._target = target_papers
        self._tier_budgets = {
            tier: max(1, round(target_papers * prop))
            for tier, prop in TIER_PROPORTIONS.items()
        }

    def select(
        self,
        papers: list[Paper],
        tier_assignments: dict[str, ReadingTier],
    ) -> list[Paper]:
        """
        Returns a deduplicated, budget-respecting selection of papers
        ordered by tier then within-tier criteria.
        """
        # Group papers by tier
        tier_buckets: dict[str, list[Paper]] = {t.value: [] for t in ReadingTier}
        for paper in papers:
            tier = tier_assignments.get(paper.arxiv_id, ReadingTier.TIER_5_EXTENDED)
            tier_buckets[tier.value].append(paper)

        # Sort each bucket
        tier_buckets[ReadingTier.TIER_1_FOUNDATION.value].sort(key=lambda p: p.year)
        tier_buckets[ReadingTier.TIER_2_METHODS.value].sort(
            key=lambda p: -(self._cluster_map.get(p.cluster_id, type("_", (), {"paper_count": 0})()).paper_count)
        )
        tier_buckets[ReadingTier.TIER_3_GAP_ADJACENT.value].sort(key=lambda p: -p.year)
        tier_buckets[ReadingTier.TIER_4_RECENT.value].sort(key=lambda p: -p.year)
        tier_buckets[ReadingTier.TIER_5_EXTENDED.value].sort(
            key=lambda p: -(self._cluster_map.get(p.cluster_id, type("_", (), {"reproducibility_score": 0})()).reproducibility_score)
        )

        # Apply budgets
        selected: list[Paper] = []
        seen: set[str] = set()

        for tier in ReadingTier:
            budget = self._tier_budgets.get(tier.value, 1)
            bucket = tier_buckets[tier.value]
            for paper in bucket:
                if paper.arxiv_id not in seen and len([p for p in selected if tier_assignments.get(p.arxiv_id) == tier]) < budget:
                    selected.append(paper)
                    seen.add(paper.arxiv_id)

        # Guarantee gap coverage: ensure at least one paper per gap cluster
        gap_cluster_ids = {
            cid for g in self._gaps for cid in g.supporting_clusters
        }
        for paper in papers:
            if paper.cluster_id in gap_cluster_ids and paper.arxiv_id not in seen:
                if len(selected) < self._target + 5:   # slight budget flex
                    selected.append(paper)
                    seen.add(paper.arxiv_id)

        # Trim to target
        return selected[:self._target]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — ANNOTATION WRITER
# ─────────────────────────────────────────────────────────────────────────────

class AnnotationWriter:
    """
    Uses GPT-4o to generate concise, actionable reading annotations
    for each selected paper.

    Annotations are generated in a single batched call (all papers in
    one prompt) to minimise API round-trips and cost.
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def annotate_batch(
        self,
        topic: str,
        papers: list[Paper],
        tier_assignments: dict[str, ReadingTier],
        cluster_map: dict[int, Cluster],
    ) -> dict[str, dict]:
        """
        Generates annotations for all papers in a single GPT-4o call.
        Returns {arxiv_id: annotation_dict}.
        """
        # Build concise paper list for the prompt
        paper_entries = []
        for i, p in enumerate(papers):
            cluster = cluster_map.get(p.cluster_id)
            tier = tier_assignments.get(p.arxiv_id, ReadingTier.TIER_2_METHODS)
            paper_entries.append(
                f'{i+1}. arxiv_id="{p.arxiv_id}" | tier={tier.value} | '
                f'year={p.year} | cluster="{cluster.label if cluster else "unknown"}"\n'
                f'   Title: {p.title}\n'
                f'   Abstract excerpt: {p.abstract[:250]}'
            )

        papers_text = "\n\n".join(paper_entries)

        prompt = f"""You are a research advisor building a reading roadmap for a PhD student in: {topic}

Annotate each of the following papers for a structured reading roadmap.

For each paper return a JSON object with:
  "arxiv_id"         : the exact arxiv_id provided
  "one_line_summary" : ≤20-word contribution summary (what the paper actually does)
  "why_read"         : 1 sentence — why this paper is in the roadmap for this topic
  "key_takeaway"     : the single most important insight to extract
  "reading_note"     : what to focus on while reading (method, results, or framing?)
  "difficulty"       : one of: easy | medium | hard
  "priority"         : one of: high | medium | low
  "estimated_mins"   : integer reading time in minutes (range 20–90)

Return a JSON array of objects, one per paper, in the same order as the input.
No markdown fences. No preamble. Valid JSON array only.

Papers to annotate:
{papers_text}"""

        try:
            response = await self._llm.ainvoke(prompt)
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            annotations: list[dict] = json.loads(content)
            return {a["arxiv_id"]: a for a in annotations if "arxiv_id" in a}
        except Exception as exc:
            logger.error("AnnotationWriter batch failed: %s", exc)
            # Return minimal fallback annotations
            return {
                p.arxiv_id: {
                    "arxiv_id":         p.arxiv_id,
                    "one_line_summary": p.title[:80],
                    "why_read":         "Core paper in the research area.",
                    "key_takeaway":     "See abstract for main contribution.",
                    "reading_note":     "Focus on the methodology section.",
                    "difficulty":       "medium",
                    "priority":         "high",
                    "estimated_mins":   AVG_MINS_PER_PAPER,
                }
                for p in papers
            }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — PATH BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class PathBuilder:
    """
    Assembles the final ReadingRoadmap from selected papers, annotations,
    and scheduling constraints (weeks available, target paper count).

    Responsibilities:
      1. Build AnnotatedPaper objects (merge Paper + annotation + tier)
      2. Build Milestone objects (group papers into named checkpoints)
      3. Build WeeklyPlan (distribute milestones across available weeks)
      4. Compute gap_coverage and hypothesis_links indexes
      5. Generate executive summary via GPT-4o
    """

    def __init__(self, llm: ChatOpenAI) -> None:
        self._llm = llm

    async def build(
        self,
        topic: str,
        papers: list[Paper],
        tier_assignments: dict[str, ReadingTier],
        annotations: dict[str, dict],
        clusters: list[Cluster],
        gaps: list[ResearchGap],
        hypotheses: list[ScoredHypothesis],
        weeks_available: int,
        session_id: str,
    ) -> ReadingRoadmap:
        """Assemble the complete ReadingRoadmap."""

        cluster_map = {c.cluster_id: c for c in clusters}

        # ── Build AnnotatedPaper objects ─────────────────────────────────────
        annotated: list[AnnotatedPaper] = []
        tier_order = list(ReadingTier)   # reading order follows tier order

        # Sort papers: first by tier order, then by within-tier criteria
        def sort_key(p: Paper) -> tuple[int, int]:
            tier = tier_assignments.get(p.arxiv_id, ReadingTier.TIER_5_EXTENDED)
            tier_idx = tier_order.index(tier)
            # Within tier: T1 ascending year, T4 descending year, others neutral
            if tier == ReadingTier.TIER_1_FOUNDATION:
                return (tier_idx, p.year)
            elif tier == ReadingTier.TIER_4_RECENT:
                return (tier_idx, -p.year)
            else:
                return (tier_idx, 0)

        ordered_papers = sorted(papers, key=sort_key)

        for reading_order, paper in enumerate(ordered_papers, 1):
            ann = annotations.get(paper.arxiv_id, {})
            cluster = cluster_map.get(paper.cluster_id)
            tier = tier_assignments.get(paper.arxiv_id, ReadingTier.TIER_5_EXTENDED)

            # Build connects_to: papers in same cluster ± adjacent tiers
            connects_to = [
                p2.arxiv_id for p2 in ordered_papers
                if p2.cluster_id == paper.cluster_id
                and p2.arxiv_id != paper.arxiv_id
            ][:3]

            annotated.append(AnnotatedPaper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                year=paper.year,
                url=paper.url,
                cluster_label=cluster.label if cluster else "Unclustered",
                tier=tier,
                reading_order=reading_order,
                one_line_summary=ann.get("one_line_summary", paper.title[:80]),
                why_read=ann.get("why_read", "Core paper in the research area."),
                key_takeaway=ann.get("key_takeaway", "See abstract."),
                reading_note=ann.get("reading_note", "Focus on methodology."),
                connects_to=connects_to,
                estimated_mins=int(ann.get("estimated_mins", AVG_MINS_PER_PAPER)),
                difficulty=ann.get("difficulty", "medium"),
                priority=ann.get("priority", "high"),
            ))

        # ── Build Milestones ─────────────────────────────────────────────────
        milestones = await self._build_milestones(
            topic, annotated, weeks_available
        )

        # ── Build WeeklyPlan ─────────────────────────────────────────────────
        weekly_plan = self._build_weekly_plan(annotated, milestones, weeks_available)

        # ── Compute indexes ───────────────────────────────────────────────────
        gap_coverage = self._build_gap_coverage(annotated, gaps, cluster_map)
        hypothesis_links = self._build_hypothesis_links(annotated, hypotheses)

        # ── Tier counts ───────────────────────────────────────────────────────
        tier_counts: dict[str, int] = {t.value: 0 for t in ReadingTier}
        for ap in annotated:
            tier_counts[ap.tier.value] += 1

        # ── Totals ────────────────────────────────────────────────────────────
        total_mins = sum(ap.estimated_mins for ap in annotated)
        total_hours = round(total_mins / 60, 1)
        hours_per_week = round(total_hours / weeks_available, 1) if weeks_available else 0.0

        # ── Executive summary ────────────────────────────────────────────────
        summary, overview = await self._generate_summary(
            topic, annotated, milestones, gaps
        )

        return ReadingRoadmap(
            session_id=session_id,
            topic=topic,
            generated_at=datetime.utcnow().isoformat(),
            annotated_papers=annotated,
            milestones=milestones,
            weekly_plan=weekly_plan,
            total_papers=len(annotated),
            total_hours_est=total_hours,
            weeks_available=weeks_available,
            hours_per_week=hours_per_week,
            tier_counts=tier_counts,
            gap_coverage=gap_coverage,
            hypothesis_links=hypothesis_links,
            executive_summary=summary,
            field_overview=overview,
        )

    async def _build_milestones(
        self,
        topic: str,
        annotated: list[AnnotatedPaper],
        weeks_available: int,
    ) -> list[Milestone]:
        """
        Creates one Milestone per tier, each with a name, learning goals,
        and self-check questions generated by GPT-4o.
        """
        # Group by tier
        tier_groups: dict[ReadingTier, list[AnnotatedPaper]] = {
            t: [] for t in ReadingTier
        }
        for ap in annotated:
            tier_groups[ap.tier].append(ap)

        # Distribute weeks proportionally across tiers that have papers
        active_tiers = [t for t in ReadingTier if tier_groups[t]]
        weeks_per_tier = max(1, weeks_available // len(active_tiers)) if active_tiers else 1

        milestone_defs = [
            ("Understand the Landscape",
             "You can describe the major research clusters and their evolution over time.",
             ReadingTier.TIER_1_FOUNDATION),
            ("Master the Technical Toolkit",
             "You can explain the dominant methods and reproduce core results.",
             ReadingTier.TIER_2_METHODS),
            ("Map the Frontier",
             "You can articulate what has been attempted near each research gap.",
             ReadingTier.TIER_3_GAP_ADJACENT),
            ("Track the Bleeding Edge",
             "You know the most recent published work and who the active groups are.",
             ReadingTier.TIER_4_RECENT),
            ("Broaden Your Perspective",
             "You have surveyed adjacent fields and identified transferable techniques.",
             ReadingTier.TIER_5_EXTENDED),
        ]

        milestones: list[Milestone] = []
        week_cursor = 1

        for i, (name, completion_desc, tier) in enumerate(milestone_defs):
            papers_in_tier = tier_groups[tier]
            if not papers_in_tier:
                continue

            paper_ids = [ap.arxiv_id for ap in papers_in_tier]
            week_end = min(week_cursor + weeks_per_tier - 1, weeks_available)

            # Generate learning goals + self-check via GPT-4o
            titles_sample = "\n".join(
                f"- {ap.title}" for ap in papers_in_tier[:5]
            )
            prompt = f"""For a PhD student in: {topic}
Reading milestone: "{name}"
Papers include:
{titles_sample}

Generate for this milestone:
  "learning_goals" : JSON array of 3–5 concrete things the student will be able to DO after this milestone (start each with an action verb)
  "self_check"     : JSON array of 3–5 questions the student should be able to answer before moving to the next milestone

Return a single JSON object. No markdown. No preamble."""

            try:
                resp = await self._llm.ainvoke(prompt)
                content = resp.content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1].lstrip("json").strip()
                data = json.loads(content)
                learning_goals = data.get("learning_goals", [])
                self_check     = data.get("self_check", [])
            except Exception:
                learning_goals = [f"Understand the core ideas in {name.lower()}"]
                self_check     = [f"Can you summarise the key papers in this tier?"]

            milestones.append(Milestone(
                milestone_id=i + 1,
                name=name,
                description=completion_desc,
                tier=tier,
                paper_ids=paper_ids,
                week_start=week_cursor,
                week_end=week_end,
                learning_goals=learning_goals,
                self_check=self_check,
            ))

            week_cursor = week_end + 1

        return milestones

    def _build_weekly_plan(
        self,
        annotated: list[AnnotatedPaper],
        milestones: list[Milestone],
        weeks_available: int,
    ) -> list[WeeklyPlan]:
        """Distributes papers across weeks based on milestone boundaries."""
        # Build week → milestone mapping
        week_milestone: dict[int, Milestone] = {}
        for ms in milestones:
            for w in range(ms.week_start, ms.week_end + 1):
                week_milestone[w] = ms

        # Distribute papers within each milestone across its weeks
        ms_paper_map: dict[int, list[AnnotatedPaper]] = {}
        for ap in annotated:
            ms = next(
                (m for m in milestones if ap.arxiv_id in m.paper_ids), None
            )
            if ms:
                ms_paper_map.setdefault(ms.milestone_id, []).append(ap)

        weekly: list[WeeklyPlan] = []
        for week in range(1, weeks_available + 1):
            ms = week_milestone.get(week)
            if ms is None:
                continue

            ms_papers = ms_paper_map.get(ms.milestone_id, [])
            ms_weeks = ms.week_end - ms.week_start + 1
            papers_per_week = max(1, len(ms_papers) // ms_weeks)

            # Assign slice of papers to this week
            week_offset = week - ms.week_start
            start_idx = week_offset * papers_per_week
            end_idx = start_idx + papers_per_week
            week_papers = ms_papers[start_idx:end_idx]

            total_mins = sum(p.estimated_mins for p in week_papers)
            weekly.append(WeeklyPlan(
                week=week,
                milestone_name=ms.name,
                papers=[p.arxiv_id for p in week_papers],
                paper_titles=[p.title for p in week_papers],
                estimated_hours=round(total_mins / 60, 1),
                focus=ms.description,
            ))

        return weekly

    def _build_gap_coverage(
        self,
        annotated: list[AnnotatedPaper],
        gaps: list[ResearchGap],
        cluster_map: dict[int, Cluster],
    ) -> dict[str, list[str]]:
        """Maps each gap title to the arxiv IDs of roadmap papers that address it."""
        # Build cluster_id → paper IDs lookup from annotated papers
        cluster_papers: dict[str, list[str]] = {}
        for ap in annotated:
            cl = next(
                (c for c in cluster_map.values() if c.label == ap.cluster_label),
                None,
            )
            if cl:
                cluster_papers.setdefault(str(cl.cluster_id), []).append(ap.arxiv_id)

        coverage: dict[str, list[str]] = {}
        for gap in gaps:
            ids: list[str] = []
            for cid in gap.supporting_clusters:
                ids.extend(cluster_papers.get(str(cid), []))
            coverage[gap.title] = ids[:6]   # cap at 6 per gap

        return coverage

    def _build_hypothesis_links(
        self,
        annotated: list[AnnotatedPaper],
        hypotheses: list[ScoredHypothesis],
    ) -> dict[str, list[str]]:
        """Maps each hypothesis ID to related paper IDs in the roadmap."""
        roadmap_ids = {ap.arxiv_id for ap in annotated}
        return {
            h.hypothesis_id: [
                pid for pid in h.related_paper_ids if pid in roadmap_ids
            ]
            for h in hypotheses
        }

    async def _generate_summary(
        self,
        topic: str,
        annotated: list[AnnotatedPaper],
        milestones: list[Milestone],
        gaps: list[ResearchGap],
    ) -> tuple[str, str]:
        """
        Generates a 2-paragraph executive summary and field overview
        via a single GPT-4o call.
        """
        milestone_names = ", ".join(m.name for m in milestones)
        gap_titles = ", ".join(g.title for g in gaps[:3])
        cluster_labels = list({ap.cluster_label for ap in annotated})[:5]

        prompt = f"""You are writing the introduction to a PhD reading roadmap in: {topic}

The roadmap has {len(annotated)} papers organised into these milestones: {milestone_names}
Key research gaps covered: {gap_titles}
Major research clusters: {", ".join(cluster_labels)}

Write two short paragraphs:
1. "executive_summary" — 3–4 sentences: what this roadmap covers, why these papers were selected, and what the student will be able to do after completing it.
2. "field_overview" — 3–4 sentences: the current state of the field, major trends, and why these gaps matter.

Return a JSON object with keys "executive_summary" and "field_overview".
No markdown fences. No preamble. Valid JSON only."""

        try:
            resp = await self._llm.ainvoke(prompt)
            content = resp.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1].lstrip("json").strip()
            data = json.loads(content)
            return data.get("executive_summary", ""), data.get("field_overview", "")
        except Exception as exc:
            logger.error("Summary generation failed: %s", exc)
            return (
                f"This reading roadmap covers {len(annotated)} key papers in {topic}.",
                "The field is rapidly evolving with significant open research questions.",
            )


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN RENDERER
# ─────────────────────────────────────────────────────────────────────────────

def _render_markdown(roadmap: ReadingRoadmap) -> str:
    """
    Renders a ReadingRoadmap as a complete Markdown document.
    Called by ReadingRoadmap.to_markdown().
    """
    lines: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        "# 📖 Reading Roadmap",
        f"## {roadmap.topic}",
        "",
        f"*Generated by ResearchFlow AI · {roadmap.generated_at[:10]}*",
        "",
        "---",
        "",
    ]

    # ── Executive summary ─────────────────────────────────────────────────────
    lines += [
        "## Overview",
        "",
        roadmap.executive_summary,
        "",
        roadmap.field_overview,
        "",
        f"**Total papers:** {roadmap.total_papers}  ",
        f"**Estimated reading time:** {roadmap.total_hours_est}h  ",
        f"**Schedule:** {roadmap.weeks_available} weeks @ ~{roadmap.hours_per_week}h/week  ",
        "",
        "---",
        "",
    ]

    # ── Tier breakdown ────────────────────────────────────────────────────────
    lines += [
        "## Tier Overview",
        "",
        "| Tier | Label | Papers |",
        "|------|-------|--------|",
    ]
    for tier in ReadingTier:
        count = roadmap.tier_counts.get(tier.value, 0)
        if count:
            lines.append(
                f"| `{tier.value}` | {TIER_LABELS[tier]} | {count} |"
            )
    lines += ["", "---", ""]

    # ── Milestones ────────────────────────────────────────────────────────────
    lines += ["## Milestones", ""]

    for ms in roadmap.milestones:
        lines += [
            f"### Milestone {ms.milestone_id}: {ms.name}",
            f"*Weeks {ms.week_start}–{ms.week_end}* · "
            f"{TIER_LABELS[ms.tier]}",
            "",
            f"**Completion:** {ms.description}",
            "",
        ]

        if ms.learning_goals:
            lines.append("**Learning goals:**")
            for goal in ms.learning_goals:
                lines.append(f"- {goal}")
            lines.append("")

        if ms.self_check:
            lines.append("**Self-check questions:**")
            for q in ms.self_check:
                lines.append(f"- {q}")
            lines.append("")

    lines += ["---", ""]

    # ── Annotated paper list ──────────────────────────────────────────────────
    lines += ["## Annotated Paper List", ""]

    current_tier: ReadingTier | None = None
    for ap in roadmap.annotated_papers:
        if ap.tier != current_tier:
            current_tier = ap.tier
            lines += [
                f"### {TIER_LABELS[ap.tier]}",
                "",
                TIER_DESCRIPTIONS[ap.tier],
                "",
            ]

        difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(ap.difficulty, "🟡")
        priority_emoji   = {"high": "⬆️", "medium": "➡️", "low": "⬇️"}.get(ap.priority, "➡️")

        lines += [
            f"#### {ap.reading_order}. {ap.title}",
            f"*{', '.join(ap.authors[:2])}{'...' if len(ap.authors) > 2 else ''}* · "
            f"{ap.year} · `{ap.arxiv_id}` · "
            f"[arXiv]({ap.url})",
            "",
            f"**{ap.one_line_summary}**",
            "",
            f"- 🎯 **Why read:** {ap.why_read}",
            f"- 💡 **Key takeaway:** {ap.key_takeaway}",
            f"- 📌 **Reading note:** {ap.reading_note}",
            f"- {difficulty_emoji} Difficulty: `{ap.difficulty}` · "
            f"{priority_emoji} Priority: `{ap.priority}` · "
            f"⏱ ~{ap.estimated_mins} min",
            "",
        ]

        if ap.connects_to:
            lines.append(
                f"*Connects to:* {', '.join(f'`{pid}`' for pid in ap.connects_to)}"
            )
            lines.append("")

    lines += ["---", ""]

    # ── Weekly plan ───────────────────────────────────────────────────────────
    if roadmap.weekly_plan:
        lines += ["## Weekly Reading Plan", ""]
        for week in roadmap.weekly_plan:
            lines += [
                f"### Week {week.week}: {week.milestone_name}",
                f"*Estimated: {week.estimated_hours}h*",
                "",
                f"**Focus:** {week.focus}",
                "",
            ]
            for pid, title in zip(week.papers, week.paper_titles):
                lines.append(f"- `{pid}` — {title}")
            lines.append("")

    lines += ["---", ""]

    # ── Gap coverage ──────────────────────────────────────────────────────────
    if roadmap.gap_coverage:
        lines += ["## Gap Coverage", ""]
        for gap_title, paper_ids in roadmap.gap_coverage.items():
            lines += [
                f"**{gap_title}**",
                f"Covered by: {', '.join(f'`{pid}`' for pid in paper_ids) or '—'}",
                "",
            ]

    lines += [
        "---",
        "",
        "*Generated by ResearchFlow AI Roadmap Agent*",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ROADMAP AGENT
# ─────────────────────────────────────────────────────────────────────────────

class RoadmapAgent:
    """
    Orchestrates the full reading roadmap pipeline:
        TierClassifier → PaperSelector → AnnotationWriter
            → PathBuilder → ReadingRoadmap

    Designed to be submitted as a LOW-priority task to the SchedulerAgent
    after gap detection and hypothesis generation complete.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        temperature: float = 0.2,
        target_papers: int = DEFAULT_TARGET,
    ) -> None:
        self._llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._target = target_papers
        self._annotator = AnnotationWriter(self._llm)
        self._builder   = PathBuilder(self._llm)

        logger.info(
            "RoadmapAgent initialised | model=%s | target=%d papers",
            model_name, target_papers,
        )

    async def run(
        self,
        topic: str,
        papers: list[Paper],
        clusters: list[Cluster],
        gaps: list[ResearchGap],
        hypotheses: list[ScoredHypothesis],
        target_papers: int | None = None,
        weeks_available: int = DEFAULT_WEEKS,
        session_id: str = "",
    ) -> ReadingRoadmap:
        """
        Full roadmap generation pipeline.

        Returns a ReadingRoadmap with annotated papers, milestones,
        weekly plan, gap coverage, and executive summary.
        """
        target = target_papers or self._target
        session_id = session_id or f"roadmap_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        start_t = time.time()

        logger.info(
            "RoadmapAgent.run() | topic='%s' | %d papers | target=%d | %dw",
            topic, len(papers), target, weeks_available,
        )

        if not papers:
            raise ValueError("No papers provided to RoadmapAgent")

        # ── Stage 1: Classify all papers into tiers ──────────────────────────
        classifier = TierClassifier(clusters, gaps)
        tier_assignments = classifier.classify_all(papers)

        tier_counts = {t.value: 0 for t in ReadingTier}
        for tier in tier_assignments.values():
            tier_counts[tier.value] += 1
        logger.info("Tier distribution: %s", tier_counts)

        # ── Stage 2: Select papers within budget ─────────────────────────────
        selector = PaperSelector(clusters, gaps, target)
        selected_papers = selector.select(papers, tier_assignments)
        logger.info("Selected %d papers from %d total", len(selected_papers), len(papers))

        # Restrict tier_assignments to selected papers only
        selected_ids = {p.arxiv_id for p in selected_papers}
        tier_assignments = {
            k: v for k, v in tier_assignments.items() if k in selected_ids
        }

        # ── Stage 3: Annotate selected papers ────────────────────────────────
        cluster_map = {c.cluster_id: c for c in clusters}
        annotations = await self._annotator.annotate_batch(
            topic, selected_papers, tier_assignments, cluster_map
        )
        logger.info("Annotations generated for %d papers", len(annotations))

        # ── Stage 4: Build the complete roadmap ──────────────────────────────
        roadmap = await self._builder.build(
            topic=topic,
            papers=selected_papers,
            tier_assignments=tier_assignments,
            annotations=annotations,
            clusters=clusters,
            gaps=gaps,
            hypotheses=hypotheses,
            weeks_available=weeks_available,
            session_id=session_id,
        )

        elapsed = time.time() - start_t
        logger.info(
            "RoadmapAgent complete | %.1fs | %d papers | %d milestones | "
            "%d weekly entries | %.1fh total reading",
            elapsed,
            roadmap.total_papers,
            len(roadmap.milestones),
            len(roadmap.weekly_plan),
            roadmap.total_hours_est,
        )

        return roadmap


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

async def _demo() -> None:
    """
    Smoke-test with synthetic data.
    Run with: python -m agents.roadmap_agent
    """
    import random
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    clusters = [
        Cluster(
            cluster_id=0, label="Zero-Shot Radiology NLP",
            paper_ids=[f"p{i}" for i in range(6)], paper_count=6,
            year_range=(2019, 2022), dominant_category="cs.CV",
            reproducibility_score=0.20, temporal_trend="stagnating",
        ),
        Cluster(
            cluster_id=1, label="Clinical Vision-Language Models",
            paper_ids=[f"p{i}" for i in range(6, 15)], paper_count=9,
            year_range=(2021, 2024), dominant_category="cs.AI",
            reproducibility_score=0.45, temporal_trend="growing",
        ),
        Cluster(
            cluster_id=2, label="Low-Resource Medical NLP",
            paper_ids=[f"p{i}" for i in range(15, 20)], paper_count=5,
            year_range=(2018, 2021), dominant_category="cs.CL",
            reproducibility_score=0.10, temporal_trend="stagnating",
        ),
    ]

    papers = []
    for i in range(20):
        cid = 0 if i < 6 else (1 if i < 15 else 2)
        yr  = [2019, 2020, 2021, 2022, 2023, 2024][i % 6]
        papers.append(Paper(
            arxiv_id=f"24{i:02d}.{i*100:05d}",
            title=f"Sample Medical AI Paper {i}: Methods for Clinical NLP",
            abstract=(
                "We present a novel approach to clinical natural language "
                "processing using vision-language pre-training. "
                "Code available at https://github.com/sample/repo."
            ),
            authors=[f"Author {chr(65+i%5)}", f"Author {chr(70+i%5)}"],
            published=f"{yr}-06-15",
            year=yr,
            categories=["cs.CV", "cs.CL"],
            url=f"https://arxiv.org/abs/24{i:02d}.{i*100:05d}",
            cluster_id=cid,
        ))

    gaps = [
        ResearchGap(
            gap_id=0,
            title="Cross-modal alignment in low-resource clinical settings",
            description="Few papers address this specific intersection.",
            supporting_clusters=[0, 2],
            evidence={}, confidence=0.78,
            suggested_methodology="architecture_proposal",
        ),
    ]

    # Minimal hypothesis stub
    @dataclass
    class _StubMethodology:
        methodology_type: Any = None
        estimated_months: int = 6
        compute_requirements: str = "1x A100"
        suggested_datasets: list = field(default_factory=list)
        suggested_baselines: list = field(default_factory=list)
        evaluation_metrics: list = field(default_factory=list)
        tools_and_frameworks: list = field(default_factory=list)
        steps: list = field(default_factory=list)
        risks: list = field(default_factory=list)
        mitigations: list = field(default_factory=list)
        description: str = ""

    @dataclass
    class _StubHypothesis:
        hypothesis_id: str = "H01_0"
        rank: int = 1
        related_paper_ids: list = field(default_factory=list)
        methodology: Any = field(default_factory=_StubMethodology)

    hypotheses = [_StubHypothesis(related_paper_ids=[papers[0].arxiv_id, papers[5].arxiv_id])]

    # Run agent
    agent = RoadmapAgent(target_papers=15)
    roadmap = await agent.run(
        topic="Multimodal Large Language Models for Medical Diagnosis",
        papers=papers,
        clusters=clusters,
        gaps=gaps,
        hypotheses=hypotheses,
        target_papers=15,
        weeks_available=8,
        session_id="demo_001",
    )

    print("\n" + "=" * 70)
    print(f"  Roadmap: {roadmap.total_papers} papers | "
          f"{roadmap.total_hours_est}h | "
          f"{len(roadmap.milestones)} milestones | "
          f"{len(roadmap.weekly_plan)} weekly entries")
    print("=" * 70)

    print("\n── TIER DISTRIBUTION ──")
    for tier, count in roadmap.tier_counts.items():
        bar = "█" * count
        print(f"  {tier:<28} {bar} ({count})")

    print("\n── MILESTONES ──")
    for ms in roadmap.milestones:
        print(f"  [{ms.milestone_id}] {ms.name} "
              f"(weeks {ms.week_start}–{ms.week_end}, "
              f"{len(ms.paper_ids)} papers)")
        for goal in ms.learning_goals[:2]:
            print(f"      ✓ {goal}")

    print("\n── WEEKLY PLAN (first 4 weeks) ──")
    for week in roadmap.weekly_plan[:4]:
        print(f"  Week {week.week} | {week.milestone_name} | "
              f"{len(week.papers)} papers | {week.estimated_hours}h")

    print("\n── FIRST 3 ANNOTATED PAPERS ──")
    for ap in roadmap.annotated_papers[:3]:
        print(f"\n  [{ap.reading_order}] {ap.title[:60]}...")
        print(f"       Tier: {ap.tier.value}")
        print(f"       {ap.one_line_summary}")
        print(f"       Why read: {ap.why_read}")
        print(f"       Key takeaway: {ap.key_takeaway}")
        print(f"       Difficulty: {ap.difficulty} | ~{ap.estimated_mins}min")

    print("\n── EXECUTIVE SUMMARY ──")
    print(" ", roadmap.executive_summary)

    print("\n── MARKDOWN PREVIEW (first 40 lines) ──")
    md_lines = roadmap.to_markdown().split("\n")
    for line in md_lines[:40]:
        print(" ", line)
    print("  ...")

    print("\nRoadmapAgent demo complete.")


if __name__ == "__main__":
    asyncio.run(_demo())
