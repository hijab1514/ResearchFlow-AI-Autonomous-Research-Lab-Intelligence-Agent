"""
pipelines/research_pipeline.py
===============================
ResearchFlow AI — End-to-End Research Intelligence Pipeline

This module is the top-level orchestrator that wires all five agents
together into a single, scheduler-dispatched, event-driven pipeline.

Flow:
  ┌──────────────────────────────────────────────────────────────────────┐
  │                   RESEARCH PIPELINE FLOW                            │
  │                                                                      │
  │  User Query                                                          │
  │      │                                                               │
  │      ▼                                                               │
  │  [STAGE 1]  QueryReformulation    (HIGH  · LLM · low CPU/RAM)       │
  │      │                                                               │
  │      ▼                                                               │
  │  [STAGE 2]  ArxivRetrieval        (HIGH  · API · low CPU/RAM)       │
  │      │                                                               │
  │      ▼                                                               │
  │  [STAGE 3]  EmbeddingGeneration   (MED   · GPU · HIGH CPU/RAM) ←─┐  │
  │      │                             scheduler resource-gated       │  │
  │      ▼                                                            │  │
  │  [STAGE 4]  Clustering + UMAP     (MED   · CPU · HIGH RAM)  ←───┘  │
  │      │                                                               │
  │      ▼                                                               │
  │  [STAGE 5]  ClusterLabeling       (MED   · LLM · med GPU)          │
  │      │                                                               │
  │      ▼                                                               │
  │  [STAGE 6]  GapDetection (ReAct)  (HIGH  · LLM · med GPU)          │
  │      │                                                               │
  │      ▼                                                               │
  │  [STAGE 7]  HypothesisGeneration  (HIGH  · LLM · med GPU)          │
  │      │                                                               │
  │      ▼                                                               │
  │  [STAGE 8]  RoadmapBuilding       (LOW   · LLM · low GPU)          │
  │      │                                                               │
  │      ▼                                                               │
  │  [STAGE 9]  ReportGeneration      (LOW   · LLM · none)             │
  │      │                                                               │
  │      ▼                                                               │
  │  PipelineResult  ──► Streamlit dashboard                            │
  └──────────────────────────────────────────────────────────────────────┘

Design principles:
  1. Every stage is a discrete, retryable async coroutine
  2. Every stage is submitted to the SchedulerAgent as a Task with the
     correct ResourceProfile — the scheduler decides when to run it
  3. The LabAgent checkpoints resource snapshots at every stage boundary
  4. Stage failures trigger configurable retry logic (exp. backoff)
  5. Progress events are emitted to an asyncio.Queue consumed by
     Streamlit's live progress bar
  6. The pipeline is fully resumable — completed stage results are cached
     so a crash mid-run only reruns from the failed stage

Key output: PipelineResult dataclass containing all agent outputs,
stage timings, resource snapshots, and the final report paths.

Usage (async):
    pipeline = ResearchPipeline(lab_agent=lab, scheduler=sched)
    async for event in pipeline.stream("Multimodal LLMs for medical diagnosis"):
        print(event)   # ProgressEvent stream for Streamlit

    # Or run fully and await result:
    result = await pipeline.run("Multimodal LLMs for medical diagnosis")

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator

from agents.lab_agent import LabAgent
from agents.research_agent import (
    Cluster,
    Paper,
    ResearchAgent,
    ResearchGap,
    ResearchReport,
)
from agents.hypothesis_agent import HypothesisAgent, ScoredHypothesis
from agents.roadmap_agent import RoadmapAgent, ReadingRoadmap
from agents.scheduler_agent import (
    Priority,
    ResourceProfile,
    SchedulerAgent,
    Task,
)
from pipelines.pipeline_base import (
    BasePipeline,
    PipelineConfig,
    PipelineError,
    RetryPolicy,
    StageResult,
    StageStatus,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

STAGE_CACHE_DIR = Path("cache/pipeline_stages")
STAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_OUT_DIR = Path("reports")
REPORTS_OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class PipelineStage(str, Enum):
    QUERY_REFORMULATION   = "query_reformulation"
    ARXIV_RETRIEVAL       = "arxiv_retrieval"
    EMBEDDING_GENERATION  = "embedding_generation"
    CLUSTERING            = "clustering"
    CLUSTER_LABELING      = "cluster_labeling"
    GAP_DETECTION         = "gap_detection"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    ROADMAP_BUILDING      = "roadmap_building"
    REPORT_GENERATION     = "report_generation"


class EventType(str, Enum):
    PIPELINE_STARTED    = "pipeline_started"
    STAGE_STARTED       = "stage_started"
    STAGE_COMPLETED     = "stage_completed"
    STAGE_FAILED        = "stage_failed"
    STAGE_RETRYING      = "stage_retrying"
    STAGE_DEFERRED      = "stage_deferred"
    RESOURCE_ALERT      = "resource_alert"
    PIPELINE_COMPLETED  = "pipeline_completed"
    PIPELINE_FAILED     = "pipeline_failed"


# ─────────────────────────────────────────────────────────────────────────────
# PROGRESS EVENTS (consumed by Streamlit st.empty() loop)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProgressEvent:
    """
    Emitted at every stage transition and resource alert.
    Streamlit's 01_research_agent.py page consumes these from the
    async generator to update the live progress UI.
    """
    event_type:    EventType
    stage:         PipelineStage | None
    message:       str
    progress_pct:  float           # 0.0–100.0 for the progress bar
    timestamp:     str = field(default_factory=lambda: datetime.utcnow().isoformat())
    detail:        dict = field(default_factory=dict)

    def __str__(self) -> str:
        ts = self.timestamp[11:19]
        stage_str = f"[{self.stage.value}] " if self.stage else ""
        return f"{ts} {stage_str}{self.message}"


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """
    Complete output of a ResearchPipeline run.
    All fields are populated on success; partial fields on failure.
    """
    run_id:             str
    query:              str
    session_id:         str
    status:             str               # "completed" | "failed" | "partial"
    started_at:         str
    completed_at:       str | None
    total_duration_s:   float

    # Stage outputs
    expanded_queries:   list[str]         = field(default_factory=list)
    papers:             list[Paper]       = field(default_factory=list)
    clusters:           list[Cluster]     = field(default_factory=list)
    gaps:               list[ResearchGap] = field(default_factory=list)
    hypotheses:         list[ScoredHypothesis] = field(default_factory=list)
    roadmap:            ReadingRoadmap | None = None
    report_md:          str               = ""
    report_path:        Path | None       = None

    # Timing breakdown
    stage_timings:      dict[str, float]  = field(default_factory=dict)

    # Resource snapshots keyed by stage name
    resource_snapshots: dict[str, dict]  = field(default_factory=dict)

    # Error info (if status != "completed")
    failed_stage:       str | None        = None
    error_message:      str               = ""

    # Scheduler stats
    scheduler_stats:    dict              = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE RESOURCE PROFILES
# ─────────────────────────────────────────────────────────────────────────────

STAGE_PROFILES: dict[PipelineStage, tuple[Priority, ResourceProfile]] = {
    PipelineStage.QUERY_REFORMULATION: (
        Priority.HIGH,
        ResourceProfile(cpu_cost="low", ram_cost="low", gpu_cost="low", estimated_duration_s=5),
    ),
    PipelineStage.ARXIV_RETRIEVAL: (
        Priority.HIGH,
        ResourceProfile(cpu_cost="low", ram_cost="low", gpu_cost="none", estimated_duration_s=20),
    ),
    PipelineStage.EMBEDDING_GENERATION: (
        Priority.MEDIUM,
        ResourceProfile(cpu_cost="high", ram_cost="medium", gpu_cost="medium", estimated_duration_s=60),
    ),
    PipelineStage.CLUSTERING: (
        Priority.MEDIUM,
        ResourceProfile(cpu_cost="high", ram_cost="medium", gpu_cost="none", estimated_duration_s=10),
    ),
    PipelineStage.CLUSTER_LABELING: (
        Priority.MEDIUM,
        ResourceProfile(cpu_cost="medium", ram_cost="low", gpu_cost="high", estimated_duration_s=30),
    ),
    PipelineStage.GAP_DETECTION: (
        Priority.HIGH,
        ResourceProfile(cpu_cost="medium", ram_cost="medium", gpu_cost="high", estimated_duration_s=45),
    ),
    PipelineStage.HYPOTHESIS_GENERATION: (
        Priority.HIGH,
        ResourceProfile(cpu_cost="low", ram_cost="low", gpu_cost="medium", estimated_duration_s=20),
    ),
    PipelineStage.ROADMAP_BUILDING: (
        Priority.LOW,
        ResourceProfile(cpu_cost="low", ram_cost="low", gpu_cost="low", estimated_duration_s=30),
    ),
    PipelineStage.REPORT_GENERATION: (
        Priority.LOW,
        ResourceProfile(cpu_cost="low", ram_cost="low", gpu_cost="none", estimated_duration_s=10),
    ),
}

# Progress percentages at each stage completion
STAGE_PROGRESS: dict[PipelineStage, float] = {
    PipelineStage.QUERY_REFORMULATION:   8.0,
    PipelineStage.ARXIV_RETRIEVAL:      20.0,
    PipelineStage.EMBEDDING_GENERATION: 38.0,
    PipelineStage.CLUSTERING:           50.0,
    PipelineStage.CLUSTER_LABELING:     62.0,
    PipelineStage.GAP_DETECTION:        76.0,
    PipelineStage.HYPOTHESIS_GENERATION:86.0,
    PipelineStage.ROADMAP_BUILDING:     94.0,
    PipelineStage.REPORT_GENERATION:   100.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# STAGE CACHE MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class StageCacheManager:
    """
    Persists completed stage outputs to disk so a mid-run crash only
    reruns from the failed stage rather than restarting from scratch.

    Cache keys: sha256(query + stage_name) → JSON file.
    Caches are invalidated manually or via TTL (default: 2h).
    """

    TTL_HOURS: float = 2.0

    def __init__(self, cache_dir: Path = STAGE_CACHE_DIR) -> None:
        self._dir = cache_dir

    def _key(self, query: str, stage: PipelineStage) -> str:
        raw = f"{query.lower().strip()}::{stage.value}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def save(self, query: str, stage: PipelineStage, data: Any) -> None:
        path = self._dir / f"{self._key(query, stage)}.json"
        try:
            path.write_text(json.dumps(data, default=str, indent=2))
            logger.debug("Stage cache saved: %s → %s", stage.value, path.name)
        except Exception as exc:
            logger.warning("Stage cache write failed: %s", exc)

    def load(self, query: str, stage: PipelineStage) -> Any | None:
        path = self._dir / f"{self._key(query, stage)}.json"
        if not path.exists():
            return None
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if age_h > self.TTL_HOURS:
            logger.debug("Stage cache expired for %s (%.1fh)", stage.value, age_h)
            return None
        try:
            logger.info("Stage cache HIT: %s (%.1fh old)", stage.value, age_h)
            return json.loads(path.read_text())
        except Exception:
            return None

    def invalidate(self, query: str, stage: PipelineStage | None = None) -> None:
        if stage:
            path = self._dir / f"{self._key(query, stage)}.json"
            path.unlink(missing_ok=True)
        else:
            for path in self._dir.glob("*.json"):
                path.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class ResearchPipeline(BasePipeline):
    """
    End-to-end research intelligence pipeline.

    Wires together:
      ResearchAgent · HypothesisAgent · RoadmapAgent
      SchedulerAgent · LabAgent · StageCacheManager

    Every stage is submitted as a prioritised Task to the SchedulerAgent.
    The SchedulerAgent's ResourcePolicy gates each stage against live
    CPU/RAM/GPU telemetry from the LabAgent — this is the OS scheduling
    integration made concrete at the pipeline level.

    Progress events are yielded from stream() for Streamlit consumption.
    """

    def __init__(
        self,
        lab_agent: LabAgent,
        scheduler: SchedulerAgent,
        config: PipelineConfig | None = None,
    ) -> None:
        super().__init__(config or PipelineConfig())

        self._lab       = lab_agent
        self._scheduler = scheduler
        self._cache     = StageCacheManager()

        # Agents (lazy-instantiated to avoid upfront API calls)
        self._research_agent:    ResearchAgent | None    = None
        self._hypothesis_agent:  HypothesisAgent | None  = None
        self._roadmap_agent:     RoadmapAgent | None     = None

        # Live progress event queue (Streamlit reads from this)
        self._event_queue: asyncio.Queue[ProgressEvent] = asyncio.Queue()

        logger.info("ResearchPipeline initialised | config=%s", self.config)

    # ── AGENT ACCESSORS (lazy) ───────────────────────────────────────────────

    @property
    def research_agent(self) -> ResearchAgent:
        if self._research_agent is None:
            self._research_agent = ResearchAgent(
                model_name=self.config.llm_model,
                max_papers=self.config.max_papers,
                verbose=self.config.verbose,
            )
        return self._research_agent

    @property
    def hypothesis_agent(self) -> HypothesisAgent:
        if self._hypothesis_agent is None:
            self._hypothesis_agent = HypothesisAgent(
                model_name=self.config.llm_model,
                hypotheses_per_gap=self.config.hypotheses_per_gap,
                max_output=self.config.max_hypotheses,
            )
        return self._hypothesis_agent

    @property
    def roadmap_agent(self) -> RoadmapAgent:
        if self._roadmap_agent is None:
            self._roadmap_agent = RoadmapAgent(
                model_name=self.config.llm_model,
                target_papers=self.config.roadmap_target_papers,
            )
        return self._roadmap_agent

    # ── MAIN ENTRY POINTS ────────────────────────────────────────────────────

    async def run(self, query: str) -> PipelineResult:
        """
        Run the full pipeline and return a PipelineResult.
        Blocks until all stages complete or a terminal failure occurs.
        """
        result = PipelineResult(
            run_id=self._make_run_id(query),
            query=query,
            session_id="",
            status="running",
            started_at=datetime.utcnow().isoformat(),
            completed_at=None,
            total_duration_s=0.0,
        )
        start_t = time.time()

        # Start LabAgent experiment tracking
        run_id = self._lab.begin_run(
            run_name=f"research_pipeline:{query[:40]}",
            params={
                "query": query,
                "llm_model": self.config.llm_model,
                "max_papers": self.config.max_papers,
            },
        )
        result.run_id = run_id

        await self._emit(EventType.PIPELINE_STARTED, None,
                         f"Pipeline started for: '{query}'", 0.0)

        try:
            await self._execute_all_stages(query, result)
            result.status = "completed"
            result.completed_at = datetime.utcnow().isoformat()
            result.total_duration_s = round(time.time() - start_t, 2)
            result.scheduler_stats = self._scheduler.status().get("stats", {})

            self._lab.end_run(notes=f"Completed in {result.total_duration_s}s")
            await self._emit(
                EventType.PIPELINE_COMPLETED, None,
                f"Pipeline complete in {result.total_duration_s:.1f}s | "
                f"{len(result.papers)} papers | {len(result.clusters)} clusters | "
                f"{len(result.gaps)} gaps | {len(result.hypotheses)} hypotheses",
                100.0,
            )

        except PipelineError as exc:
            result.status = "failed"
            result.failed_stage = exc.stage
            result.error_message = str(exc)
            result.total_duration_s = round(time.time() - start_t, 2)
            self._lab.fail_run(reason=str(exc))
            await self._emit(
                EventType.PIPELINE_FAILED, None,
                f"Pipeline failed at stage [{exc.stage}]: {exc}", 0.0,
                {"error": str(exc), "stage": exc.stage},
            )
            logger.error("Pipeline failed: %s", exc)

        except Exception as exc:
            result.status = "failed"
            result.error_message = f"Unexpected error: {exc}\n{traceback.format_exc()}"
            result.total_duration_s = round(time.time() - start_t, 2)
            self._lab.fail_run(reason=str(exc))
            logger.exception("Unexpected pipeline error")

        return result

    async def stream(self, query: str) -> AsyncIterator[ProgressEvent]:
        """
        Async generator that yields ProgressEvents as the pipeline runs.
        Use this in Streamlit with an async for loop to update the UI.

        Example (Streamlit):
            async for event in pipeline.stream(query):
                progress_bar.progress(event.progress_pct / 100)
                status_text.text(event.message)
        """
        # Launch pipeline in background task
        pipeline_task = asyncio.create_task(self.run(query))

        while not pipeline_task.done():
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=0.5
                )
                yield event
            except asyncio.TimeoutError:
                continue

        # Drain any remaining events
        while not self._event_queue.empty():
            yield self._event_queue.get_nowait()

        # Propagate exceptions from the pipeline task
        exc = pipeline_task.exception()
        if exc:
            raise exc

    # ── STAGE ORCHESTRATION ──────────────────────────────────────────────────

    async def _execute_all_stages(
        self, query: str, result: PipelineResult
    ) -> None:
        """
        Executes all 9 pipeline stages sequentially, submitting each to the
        SchedulerAgent and waiting for completion before advancing.

        Sequential execution is intentional — stages have strict data
        dependencies (can't cluster without embeddings). The SchedulerAgent
        still adds value by: gating each stage behind resource checks,
        logging every dispatch/deferral to the Gantt chart, and providing
        backpressure when the system is overloaded.
        """

        # Stage 1 — Query Reformulation
        expanded = await self._run_stage(
            stage=PipelineStage.QUERY_REFORMULATION,
            query=query,
            coro_fn=self.research_agent.reformulate_query,
            args=(query,),
        )
        result.expanded_queries = expanded or [query]

        # Stage 2 — arXiv Retrieval
        papers = await self._run_stage(
            stage=PipelineStage.ARXIV_RETRIEVAL,
            query=query,
            coro_fn=self.research_agent.retrieve_papers,
            args=(result.expanded_queries,),
        )
        if not papers:
            raise PipelineError(
                stage=PipelineStage.ARXIV_RETRIEVAL.value,
                message=f"No papers retrieved for query: '{query}'",
            )
        result.papers = papers
        self._lab.record_metric("papers_retrieved", float(len(papers)))
        self._lab.checkpoint()

        # Stage 3 — Embedding Generation (resource-gated by scheduler)
        embeddings = await self._run_stage(
            stage=PipelineStage.EMBEDDING_GENERATION,
            query=query,
            coro_fn=self.research_agent.embed_papers,
            args=(result.papers,),
        )
        self._lab.record_metric("embedding_dim", float(embeddings.shape[1]))
        self._lab.checkpoint()

        # Stage 4 — Clustering + UMAP (RAM-intensive, scheduler-gated)
        cluster_result = await self._run_stage(
            stage=PipelineStage.CLUSTERING,
            query=query,
            coro_fn=self.research_agent.cluster_papers,
            args=(result.papers, embeddings),
        )
        papers_clustered, umap_coords = cluster_result
        result.papers = papers_clustered
        n_clusters = len(set(p.cluster_id for p in papers_clustered if p.cluster_id >= 0))
        self._lab.record_metric("clusters_found", float(n_clusters))
        self._lab.checkpoint()

        # Stage 5 — Cluster Labeling (LLM + GPU)
        clusters = await self._run_stage(
            stage=PipelineStage.CLUSTER_LABELING,
            query=query,
            coro_fn=self.research_agent.label_clusters,
            args=(result.papers, embeddings),
        )
        result.clusters = clusters

        # Stage 6 — Gap Detection ReAct Agent (highest-value stage)
        gaps = await self._run_stage(
            stage=PipelineStage.GAP_DETECTION,
            query=query,
            coro_fn=self.research_agent.detect_gaps,
            args=(query, result.papers, result.clusters, embeddings),
        )
        result.gaps = gaps
        self._lab.record_metric("gaps_detected", float(len(gaps)))
        self._lab.checkpoint()

        # Stage 7 — Hypothesis Generation
        hypotheses = await self._run_stage(
            stage=PipelineStage.HYPOTHESIS_GENERATION,
            query=query,
            coro_fn=self.hypothesis_agent.run,
            args=(query, result.gaps, result.clusters, result.papers),
        )
        result.hypotheses = hypotheses or []
        self._lab.record_metric("hypotheses_generated", float(len(result.hypotheses)))

        # Stage 8 — Roadmap Building
        roadmap = await self._run_stage(
            stage=PipelineStage.ROADMAP_BUILDING,
            query=query,
            coro_fn=self.roadmap_agent.run,
            args=(query, result.papers, result.clusters,
                  result.gaps, result.hypotheses),
            kwargs={
                "target_papers": self.config.roadmap_target_papers,
                "weeks_available": self.config.roadmap_weeks,
                "session_id": result.run_id,
            },
        )
        result.roadmap = roadmap

        # Stage 9 — Report Generation
        report_md, report_path = await self._run_stage(
            stage=PipelineStage.REPORT_GENERATION,
            query=query,
            coro_fn=self._generate_report,
            args=(query, result),
        )
        result.report_md   = report_md
        result.report_path = report_path
        result.session_id  = result.run_id

    # ── STAGE RUNNER ─────────────────────────────────────────────────────────

    async def _run_stage(
        self,
        stage: PipelineStage,
        query: str,
        coro_fn,
        args: tuple = (),
        kwargs: dict | None = None,
        use_cache: bool = True,
    ) -> Any:
        """
        Submits a single pipeline stage to the SchedulerAgent as a Task.

        Handles:
          - Stage cache lookup (skip if already completed this session)
          - SchedulerAgent submission with correct Priority + ResourceProfile
          - Retry logic (exponential backoff, max_retries from config)
          - LabAgent resource checkpoint before + after
          - ProgressEvent emission for the Streamlit stream
          - Stage timing recording
        """
        priority, resource = STAGE_PROFILES[stage]

        await self._emit(
            EventType.STAGE_STARTED, stage,
            f"Starting: {stage.value.replace('_', ' ').title()}",
            STAGE_PROGRESS.get(stage, 0) - 5,
        )

        # Check stage cache (allows resuming after crash)
        if use_cache and self.config.use_stage_cache:
            cached = self._cache.load(query, stage)
            if cached is not None:
                await self._emit(
                    EventType.STAGE_COMPLETED, stage,
                    f"Loaded from cache: {stage.value}",
                    STAGE_PROGRESS[stage],
                    {"cached": True},
                )
                return cached

        # Capture resource state before dispatch
        snap_before = self._resource_dict()

        retry_policy = self.config.retry_policy
        last_exc: Exception | None = None

        for attempt in range(retry_policy.max_retries + 1):
            if attempt > 0:
                backoff = retry_policy.backoff_s * (2 ** (attempt - 1))
                await self._emit(
                    EventType.STAGE_RETRYING, stage,
                    f"Retrying {stage.value} (attempt {attempt + 1}/{retry_policy.max_retries + 1}) "
                    f"after {backoff:.1f}s backoff",
                    STAGE_PROGRESS.get(stage, 0) - 5,
                    {"attempt": attempt + 1, "backoff_s": backoff},
                )
                await asyncio.sleep(backoff)

            try:
                stage_start = time.time()

                # Build Task wrapper (runs coro in thread pool via scheduler)
                def _make_runner(fn, a, kw):
                    def _runner():
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            return loop.run_until_complete(fn(*a, **(kw or {})))
                        finally:
                            loop.close()
                    return _runner

                task = Task(
                    name=stage.value,
                    fn=_make_runner(coro_fn, args, kwargs),
                    priority=priority,
                    resource=resource,
                )
                future = self._scheduler.submit(task)

                # Wait for scheduler to dispatch and complete
                result = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: future.result(timeout=self.config.stage_timeout_s)
                )

                stage_time = round(time.time() - stage_start, 2)

                # Record timing + resource snapshot
                self._current_stage_timings[stage.value] = stage_time
                snap_after = self._resource_dict()

                self._lab.record_metric(f"{stage.value}_duration_s", stage_time)
                self._lab.checkpoint()

                # Save to stage cache
                if use_cache and self.config.use_stage_cache:
                    try:
                        self._cache.save(query, stage, _serialisable(result))
                    except Exception:
                        pass   # cache write failure is non-fatal

                await self._emit(
                    EventType.STAGE_COMPLETED, stage,
                    f"✓ {stage.value.replace('_', ' ').title()} "
                    f"completed in {stage_time:.1f}s",
                    STAGE_PROGRESS[stage],
                    {
                        "duration_s":    stage_time,
                        "attempt":       attempt + 1,
                        "resource_before": snap_before,
                        "resource_after":  snap_after,
                    },
                )

                logger.info(
                    "Stage COMPLETED: %s | %.2fs | attempt %d",
                    stage.value, stage_time, attempt + 1,
                )
                return result

            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Stage FAILED: %s | attempt %d | %s",
                    stage.value, attempt + 1, exc,
                )
                await self._emit(
                    EventType.STAGE_FAILED, stage,
                    f"✗ {stage.value} failed (attempt {attempt + 1}): {exc}",
                    STAGE_PROGRESS.get(stage, 0) - 5,
                    {"error": str(exc), "attempt": attempt + 1},
                )

                if attempt < retry_policy.max_retries:
                    continue   # retry loop handles backoff
                break

        # All retries exhausted
        raise PipelineError(
            stage=stage.value,
            message=f"Stage '{stage.value}' failed after {retry_policy.max_retries + 1} attempts: {last_exc}",
        )

    # ── REPORT GENERATION ────────────────────────────────────────────────────

    async def _generate_report(
        self, query: str, result: PipelineResult
    ) -> tuple[str, Path]:
        """
        Compiles all pipeline outputs into a structured Markdown report
        and persists it to reports/<run_id>.md

        This is the lightweight built-in report; the full report generator
        in reports/report_generator.py produces the PDF version.
        """
        lines: list[str] = [
            f"# ResearchFlow AI — Research Intelligence Report",
            f"",
            f"**Query:** {query}  ",
            f"**Run ID:** `{result.run_id}`  ",
            f"**Generated:** {datetime.utcnow().isoformat()[:19]}  ",
            f"",
            f"---",
            f"",
            f"## Executive Summary",
            f"",
            f"- **Papers retrieved:** {len(result.papers)}",
            f"- **Clusters identified:** {len(result.clusters)}",
            f"- **Research gaps:** {len(result.gaps)}",
            f"- **Thesis hypotheses:** {len(result.hypotheses)}",
            f"- **Roadmap papers:** {result.roadmap.total_papers if result.roadmap else 0}",
            f"- **Total pipeline time:** {result.total_duration_s:.1f}s",
            f"",
            f"---",
            f"",
        ]

        # Clusters
        lines += ["## Research Clusters", ""]
        for c in result.clusters:
            lines.append(
                f"- **[{c.cluster_id}] {c.label}** — "
                f"{c.paper_count} papers · {c.temporal_trend} · "
                f"repro={c.reproducibility_score:.2f}"
            )
        lines += ["", "---", ""]

        # Gaps
        lines += ["## Research Gaps", ""]
        for g in result.gaps:
            lines += [
                f"### Gap {g.gap_id}: {g.title}",
                f"*Confidence: {g.confidence:.2f} · Method: {g.suggested_methodology}*",
                f"",
                g.description,
                f"",
            ]
        lines += ["---", ""]

        # Hypotheses
        lines += ["## Thesis Hypotheses", ""]
        for h in result.hypotheses:
            lines += [
                f"### {h.rank}. {h.one_liner}",
                f"*Composite score: {h.composite_score:.3f} · "
                f"Novelty: {h.novelty_score:.3f} · "
                f"Feasibility: {h.feasibility_score:.3f}*",
                f"",
                f"**Q:** {h.research_question}",
                f"",
                f"**H₁:** {h.hypothesis_statement}",
                f"",
                f"**Method:** {h.methodology.methodology_type.value} · "
                f"~{h.methodology.estimated_months} months",
                f"",
            ]
        lines += ["---", ""]

        # Stage timings
        lines += ["## Pipeline Execution", ""]
        lines += ["| Stage | Duration |", "|-------|----------|"]
        for stage, t in self._current_stage_timings.items():
            lines.append(f"| `{stage}` | {t:.2f}s |")
        lines += ["", "---", ""]

        # Roadmap summary
        if result.roadmap:
            lines += [
                "## Reading Roadmap Summary",
                "",
                result.roadmap.executive_summary,
                "",
                f"**Total:** {result.roadmap.total_papers} papers · "
                f"{result.roadmap.total_hours_est}h · "
                f"{result.roadmap.weeks_available} weeks",
                "",
                "---",
                "",
            ]

        lines.append("*Generated by ResearchFlow AI*")
        report_md = "\n".join(lines)

        # Write to disk
        report_path = REPORTS_OUT_DIR / f"{result.run_id}_research_report.md"
        report_path.write_text(report_md)
        logger.info("Report written to %s", report_path)

        return report_md, report_path

    # ── HELPERS ──────────────────────────────────────────────────────────────

    async def _emit(
        self,
        event_type: EventType,
        stage: PipelineStage | None,
        message: str,
        progress_pct: float,
        detail: dict | None = None,
    ) -> None:
        event = ProgressEvent(
            event_type=event_type,
            stage=stage,
            message=message,
            progress_pct=max(0.0, min(100.0, progress_pct)),
            detail=detail or {},
        )
        await self._event_queue.put(event)
        logger.debug("Event emitted: %s", event)

    def _resource_dict(self) -> dict:
        snap = self._lab.snapshot()
        if snap is None:
            return {}
        return {
            "cpu_pct": snap.cpu.overall_pct,
            "ram_pct": snap.memory.used_pct,
            "gpu_pct": snap.gpu.vram_used_pct if snap.gpu.available else None,
            "bottleneck": snap.bottleneck.value,
        }

    @staticmethod
    def _make_run_id(query: str) -> str:
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        slug = hashlib.md5(query.encode()).hexdigest()[:6]
        return f"rp_{ts}_{slug}"

    @property
    def _current_stage_timings(self) -> dict[str, float]:
        """Per-run stage timing accumulator (reset on each run() call)."""
        if not hasattr(self, "_stage_timings"):
            self._stage_timings: dict[str, float] = {}
        return self._stage_timings


# ─────────────────────────────────────────────────────────────────────────────
# SERIALISATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _serialisable(obj: Any) -> Any:
    """
    Recursively convert an object to a JSON-serialisable form.
    Used when saving stage results to the cache.
    """
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        try:
            return asdict(obj)
        except Exception:
            return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_serialisable(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialisable(v) for k, v in obj.items()}
    if hasattr(obj, "tolist"):   # numpy array
        return obj.tolist()
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

async def _demo() -> None:
    """
    End-to-end pipeline smoke test.
    Run with: python -m pipelines.research_pipeline
    """
    import os
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    from agents.lab_agent import LabAgent
    from agents.scheduler_agent import SchedulerAgent
    from pipelines.pipeline_base import PipelineConfig

    lab = LabAgent()
    lab.start()

    sched = SchedulerAgent(lab_agent=lab, max_workers=2)
    sched.start()

    config = PipelineConfig(
        llm_model="gpt-4o",
        max_papers=30,
        hypotheses_per_gap=1,
        max_hypotheses=3,
        roadmap_target_papers=10,
        roadmap_weeks=4,
        use_stage_cache=True,
        verbose=False,
    )

    pipeline = ResearchPipeline(
        lab_agent=lab,
        scheduler=sched,
        config=config,
    )

    query = "Multimodal Large Language Models for Medical Diagnosis"
    print(f"\nRunning pipeline for: '{query}'\n")
    print("─" * 60)

    async for event in pipeline.stream(query):
        bar_filled = int(event.progress_pct / 5)
        bar = "█" * bar_filled + "░" * (20 - bar_filled)
        print(f"  [{bar}] {event.progress_pct:5.1f}%  {event.message}")

    print("─" * 60)
    print("\nDone. Check reports/ for the output.")

    sched.stop()
    lab.stop()


if __name__ == "__main__":
    asyncio.run(_demo())
