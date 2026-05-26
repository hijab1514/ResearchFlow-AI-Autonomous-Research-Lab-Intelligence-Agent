"""
pipelines/pipeline_base.py
==========================
ResearchFlow AI — Abstract Pipeline Base Class

Provides the shared contract, configuration, retry policy, and error
types that all pipelines (ResearchPipeline, LabPipeline) inherit from.

Keeping this separate enforces the open/closed principle:
  - New pipelines extend BasePipeline without touching existing ones
  - Config, retry, and error types stay in one place
  - Hooks (on_stage_start / on_stage_complete / on_stage_fail) allow
    middleware injection (e.g. W&B logging, Slack alerts) without
    modifying pipeline logic

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE STATUS
# ─────────────────────────────────────────────────────────────────────────────

class StageStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"
    CACHED    = "cached"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE RESULT (lightweight record per stage)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StageResult:
    """Lightweight record of a single stage execution."""
    stage_name:    str
    status:        StageStatus
    duration_s:    float           = 0.0
    attempt:       int             = 1
    error:         str             = ""
    output_size:   int             = 0   # e.g. len(papers), embedding dim
    cached:        bool            = False


# ─────────────────────────────────────────────────────────────────────────────
# RETRY POLICY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetryPolicy:
    """
    Exponential back-off retry configuration used by _run_stage().

    max_retries=2, backoff_s=2.0 means:
      attempt 1: immediate
      attempt 2: wait 2s
      attempt 3: wait 4s
      → total 3 attempts, then PipelineError raised
    """
    max_retries:    int   = 2
    backoff_s:      float = 2.0
    retryable_exceptions: tuple = (Exception,)   # retry on any exception by default


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """
    Central configuration for pipeline behaviour.
    All agent-level defaults are expressed here so the pipeline can be
    configured from a single object (loaded from configs/pipeline_config.yaml).
    """
    # LLM
    llm_model:               str   = "gpt-4o"
    temperature:             float = 0.0

    # arXiv retrieval
    max_papers:              int   = 80

    # Hypothesis generation
    hypotheses_per_gap:      int   = 2
    max_hypotheses:          int   = 5

    # Roadmap
    roadmap_target_papers:   int   = 20
    roadmap_weeks:           int   = 8

    # Execution
    stage_timeout_s:         float = 300.0    # max seconds to wait per stage
    use_stage_cache:         bool  = True
    verbose:                 bool  = True

    # Retry
    retry_policy:            RetryPolicy = field(default_factory=RetryPolicy)

    def __str__(self) -> str:
        return (
            f"PipelineConfig(model={self.llm_model}, papers={self.max_papers}, "
            f"hypotheses={self.max_hypotheses}, roadmap={self.roadmap_target_papers}p)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE ERROR
# ─────────────────────────────────────────────────────────────────────────────

class PipelineError(Exception):
    """
    Raised when a pipeline stage fails all retries.
    Carries the stage name so callers can report exactly where failure occurred.
    """
    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


# ─────────────────────────────────────────────────────────────────────────────
# BASE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class BasePipeline(ABC):
    """
    Abstract base class for all ResearchFlow AI pipelines.

    Subclasses must implement run() and optionally override the hook methods
    to inject logging, metrics, or alerting at each stage boundary.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._stage_results: list[StageResult] = []

        # Optional hook callbacks
        self._on_stage_start_hooks:    list[Callable] = []
        self._on_stage_complete_hooks: list[Callable] = []
        self._on_stage_fail_hooks:     list[Callable] = []

    # ── ABSTRACT INTERFACE ───────────────────────────────────────────────────

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Execute the full pipeline. Must be implemented by subclasses."""
        ...

    # ── HOOK REGISTRATION ────────────────────────────────────────────────────

    def on_stage_start(self, fn: Callable) -> None:
        """Register a callback fired when any stage starts. fn(stage_name: str)"""
        self._on_stage_start_hooks.append(fn)

    def on_stage_complete(self, fn: Callable) -> None:
        """Register a callback fired when any stage completes. fn(result: StageResult)"""
        self._on_stage_complete_hooks.append(fn)

    def on_stage_fail(self, fn: Callable) -> None:
        """Register a callback fired when any stage fails. fn(stage_name: str, error: str)"""
        self._on_stage_fail_hooks.append(fn)

    def _fire_start_hooks(self, stage_name: str) -> None:
        for fn in self._on_stage_start_hooks:
            try:
                fn(stage_name)
            except Exception as exc:
                logger.warning("on_stage_start hook error: %s", exc)

    def _fire_complete_hooks(self, result: StageResult) -> None:
        for fn in self._on_stage_complete_hooks:
            try:
                fn(result)
            except Exception as exc:
                logger.warning("on_stage_complete hook error: %s", exc)

    def _fire_fail_hooks(self, stage_name: str, error: str) -> None:
        for fn in self._on_stage_fail_hooks:
            try:
                fn(stage_name, error)
            except Exception as exc:
                logger.warning("on_stage_fail hook error: %s", exc)

    # ── STATUS ───────────────────────────────────────────────────────────────

    @property
    def stage_results(self) -> list[StageResult]:
        return list(self._stage_results)

    def stage_summary(self) -> dict[str, Any]:
        return {
            "total_stages":    len(self._stage_results),
            "completed":       sum(1 for s in self._stage_results if s.status == StageStatus.COMPLETED),
            "failed":          sum(1 for s in self._stage_results if s.status == StageStatus.FAILED),
            "cached":          sum(1 for s in self._stage_results if s.cached),
            "total_duration_s": sum(s.duration_s for s in self._stage_results),
        }
