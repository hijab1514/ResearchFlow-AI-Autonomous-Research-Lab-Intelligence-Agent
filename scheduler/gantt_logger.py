"""
scheduler/gantt_logger.py
=========================
ResearchFlow AI — Execution Timeline Recorder & Gantt Chart Data Engine

This module records every scheduling event (dispatch, defer, complete,
fail, force-dispatch) as a structured timeline entry and exposes the
data in multiple formats for the Streamlit Gantt chart component.

The Gantt chart is one of the most important visualisations in the
project — it is the standard representation used in OS coursework to
analyse CPU burst scheduling algorithms (SJF, Round Robin, Priority
Scheduling). Having it rendered live from real data is what makes the
OS integration in ResearchFlow AI tangible and presentable.

What gets recorded per event:
  ┌─────────────────────────────────────────────────────────────────┐
  │  GanttEntry                                                     │
  │  ─────────                                                      │
  │  task_id / task_name / stage_group                              │
  │  priority (nominal) / effective_priority (post-aging)           │
  │  resource profile (cpu/ram/gpu cost)                            │
  │  event_type  (QUEUED→DEFERRED→DISPATCHED→COMPLETED/FAILED)      │
  │  timestamps  (queued_at, dispatch_at, complete_at)              │
  │  durations   (wait_time_s, burst_time_s, turnaround_s)          │
  │  defer_count / force_dispatched flag                            │
  │  resource_snapshot  (live CPU%/RAM%/GPU% at dispatch)           │
  │  scheduler_state   (RUNNING / THROTTLED at dispatch time)       │
  └─────────────────────────────────────────────────────────────────┘

Outputs:
  1. JSONL log file          → logs/gantt_timeline.jsonl
  2. In-memory event list    → fast reads for dashboard
  3. Plotly-ready dataframe  → GanttLogger.to_plotly_df()
  4. OS metrics summary      → GanttLogger.scheduling_metrics()

OS Scheduling Metrics computed:
  - Average Waiting Time     (W̄)  = mean time from queue to dispatch
  - Average Turnaround Time  (T̄)  = mean time from queue to completion
  - Average Burst Time       (B̄)  = mean execution time
  - CPU Utilisation          (U)   = % of wall time tasks were running
  - Throughput               (X)   = tasks completed per minute
  - Scheduling Overhead      (S)   = total deferral time / total time
  - Starvation Events        (SE)  = count of force-dispatched tasks

These are the exact metrics an OS professor expects to see in a
scheduling algorithm analysis.

Usage:
    logger = GanttLogger()

    # Record lifecycle events
    logger.on_queued(entry)                       # task enters queue
    logger.on_deferred(entry, reason, snap)       # task requeued
    logger.on_dispatched(entry, snap, sched_state) # task sent to worker
    logger.on_completed(entry, duration_s)        # task finished OK
    logger.on_failed(entry, error)                # task failed

    # Dashboard reads
    df   = logger.to_plotly_df()                  # Plotly timeline df
    rows = logger.get_entries(last_n=50)          # raw entry list
    met  = logger.scheduling_metrics()            # OS metrics dict

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LOG_PATH   = Path("logs/gantt_timeline.jsonl")
MAX_MEMORY_ENTRIES = 1000   # rolling in-memory buffer cap


# ─────────────────────────────────────────────────────────────────────────────
# EVENT TYPES
# ─────────────────────────────────────────────────────────────────────────────

class GanttEventType(str, Enum):
    """
    Lifecycle events recorded in the Gantt log.
    Maps to visual markers / bar colours in the Streamlit component.
    """
    QUEUED          = "queued"          # task entered priority queue
    DEFERRED        = "deferred"        # task requeued after resource check
    DISPATCHED      = "dispatched"      # task sent to worker thread
    COMPLETED       = "completed"       # task finished successfully
    FAILED          = "failed"          # task failed (all retries exhausted)
    FORCE_DISPATCHED = "force_dispatched" # dispatched despite resource limits
    CANCELLED       = "cancelled"       # task removed from queue
    THROTTLED       = "throttled"       # scheduler state changed to throttled
    UNTHROTTLED     = "unthrottled"     # scheduler state restored


# Colour mapping for Plotly Gantt bars
EVENT_COLOURS: dict[str, str] = {
    GanttEventType.QUEUED:           "#6B7280",   # grey
    GanttEventType.DEFERRED:         "#F59E0B",   # amber
    GanttEventType.DISPATCHED:       "#3B82F6",   # blue
    GanttEventType.COMPLETED:        "#10B981",   # green
    GanttEventType.FAILED:           "#EF4444",   # red
    GanttEventType.FORCE_DISPATCHED: "#8B5CF6",   # purple
    GanttEventType.CANCELLED:        "#9CA3AF",   # light grey
    GanttEventType.THROTTLED:        "#F97316",   # orange
    GanttEventType.UNTHROTTLED:      "#22C55E",   # light green
}

# Stage group colours (for Gantt bar fill by pipeline stage)
STAGE_COLOURS: dict[str, str] = {
    "system":       "#1F2937",
    "retrieval":    "#2563EB",
    "embedding":    "#7C3AED",
    "clustering":   "#DB2777",
    "labeling":     "#D97706",
    "gap_detection": "#DC2626",
    "hypothesis":   "#059669",
    "roadmap":      "#0891B2",
    "report":       "#6D28D9",
    "utility":      "#6B7280",
    "general":      "#374151",
}


# ─────────────────────────────────────────────────────────────────────────────
# GANTT ENTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GanttEntry:
    """
    A single timeline event record.

    Each task generates multiple GanttEntry records across its lifecycle:
      QUEUED → (0 or more DEFERRED) → DISPATCHED → COMPLETED | FAILED

    The Plotly Gantt chart renders the DISPATCHED→COMPLETED interval as
    the task bar. DEFERRED events are shown as markers on the timeline.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    entry_id:          str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    task_id:           str = ""
    task_name:         str = ""
    stage_group:       str = "general"

    # ── Scheduling metadata ───────────────────────────────────────────────────
    priority:          str = "MEDIUM"        # nominal priority name
    effective_priority: str = "MEDIUM"       # post-aging priority name
    priority_int:      int = 2               # numeric (for sorting)
    cpu_cost:          str = "medium"
    ram_cost:          str = "medium"
    gpu_cost:          str = "none"

    # ── Event type ────────────────────────────────────────────────────────────
    event_type:        str = GanttEventType.QUEUED

    # ── Timestamps (ISO strings) ──────────────────────────────────────────────
    queued_at:         str = ""
    dispatch_at:       str = ""
    complete_at:       str = ""
    event_at:          str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Durations (seconds) ───────────────────────────────────────────────────
    wait_time_s:       float = 0.0    # queued_at → dispatch_at
    burst_time_s:      float = 0.0    # dispatch_at → complete_at
    turnaround_s:      float = 0.0    # queued_at → complete_at

    # ── Scheduling behaviour ──────────────────────────────────────────────────
    defer_count:       int  = 0
    force_dispatched:  bool = False
    scheduler_state:   str  = "running"   # "running" | "throttled"

    # ── Resource snapshot at event time ──────────────────────────────────────
    cpu_pct:           float | None = None
    ram_pct:           float | None = None
    gpu_pct:           float | None = None
    bottleneck:        str          = "none"

    # ── Error info (for FAILED events) ───────────────────────────────────────
    error:             str = ""

    # ── Defer reason (for DEFERRED events) ───────────────────────────────────
    defer_reason:      str = ""

    # ── Plotly rendering helpers ──────────────────────────────────────────────
    @property
    def bar_colour(self) -> str:
        return EVENT_COLOURS.get(self.event_type, "#6B7280")

    @property
    def stage_colour(self) -> str:
        return STAGE_COLOURS.get(self.stage_group, "#374151")

    @property
    def display_label(self) -> str:
        """Short label for Gantt bar annotation."""
        extra = f" (×{self.defer_count} deferred)" if self.defer_count > 0 else ""
        forced = " ⚡FORCED" if self.force_dispatched else ""
        return f"{self.task_name}{extra}{forced}"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}

    def to_plotly_row(self) -> dict:
        """
        Row format expected by plotly.figure_factory.create_gantt()
        and the custom Streamlit Gantt component.
        """
        return {
            "Task":         self.display_label,
            "Start":        self.dispatch_at or self.queued_at,
            "Finish":       self.complete_at or self.dispatch_at,
            "Resource":     self.stage_group,
            "Description":  (
                f"Priority: {self.effective_priority} | "
                f"Wait: {self.wait_time_s:.2f}s | "
                f"Burst: {self.burst_time_s:.2f}s | "
                f"CPU: {self.cpu_pct or 0:.1f}% | "
                f"RAM: {self.ram_pct or 0:.1f}%"
            ),
            "Color":        self.stage_colour,
            "EventType":    self.event_type,
            "DeferCount":   self.defer_count,
            "ForceDispatched": self.force_dispatched,
            "WaitTime":     self.wait_time_s,
            "BurstTime":    self.burst_time_s,
            "TurnaroundTime": self.turnaround_s,
        }


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULING METRICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchedulingMetrics:
    """
    OS scheduling performance metrics computed from completed Gantt entries.

    These are the standard metrics used in OS scheduling analysis:

      Average Waiting Time (AWT):
        AWT = (1/n) Σ (dispatch_time - arrival_time)
        Lower is better. High AWT indicates queue congestion or
        resource contention causing excessive deferral.

      Average Turnaround Time (ATT):
        ATT = (1/n) Σ (completion_time - arrival_time)
        = AWT + Average Burst Time
        The total time from submission to result.

      Average Burst Time (ABT):
        ABT = (1/n) Σ burst_time
        The actual execution time, unaffected by scheduling decisions.

      CPU Utilisation (U):
        U = total_burst_time / total_wall_time
        Fraction of wall time spent doing useful work.
        Higher is better (100% = no idle time).

      Throughput (X):
        X = completed_tasks / total_wall_time_minutes
        Tasks completed per minute.

      Scheduling Overhead (SO):
        SO = total_deferral_time / total_time
        Fraction of time wasted on deferrals.
        Lower is better.

      Starvation Events (SE):
        Count of tasks force-dispatched due to excessive deferral.
        Ideally 0.
    """
    # Core metrics
    avg_waiting_time_s:     float = 0.0
    avg_turnaround_time_s:  float = 0.0
    avg_burst_time_s:       float = 0.0
    cpu_utilisation_pct:    float = 0.0
    throughput_per_min:     float = 0.0
    scheduling_overhead_pct: float = 0.0
    starvation_events:      int   = 0

    # Counts
    total_tasks:            int   = 0
    completed_tasks:        int   = 0
    failed_tasks:           int   = 0
    deferred_tasks:         int   = 0
    force_dispatched_tasks: int   = 0

    # Priority breakdown
    avg_wait_by_priority:   dict[str, float] = field(default_factory=dict)
    completed_by_priority:  dict[str, int]   = field(default_factory=dict)

    # Resource context
    avg_cpu_at_dispatch:    float = 0.0
    avg_ram_at_dispatch:    float = 0.0
    peak_cpu_at_dispatch:   float = 0.0
    peak_ram_at_dispatch:   float = 0.0

    # Timeline bounds
    first_event_at:         str = ""
    last_event_at:          str = ""
    wall_time_s:            float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        """Human-readable summary for CLI / log output."""
        return [
            f"{'─' * 50}",
            f"  OS SCHEDULING METRICS",
            f"{'─' * 50}",
            f"  Avg Waiting Time    (AWT) : {self.avg_waiting_time_s:.3f}s",
            f"  Avg Turnaround Time (ATT) : {self.avg_turnaround_time_s:.3f}s",
            f"  Avg Burst Time      (ABT) : {self.avg_burst_time_s:.3f}s",
            f"  CPU Utilisation     (U)   : {self.cpu_utilisation_pct:.1f}%",
            f"  Throughput          (X)   : {self.throughput_per_min:.2f} tasks/min",
            f"  Scheduling Overhead (SO)  : {self.scheduling_overhead_pct:.1f}%",
            f"  Starvation Events   (SE)  : {self.starvation_events}",
            f"{'─' * 50}",
            f"  Tasks: {self.completed_tasks} done / {self.failed_tasks} failed / "
            f"{self.force_dispatched_tasks} forced / {self.deferred_tasks} deferred",
            f"  Wall time: {self.wall_time_s:.1f}s",
            f"{'─' * 50}",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# GANTT LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class GanttLogger:
    """
    Records scheduling lifecycle events and exposes them as a structured
    timeline for Plotly Gantt chart rendering and OS metric analysis.

    Thread-safe — all public methods acquire self._lock.

    Internal storage:
      self._entries : list[GanttEntry]  (rolling, capped at MAX_MEMORY_ENTRIES)
      self._task_states : dict[task_id → partial GanttEntry]
                          Accumulates timestamps across a task's lifecycle
                          so we can compute wait_time_s, burst_time_s, and
                          turnaround_s correctly.

    The JSONL log is append-only — never rewritten — so it survives
    process restarts and accumulates a full session history.
    """

    def __init__(
        self,
        log_path: Path = DEFAULT_LOG_PATH,
        max_memory: int = MAX_MEMORY_ENTRIES,
    ) -> None:
        self._log_path   = log_path
        self._max_memory = max_memory
        self._entries:    list[GanttEntry]          = []
        self._task_states: dict[str, GanttEntry]    = {}   # partial accumulator per task
        self._lock        = threading.Lock()
        self._session_start: float = time.time()

        # Ensure log directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("GanttLogger initialised | log=%s", self._log_path)

    # ── LIFECYCLE RECORDERS ──────────────────────────────────────────────────

    def on_queued(
        self,
        task_id:      str,
        task_name:    str,
        priority:     str         = "MEDIUM",
        priority_int: int         = 2,
        cpu_cost:     str         = "medium",
        ram_cost:     str         = "medium",
        gpu_cost:     str         = "none",
        stage_group:  str         = "general",
    ) -> GanttEntry:
        """
        Record a task entering the priority queue.
        Initialises the partial accumulator for this task_id.
        """
        now = _iso_now()
        entry = GanttEntry(
            task_id=task_id,
            task_name=task_name,
            stage_group=stage_group,
            priority=priority,
            effective_priority=priority,
            priority_int=priority_int,
            cpu_cost=cpu_cost,
            ram_cost=ram_cost,
            gpu_cost=gpu_cost,
            event_type=GanttEventType.QUEUED,
            queued_at=now,
            event_at=now,
        )

        with self._lock:
            self._task_states[task_id] = entry
            self._append(entry)

        logger.debug("Gantt QUEUED: '%s' [%s]", task_name, task_id)
        return entry

    def on_deferred(
        self,
        task_id:      str,
        defer_count:  int,
        reason:       str         = "",
        cpu_pct:      float | None = None,
        ram_pct:      float | None = None,
        gpu_pct:      float | None = None,
        bottleneck:   str         = "none",
        scheduler_state: str      = "running",
    ) -> GanttEntry | None:
        """
        Record a task being requeued after a resource check failure.
        The defer_count tracks how many times this has happened.
        """
        with self._lock:
            base = self._task_states.get(task_id)
            if base is None:
                return None

            now = _iso_now()
            entry = GanttEntry(
                task_id=task_id,
                task_name=base.task_name,
                stage_group=base.stage_group,
                priority=base.priority,
                effective_priority=base.effective_priority,
                priority_int=base.priority_int,
                cpu_cost=base.cpu_cost,
                ram_cost=base.ram_cost,
                gpu_cost=base.gpu_cost,
                event_type=GanttEventType.DEFERRED,
                queued_at=base.queued_at,
                event_at=now,
                defer_count=defer_count,
                defer_reason=reason,
                scheduler_state=scheduler_state,
                cpu_pct=cpu_pct,
                ram_pct=ram_pct,
                gpu_pct=gpu_pct,
                bottleneck=bottleneck,
            )

            # Update the accumulator
            base.defer_count = defer_count
            self._append(entry)

        logger.debug(
            "Gantt DEFERRED: '%s' (×%d) | %s",
            base.task_name, defer_count, reason[:60],
        )
        return entry

    def on_dispatched(
        self,
        task_id:          str,
        cpu_pct:          float | None = None,
        ram_pct:          float | None = None,
        gpu_pct:          float | None = None,
        bottleneck:       str          = "none",
        scheduler_state:  str          = "running",
        force_dispatched: bool         = False,
        effective_priority: str        = "",
    ) -> GanttEntry | None:
        """
        Record a task being dispatched to a worker thread.
        Computes wait_time_s = dispatch_at - queued_at.
        """
        with self._lock:
            base = self._task_states.get(task_id)
            if base is None:
                return None

            now = _iso_now()
            wait_s = _seconds_between(base.queued_at, now)
            event_type = (
                GanttEventType.FORCE_DISPATCHED
                if force_dispatched
                else GanttEventType.DISPATCHED
            )

            base.dispatch_at       = now
            base.wait_time_s       = wait_s
            base.force_dispatched  = force_dispatched
            base.scheduler_state   = scheduler_state
            base.cpu_pct           = cpu_pct
            base.ram_pct           = ram_pct
            base.gpu_pct           = gpu_pct
            base.bottleneck        = bottleneck
            if effective_priority:
                base.effective_priority = effective_priority

            entry = GanttEntry(
                task_id=task_id,
                task_name=base.task_name,
                stage_group=base.stage_group,
                priority=base.priority,
                effective_priority=base.effective_priority,
                priority_int=base.priority_int,
                cpu_cost=base.cpu_cost,
                ram_cost=base.ram_cost,
                gpu_cost=base.gpu_cost,
                event_type=event_type,
                queued_at=base.queued_at,
                dispatch_at=now,
                event_at=now,
                wait_time_s=wait_s,
                defer_count=base.defer_count,
                force_dispatched=force_dispatched,
                scheduler_state=scheduler_state,
                cpu_pct=cpu_pct,
                ram_pct=ram_pct,
                gpu_pct=gpu_pct,
                bottleneck=bottleneck,
            )
            self._append(entry)

        logger.debug(
            "Gantt DISPATCHED: '%s' wait=%.2fs force=%s state=%s",
            base.task_name, wait_s, force_dispatched, scheduler_state,
        )
        return entry

    def on_completed(
        self,
        task_id:     str,
        burst_time_s: float = 0.0,
    ) -> GanttEntry | None:
        """
        Record a task completing successfully.
        Computes burst_time_s and turnaround_s.
        """
        with self._lock:
            base = self._task_states.get(task_id)
            if base is None:
                return None

            now = _iso_now()

            # Use provided burst_time_s if given; else compute from timestamps
            if burst_time_s <= 0.0 and base.dispatch_at:
                burst_time_s = _seconds_between(base.dispatch_at, now)

            turnaround_s = _seconds_between(base.queued_at, now)

            base.complete_at    = now
            base.burst_time_s   = burst_time_s
            base.turnaround_s   = turnaround_s

            entry = GanttEntry(
                task_id=task_id,
                task_name=base.task_name,
                stage_group=base.stage_group,
                priority=base.priority,
                effective_priority=base.effective_priority,
                priority_int=base.priority_int,
                cpu_cost=base.cpu_cost,
                ram_cost=base.ram_cost,
                gpu_cost=base.gpu_cost,
                event_type=GanttEventType.COMPLETED,
                queued_at=base.queued_at,
                dispatch_at=base.dispatch_at,
                complete_at=now,
                event_at=now,
                wait_time_s=base.wait_time_s,
                burst_time_s=burst_time_s,
                turnaround_s=turnaround_s,
                defer_count=base.defer_count,
                force_dispatched=base.force_dispatched,
                scheduler_state=base.scheduler_state,
                cpu_pct=base.cpu_pct,
                ram_pct=base.ram_pct,
                gpu_pct=base.gpu_pct,
                bottleneck=base.bottleneck,
            )
            self._append(entry)

            # Clean up accumulator
            del self._task_states[task_id]

        logger.debug(
            "Gantt COMPLETED: '%s' burst=%.2fs turnaround=%.2fs",
            base.task_name, burst_time_s, turnaround_s,
        )
        return entry

    def on_failed(
        self,
        task_id:  str,
        error:    str = "",
    ) -> GanttEntry | None:
        """Record a task failing after all retries are exhausted."""
        with self._lock:
            base = self._task_states.get(task_id)
            if base is None:
                return None

            now = _iso_now()
            burst_time_s = (
                _seconds_between(base.dispatch_at, now)
                if base.dispatch_at
                else 0.0
            )
            turnaround_s = _seconds_between(base.queued_at, now)

            entry = GanttEntry(
                task_id=task_id,
                task_name=base.task_name,
                stage_group=base.stage_group,
                priority=base.priority,
                effective_priority=base.effective_priority,
                priority_int=base.priority_int,
                cpu_cost=base.cpu_cost,
                ram_cost=base.ram_cost,
                gpu_cost=base.gpu_cost,
                event_type=GanttEventType.FAILED,
                queued_at=base.queued_at,
                dispatch_at=base.dispatch_at,
                complete_at=now,
                event_at=now,
                wait_time_s=base.wait_time_s,
                burst_time_s=burst_time_s,
                turnaround_s=turnaround_s,
                defer_count=base.defer_count,
                force_dispatched=base.force_dispatched,
                scheduler_state=base.scheduler_state,
                error=error[:200],
            )
            self._append(entry)
            del self._task_states[task_id]

        logger.warning(
            "Gantt FAILED: '%s' | %s",
            base.task_name, error[:80],
        )
        return entry

    def on_cancelled(self, task_id: str) -> GanttEntry | None:
        """Record a task being cancelled before dispatch."""
        with self._lock:
            base = self._task_states.pop(task_id, None)
            if base is None:
                return None

            entry = GanttEntry(
                task_id=task_id,
                task_name=base.task_name,
                stage_group=base.stage_group,
                priority=base.priority,
                event_type=GanttEventType.CANCELLED,
                queued_at=base.queued_at,
                event_at=_iso_now(),
                defer_count=base.defer_count,
            )
            self._append(entry)

        return entry

    def on_throttle_change(
        self,
        throttled: bool,
        cpu_pct: float = 0.0,
        reason: str = "",
    ) -> GanttEntry:
        """Record a scheduler throttle state transition."""
        event_type = GanttEventType.THROTTLED if throttled else GanttEventType.UNTHROTTLED
        entry = GanttEntry(
            task_id=f"sched_{uuid.uuid4().hex[:6]}",
            task_name="__scheduler__",
            stage_group="system",
            event_type=event_type,
            event_at=_iso_now(),
            cpu_pct=cpu_pct,
            defer_reason=reason,
        )
        with self._lock:
            self._append(entry)
        logger.info(
            "Gantt THROTTLE: %s | CPU=%.1f%% | %s",
            event_type, cpu_pct, reason[:60],
        )
        return entry

    # ── READ API ─────────────────────────────────────────────────────────────

    def get_entries(
        self,
        last_n:     int | None          = None,
        event_type: GanttEventType | None = None,
        stage_group: str | None          = None,
    ) -> list[GanttEntry]:
        """
        Return entries from the in-memory buffer.
        Optionally filter by event_type and/or stage_group.
        """
        with self._lock:
            entries = list(self._entries)

        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        if stage_group:
            entries = [e for e in entries if e.stage_group == stage_group]
        if last_n:
            entries = entries[-last_n:]

        return entries

    def get_completed_entries(self) -> list[GanttEntry]:
        """Return only COMPLETED entries — used for metric computation."""
        return self.get_entries(event_type=GanttEventType.COMPLETED)

    def get_entries_as_dicts(self, last_n: int | None = None) -> list[dict]:
        return [e.to_dict() for e in self.get_entries(last_n=last_n)]

    # ── PLOTLY DATAFRAME ────────────────────────────────────────────────────

    def to_plotly_df(self) -> list[dict]:
        """
        Returns a list of row dicts in Plotly Gantt format.

        Each completed task produces ONE bar entry:
          Start  = dispatch_at
          Finish = complete_at
          Task   = display_label
          Color  = stage_colour

        Deferred events are included as zero-width markers for annotation.

        This list is passed directly to the Streamlit gantt_chart.py
        component which renders it via plotly.figure_factory.create_gantt()
        or plotly.express.timeline().
        """
        rows: list[dict] = []

        with self._lock:
            entries = list(self._entries)

        for entry in entries:
            if entry.event_type in (
                GanttEventType.COMPLETED,
                GanttEventType.FAILED,
                GanttEventType.FORCE_DISPATCHED,
            ):
                if entry.dispatch_at and entry.complete_at:
                    rows.append(entry.to_plotly_row())

        # Also add deferred markers as zero-width entries
        for entry in entries:
            if entry.event_type == GanttEventType.DEFERRED:
                row = {
                    "Task":        f"↩ {entry.task_name} (deferred #{entry.defer_count})",
                    "Start":       entry.event_at,
                    "Finish":      entry.event_at,   # zero-width marker
                    "Resource":    "deferred",
                    "Description": f"Deferred: {entry.defer_reason[:80]}",
                    "Color":       EVENT_COLOURS[GanttEventType.DEFERRED],
                    "EventType":   GanttEventType.DEFERRED,
                    "DeferCount":  entry.defer_count,
                    "WaitTime":    0.0,
                    "BurstTime":   0.0,
                    "TurnaroundTime": 0.0,
                    "ForceDispatched": False,
                }
                rows.append(row)

        return rows

    # ── OS SCHEDULING METRICS ────────────────────────────────────────────────

    def scheduling_metrics(self) -> SchedulingMetrics:
        """
        Compute all OS scheduling metrics from completed entries.

        This is the core analytical output — maps directly to the
        metrics an OS professor expects in a scheduling algorithm report.
        """
        with self._lock:
            entries = list(self._entries)

        completed = [e for e in entries if e.event_type == GanttEventType.COMPLETED]
        failed    = [e for e in entries if e.event_type == GanttEventType.FAILED]
        deferred  = [e for e in entries if e.event_type == GanttEventType.DEFERRED]
        forced    = [e for e in entries if e.event_type == GanttEventType.FORCE_DISPATCHED]
        all_final = completed + failed

        if not all_final:
            return SchedulingMetrics()

        # ── Core time metrics ────────────────────────────────────────────────
        wait_times       = [e.wait_time_s    for e in all_final if e.wait_time_s > 0]
        burst_times      = [e.burst_time_s   for e in completed if e.burst_time_s > 0]
        turnaround_times = [e.turnaround_s   for e in all_final if e.turnaround_s > 0]

        awt = _mean(wait_times)
        abт = _mean(burst_times)
        att = _mean(turnaround_times)

        # ── Wall time ────────────────────────────────────────────────────────
        all_times = [e.event_at for e in entries if e.event_at]
        if len(all_times) >= 2:
            first = min(all_times)
            last  = max(all_times)
            wall_s = _seconds_between(first, last)
        else:
            first = last = ""
            wall_s = time.time() - self._session_start

        # ── CPU Utilisation: total burst / wall time ─────────────────────────
        total_burst = sum(burst_times)
        cpu_util = min(100.0, (total_burst / wall_s * 100)) if wall_s > 0 else 0.0

        # ── Throughput: completed per minute ────────────────────────────────
        throughput = (len(completed) / (wall_s / 60)) if wall_s > 0 else 0.0

        # ── Scheduling overhead: total wait / total time ─────────────────────
        total_wait = sum(wait_times)
        overhead   = min(100.0, (total_wait / (total_wait + total_burst) * 100)) \
                     if (total_wait + total_burst) > 0 else 0.0

        # ── Starvation events ────────────────────────────────────────────────
        starvation = sum(1 for e in entries if e.force_dispatched)

        # ── Per-priority breakdown ────────────────────────────────────────────
        wait_by_pri: dict[str, list[float]] = defaultdict(list)
        count_by_pri: dict[str, int] = defaultdict(int)
        for e in all_final:
            wait_by_pri[e.effective_priority].append(e.wait_time_s)
            if e.event_type == GanttEventType.COMPLETED:
                count_by_pri[e.effective_priority] += 1

        # ── Resource at dispatch ─────────────────────────────────────────────
        dispatched = [e for e in entries if e.event_type == GanttEventType.DISPATCHED]
        cpu_at_d   = [e.cpu_pct for e in dispatched if e.cpu_pct is not None]
        ram_at_d   = [e.ram_pct for e in dispatched if e.ram_pct is not None]

        return SchedulingMetrics(
            avg_waiting_time_s=round(awt, 3),
            avg_turnaround_time_s=round(att, 3),
            avg_burst_time_s=round(abт, 3),
            cpu_utilisation_pct=round(cpu_util, 2),
            throughput_per_min=round(throughput, 3),
            scheduling_overhead_pct=round(overhead, 2),
            starvation_events=starvation,
            total_tasks=len(set(e.task_id for e in entries if e.task_id)),
            completed_tasks=len(completed),
            failed_tasks=len(failed),
            deferred_tasks=len(deferred),
            force_dispatched_tasks=len(forced),
            avg_wait_by_priority={k: round(_mean(v), 3) for k, v in wait_by_pri.items()},
            completed_by_priority=dict(count_by_pri),
            avg_cpu_at_dispatch=round(_mean(cpu_at_d), 2),
            avg_ram_at_dispatch=round(_mean(ram_at_d), 2),
            peak_cpu_at_dispatch=round(max(cpu_at_d, default=0.0), 2),
            peak_ram_at_dispatch=round(max(ram_at_d, default=0.0), 2),
            first_event_at=first,
            last_event_at=last,
            wall_time_s=round(wall_s, 2),
        )

    def metrics_dict(self) -> dict:
        return self.scheduling_metrics().to_dict()

    # ── SESSION MANAGEMENT ───────────────────────────────────────────────────

    def clear(self) -> int:
        """
        Clear the in-memory buffer (does NOT clear the JSONL log).
        Returns the number of entries cleared.
        """
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._task_states.clear()
        self._session_start = time.time()
        logger.info("GanttLogger cleared: %d entries removed", count)
        return count

    def clear_log_file(self) -> None:
        """Truncate the JSONL log file (destructive — for testing only)."""
        try:
            self._log_path.write_text("")
            logger.info("Gantt log file cleared: %s", self._log_path)
        except Exception as exc:
            logger.warning("Could not clear log file: %s", exc)

    def load_from_log(self, last_n: int = 500) -> int:
        """
        Load the last N entries from the JSONL log file into memory.
        Useful for restoring Gantt state after a process restart.
        Returns the number of entries loaded.
        """
        if not self._log_path.exists():
            return 0

        loaded = 0
        try:
            lines = self._log_path.read_text().strip().split("\n")
            for line in lines[-last_n:]:
                if not line.strip():
                    continue
                data = json.loads(line)
                entry = GanttEntry(**{
                    k: v for k, v in data.items()
                    if k in GanttEntry.__dataclass_fields__
                })
                self._entries.append(entry)
                loaded += 1
        except Exception as exc:
            logger.warning("GanttLogger load_from_log error: %s", exc)

        logger.info("GanttLogger: loaded %d entries from log", loaded)
        return loaded

    # ── STATS PROPERTIES ────────────────────────────────────────────────────

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def active_task_count(self) -> int:
        """Number of tasks currently in progress (queued/running)."""
        with self._lock:
            return len(self._task_states)

    def __repr__(self) -> str:
        return (
            f"GanttLogger(entries={self.entry_count}, "
            f"active={self.active_task_count}, "
            f"log={self._log_path})"
        )

    # ── INTERNAL ────────────────────────────────────────────────────────────

    def _append(self, entry: GanttEntry) -> None:
        """
        Append entry to memory buffer and write to JSONL log.
        Enforces MAX_MEMORY_ENTRIES rolling cap.
        Must be called with self._lock held.
        """
        self._entries.append(entry)

        # Rolling cap: drop oldest entries when over limit
        if len(self._entries) > self._max_memory:
            self._entries = self._entries[-self._max_memory:]

        # Write to JSONL log (best-effort)
        try:
            with open(self._log_path, "a") as f:
                f.write(json.dumps(entry.to_dict(), default=str) + "\n")
        except Exception as exc:
            logger.warning("GanttLogger write error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seconds_between(iso_start: str, iso_end: str) -> float:
    """Parse two ISO timestamps and return elapsed seconds. Returns 0.0 on error."""
    try:
        t0 = datetime.fromisoformat(iso_start)
        t1 = datetime.fromisoformat(iso_end)
        return max(0.0, (t1 - t0).total_seconds())
    except Exception:
        return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Simulate a full pipeline run through the GanttLogger and print
    the resulting timeline, Plotly rows, and OS metrics.

    Run with: python -m scheduler.gantt_logger
    """
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    gantt = GanttLogger(log_path=Path("logs/demo_gantt.jsonl"))

    # Pipeline task definitions for demo
    tasks = [
        ("t01", "query_reformulation",   "HIGH",   2, "low",    "low",    "none",    "retrieval"),
        ("t02", "arxiv_retrieval",        "HIGH",   2, "low",    "low",    "none",    "retrieval"),
        ("t03", "abstract_embedding",    "MEDIUM", 3, "high",   "medium", "medium",  "embedding"),
        ("t04", "faiss_index_build",     "MEDIUM", 3, "medium", "medium", "medium",  "embedding"),
        ("t05", "hdbscan_clustering",    "MEDIUM", 3, "high",   "medium", "none",    "clustering"),
        ("t06", "umap_reduction",        "LOW",    4, "high",   "high",   "none",    "clustering"),
        ("t07", "cluster_labeling",      "MEDIUM", 3, "medium", "low",    "high",    "labeling"),
        ("t08", "gap_analysis_react",    "HIGH",   2, "medium", "medium", "high",    "gap_detection"),
        ("t09", "hypothesis_generation", "HIGH",   2, "low",    "low",    "medium",  "hypothesis"),
        ("t10", "roadmap_building",      "LOW",    4, "low",    "low",    "low",     "roadmap"),
        ("t11", "report_generation",     "LOW",    4, "low",    "low",    "none",    "report"),
    ]

    print("\n═══ SIMULATING PIPELINE ═══\n")

    for tid, name, pri, pri_int, cpu, ram, gpu, group in tasks:
        # Queue
        gantt.on_queued(tid, name, pri, pri_int, cpu, ram, gpu, group)
        time.sleep(0.02)

        # Maybe defer once or twice
        n_defers = random.randint(0, 2)
        for d in range(1, n_defers + 1):
            gantt.on_deferred(
                tid, d,
                reason=f"CPU {random.randint(70, 92)}% > threshold",
                cpu_pct=float(random.randint(70, 92)),
                ram_pct=float(random.randint(40, 65)),
            )
            time.sleep(0.05)

        # Dispatch
        force = n_defers >= 2
        gantt.on_dispatched(
            tid,
            cpu_pct=float(random.randint(30, 75)),
            ram_pct=float(random.randint(40, 65)),
            gpu_pct=float(random.randint(20, 60)) if gpu != "none" else None,
            bottleneck="none",
            scheduler_state="throttled" if random.random() < 0.3 else "running",
            force_dispatched=force,
        )

        # Execute
        burst = random.uniform(0.1, 0.8)
        time.sleep(burst)

        # Complete or fail
        if random.random() < 0.9:
            gantt.on_completed(tid, burst_time_s=burst)
        else:
            gantt.on_failed(tid, error="Simulated timeout")

    # Throttle event
    gantt.on_throttle_change(True, cpu_pct=88.5, reason="CPU sustained high")
    time.sleep(0.1)
    gantt.on_throttle_change(False, cpu_pct=52.0)

    # ── Print timeline ────────────────────────────────────────────────────────
    print(f"\n═══ GANTT TIMELINE ({gantt.entry_count} events) ═══")
    print(f"\n{'Task':<30} {'Event':<18} {'Wait':>8} {'Burst':>8} {'Defer':>6} {'CPU':>6}")
    print("─" * 80)
    for e in gantt.get_entries():
        if e.task_name == "__scheduler__":
            print(f"  ── {e.event_type.upper()} (CPU={e.cpu_pct or 0:.1f}%) ──")
            continue
        print(
            f"{e.task_name:<30} {e.event_type:<18} "
            f"{e.wait_time_s:>7.2f}s "
            f"{e.burst_time_s:>7.2f}s "
            f"{e.defer_count:>6} "
            f"{(e.cpu_pct or 0):>5.1f}%"
        )

    # ── Plotly rows ───────────────────────────────────────────────────────────
    plotly_rows = gantt.to_plotly_df()
    print(f"\n═══ PLOTLY-READY ROWS: {len(plotly_rows)} ═══")
    for row in plotly_rows[:5]:
        print(f"  {row['Task'][:40]:<40} | {row['Start'][:19]} → {row['Finish'][:19]}")
    if len(plotly_rows) > 5:
        print(f"  ... and {len(plotly_rows) - 5} more")

    # ── OS Metrics ────────────────────────────────────────────────────────────
    metrics = gantt.scheduling_metrics()
    print()
    for line in metrics.summary_lines():
        print(line)

    print(f"\n  Per-priority avg wait:")
    for pri, awt in metrics.avg_wait_by_priority.items():
        print(f"    {pri:<10} {awt:.3f}s")

    print(f"\n{gantt}")


if __name__ == "__main__":
    _demo()
