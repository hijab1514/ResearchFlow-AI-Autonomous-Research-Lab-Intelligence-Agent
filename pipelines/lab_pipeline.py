"""
pipelines/lab_pipeline.py
=========================
ResearchFlow AI — Lab Assistant Pipeline

The LabPipeline is the operational backbone of ResearchFlow AI's
OS-integration layer. Where ResearchPipeline is query-driven and
runs once per research session, the LabPipeline runs continuously
as a long-lived daemon that:

  1. MONITORS  — polls system telemetry every 2s via LabAgent
  2. SCHEDULES — gates ResearchPipeline stage dispatch based on
                 live resource state via SchedulerAgent
  3. LOGS      — writes structured JSONL logs for every event
  4. ALERTS    — routes alerts to Streamlit toast queue, log file,
                 and optional webhook

Architecture:
  ┌──────────────────────────────────────────────────────────────────────┐
  │                      LAB PIPELINE LOOP                              │
  │                                                                      │
  │   ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐  │
  │   │ SystemMonitor│─────▶│ AlertRouter  │─────▶│ Streamlit Queue  │  │
  │   │ (background) │      │              │─────▶│ Log File         │  │
  │   └──────┬───────┘      └──────────────┘      │ Webhook (opt.)   │  │
  │          │                                     └──────────────────┘  │
  │          ▼                                                           │
  │   ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐  │
  │   │ ResourceState│─────▶│ Scheduler    │─────▶│ Task Dispatch    │  │
  │   │ Evaluator    │      │ Agent        │      │ / Deferral       │  │
  │   └──────────────┘      └──────────────┘      └──────────────────┘  │
  │          │                                                           │
  │          ▼                                                           │
  │   ┌──────────────┐      ┌──────────────┐      ┌──────────────────┐  │
  │   │ Experiment   │─────▶│ Metric       │─────▶│ Run Reports      │  │
  │   │ Tracker      │      │ Logger       │      │ (MD + JSON)      │  │
  │   └──────────────┘      └──────────────┘      └──────────────────┘  │
  └──────────────────────────────────────────────────────────────────────┘

OS Concepts demonstrated end-to-end in this file:
  - Daemon process management   : background thread lifecycle
  - Resource-aware scheduling   : dispatch gating on live telemetry
  - Process accounting          : per-stage resource snapshots
  - Producer-consumer           : alert queue → dashboard consumer
  - Adaptive policy             : throttle ↔ unthrottle transitions
  - Structured logging          : JSONL event log, queryable offline

Key design:
  LabPipeline wraps LabAgent + SchedulerAgent + a persistent event loop
  that coordinates between them. It exposes:
    - start() / stop()           : lifecycle
    - attach_research_pipeline() : connect a ResearchPipeline for joint monitoring
    - status()                   : live dashboard snapshot
    - get_log_tail()             : last N log entries for Streamlit log viewer
    - generate_session_report()  : full Markdown report for a completed session

Usage:
    lab = LabPipeline()
    lab.start()

    # Attach the research pipeline so LabPipeline can observe its stages
    lab.attach_research_pipeline(research_pipeline)

    # Run research (LabPipeline monitors the whole thing)
    result = await research_pipeline.run(query)

    # Get the lab report for that session
    report = lab.generate_session_report(result.run_id)

    lab.stop()

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque

from agents.lab_agent import (
    Alert,
    AlertLevel,
    BottleneckType,
    ExperimentRun,
    LabAgent,
    SystemSnapshot,
)
from agents.scheduler_agent import SchedulerAgent, SchedulerState
from pipelines.pipeline_base import BasePipeline, PipelineConfig, PipelineError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

LAB_LOG_PATH   = Path("logs/lab_pipeline.jsonl")
REPORT_DIR     = Path("reports")
SESSION_DIR    = Path("experiment_tracker")

for _d in (LAB_LOG_PATH.parent, REPORT_DIR, SESSION_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# How often the lab pipeline supervision loop ticks (seconds)
SUPERVISION_TICK_S: float = float(os.getenv("LAB_SUPERVISION_TICK_S", "3.0"))

# Rolling in-memory log buffer for Streamlit log viewer
LOG_BUFFER_SIZE: int = 500


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class LabEventType(str, Enum):
    LAB_STARTED          = "lab_started"
    LAB_STOPPED          = "lab_stopped"
    RESOURCE_SNAPSHOT    = "resource_snapshot"
    ALERT_RECEIVED       = "alert_received"
    BOTTLENECK_DETECTED  = "bottleneck_detected"
    BOTTLENECK_CLEARED   = "bottleneck_cleared"
    SCHEDULER_THROTTLED  = "scheduler_throttled"
    SCHEDULER_UNTHROTTLED = "scheduler_unthrottled"
    EXPERIMENT_STARTED   = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"
    EXPERIMENT_FAILED    = "experiment_failed"
    STAGE_CHECKPOINT     = "stage_checkpoint"
    REPORT_GENERATED     = "report_generated"
    WEBHOOK_SENT         = "webhook_sent"
    WEBHOOK_FAILED       = "webhook_failed"


class AlertSeverity(str, Enum):
    """Normalised severity for routing decisions."""
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LabEvent:
    """
    Structured event record — written to the JSONL log and
    optionally pushed to Streamlit's live event feed.
    """
    event_id:    str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    event_type:  LabEventType = LabEventType.RESOURCE_SNAPSHOT
    timestamp:   str = field(default_factory=lambda: datetime.utcnow().isoformat())
    message:     str = ""
    severity:    AlertSeverity = AlertSeverity.LOW
    data:        dict = field(default_factory=dict)
    session_id:  str = ""

    def to_log_line(self) -> str:
        return json.dumps(asdict(self), default=str)


@dataclass
class ResourceTrend:
    """
    Tracks a resource metric over a short rolling window to detect
    sustained pressure (not just momentary spikes).

    Used by the AdaptiveThrottleController to make smoother throttle
    decisions than a single-sample threshold check.
    """
    metric_name:    str
    window_size:    int = 10
    _values:        Deque[float] = field(default_factory=lambda: deque(maxlen=10))

    def push(self, value: float) -> None:
        self._values.append(value)

    @property
    def mean(self) -> float:
        return sum(self._values) / len(self._values) if self._values else 0.0

    @property
    def max(self) -> float:
        return max(self._values) if self._values else 0.0

    @property
    def is_sustained_high(self) -> bool:
        """True if the last 5 samples are ALL above 80% (sustained pressure)."""
        if len(self._values) < 5:
            return False
        return all(v > 80.0 for v in list(self._values)[-5:])

    @property
    def is_sustained_low(self) -> bool:
        """True if the last 5 samples are ALL below 60% (safe to unthrottle)."""
        if len(self._values) < 5:
            return False
        return all(v < 60.0 for v in list(self._values)[-5:])


@dataclass
class SessionSummary:
    """
    Aggregated statistics for a completed research session — the
    basis of the lab report generated at session end.
    """
    session_id:         str
    query:              str
    started_at:         str
    ended_at:           str
    total_duration_s:   float

    # Resource utilisation across the session
    cpu_mean:           float
    cpu_peak:           float
    ram_mean:           float
    ram_peak:           float
    gpu_mean:           float
    gpu_peak:           float

    # Scheduling statistics
    total_tasks:        int
    completed_tasks:    int
    deferred_tasks:     int
    force_dispatched:   int
    throttle_events:    int
    avg_wait_s:         float
    avg_burst_s:        float

    # Alert summary
    total_alerts:       int
    critical_alerts:    int
    warning_alerts:     int
    bottleneck_types:   list[str]

    # Stage breakdown
    stage_timings:      dict[str, float]
    stage_resource_peaks: dict[str, dict]

    # Output counts
    papers_retrieved:   int
    clusters_found:     int
    gaps_detected:      int
    hypotheses_generated: int


# ─────────────────────────────────────────────────────────────────────────────
# ALERT ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class AlertRouter:
    """
    Routes alerts from LabAgent's alert queue to multiple sinks:
      1. In-memory deque  → Streamlit toast / alert feed
      2. JSONL log file   → persistent audit trail
      3. Webhook (opt.)   → external integrations (Slack, PagerDuty)

    Severity mapping:
      AlertLevel.INFO     → AlertSeverity.LOW
      AlertLevel.WARNING  → AlertSeverity.MEDIUM / HIGH (resource-dependent)
      AlertLevel.CRITICAL → AlertSeverity.CRITICAL
    """

    SEVERITY_MAP: dict[AlertLevel, AlertSeverity] = {
        AlertLevel.INFO:     AlertSeverity.LOW,
        AlertLevel.WARNING:  AlertSeverity.MEDIUM,
        AlertLevel.CRITICAL: AlertSeverity.CRITICAL,
    }

    def __init__(
        self,
        streamlit_queue: queue.Queue,
        log_path: Path = LAB_LOG_PATH,
        webhook_url: str | None = None,
    ) -> None:
        self._st_queue    = streamlit_queue
        self._log_path    = log_path
        self._webhook_url = webhook_url
        self._lock        = threading.Lock()

        # In-memory circular buffer for Streamlit log viewer
        self._buffer: Deque[LabEvent] = deque(maxlen=LOG_BUFFER_SIZE)

        # Alert counts for session summary
        self._counts: dict[AlertSeverity, int] = {s: 0 for s in AlertSeverity}

    def route(self, alert: Alert, session_id: str = "") -> None:
        """Route a single alert to all configured sinks."""
        severity = self._classify_severity(alert)
        self._counts[severity] += 1

        event = LabEvent(
            event_type=LabEventType.ALERT_RECEIVED,
            message=alert.message,
            severity=severity,
            data={
                "alert_id":   alert.alert_id,
                "level":      alert.level.value,
                "metric":     alert.metric,
                "value":      alert.value,
                "threshold":  alert.threshold,
                "bottleneck": alert.bottleneck.value,
            },
            session_id=session_id,
        )

        with self._lock:
            self._buffer.append(event)

        # Sink 1: Streamlit queue (non-blocking)
        try:
            self._st_queue.put_nowait(event)
        except queue.Full:
            pass

        # Sink 2: JSONL log
        self._write_log(event)

        # Sink 3: Webhook (fire-and-forget for high/critical)
        if self._webhook_url and severity in (AlertSeverity.HIGH, AlertSeverity.CRITICAL):
            threading.Thread(
                target=self._send_webhook,
                args=(event,),
                daemon=True,
            ).start()

        logger.debug("Alert routed [%s]: %s", severity.value, alert.message)

    def route_event(self, event: LabEvent) -> None:
        """Route a generic LabEvent (non-alert) to buffer and log."""
        with self._lock:
            self._buffer.append(event)
        self._write_log(event)

    def get_recent(self, n: int = 50) -> list[LabEvent]:
        """Return the last n events from the in-memory buffer."""
        with self._lock:
            return list(self._buffer)[-n:]

    @property
    def counts(self) -> dict[str, int]:
        return {k.value: v for k, v in self._counts.items()}

    def _classify_severity(self, alert: Alert) -> AlertSeverity:
        if alert.level == AlertLevel.CRITICAL:
            return AlertSeverity.CRITICAL
        if alert.level == AlertLevel.WARNING:
            # Escalate to HIGH if it involves OOM or GPU saturation
            if alert.bottleneck in (BottleneckType.OOM_RISK, BottleneckType.GPU_BOUND):
                return AlertSeverity.HIGH
            return AlertSeverity.MEDIUM
        return AlertSeverity.LOW

    def _write_log(self, event: LabEvent) -> None:
        try:
            with open(self._log_path, "a") as f:
                f.write(event.to_log_line() + "\n")
        except Exception as exc:
            logger.warning("AlertRouter log write failed: %s", exc)

    def _send_webhook(self, event: LabEvent) -> None:
        """Fire-and-forget webhook POST (best-effort)."""
        try:
            import urllib.request
            payload = json.dumps({
                "text":      f"[ResearchFlow] {event.severity.value.upper()}: {event.message}",
                "timestamp": event.timestamp,
                "data":      event.data,
            }).encode()
            req = urllib.request.Request(
                self._webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                status = resp.status
            self._write_log(LabEvent(
                event_type=LabEventType.WEBHOOK_SENT,
                message=f"Webhook sent (HTTP {status})",
                data={"url": self._webhook_url[:40]},
            ))
        except Exception as exc:
            self._write_log(LabEvent(
                event_type=LabEventType.WEBHOOK_FAILED,
                message=f"Webhook failed: {exc}",
                severity=AlertSeverity.LOW,
            ))


# ─────────────────────────────────────────────────────────────────────────────
# ADAPTIVE THROTTLE CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveThrottleController:
    """
    Manages the scheduler's throttle state based on sustained resource trends,
    not just instantaneous spikes.

    This implements a hysteresis-based controller — a classic feedback control
    pattern used in OS schedulers to avoid rapid oscillation between states:

      NORMAL → THROTTLED  : when CPU/RAM trends show sustained high load
      THROTTLED → NORMAL  : only when BOTH trends show sustained low load

    The hysteresis gap (high threshold > low threshold) prevents the system
    from rapidly bouncing between states, which would cause worse performance
    than staying in one state.

    Analogous to Linux's load balancer hysteresis (busy_factor, imbalance_pct).
    """

    def __init__(self, scheduler: SchedulerAgent) -> None:
        self._scheduler  = scheduler
        self._cpu_trend  = ResourceTrend("cpu_overall")
        self._ram_trend  = ResourceTrend("ram_used_pct")
        self._gpu_trend  = ResourceTrend("gpu_vram_pct")
        self._is_throttled = False
        self._throttle_events: list[dict] = []

    def update(self, snapshot: SystemSnapshot) -> str | None:
        """
        Feed the latest snapshot. Returns a human-readable action string
        if a throttle state change occurred, else None.
        """
        self._cpu_trend.push(snapshot.cpu.overall_pct)
        self._ram_trend.push(snapshot.memory.used_pct)
        if snapshot.gpu.available:
            self._gpu_trend.push(snapshot.gpu.vram_used_pct)

        action: str | None = None

        if not self._is_throttled:
            # Check if we should throttle
            should_throttle = (
                self._cpu_trend.is_sustained_high
                or self._ram_trend.is_sustained_high
                or snapshot.bottleneck == BottleneckType.OOM_RISK
            )
            if should_throttle:
                self._is_throttled = True
                ts = datetime.utcnow().isoformat()
                self._throttle_events.append({
                    "action": "throttle",
                    "timestamp": ts,
                    "cpu_mean": self._cpu_trend.mean,
                    "ram_mean": self._ram_trend.mean,
                    "reason": snapshot.bottleneck_detail,
                })
                action = (
                    f"THROTTLED: CPU {self._cpu_trend.mean:.1f}% sustained | "
                    f"RAM {self._ram_trend.mean:.1f}% | "
                    f"{snapshot.bottleneck_detail}"
                )
                logger.warning("AdaptiveThrottle: %s", action)

        else:
            # Check if we can unthrottle (both CPU AND RAM must be sustainably low)
            can_unthrottle = (
                self._cpu_trend.is_sustained_low
                and self._ram_trend.is_sustained_low
                and snapshot.bottleneck == BottleneckType.NONE
            )
            if can_unthrottle:
                self._is_throttled = False
                ts = datetime.utcnow().isoformat()
                self._throttle_events.append({
                    "action": "unthrottle",
                    "timestamp": ts,
                    "cpu_mean": self._cpu_trend.mean,
                    "ram_mean": self._ram_trend.mean,
                })
                action = (
                    f"UNTHROTTLED: CPU {self._cpu_trend.mean:.1f}% | "
                    f"RAM {self._ram_trend.mean:.1f}% — resources recovered"
                )
                logger.info("AdaptiveThrottle: %s", action)

        return action

    @property
    def is_throttled(self) -> bool:
        return self._is_throttled

    @property
    def throttle_event_count(self) -> int:
        return len([e for e in self._throttle_events if e["action"] == "throttle"])


# ─────────────────────────────────────────────────────────────────────────────
# SESSION TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class SessionTracker:
    """
    Builds a SessionSummary from the resource snapshots, alert counts,
    and scheduler stats accumulated during a pipeline run.

    Called by LabPipeline.generate_session_report() after a run completes.
    """

    def build_summary(
        self,
        session_id: str,
        query: str,
        started_at: str,
        ended_at: str,
        snapshots: list[dict],
        alert_counts: dict[str, int],
        bottleneck_types: list[str],
        scheduler_stats: dict,
        stage_timings: dict[str, float],
        stage_peaks: dict[str, dict],
        output_counts: dict[str, int],
    ) -> SessionSummary:

        def _safe_mean(vals: list[float]) -> float:
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        def _safe_peak(vals: list[float]) -> float:
            return round(max(vals), 2) if vals else 0.0

        # Extract time-series from snapshots
        cpu_series  = [s.get("cpu", {}).get("overall_pct", 0.0) for s in snapshots]
        ram_series  = [s.get("memory", {}).get("used_pct", 0.0) for s in snapshots]
        gpu_series  = [
            s.get("gpu", {}).get("vram_used_pct", 0.0)
            for s in snapshots
            if s.get("gpu", {}).get("available", False)
        ]

        started_dt = datetime.fromisoformat(started_at)
        ended_dt   = datetime.fromisoformat(ended_at)
        duration   = round((ended_dt - started_dt).total_seconds(), 2)

        return SessionSummary(
            session_id=session_id,
            query=query,
            started_at=started_at,
            ended_at=ended_at,
            total_duration_s=duration,
            cpu_mean=_safe_mean(cpu_series),
            cpu_peak=_safe_peak(cpu_series),
            ram_mean=_safe_mean(ram_series),
            ram_peak=_safe_peak(ram_series),
            gpu_mean=_safe_mean(gpu_series),
            gpu_peak=_safe_peak(gpu_series),
            total_tasks=scheduler_stats.get("total_submitted", 0),
            completed_tasks=scheduler_stats.get("total_completed", 0),
            deferred_tasks=scheduler_stats.get("total_deferred", 0),
            force_dispatched=scheduler_stats.get("total_force_dispatched", 0),
            throttle_events=scheduler_stats.get("total_throttle_events", 0),
            avg_wait_s=scheduler_stats.get("avg_wait_time_s", 0.0),
            avg_burst_s=scheduler_stats.get("avg_burst_time_s", 0.0),
            total_alerts=sum(alert_counts.values()),
            critical_alerts=alert_counts.get("critical", 0),
            warning_alerts=alert_counts.get("medium", 0) + alert_counts.get("high", 0),
            bottleneck_types=list(set(bottleneck_types)),
            stage_timings=stage_timings,
            stage_resource_peaks=stage_peaks,
            papers_retrieved=output_counts.get("papers", 0),
            clusters_found=output_counts.get("clusters", 0),
            gaps_detected=output_counts.get("gaps", 0),
            hypotheses_generated=output_counts.get("hypotheses", 0),
        )


# ─────────────────────────────────────────────────────────────────────────────
# LAB PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class LabPipeline(BasePipeline):
    """
    Long-lived lab assistant pipeline that monitors, schedules, logs,
    and alerts across the lifetime of a ResearchFlow AI session.

    Lifecycle:
      start()  → launches SystemMonitor + supervision loop
        │
        ├── attach_research_pipeline() → observes stage events
        │
        ├── supervision loop (every 3s):
        │     drain alerts → route → update throttle controller
        │     → emit resource snapshot event → log
        │
        └── stop() → graceful shutdown, final session report

    The supervision loop is the 'OS kernel tick' of ResearchFlow AI:
    it runs independently of any research job, continuously maintaining
    the health of the system and the scheduler.
    """

    def __init__(
        self,
        lab_agent: LabAgent | None = None,
        scheduler: SchedulerAgent | None = None,
        config: PipelineConfig | None = None,
        streamlit_queue: queue.Queue | None = None,
        webhook_url: str | None = None,
    ) -> None:
        super().__init__(config or PipelineConfig())

        # Core agents (can be injected or created here)
        self._lab        = lab_agent or LabAgent()
        self._scheduler  = scheduler  # may be None if used standalone

        # Streamlit event feed (dashboard reads from this queue)
        self._st_queue: queue.Queue = streamlit_queue or queue.Queue(maxsize=500)

        # Sub-components
        self._router     = AlertRouter(self._st_queue, LAB_LOG_PATH, webhook_url)
        self._throttle   = AdaptiveThrottleController(self._scheduler) if scheduler else None
        self._tracker    = SessionTracker()

        # Supervision thread
        self._stop_event   = threading.Event()
        self._sup_thread: threading.Thread | None = None

        # Session state
        self._session_id:    str = ""
        self._session_start: str = ""
        self._session_snapshots: list[dict] = []
        self._session_bottlenecks: list[str] = []
        self._session_stage_timings: dict[str, float] = {}
        self._session_stage_peaks: dict[str, dict] = {}
        self._session_output_counts: dict[str, int] = {}
        self._is_session_active: bool = False

        # Bottleneck state tracking (for DETECTED / CLEARED events)
        self._last_bottleneck: BottleneckType = BottleneckType.NONE

        logger.info("LabPipeline initialised | webhook=%s", bool(webhook_url))

    # ── LIFECYCLE ────────────────────────────────────────────────────────────

    async def run(self, *args, **kwargs) -> Any:
        """
        BasePipeline compliance. LabPipeline is started via start(),
        not run() — this method is a no-op for direct callers.
        """
        self.start()

    def start(self) -> None:
        """
        Boot the LabPipeline:
          1. Start the LabAgent's SystemMonitor background thread
          2. Launch the supervision loop daemon thread
        """
        self._lab.start()
        self._stop_event.clear()

        self._sup_thread = threading.Thread(
            target=self._supervision_loop,
            name="researchflow-lab-pipeline",
            daemon=True,
        )
        self._sup_thread.start()

        self._emit_event(LabEvent(
            event_type=LabEventType.LAB_STARTED,
            message="LabPipeline started — monitor + supervision loop active",
            severity=AlertSeverity.LOW,
        ))
        logger.info("LabPipeline started")

    def stop(self) -> None:
        """
        Graceful shutdown:
          1. Signal supervision loop to exit
          2. Stop LabAgent (joins SystemMonitor thread)
          3. Emit LAB_STOPPED event
        """
        logger.info("LabPipeline stopping...")
        self._stop_event.set()

        if self._sup_thread and self._sup_thread.is_alive():
            self._sup_thread.join(timeout=SUPERVISION_TICK_S * 3)

        self._lab.stop()

        self._emit_event(LabEvent(
            event_type=LabEventType.LAB_STOPPED,
            message="LabPipeline stopped",
            severity=AlertSeverity.LOW,
        ))
        logger.info("LabPipeline stopped")

    @property
    def is_running(self) -> bool:
        return self._sup_thread is not None and self._sup_thread.is_alive()

    # ── RESEARCH PIPELINE ATTACHMENT ─────────────────────────────────────────

    def attach_research_pipeline(self, pipeline: Any) -> None:
        """
        Register hooks on a ResearchPipeline so LabPipeline can observe
        every stage start, completion, and failure without tight coupling.

        Hooks are lightweight callbacks — they record resource peaks per
        stage and update the session summary in real time.
        """
        pipeline.on_stage_start(self._on_stage_start)
        pipeline.on_stage_complete(self._on_stage_complete)
        pipeline.on_stage_fail(self._on_stage_fail)
        logger.info("LabPipeline attached to ResearchPipeline")

    def _on_stage_start(self, stage_name: str) -> None:
        """Hook: fired when a ResearchPipeline stage starts."""
        snap = self._lab.snapshot()
        self._emit_event(LabEvent(
            event_type=LabEventType.STAGE_CHECKPOINT,
            message=f"Stage started: {stage_name}",
            severity=AlertSeverity.LOW,
            data={
                "stage": stage_name,
                "phase": "start",
                "resource": self._snap_dict(snap),
            },
            session_id=self._session_id,
        ))

    def _on_stage_complete(self, stage_result: Any) -> None:
        """Hook: fired when a ResearchPipeline stage completes."""
        stage_name = stage_result.stage_name
        snap = self._lab.snapshot()
        resource_peak = self._snap_dict(snap)

        # Record stage timing and peak resource for session report
        self._session_stage_timings[stage_name] = stage_result.duration_s
        self._session_stage_peaks[stage_name]   = resource_peak

        self._emit_event(LabEvent(
            event_type=LabEventType.STAGE_CHECKPOINT,
            message=f"Stage completed: {stage_name} in {stage_result.duration_s:.2f}s",
            severity=AlertSeverity.LOW,
            data={
                "stage":        stage_name,
                "phase":        "complete",
                "duration_s":   stage_result.duration_s,
                "attempt":      stage_result.attempt,
                "resource":     resource_peak,
            },
            session_id=self._session_id,
        ))

    def _on_stage_fail(self, stage_name: str, error: str) -> None:
        """Hook: fired when a ResearchPipeline stage fails."""
        self._emit_event(LabEvent(
            event_type=LabEventType.STAGE_CHECKPOINT,
            message=f"Stage FAILED: {stage_name} — {error[:120]}",
            severity=AlertSeverity.HIGH,
            data={"stage": stage_name, "phase": "fail", "error": error},
            session_id=self._session_id,
        ))

    # ── SESSION MANAGEMENT ────────────────────────────────────────────────────

    def begin_session(
        self,
        session_id: str,
        query: str,
        params: dict | None = None,
    ) -> None:
        """
        Call this when a new research pipeline run begins.
        Resets session-level accumulators and starts LabAgent run tracking.
        """
        self._session_id          = session_id
        self._session_start       = datetime.utcnow().isoformat()
        self._session_snapshots   = []
        self._session_bottlenecks = []
        self._session_stage_timings = {}
        self._session_stage_peaks   = {}
        self._session_output_counts = {}
        self._is_session_active     = True

        self._lab.begin_run(
            run_name=f"session:{query[:40]}",
            params={"session_id": session_id, "query": query, **(params or {})},
        )

        self._emit_event(LabEvent(
            event_type=LabEventType.EXPERIMENT_STARTED,
            message=f"Session started: '{query[:50]}'",
            severity=AlertSeverity.LOW,
            data={"session_id": session_id, "query": query},
            session_id=session_id,
        ))
        logger.info("LabPipeline session started: %s", session_id)

    def end_session(
        self,
        output_counts: dict[str, int] | None = None,
        notes: str = "",
    ) -> str:
        """
        Call when a research pipeline run completes.
        Finalises LabAgent run, emits EXPERIMENT_COMPLETED, returns session_id.
        """
        if not self._is_session_active:
            return self._session_id

        self._session_output_counts = output_counts or {}
        self._is_session_active = False

        completed_run = self._lab.end_run(notes=notes)

        self._emit_event(LabEvent(
            event_type=LabEventType.EXPERIMENT_COMPLETED,
            message=f"Session completed: {self._session_id}",
            severity=AlertSeverity.LOW,
            data={
                "session_id": self._session_id,
                "output_counts": self._session_output_counts,
                "notes": notes,
            },
            session_id=self._session_id,
        ))

        logger.info("LabPipeline session ended: %s", self._session_id)
        return self._session_id

    def fail_session(self, reason: str = "") -> None:
        """Call when a research pipeline run fails."""
        self._is_session_active = False
        self._lab.fail_run(reason=reason)
        self._emit_event(LabEvent(
            event_type=LabEventType.EXPERIMENT_FAILED,
            message=f"Session FAILED: {reason[:120]}",
            severity=AlertSeverity.HIGH,
            data={"session_id": self._session_id, "reason": reason},
            session_id=self._session_id,
        ))

    # ── SUPERVISION LOOP (background thread) ─────────────────────────────────

    def _supervision_loop(self) -> None:
        """
        Background daemon that runs every SUPERVISION_TICK_S seconds:

          1. Drain LabAgent's alert queue → route alerts
          2. Pull latest SystemSnapshot → update throttle controller
          3. Detect bottleneck state transitions (DETECTED / CLEARED)
          4. Record session snapshot if a session is active
          5. Sleep until next tick

        This is the 'heartbeat' of the entire system — analogous to the
        kernel's periodic timer interrupt that drives process accounting,
        load balancing, and I/O scheduling decisions.
        """
        logger.debug("LabPipeline supervision loop started")

        while not self._stop_event.is_set():
            tick_start = time.time()

            try:
                self._supervision_tick()
            except Exception as exc:
                logger.error("Supervision tick error: %s", exc, exc_info=True)

            # Precise sleep for remainder of tick interval
            elapsed = time.time() - tick_start
            sleep_s = max(0.0, SUPERVISION_TICK_S - elapsed)
            self._stop_event.wait(timeout=sleep_s)

        logger.debug("LabPipeline supervision loop exited")

    def _supervision_tick(self) -> None:
        """Execute one supervision tick."""

        # ── 1: Drain and route all pending alerts ───────────────────────────
        alerts = self._lab.drain_alerts(max_items=100)
        for alert in alerts:
            self._router.route(alert, self._session_id)
            if self._is_session_active:
                self._lab._tracker.record_alert(alert)

        # ── 2: Get latest snapshot ──────────────────────────────────────────
        snap = self._lab.snapshot()
        if snap is None:
            return

        # ── 3: Update adaptive throttle controller ──────────────────────────
        if self._throttle:
            action = self._throttle.update(snap)
            if action:
                event_type = (
                    LabEventType.SCHEDULER_THROTTLED
                    if self._throttle.is_throttled
                    else LabEventType.SCHEDULER_UNTHROTTLED
                )
                self._emit_event(LabEvent(
                    event_type=event_type,
                    message=action,
                    severity=AlertSeverity.MEDIUM if self._throttle.is_throttled else AlertSeverity.LOW,
                    data=self._snap_dict(snap),
                    session_id=self._session_id,
                ))

        # ── 4: Detect bottleneck state transitions ──────────────────────────
        current_bt = snap.bottleneck
        if current_bt != self._last_bottleneck:
            if current_bt != BottleneckType.NONE:
                # New bottleneck detected
                self._emit_event(LabEvent(
                    event_type=LabEventType.BOTTLENECK_DETECTED,
                    message=f"Bottleneck detected: {current_bt.value} — {snap.bottleneck_detail}",
                    severity=AlertSeverity.HIGH if current_bt == BottleneckType.OOM_RISK else AlertSeverity.MEDIUM,
                    data={
                        "bottleneck": current_bt.value,
                        "detail":     snap.bottleneck_detail,
                        "resource":   self._snap_dict(snap),
                    },
                    session_id=self._session_id,
                ))
                if self._is_session_active:
                    self._session_bottlenecks.append(current_bt.value)
            else:
                # Bottleneck cleared
                self._emit_event(LabEvent(
                    event_type=LabEventType.BOTTLENECK_CLEARED,
                    message=f"Bottleneck cleared (was: {self._last_bottleneck.value})",
                    severity=AlertSeverity.LOW,
                    data={"previous_bottleneck": self._last_bottleneck.value},
                    session_id=self._session_id,
                ))
            self._last_bottleneck = current_bt

        # ── 5: Record session snapshot ──────────────────────────────────────
        if self._is_session_active:
            snap_dict = self._snap_dict(snap)
            self._session_snapshots.append(snap_dict)
            self._lab.checkpoint()

    # ── REPORT GENERATION ────────────────────────────────────────────────────

    def generate_session_report(
        self,
        session_id: str | None = None,
        scheduler_stats: dict | None = None,
    ) -> str:
        """
        Generates a structured Markdown lab report for the completed session.
        Combines session summary, resource utilisation tables, scheduling
        analysis (Gantt-style), and alert log.

        Returns the report as a Markdown string. Also writes to reports/.
        """
        sid = session_id or self._session_id

        summary = self._tracker.build_summary(
            session_id=sid,
            query="Research session",
            started_at=self._session_start or datetime.utcnow().isoformat(),
            ended_at=datetime.utcnow().isoformat(),
            snapshots=self._session_snapshots,
            alert_counts=self._router.counts,
            bottleneck_types=self._session_bottlenecks,
            scheduler_stats=scheduler_stats or (
                self._scheduler.status().get("stats", {})
                if self._scheduler else {}
            ),
            stage_timings=self._session_stage_timings,
            stage_peaks=self._session_stage_peaks,
            output_counts=self._session_output_counts,
        )

        report = self._render_report(summary)

        # Persist
        path = REPORT_DIR / f"{sid}_lab_report.md"
        try:
            path.write_text(report)
            self._emit_event(LabEvent(
                event_type=LabEventType.REPORT_GENERATED,
                message=f"Lab report written to {path}",
                data={"path": str(path), "session_id": sid},
            ))
            logger.info("Lab report written to %s", path)
        except Exception as exc:
            logger.warning("Lab report write failed: %s", exc)

        return report

    def _render_report(self, s: SessionSummary) -> str:
        """Renders a SessionSummary as a Markdown document."""

        lines: list[str] = [
            "# ResearchFlow AI — Lab Report",
            "",
            f"**Session ID:** `{s.session_id}`  ",
            f"**Query:** {s.query}  ",
            f"**Started:** {s.started_at[:19]}  ",
            f"**Ended:** {s.ended_at[:19]}  ",
            f"**Duration:** {s.total_duration_s:.1f}s  ",
            "",
            "---",
            "",
            "## Resource Utilisation",
            "",
            "| Metric    | Mean    | Peak    |",
            "|-----------|---------|---------|",
            f"| CPU       | {s.cpu_mean:.1f}%  | {s.cpu_peak:.1f}%  |",
            f"| RAM       | {s.ram_mean:.1f}%  | {s.ram_peak:.1f}%  |",
            f"| GPU VRAM  | {s.gpu_mean:.1f}%  | {s.gpu_peak:.1f}%  |",
            "",
            "---",
            "",
            "## Scheduling Analysis",
            "",
            "| Metric                  | Value |",
            "|-------------------------|-------|",
            f"| Total tasks submitted   | {s.total_tasks} |",
            f"| Tasks completed         | {s.completed_tasks} |",
            f"| Tasks deferred          | {s.deferred_tasks} |",
            f"| Force-dispatched        | {s.force_dispatched} |",
            f"| Throttle events         | {s.throttle_events} |",
            f"| Avg wait time           | {s.avg_wait_s:.2f}s |",
            f"| Avg burst time          | {s.avg_burst_s:.2f}s |",
            "",
        ]

        # Stage timing table
        if s.stage_timings:
            lines += [
                "## Stage Execution Timeline",
                "",
                "| Stage | Duration | CPU Peak | RAM Peak |",
                "|-------|----------|----------|----------|",
            ]
            for stage_name, dur in s.stage_timings.items():
                peaks = s.stage_resource_peaks.get(stage_name, {})
                cpu_p = peaks.get("cpu_pct", "—")
                ram_p = peaks.get("ram_pct", "—")
                cpu_str = f"{cpu_p:.1f}%" if isinstance(cpu_p, float) else str(cpu_p)
                ram_str = f"{ram_p:.1f}%" if isinstance(ram_p, float) else str(ram_p)
                lines.append(f"| `{stage_name}` | {dur:.2f}s | {cpu_str} | {ram_str} |")
            lines += [""]

        lines += [
            "---",
            "",
            "## Alerts Summary",
            "",
            f"**Total alerts fired:** {s.total_alerts}  ",
            f"**Critical:** {s.critical_alerts}  ",
            f"**Warning:** {s.warning_alerts}  ",
            "",
        ]

        if s.bottleneck_types:
            lines += [
                "**Bottleneck types encountered:**",
                "",
            ]
            for bt in s.bottleneck_types:
                lines.append(f"- `{bt}`")
            lines.append("")

        lines += [
            "---",
            "",
            "## Pipeline Outputs",
            "",
            f"- Papers retrieved:     {s.papers_retrieved}",
            f"- Clusters identified:  {s.clusters_found}",
            f"- Research gaps:        {s.gaps_detected}",
            f"- Thesis hypotheses:    {s.hypotheses_generated}",
            "",
            "---",
            "",
        ]

        # Recent log entries
        recent = self._router.get_recent(20)
        if recent:
            lines += [
                "## Recent Event Log (last 20)",
                "",
                "| Time | Type | Severity | Message |",
                "|------|------|----------|---------|",
            ]
            for ev in recent:
                ts  = ev.timestamp[11:19] if hasattr(ev, "timestamp") else "—"
                et  = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                sev = ev.severity.value if hasattr(ev.severity, "value") else str(ev.severity)
                msg = ev.message[:60]
                lines.append(f"| {ts} | `{et}` | {sev} | {msg} |")
            lines.append("")

        lines += ["---", "", "*Generated by ResearchFlow AI Lab Pipeline*"]
        return "\n".join(lines)

    # ── STATUS & DASHBOARD API ────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Returns a complete status snapshot for the Streamlit Lab Assistant
        dashboard page (02_lab_assistant.py).
        """
        lab_summary    = self._lab.status_summary()
        sched_status   = self._scheduler.status() if self._scheduler else {}
        gantt_events   = (
            self._scheduler.gantt_events(last_n=50)
            if self._scheduler else []
        )
        recent_events  = self._router.get_recent(20)
        alert_counts   = self._router.counts

        return {
            "lab":              lab_summary,
            "scheduler":        sched_status,
            "gantt_events":     gantt_events,
            "recent_events":    [asdict(e) for e in recent_events],
            "alert_counts":     alert_counts,
            "is_throttled":     self._throttle.is_throttled if self._throttle else False,
            "throttle_events":  self._throttle.throttle_event_count if self._throttle else 0,
            "session_active":   self._is_session_active,
            "session_id":       self._session_id,
            "supervision_running": self.is_running,
            "session_snapshots_count": len(self._session_snapshots),
            "bottleneck_history": self._session_bottlenecks[-10:],
        }

    def get_log_tail(self, n: int = 100) -> list[dict]:
        """
        Returns the last n log entries for the Streamlit log viewer.
        Falls back to reading from the JSONL file if the in-memory
        buffer has been cleared.
        """
        recent = self._router.get_recent(n)
        if recent:
            return [asdict(e) for e in recent]

        # Read from JSONL file
        entries: list[dict] = []
        try:
            lines = LAB_LOG_PATH.read_text().strip().split("\n")
            for line in lines[-n:]:
                if line.strip():
                    entries.append(json.loads(line))
        except Exception:
            pass
        return entries

    def drain_streamlit_events(self, max_items: int = 50) -> list[LabEvent]:
        """
        Drains the Streamlit event queue. Called on every page refresh
        cycle by the 02_lab_assistant.py Streamlit page.
        """
        events: list[LabEvent] = []
        for _ in range(max_items):
            try:
                events.append(self._st_queue.get_nowait())
            except queue.Empty:
                break
        return events

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _emit_event(self, event: LabEvent) -> None:
        """Write event to buffer + log."""
        self._router.route_event(event)

    @staticmethod
    def _snap_dict(snap: SystemSnapshot | None) -> dict:
        if snap is None:
            return {}
        return {
            "cpu_pct":    snap.cpu.overall_pct,
            "ram_pct":    snap.memory.used_pct,
            "ram_avail":  snap.memory.available_gb,
            "gpu_pct":    snap.gpu.vram_used_pct if snap.gpu.available else None,
            "bottleneck": snap.bottleneck.value,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Smoke-test: starts the LabPipeline, simulates a 10-second
    session with stage events, then generates and prints the lab report.

    Run with: python -m pipelines.lab_pipeline
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    from agents.lab_agent import LabAgent
    from agents.scheduler_agent import SchedulerAgent

    lab_agent = LabAgent()
    scheduler = SchedulerAgent(lab_agent=lab_agent, max_workers=2)
    scheduler.start()

    lab_pipeline = LabPipeline(
        lab_agent=lab_agent,
        scheduler=scheduler,
    )
    lab_pipeline.start()

    print("\nLabPipeline warming up (3s)...")
    time.sleep(3)

    # Simulate a session
    session_id = "demo_session_001"
    lab_pipeline.begin_session(
        session_id=session_id,
        query="Multimodal LLMs for Medical Diagnosis",
        params={"max_papers": 50},
    )

    # Simulate stage checkpoints
    from pipelines.pipeline_base import StageResult, StageStatus

    fake_stages = [
        ("query_reformulation",   1.2),
        ("arxiv_retrieval",       8.5),
        ("embedding_generation",  22.3),
        ("clustering",            4.1),
        ("cluster_labeling",      12.7),
        ("gap_detection",         31.4),
        ("hypothesis_generation", 14.2),
        ("roadmap_building",      18.9),
        ("report_generation",     3.1),
    ]

    for stage_name, duration in fake_stages:
        lab_pipeline._on_stage_start(stage_name)
        time.sleep(0.3)
        lab_pipeline._on_stage_complete(
            StageResult(
                stage_name=stage_name,
                status=StageStatus.COMPLETED,
                duration_s=duration,
                attempt=1,
            )
        )

    lab_pipeline.end_session(
        output_counts={
            "papers": 72, "clusters": 7,
            "gaps": 4, "hypotheses": 5,
        },
        notes="Demo session completed successfully",
    )

    time.sleep(1)

    # Status snapshot
    status = lab_pipeline.status()
    print("\n── LIVE STATUS ──")
    print(f"  Session active:    {status['session_active']}")
    print(f"  Alert counts:      {status['alert_counts']}")
    print(f"  Throttle events:   {status['throttle_events']}")
    print(f"  Bottleneck history:{status['bottleneck_history']}")

    # Log tail
    log = lab_pipeline.get_log_tail(10)
    print(f"\n── LOG TAIL (last {len(log)} entries) ──")
    for entry in log[-5:]:
        print(f"  [{entry.get('severity','?'):8}] {entry.get('message','')[:70]}")

    # Generate session report
    sched_stats = scheduler.status().get("stats", {})
    report = lab_pipeline.generate_session_report(
        session_id=session_id,
        scheduler_stats=sched_stats,
    )
    print("\n── LAB REPORT (first 40 lines) ──")
    for line in report.split("\n")[:40]:
        print(" ", line)

    lab_pipeline.stop()
    scheduler.stop()
    print("\nLabPipeline demo complete.")


if __name__ == "__main__":
    _demo()
