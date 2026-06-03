"""
monitoring/alert_manager.py
============================
ResearchFlow AI — Alert Manager

This is the final module in monitoring/ and the top of the telemetry
stack. It consumes BottleneckReports, raw threshold breaches, and
ProcessTracker anomalies — then routes structured alerts to:

  1. Streamlit toast queue      → live dashboard notifications
  2. JSONL structured log       → persistent audit trail
  3. In-memory alert feed       → 02_lab_assistant.py alert panel
  4. Webhook (optional)         → Slack / PagerDuty integration
  5. LabAgent experiment record → per-run alert accounting

Architecture:
  ┌─────────────────────────────────────────────────────────────────┐
  │                    ALERT PIPELINE                               │
  │                                                                 │
  │  BottleneckReport ──┐                                           │
  │  ThresholdBreach  ──┼──► AlertEvaluator ──► Alert              │
  │  ProcessAnomaly   ──┘         │               │                 │
  │                               │               ▼                 │
  │                          dedup/rate     AlertRouter             │
  │                          limiting         │   │   │             │
  │                                     Toast Log  Webhook          │
  └─────────────────────────────────────────────────────────────────┘

Alert design principles:
  1. Deduplication   — identical alerts within DEDUP_WINDOW_S are suppressed
  2. Rate limiting   — max RATE_LIMIT_PER_MIN alerts per metric per minute
  3. Severity levels — INFO / WARNING / CRITICAL with distinct routing
  4. Auto-resolution — alerts auto-resolve when the condition clears
  5. Alert grouping  — related alerts bundled into a single notification

OS Concepts demonstrated:
  - Interrupt-driven alerting   : threshold crossing triggers immediate alert
  - Alert deduplication         : equivalent to interrupt coalescing
  - Rate limiting               : analogous to interrupt throttling (NAPI)
  - Structured logging          : kernel audit trail equivalent
  - Event-driven architecture   : producer-consumer alert pipeline

Alert taxonomy:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Source              │  Trigger condition                       │
  ├──────────────────────┼──────────────────────────────────────────┤
  │  CPU threshold       │  overall% > WARNING_PCT or CRIT_PCT      │
  │  RAM threshold       │  used% > WARNING_PCT                     │
  │  OOM risk            │  from BottleneckDetector OOM_RISK signal  │
  │  GPU VRAM            │  vram% > WARNING_PCT or CRIT_PCT         │
  │  GPU thermal         │  temperature > THERMAL_THRESHOLD         │
  │  Disk I/O            │  busy% > IO_BUSY_THRESHOLD               │
  │  Process anomaly     │  2σ CPU or RAM deviation per PID         │
  │  Zombie process      │  status == "zombie" detected             │
  │  Scheduler throttle  │  scheduler state → THROTTLED             │
  │  Task starvation     │  defer_count > MAX_DEFER                 │
  │  Bottleneck change   │  primary_type changed                    │
  └─────────────────────────────────────────────────────────────────┘

Usage:
    mgr = AlertManager()
    mgr.start()

    # Feed from bottleneck detector
    mgr.process_bottleneck(bottleneck_report)

    # Feed from raw telemetry
    mgr.process_snapshot(sys_snapshot, gpu_snapshot)

    # Feed from process tracker
    mgr.process_process_report(proc_tree_report)

    # Drain for Streamlit dashboard
    alerts = mgr.drain_toasts(max_items=20)
    feed   = mgr.get_feed(last_n=50)

    mgr.stop()

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Deque

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

ALERT_LOG_PATH: Path  = Path("logs/alerts.jsonl")
MAX_FEED_SIZE:  int   = 500     # in-memory rolling feed cap
MAX_TOAST_QUEUE: int  = 100     # Streamlit toast queue cap

# Deduplication: suppress identical alerts within this window
DEDUP_WINDOW_S: float = float(os.getenv("ALERT_DEDUP_WINDOW_S", "30.0"))

# Rate limiting: max alerts per metric per 60-second window
RATE_LIMIT_PER_MIN: int = int(os.getenv("ALERT_RATE_LIMIT", "3"))

# Thresholds (can be overridden via env)
CPU_WARN_PCT:   float = float(os.getenv("ALERT_CPU_WARN",    "80.0"))
CPU_CRIT_PCT:   float = float(os.getenv("ALERT_CPU_CRIT",    "92.0"))
RAM_WARN_PCT:   float = float(os.getenv("ALERT_RAM_WARN",    "80.0"))
RAM_CRIT_PCT:   float = float(os.getenv("ALERT_RAM_CRIT",    "92.0"))
SWAP_WARN_PCT:  float = float(os.getenv("ALERT_SWAP_WARN",   "30.0"))
GPU_WARN_PCT:   float = float(os.getenv("ALERT_GPU_WARN",    "85.0"))
GPU_CRIT_PCT:   float = float(os.getenv("ALERT_GPU_CRIT",    "95.0"))
GPU_TEMP_WARN:  float = float(os.getenv("ALERT_GPU_TEMP",    "80.0"))
DISK_BUSY_WARN: float = float(os.getenv("ALERT_DISK_BUSY",   "70.0"))


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


class AlertCategory(str, Enum):
    CPU         = "cpu"
    MEMORY      = "memory"
    DISK        = "disk"
    GPU         = "gpu"
    PROCESS     = "process"
    SCHEDULER   = "scheduler"
    BOTTLENECK  = "bottleneck"
    SYSTEM      = "system"


class AlertStatus(str, Enum):
    ACTIVE   = "active"
    RESOLVED = "resolved"
    SILENCED = "silenced"


# Streamlit toast colours per severity
TOAST_ICONS: dict[str, str] = {
    AlertSeverity.INFO:     "ℹ️",
    AlertSeverity.WARNING:  "⚠️",
    AlertSeverity.CRITICAL: "🔴",
}


# ─────────────────────────────────────────────────────────────────────────────
# ALERT DATA MODEL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Alert:
    """
    A single structured alert event.

    alert_key is a stable identifier for deduplication:
      "{category}:{metric}:{severity}" — same key suppressed within DEDUP_WINDOW_S.

    fingerprint is a hash of (alert_key + rounded_value) for exact deduplication.
    """
    alert_id:    str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    alert_key:   str = ""           # stable dedup key
    fingerprint: str = ""           # exact dedup hash

    # Classification
    severity:    AlertSeverity  = AlertSeverity.INFO
    category:    AlertCategory  = AlertCategory.SYSTEM
    status:      AlertStatus    = AlertStatus.ACTIVE

    # Content
    title:       str   = ""
    message:     str   = ""
    metric:      str   = ""         # e.g. "cpu_pct", "ram_used_pct"
    value:       float = 0.0        # current metric value
    threshold:   float = 0.0        # the threshold that was breached

    # Context
    timestamp:   str   = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source:      str   = ""         # which monitor produced this
    detail:      dict  = field(default_factory=dict)   # extra context

    # Resolution
    resolved_at: str   = ""
    duration_s:  float = 0.0

    @property
    def age_s(self) -> float:
        try:
            t0 = datetime.fromisoformat(self.timestamp)
            return (datetime.now(timezone.utc) - t0).total_seconds()
        except Exception:
            return 0.0

    @property
    def icon(self) -> str:
        return TOAST_ICONS.get(self.severity, "ℹ️")

    @property
    def toast_text(self) -> str:
        return f"{self.icon} **{self.title}** — {self.message}"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()}

    def to_log_line(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# ─────────────────────────────────────────────────────────────────────────────
# ALERT EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────

class AlertEvaluator:
    """
    Evaluates raw telemetry snapshots against threshold rules and
    produces Alert objects.

    Each evaluate_*() method checks one signal source and returns
    0..N Alert objects. The AlertManager collects them all, deduplicates,
    rate-limits, and routes to sinks.
    """

    def evaluate_cpu(
        self,
        cpu_pct: float,
        per_core: list[float] | None = None,
    ) -> list[Alert]:
        alerts: list[Alert] = []

        if cpu_pct >= CPU_CRIT_PCT:
            alerts.append(Alert(
                alert_key=f"cpu:overall:critical",
                fingerprint=f"cpu:crit:{int(cpu_pct // 5) * 5}",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.CPU,
                title="CPU Critical",
                message=f"Overall CPU {cpu_pct:.1f}% — pipeline tasks will be throttled",
                metric="cpu_pct", value=cpu_pct, threshold=CPU_CRIT_PCT,
                source="system_monitor",
                detail={"per_core": per_core or []},
            ))
        elif cpu_pct >= CPU_WARN_PCT:
            alerts.append(Alert(
                alert_key=f"cpu:overall:warning",
                fingerprint=f"cpu:warn:{int(cpu_pct // 5) * 5}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.CPU,
                title="CPU High",
                message=f"CPU {cpu_pct:.1f}% — high-cost tasks being deferred",
                metric="cpu_pct", value=cpu_pct, threshold=CPU_WARN_PCT,
                source="system_monitor",
            ))

        # Hot core alert (individual core > 98%)
        if per_core:
            for i, core_pct in enumerate(per_core):
                if core_pct >= 98.0:
                    alerts.append(Alert(
                        alert_key=f"cpu:core{i}:critical",
                        fingerprint=f"cpu:core{i}:crit",
                        severity=AlertSeverity.WARNING,
                        category=AlertCategory.CPU,
                        title=f"CPU Core {i} Saturated",
                        message=f"Core {i} at {core_pct:.1f}% — single-threaded bottleneck",
                        metric=f"cpu_core_{i}_pct", value=core_pct, threshold=98.0,
                        source="system_monitor",
                    ))

        return alerts

    def evaluate_memory(
        self,
        ram_used_pct: float,
        swap_pct: float,
        available_gb: float,
    ) -> list[Alert]:
        alerts: list[Alert] = []

        if ram_used_pct >= RAM_CRIT_PCT:
            alerts.append(Alert(
                alert_key="memory:ram:critical",
                fingerprint=f"ram:crit:{int(ram_used_pct // 2) * 2}",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.MEMORY,
                title="RAM Critical — OOM Risk",
                message=(
                    f"RAM {ram_used_pct:.1f}% used "
                    f"({available_gb:.2f}GB free) — OOM kill possible"
                ),
                metric="ram_used_pct", value=ram_used_pct, threshold=RAM_CRIT_PCT,
                source="system_monitor",
                detail={"available_gb": available_gb, "swap_pct": swap_pct},
            ))
        elif ram_used_pct >= RAM_WARN_PCT:
            alerts.append(Alert(
                alert_key="memory:ram:warning",
                fingerprint=f"ram:warn:{int(ram_used_pct // 5) * 5}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.MEMORY,
                title="RAM High",
                message=f"RAM {ram_used_pct:.1f}% used — memory-intensive tasks deferred",
                metric="ram_used_pct", value=ram_used_pct, threshold=RAM_WARN_PCT,
                source="system_monitor",
            ))

        if swap_pct >= SWAP_WARN_PCT:
            alerts.append(Alert(
                alert_key="memory:swap:warning",
                fingerprint=f"swap:warn:{int(swap_pct // 10) * 10}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.MEMORY,
                title="Swap Usage Elevated",
                message=f"Swap {swap_pct:.1f}% used — system is paging; performance degraded",
                metric="swap_pct", value=swap_pct, threshold=SWAP_WARN_PCT,
                source="system_monitor",
            ))

        return alerts

    def evaluate_disk(self, disk_busy_pct: float, io_mb_s: float) -> list[Alert]:
        alerts: list[Alert] = []
        if disk_busy_pct >= DISK_BUSY_WARN:
            alerts.append(Alert(
                alert_key="disk:busy:warning",
                fingerprint=f"disk:warn:{int(disk_busy_pct // 10) * 10}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.DISK,
                title="Disk I/O Bottleneck",
                message=(
                    f"Disk busy {disk_busy_pct:.1f}% | "
                    f"I/O {io_mb_s:.1f}MB/s — embedding cache or model load"
                ),
                metric="disk_busy_pct", value=disk_busy_pct, threshold=DISK_BUSY_WARN,
                source="system_monitor",
                detail={"io_mb_s": io_mb_s},
            ))
        return alerts

    def evaluate_gpu(
        self,
        vram_pct: float,
        temperature_c: float,
        utilisation_pct: float,
        device_name: str = "",
    ) -> list[Alert]:
        alerts: list[Alert] = []

        if vram_pct >= GPU_CRIT_PCT:
            alerts.append(Alert(
                alert_key="gpu:vram:critical",
                fingerprint=f"gpu:vram:crit:{int(vram_pct // 2) * 2}",
                severity=AlertSeverity.CRITICAL,
                category=AlertCategory.GPU,
                title="GPU VRAM Critical",
                message=(
                    f"VRAM {vram_pct:.1f}% used ({device_name}) — "
                    f"CUDA OOM imminent; all GPU tasks paused"
                ),
                metric="gpu_vram_pct", value=vram_pct, threshold=GPU_CRIT_PCT,
                source="gpu_monitor",
                detail={"device": device_name, "utilisation": utilisation_pct},
            ))
        elif vram_pct >= GPU_WARN_PCT:
            alerts.append(Alert(
                alert_key="gpu:vram:warning",
                fingerprint=f"gpu:vram:warn:{int(vram_pct // 5) * 5}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.GPU,
                title="GPU VRAM High",
                message=f"VRAM {vram_pct:.1f}% — GPU-heavy tasks being deferred",
                metric="gpu_vram_pct", value=vram_pct, threshold=GPU_WARN_PCT,
                source="gpu_monitor",
            ))

        if temperature_c >= GPU_TEMP_WARN:
            alerts.append(Alert(
                alert_key="gpu:thermal:warning",
                fingerprint=f"gpu:thermal:{int(temperature_c // 5) * 5}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.GPU,
                title="GPU Thermal Alert",
                message=(
                    f"GPU {temperature_c:.0f}°C ({device_name}) — "
                    f"thermal throttling may reduce inference speed"
                ),
                metric="gpu_temperature_c", value=temperature_c, threshold=GPU_TEMP_WARN,
                source="gpu_monitor",
            ))

        return alerts

    def evaluate_process_anomaly(
        self,
        pid: int,
        name: str,
        cpu_pct: float,
        mem_mb: float,
        cpu_zscore: float,
        mem_zscore: float,
    ) -> list[Alert]:
        alerts: list[Alert] = []

        if abs(cpu_zscore) > 2.0:
            alerts.append(Alert(
                alert_key=f"process:cpu_anomaly:{pid}",
                fingerprint=f"proc:cpu:{pid}:{int(cpu_pct // 10) * 10}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PROCESS,
                title=f"Process CPU Anomaly",
                message=(
                    f"PID {pid} '{name}' CPU={cpu_pct:.1f}% "
                    f"(z={cpu_zscore:.1f}) — unexpected spike above baseline"
                ),
                metric="process_cpu_pct", value=cpu_pct, threshold=0.0,
                source="process_tracker",
                detail={"pid": pid, "name": name, "zscore": cpu_zscore},
            ))

        if abs(mem_zscore) > 2.0:
            alerts.append(Alert(
                alert_key=f"process:mem_anomaly:{pid}",
                fingerprint=f"proc:mem:{pid}:{int(mem_mb // 100) * 100}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.PROCESS,
                title=f"Process Memory Anomaly",
                message=(
                    f"PID {pid} '{name}' RSS={mem_mb:.0f}MB "
                    f"(z={mem_zscore:.1f}) — memory spike above baseline"
                ),
                metric="process_mem_mb", value=mem_mb, threshold=0.0,
                source="process_tracker",
                detail={"pid": pid, "name": name, "zscore": mem_zscore},
            ))

        return alerts

    def evaluate_zombie(self, pid: int, name: str) -> Alert:
        return Alert(
            alert_key=f"process:zombie:{pid}",
            fingerprint=f"zombie:{pid}",
            severity=AlertSeverity.WARNING,
            category=AlertCategory.PROCESS,
            title="Zombie Process Detected",
            message=(
                f"PID {pid} '{name}' is in zombie state — "
                f"parent process failed to reap child"
            ),
            metric="process_status", value=0.0, threshold=0.0,
            source="process_tracker",
            detail={"pid": pid, "name": name},
        )

    def evaluate_bottleneck(
        self,
        bottleneck_type: str,
        confidence: float,
        detail: str,
        prev_type: str = "none",
    ) -> list[Alert]:
        alerts: list[Alert] = []

        if bottleneck_type == prev_type:
            return alerts   # no change — don't re-alert

        if bottleneck_type != "none":
            severity = (
                AlertSeverity.CRITICAL
                if bottleneck_type in ("oom_risk", "mixed")
                else AlertSeverity.WARNING
            )
            alerts.append(Alert(
                alert_key=f"bottleneck:change:{bottleneck_type}",
                fingerprint=f"bottleneck:{bottleneck_type}:{int(confidence * 10)}",
                severity=severity,
                category=AlertCategory.BOTTLENECK,
                title=f"Bottleneck Detected: {bottleneck_type.replace('_', ' ').upper()}",
                message=f"{detail[:120]} (confidence={confidence:.2f})",
                metric="bottleneck_type", value=confidence, threshold=0.0,
                source="bottleneck_detector",
                detail={"type": bottleneck_type, "confidence": confidence},
            ))
        else:
            # Bottleneck cleared
            alerts.append(Alert(
                alert_key="bottleneck:cleared",
                fingerprint=f"bottleneck:cleared:{prev_type}",
                severity=AlertSeverity.INFO,
                category=AlertCategory.BOTTLENECK,
                title="Bottleneck Cleared",
                message=f"System resources recovered (was: {prev_type})",
                metric="bottleneck_type", value=0.0, threshold=0.0,
                source="bottleneck_detector",
            ))

        return alerts

    def evaluate_scheduler(
        self,
        event: str,   # "throttled" | "unthrottled" | "task_starvation"
        detail: str = "",
        task_name: str = "",
        defer_count: int = 0,
    ) -> Alert | None:
        if event == "throttled":
            return Alert(
                alert_key="scheduler:throttled",
                fingerprint="scheduler:throttled",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SCHEDULER,
                title="Scheduler Throttled",
                message=f"Concurrency reduced to 1 — {detail[:80]}",
                metric="scheduler_state", value=1.0, threshold=0.0,
                source="scheduler_agent",
            )
        if event == "unthrottled":
            return Alert(
                alert_key="scheduler:unthrottled",
                fingerprint="scheduler:unthrottled",
                severity=AlertSeverity.INFO,
                category=AlertCategory.SCHEDULER,
                title="Scheduler Restored",
                message=f"Normal concurrency restored — {detail[:80]}",
                metric="scheduler_state", value=0.0, threshold=0.0,
                source="scheduler_agent",
            )
        if event == "task_starvation":
            return Alert(
                alert_key=f"scheduler:starvation:{task_name}",
                fingerprint=f"starvation:{task_name}:{defer_count // 5}",
                severity=AlertSeverity.WARNING,
                category=AlertCategory.SCHEDULER,
                title="Task Starvation",
                message=(
                    f"'{task_name}' force-dispatched after {defer_count} deferrals "
                    f"— resource contention causing scheduling delays"
                ),
                metric="defer_count", value=float(defer_count), threshold=0.0,
                source="scheduler_agent",
                detail={"task_name": task_name, "defer_count": defer_count},
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION & RATE LIMITING
# ─────────────────────────────────────────────────────────────────────────────

class AlertDeduplicator:
    """
    Suppresses duplicate alerts using two strategies:

    1. Exact deduplication: identical fingerprint within DEDUP_WINDOW_S → skip
    2. Rate limiting: same alert_key more than RATE_LIMIT_PER_MIN times
       within 60s → skip

    Analogous to interrupt coalescing (NAPI, NAPI polling in Linux):
    when interrupts arrive faster than the handler can process them,
    the kernel coalesces them into a single notification rather than
    flooding the interrupt handler.
    """

    def __init__(self) -> None:
        # fingerprint → last_seen_epoch
        self._seen: dict[str, float] = {}
        # alert_key → deque of timestamps in last 60s
        self._rate: dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=60))
        self._lock = threading.Lock()

    def should_emit(self, alert: Alert) -> bool:
        """Returns True if the alert should be emitted (not suppressed)."""
        now = time.time()

        with self._lock:
            # Exact dedup
            last = self._seen.get(alert.fingerprint, 0.0)
            if now - last < DEDUP_WINDOW_S:
                return False
            self._seen[alert.fingerprint] = now

            # Rate limit
            bucket = self._rate[alert.alert_key]
            # Prune old timestamps
            cutoff = now - 60.0
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= RATE_LIMIT_PER_MIN:
                return False
            bucket.append(now)

        return True

    def invalidate(self, fingerprint: str) -> None:
        """Force-expire a fingerprint (used when a condition clears)."""
        with self._lock:
            self._seen.pop(fingerprint, None)

    def clear(self) -> None:
        with self._lock:
            self._seen.clear()
            self._rate.clear()


# ─────────────────────────────────────────────────────────────────────────────
# ALERT ROUTER
# ─────────────────────────────────────────────────────────────────────────────

class AlertRouter:
    """
    Routes alerts to configured sinks.

    Sinks:
      1. toast_queue   — Streamlit queue.Queue (non-blocking, size-capped)
      2. feed          — in-memory deque for alert history panel
      3. log file      — JSONL append-only log
      4. callbacks     — registered callback functions (e.g. LabAgent.record_alert)
      5. webhook       — HTTP POST (optional, best-effort, background thread)
    """

    def __init__(
        self,
        log_path:    Path = ALERT_LOG_PATH,
        webhook_url: str  | None = None,
    ) -> None:
        self._log_path   = log_path
        self._webhook_url = webhook_url
        self._toast_q:   queue.Queue = queue.Queue(maxsize=MAX_TOAST_QUEUE)
        self._feed:      Deque[Alert] = deque(maxlen=MAX_FEED_SIZE)
        self._callbacks: list[Callable[[Alert], None]] = []
        self._lock       = threading.Lock()
        self._counts:    dict[str, int] = defaultdict(int)

        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def route(self, alert: Alert) -> None:
        """Route a single alert to all configured sinks."""
        with self._lock:
            self._feed.append(alert)
            self._counts[alert.severity.value] += 1

        # Sink 1: Streamlit toast (non-blocking)
        try:
            self._toast_q.put_nowait(alert)
        except queue.Full:
            pass

        # Sink 2: JSONL log
        try:
            with open(self._log_path, "a") as f:
                f.write(alert.to_log_line() + "\n")
        except Exception as exc:
            logger.warning("Alert log write failed: %s", exc)

        # Sink 3: Callbacks
        for cb in list(self._callbacks):
            try:
                cb(alert)
            except Exception as exc:
                logger.warning("Alert callback error: %s", exc)

        # Sink 4: Webhook (critical/warning only, best-effort)
        if self._webhook_url and alert.severity != AlertSeverity.INFO:
            threading.Thread(
                target=self._post_webhook,
                args=(alert,),
                daemon=True,
            ).start()

        # Log to module logger
        log_fn = {
            AlertSeverity.INFO:     logger.info,
            AlertSeverity.WARNING:  logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
        }[alert.severity]
        log_fn("[ALERT %s] %s — %s", alert.severity.value.upper(),
               alert.title, alert.message[:80])

    def register_callback(self, cb: Callable[[Alert], None]) -> None:
        self._callbacks.append(cb)

    def drain_toasts(self, max_items: int = 50) -> list[Alert]:
        """Drain the Streamlit toast queue. Non-blocking."""
        alerts: list[Alert] = []
        for _ in range(max_items):
            try:
                alerts.append(self._toast_q.get_nowait())
            except queue.Empty:
                break
        return alerts

    def get_feed(self, last_n: int | None = None) -> list[Alert]:
        with self._lock:
            feed = list(self._feed)
        return feed[-last_n:] if last_n else feed

    @property
    def counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counts)

    def _post_webhook(self, alert: Alert) -> None:
        try:
            import urllib.request
            payload = json.dumps({
                "text":      f"[ResearchFlow] {alert.severity.value.upper()}: {alert.title}",
                "message":   alert.message,
                "timestamp": alert.timestamp,
                "metric":    alert.metric,
                "value":     alert.value,
            }).encode()
            req = urllib.request.Request(
                self._webhook_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as exc:
            logger.debug("Webhook POST failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# ALERT MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class AlertManager:
    """
    Top-level alert orchestrator for ResearchFlow AI.

    Composes AlertEvaluator, AlertDeduplicator, and AlertRouter into
    a single clean API consumed by the monitoring stack and Streamlit UI.

    The AlertManager is the final consumer in the monitoring pipeline:
      SystemMonitor → BottleneckDetector → AlertManager → Dashboard

    Thread-safe: all process_*() methods can be called from the
    SystemMonitor background thread without concern.
    """

    def __init__(
        self,
        log_path:    Path = ALERT_LOG_PATH,
        webhook_url: str  | None = None,
    ) -> None:
        self._evaluator  = AlertEvaluator()
        self._dedup      = AlertDeduplicator()
        self._router     = AlertRouter(log_path, webhook_url)
        self._prev_bottleneck: str = "none"
        self._lock       = threading.Lock()

        logger.info("AlertManager initialised | log=%s", log_path)

    # ── PUBLIC PROCESS METHODS ───────────────────────────────────────────────

    def process_snapshot(
        self,
        sys_snap:  object,
        gpu_snap:  object | None = None,
    ) -> int:
        """
        Evaluate a SystemSnapshot (+ optional GPUSnapshot) against all
        threshold rules and route any generated alerts.
        Returns the number of alerts emitted.
        """
        alerts: list[Alert] = []

        # CPU
        cpu = getattr(sys_snap, "cpu", None)
        if cpu:
            alerts += self._evaluator.evaluate_cpu(
                cpu_pct=getattr(cpu, "overall_pct", 0.0),
                per_core=getattr(cpu, "per_core_pct", []),
            )

        # Memory
        mem = getattr(sys_snap, "memory", None)
        if mem:
            alerts += self._evaluator.evaluate_memory(
                ram_used_pct=getattr(mem, "used_pct",       0.0),
                swap_pct=    getattr(mem, "swap_pct",       0.0),
                available_gb=getattr(mem, "available_gb",  16.0),
            )

        # Disk
        disk = getattr(sys_snap, "disk", None)
        if disk:
            io_mb = getattr(disk, "read_mb_s", 0.0) + getattr(disk, "write_mb_s", 0.0)
            alerts += self._evaluator.evaluate_disk(
                disk_busy_pct=getattr(disk, "busy_pct", 0.0),
                io_mb_s=io_mb,
            )

        # GPU
        if gpu_snap and getattr(gpu_snap, "is_available", False):
            primary = getattr(gpu_snap, "primary", None)
            if primary:
                alerts += self._evaluator.evaluate_gpu(
                    vram_pct=       getattr(primary, "vram_used_pct",   0.0),
                    temperature_c=  getattr(primary, "temperature_c",   0.0),
                    utilisation_pct=getattr(primary, "utilisation_pct", 0.0),
                    device_name=    getattr(primary, "device_name",     "GPU"),
                )

        return self._emit_batch(alerts)

    def process_bottleneck(self, report: object) -> int:
        """
        Evaluate a BottleneckReport and emit a transition alert when
        the bottleneck type changes.
        """
        btype      = str(getattr(report, "primary_type",
                         getattr(getattr(report, "primary_type", None),
                                 "value", "none")))
        # Handle both string and enum
        if hasattr(btype, "value"):
            btype = btype.value

        confidence = float(getattr(report, "confidence", 0.5))
        detail     = str(getattr(report, "detail", ""))

        with self._lock:
            prev = self._prev_bottleneck

        alerts = self._evaluator.evaluate_bottleneck(
            bottleneck_type=btype,
            confidence=confidence,
            detail=detail,
            prev_type=prev,
        )

        with self._lock:
            self._prev_bottleneck = btype

        return self._emit_batch(alerts)

    def process_process_report(self, report: object) -> int:
        """
        Evaluate a ProcessTreeReport for anomalies and zombies.
        """
        alerts: list[Alert] = []
        processes = getattr(report, "processes", [])

        for proc in processes:
            if getattr(proc, "is_zombie", False):
                alerts.append(self._evaluator.evaluate_zombie(
                    pid=getattr(proc, "pid", 0),
                    name=getattr(proc, "name", "?"),
                ))

            if getattr(proc, "is_cpu_anomaly", False) or getattr(proc, "is_mem_anomaly", False):
                alerts += self._evaluator.evaluate_process_anomaly(
                    pid=      getattr(proc, "pid",         0),
                    name=     getattr(proc, "name",        "?"),
                    cpu_pct=  getattr(proc, "cpu_pct",     0.0),
                    mem_mb=   getattr(proc, "mem_rss_mb",  0.0),
                    cpu_zscore=getattr(proc, "cpu_zscore", 0.0),
                    mem_zscore=getattr(proc, "mem_zscore", 0.0),
                )

        return self._emit_batch(alerts)

    def emit_scheduler_event(
        self,
        event:       str,
        detail:      str = "",
        task_name:   str = "",
        defer_count: int = 0,
    ) -> int:
        """Emit a scheduler-level alert (throttle, starvation, etc.)."""
        alert = self._evaluator.evaluate_scheduler(
            event=event, detail=detail,
            task_name=task_name, defer_count=defer_count,
        )
        if alert:
            return self._emit_batch([alert])
        return 0

    def emit_custom(
        self,
        title:    str,
        message:  str,
        severity: AlertSeverity = AlertSeverity.INFO,
        category: AlertCategory = AlertCategory.SYSTEM,
        detail:   dict | None   = None,
    ) -> Alert:
        """Emit a custom alert directly (for pipeline-level events)."""
        alert = Alert(
            alert_key=f"custom:{category.value}:{title[:20]}",
            fingerprint=f"custom:{title[:20]}",
            severity=severity,
            category=category,
            title=title,
            message=message,
            source="custom",
            detail=detail or {},
        )
        self._emit_batch([alert])
        return alert

    # ── DASHBOARD API ────────────────────────────────────────────────────────

    def drain_toasts(self, max_items: int = 50) -> list[Alert]:
        """Drain Streamlit toast queue. Call on every dashboard rerun."""
        return self._router.drain_toasts(max_items)

    def get_feed(self, last_n: int | None = None) -> list[Alert]:
        """Return alert history for the 02_lab_assistant.py alert panel."""
        return self._router.get_feed(last_n)

    def get_feed_dicts(self, last_n: int | None = None) -> list[dict]:
        return [a.to_dict() for a in self.get_feed(last_n)]

    def register_callback(self, cb: Callable[[Alert], None]) -> None:
        """Register a callback invoked on every emitted alert."""
        self._router.register_callback(cb)

    @property
    def counts(self) -> dict[str, int]:
        """Alert counts by severity — for dashboard summary."""
        return self._router.counts

    def reset_dedup(self) -> None:
        """Clear deduplication state (call at start of new pipeline run)."""
        self._dedup.clear()
        with self._lock:
            self._prev_bottleneck = "none"

    # ── INTERNAL ────────────────────────────────────────────────────────────

    def _emit_batch(self, alerts: list[Alert]) -> int:
        """Dedup + rate-limit + route a batch of candidate alerts."""
        emitted = 0
        for alert in alerts:
            if self._dedup.should_emit(alert):
                self._router.route(alert)
                emitted += 1
        return emitted

    def status(self) -> dict:
        return {
            "counts":             self.counts,
            "feed_depth":         len(self._router.get_feed()),
            "prev_bottleneck":    self._prev_bottleneck,
        }

    def __repr__(self) -> str:
        c = self.counts
        return (
            f"AlertManager("
            f"info={c.get('info',0)}, "
            f"warning={c.get('warning',0)}, "
            f"critical={c.get('critical',0)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_manager: AlertManager | None = None


def get_alert_manager() -> AlertManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = AlertManager()
    return _default_manager


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Simulate a pipeline run generating alerts across all categories.
    Run with: python -m monitoring.alert_manager
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    mgr = AlertManager(log_path=Path("logs/demo_alerts.jsonl"))

    # Register a callback
    received: list[Alert] = []
    mgr.register_callback(received.append)

    print("\n═══ ALERT MANAGER DEMO ═══\n")

    # ── CPU alerts ────────────────────────────────────────────────────────────
    print("  Simulating CPU overload...")
    class FakeCPU:
        overall_pct = 91.0
        per_core_pct = [99.0, 45.0, 88.0, 92.0]
        load_avg_1m = 7.2
        logical_cores = 4

    class FakeMem:
        used_pct = 55.0; swap_pct = 5.0; available_gb = 6.0
        used_gb = 10.0; total_gb = 16.0; pressure_index = 40.0

    class FakeDisk:
        busy_pct = 8.0; read_mb_s = 5.0; write_mb_s = 2.0

    class FakeSys:
        cpu = FakeCPU(); memory = FakeMem(); disk = FakeDisk()

    mgr.process_snapshot(FakeSys())

    # ── Memory pressure ───────────────────────────────────────────────────────
    print("  Simulating memory pressure...")
    FakeMem.used_pct = 88.0; FakeMem.available_gb = 1.9; FakeMem.swap_pct = 35.0
    mgr.process_snapshot(FakeSys())

    # ── GPU alert ─────────────────────────────────────────────────────────────
    print("  Simulating GPU VRAM saturation...")
    class FakeGPUDevice:
        vram_used_pct = 91.0; temperature_c = 82.0
        utilisation_pct = 95.0; device_name = "NVIDIA RTX 4090"
        is_warning = True; is_critical = False

    class FakeGPU:
        is_available = True; primary = FakeGPUDevice()

    mgr.process_snapshot(FakeSys(), FakeGPU())

    # ── Bottleneck change ─────────────────────────────────────────────────────
    print("  Simulating bottleneck transition...")
    class FakeBN:
        primary_type = "cpu_bound"; confidence = 0.87
        detail = "CPU 91% sustained for 5 ticks — concurrency reduced"

    mgr.process_bottleneck(FakeBN())

    # ── Process anomaly ───────────────────────────────────────────────────────
    print("  Simulating process anomaly...")
    class FakeProc:
        pid=12345; name="embedding_worker"; is_zombie=False
        cpu_pct=95.0; mem_rss_mb=3200.0
        is_cpu_anomaly=True; is_mem_anomaly=True
        cpu_zscore=3.8; mem_zscore=2.4

    class FakeProcReport:
        processes = [FakeProc()]

    mgr.process_process_report(FakeProcReport())

    # ── Scheduler events ──────────────────────────────────────────────────────
    print("  Simulating scheduler throttle + starvation...")
    mgr.emit_scheduler_event("throttled", "CPU 91% > 85% threshold")
    mgr.emit_scheduler_event("task_starvation",
                             task_name="umap_reduction", defer_count=32)

    # ── Dedup test ────────────────────────────────────────────────────────────
    print("  Testing deduplication (same alert emitted 5× → should appear 1×)...")
    before = len(received)
    for _ in range(5):
        mgr.emit_custom("Test Alert", "Should be deduped",
                        AlertSeverity.INFO, AlertCategory.SYSTEM)
    after = len(received)
    print(f"    Emitted 5×, arrived: {after - before} (expected 1)")

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n═══ RESULTS ═══")
    print(f"  Total alerts received: {len(received)}")
    print(f"  Counts: {mgr.counts}")
    print(f"\n  Alert feed (last 10):")
    for a in mgr.get_feed(last_n=10):
        print(
            f"  [{a.severity.value.upper():<8}] "
            f"{a.category.value:<12} "
            f"{a.title[:35]:<35} | {a.message[:50]}"
        )

    print(f"\n{mgr}")
    print(f"\n  Log written to: logs/demo_alerts.jsonl")


if __name__ == "__main__":
    _demo()
