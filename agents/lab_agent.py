"""
agents/lab_agent.py
===================
ResearchFlow AI — OS-Integrated Lab Assistant Agent

Responsibilities:
  1. Continuously monitors CPU, RAM, disk, GPU, and process-level telemetry
     in a background thread using psutil + torch.cuda
  2. Detects resource bottlenecks (CPU-bound, I/O-bound, OOM-risk) via
     heuristic rules and 2-sigma anomaly detection on rolling baselines
  3. Fires structured alerts (INFO / WARNING / CRITICAL) to a thread-safe
     event queue consumed by the Streamlit dashboard
  4. Tracks all pipeline experiment runs — params, metrics, resource
     snapshots, status, and wall-clock timing
  5. Generates structured experiment reports (Markdown + JSON) at the end
     of each pipeline session
  6. Exposes a clean async API so the SchedulerAgent can query live state
     before dispatching resource-intensive tasks

OS Concepts demonstrated:
  - Process monitoring   : psutil.process_iter(), own PID tree
  - Memory management    : virtual_memory(), swap detection, OOM-risk flag
  - CPU scheduling view  : per-core utilisation, sustained load detection
  - Background threads   : threading.Thread daemon for non-blocking polling
  - Producer-consumer    : queue.Queue for alert events (monitor → dashboard)
  - Anomaly detection    : rolling 2σ baseline per metric

Usage:
    agent = LabAgent()
    agent.start()                      # launches background monitor thread
    snapshot = agent.snapshot()        # get latest SystemSnapshot
    agent.begin_run("my_experiment")   # start tracking a pipeline run
    agent.record_metric("loss", 0.42)  # log a metric
    agent.end_run()                    # finalise and persist run record
    agent.stop()                       # graceful shutdown

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import json
import logging
import os
import queue
import statistics
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Deque

import psutil

# GPU monitoring (optional — only available when PyTorch + CUDA present)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL_S: float = float(os.getenv("MONITOR_POLL_INTERVAL", "2.0"))
ROLLING_WINDOW: int = 30          # samples kept for rolling baseline (~60s)
ANOMALY_SIGMA: float = 2.0        # standard deviations for anomaly flag

# Alert thresholds (overridable via env / scheduler_config.yaml)
CPU_WARN_PCT: float = float(os.getenv("SCHEDULER_CPU_THRESHOLD", "85"))
CPU_CRIT_PCT: float = 95.0
RAM_WARN_PCT: float = float(os.getenv("SCHEDULER_RAM_THRESHOLD", "80"))   # % used
RAM_CRIT_PCT: float = 92.0
SWAP_WARN_PCT: float = 40.0
GPU_WARN_PCT: float = float(os.getenv("SCHEDULER_GPU_THRESHOLD", "90"))
GPU_CRIT_PCT: float = 97.0
DISK_IO_WARN_MB_S: float = 200.0   # MB/s sustained read+write

LOG_DIR = Path("logs")
REPORTS_DIR = Path("reports")
EXPERIMENT_DIR = Path("experiment_tracker")
for _d in (LOG_DIR, REPORTS_DIR, EXPERIMENT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class AlertLevel(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class BottleneckType(str, Enum):
    NONE      = "none"
    CPU_BOUND = "cpu_bound"
    IO_BOUND  = "io_bound"
    MEM_BOUND = "mem_bound"
    GPU_BOUND = "gpu_bound"
    OOM_RISK  = "oom_risk"


class RunStatus(str, Enum):
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    ABORTED   = "aborted"


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CPUStats:
    """Per-core and aggregate CPU utilisation."""
    overall_pct: float
    per_core_pct: list[float]
    core_count_logical: int
    core_count_physical: int
    load_avg_1m: float
    load_avg_5m: float
    freq_mhz: float
    ctx_switches_per_s: float


@dataclass
class MemoryStats:
    """RAM and swap utilisation."""
    total_gb: float
    used_gb: float
    available_gb: float
    used_pct: float
    swap_total_gb: float
    swap_used_gb: float
    swap_pct: float


@dataclass
class DiskStats:
    """Disk I/O counters and throughput."""
    read_mb_s: float
    write_mb_s: float
    read_count_per_s: float
    write_count_per_s: float
    io_busy_pct: float


@dataclass
class GPUStats:
    """CUDA GPU memory utilisation (if available)."""
    available: bool
    device_name: str
    vram_total_gb: float
    vram_used_gb: float
    vram_reserved_gb: float
    vram_used_pct: float
    vram_reserved_pct: float


@dataclass
class ProcessInfo:
    """Snapshot of a single monitored process."""
    pid: int
    name: str
    cpu_pct: float
    mem_mb: float
    mem_pct: float
    status: str
    threads: int
    is_anomaly: bool = False


@dataclass
class SystemSnapshot:
    """
    Complete system telemetry snapshot — captured every POLL_INTERVAL_S.
    This is the core data structure streamed to the Streamlit dashboard.
    """
    timestamp: str
    cpu: CPUStats
    memory: MemoryStats
    disk: DiskStats
    gpu: GPUStats
    processes: list[ProcessInfo]
    bottleneck: BottleneckType
    bottleneck_detail: str


@dataclass
class Alert:
    """A structured alert event emitted by the monitor and consumed by the dashboard."""
    alert_id: str
    level: AlertLevel
    message: str
    metric: str
    value: float
    threshold: float
    timestamp: str
    bottleneck: BottleneckType


@dataclass
class ExperimentRun:
    """
    Tracks a single pipeline execution — analogous to a W&B run but built-in.
    Persisted as JSON under experiment_tracker/.
    """
    run_id: str
    run_name: str
    status: RunStatus
    started_at: str
    ended_at: str | None
    duration_s: float | None
    params: dict[str, Any]
    metrics: dict[str, list[float]]          # metric_name → time-series values
    resource_snapshots: list[dict]           # SystemSnapshot dicts at key moments
    bottlenecks_detected: list[str]
    alerts_fired: list[dict]
    notes: str


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING BASELINE (for 2σ anomaly detection)
# ─────────────────────────────────────────────────────────────────────────────

class RollingBaseline:
    """
    Maintains a fixed-size rolling window of metric samples and computes
    a 2σ anomaly flag for each new observation.

    Implements the OS-inspired principle of adaptive process accounting:
    the threshold is not static — it evolves with the system's observed
    normal behaviour, just as modern OS schedulers adjust to workload.
    """

    def __init__(self, window: int = ROLLING_WINDOW, sigma: float = ANOMALY_SIGMA) -> None:
        self._window = window
        self._sigma = sigma
        self._samples: Deque[float] = deque(maxlen=window)

    def update(self, value: float) -> bool:
        """
        Add a new sample. Returns True if the value is anomalous (> 2σ from mean).
        Requires at least 10 samples before flagging anomalies.
        """
        is_anomaly = False

        if len(self._samples) >= 10:
            mean = statistics.mean(self._samples)
            try:
                stdev = statistics.stdev(self._samples)
            except statistics.StatisticsError:
                stdev = 0.0

            if stdev > 0 and abs(value - mean) > self._sigma * stdev:
                is_anomaly = True

        self._samples.append(value)
        return is_anomaly

    @property
    def mean(self) -> float:
        return statistics.mean(self._samples) if self._samples else 0.0

    @property
    def stdev(self) -> float:
        try:
            return statistics.stdev(self._samples) if len(self._samples) >= 2 else 0.0
        except statistics.StatisticsError:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckDetector:
    """
    Classifies the current system state into a bottleneck type using
    heuristic rules derived from OS performance analysis methodology.

    Rules (evaluated in priority order):
      OOM_RISK  : RAM > 92% OR swap > 40%
      GPU_BOUND : GPU VRAM > 90%
      MEM_BOUND : RAM > 80% AND CPU < 60% (memory pressure without CPU load)
      IO_BOUND  : disk_io_busy > 70% AND CPU < 50% (classic I/O wait pattern)
      CPU_BOUND : CPU > 85% sustained
      NONE      : no significant pressure detected
    """

    def classify(
        self,
        snapshot: SystemSnapshot,
        sustained_cpu_high: bool = False,
    ) -> tuple[BottleneckType, str]:
        cpu = snapshot.cpu.overall_pct
        ram = snapshot.memory.used_pct
        swap = snapshot.memory.swap_pct
        gpu = snapshot.gpu.vram_used_pct if snapshot.gpu.available else 0.0
        disk_busy = snapshot.disk.io_busy_pct

        if ram > RAM_CRIT_PCT or swap > SWAP_WARN_PCT:
            return (
                BottleneckType.OOM_RISK,
                f"OOM risk: RAM {ram:.1f}% used"
                + (f", swap {swap:.1f}% used" if swap > SWAP_WARN_PCT else ""),
            )

        if snapshot.gpu.available and gpu > GPU_WARN_PCT:
            return (
                BottleneckType.GPU_BOUND,
                f"GPU VRAM saturated: {gpu:.1f}% used "
                f"({snapshot.gpu.vram_used_gb:.2f} / {snapshot.gpu.vram_total_gb:.2f} GB)",
            )

        if ram > RAM_WARN_PCT and cpu < 60.0:
            return (
                BottleneckType.MEM_BOUND,
                f"Memory pressure: RAM {ram:.1f}% used, CPU only {cpu:.1f}% "
                f"— pipeline likely stalled on memory allocation",
            )

        if disk_busy > 70.0 and cpu < 50.0:
            return (
                BottleneckType.IO_BOUND,
                f"I/O bottleneck: disk busy {disk_busy:.1f}%, CPU {cpu:.1f}% "
                f"— likely embedding cache miss or large file read",
            )

        if sustained_cpu_high or cpu > CPU_CRIT_PCT:
            return (
                BottleneckType.CPU_BOUND,
                f"CPU saturated: {cpu:.1f}% overall "
                f"(cores: {[f'{c:.0f}%' for c in snapshot.cpu.per_core_pct]})",
            )

        return BottleneckType.NONE, "System resources within normal operating range"


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM MONITOR (background thread)
# ─────────────────────────────────────────────────────────────────────────────

class SystemMonitor:
    """
    Background daemon thread that polls all system metrics every
    POLL_INTERVAL_S seconds and writes to a shared snapshot buffer.

    Implements the producer side of a producer-consumer pattern:
      - Produces: SystemSnapshot objects → _latest_snapshot
      - Produces: Alert events          → alert_queue (thread-safe)

    This is analogous to an OS kernel's periodic accounting interrupt
    (e.g. Linux's scheduler_tick()) that samples process statistics.
    """

    def __init__(self, alert_queue: queue.Queue) -> None:
        self._alert_queue = alert_queue
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_snapshot: SystemSnapshot | None = None
        self._thread: threading.Thread | None = None

        # Rolling baselines for anomaly detection — one per key metric
        self._baselines: dict[str, RollingBaseline] = {
            "cpu_overall": RollingBaseline(),
            "ram_used_pct": RollingBaseline(),
            "disk_read_mb": RollingBaseline(),
            "disk_write_mb": RollingBaseline(),
            "gpu_vram_pct": RollingBaseline(),
        }

        # For sustained CPU detection (CPU > threshold for N consecutive polls)
        self._cpu_high_streak: int = 0
        self._cpu_high_streak_threshold: int = 3   # ~6 seconds at 2s poll

        # Disk I/O delta tracking (counters are cumulative)
        self._prev_disk_counters = psutil.disk_io_counters()
        self._prev_disk_time = time.time()
        self._prev_ctx_switches = psutil.cpu_stats().ctx_switches

        # Process baseline (cpu% per PID for anomaly detection)
        self._process_baselines: dict[int, RollingBaseline] = {}

        # Own PID + children
        self._own_pid = os.getpid()
        self._own_process = psutil.Process(self._own_pid)

        self._detector = BottleneckDetector()

        logger.info("SystemMonitor initialised | PID=%d | poll=%.1fs", self._own_pid, POLL_INTERVAL_S)

    # ── THREAD CONTROL ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background polling thread."""
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="researchflow-monitor",
            daemon=True,          # dies automatically when main process exits
        )
        self._thread.start()
        logger.info("SystemMonitor started (daemon thread: %s)", self._thread.name)

    def stop(self) -> None:
        """Signal the polling thread to exit gracefully."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=POLL_INTERVAL_S * 2)
        logger.info("SystemMonitor stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── SNAPSHOT ACCESS ──────────────────────────────────────────────────────

    def latest_snapshot(self) -> SystemSnapshot | None:
        """Thread-safe read of the most recent system snapshot."""
        with self._lock:
            return self._latest_snapshot

    # ── INTERNAL POLLING LOOP ────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """
        Main background loop. Runs every POLL_INTERVAL_S.
        Collects all metrics, builds a SystemSnapshot, detects bottlenecks,
        checks alert thresholds, and updates the shared snapshot buffer.
        """
        logger.debug("Monitor poll loop started")

        while not self._stop_event.is_set():
            loop_start = time.time()

            try:
                snapshot = self._collect_snapshot()
                with self._lock:
                    self._latest_snapshot = snapshot
                self._check_alerts(snapshot)

            except Exception as exc:
                logger.error("Monitor poll error: %s", exc, exc_info=True)

            # Sleep for remainder of poll interval (accurate timing)
            elapsed = time.time() - loop_start
            sleep_time = max(0.0, POLL_INTERVAL_S - elapsed)
            self._stop_event.wait(timeout=sleep_time)

        logger.debug("Monitor poll loop exited")

    # ── METRIC COLLECTION ────────────────────────────────────────────────────

    def _collect_cpu(self) -> CPUStats:
        overall = psutil.cpu_percent(interval=None)
        per_core = psutil.cpu_percent(percpu=True, interval=None)
        count_log = psutil.cpu_count(logical=True)
        count_phy = psutil.cpu_count(logical=False) or 1

        load = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0.0, 0.0, 0.0)
        freq = psutil.cpu_freq()
        freq_mhz = freq.current if freq else 0.0

        now_ctx = psutil.cpu_stats().ctx_switches
        ctx_delta = max(0, now_ctx - self._prev_ctx_switches)
        self._prev_ctx_switches = now_ctx

        return CPUStats(
            overall_pct=overall,
            per_core_pct=list(per_core),
            core_count_logical=count_log or 1,
            core_count_physical=count_phy,
            load_avg_1m=load[0],
            load_avg_5m=load[1],
            freq_mhz=freq_mhz,
            ctx_switches_per_s=ctx_delta / POLL_INTERVAL_S,
        )

    def _collect_memory(self) -> MemoryStats:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        def to_gb(b: int) -> float:
            return round(b / (1024 ** 3), 3)

        return MemoryStats(
            total_gb=to_gb(vm.total),
            used_gb=to_gb(vm.used),
            available_gb=to_gb(vm.available),
            used_pct=vm.percent,
            swap_total_gb=to_gb(sw.total),
            swap_used_gb=to_gb(sw.used),
            swap_pct=sw.percent,
        )

    def _collect_disk(self) -> DiskStats:
        now = time.time()
        dt = now - self._prev_disk_time
        if dt <= 0:
            dt = POLL_INTERVAL_S

        try:
            curr = psutil.disk_io_counters()
        except Exception:
            return DiskStats(0, 0, 0, 0, 0)

        if curr is None or self._prev_disk_counters is None:
            self._prev_disk_counters = curr
            self._prev_disk_time = now
            return DiskStats(0, 0, 0, 0, 0)

        def mb(b: int) -> float:
            return b / (1024 ** 2)

        read_mb_s  = mb(curr.read_bytes  - self._prev_disk_counters.read_bytes)  / dt
        write_mb_s = mb(curr.write_bytes - self._prev_disk_counters.write_bytes) / dt
        read_c_s   = (curr.read_count  - self._prev_disk_counters.read_count)  / dt
        write_c_s  = (curr.write_count - self._prev_disk_counters.write_count) / dt

        # io_busy_pct: fraction of time disk was busy (Linux: busy_time field)
        busy_pct = 0.0
        if hasattr(curr, "busy_time") and hasattr(self._prev_disk_counters, "busy_time"):
            delta_busy_ms = curr.busy_time - self._prev_disk_counters.busy_time
            busy_pct = min(100.0, (delta_busy_ms / 1000.0) / dt * 100.0)

        self._prev_disk_counters = curr
        self._prev_disk_time = now

        return DiskStats(
            read_mb_s=round(max(0.0, read_mb_s), 3),
            write_mb_s=round(max(0.0, write_mb_s), 3),
            read_count_per_s=round(max(0.0, read_c_s), 1),
            write_count_per_s=round(max(0.0, write_c_s), 1),
            io_busy_pct=round(busy_pct, 1),
        )

    def _collect_gpu(self) -> GPUStats:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return GPUStats(
                available=False,
                device_name="N/A",
                vram_total_gb=0.0,
                vram_used_gb=0.0,
                vram_reserved_gb=0.0,
                vram_used_pct=0.0,
                vram_reserved_pct=0.0,
            )

        def to_gb(b: int) -> float:
            return round(b / (1024 ** 3), 3)

        device = torch.cuda.current_device()
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory
        mem_stats = torch.cuda.memory_stats(device)
        allocated = mem_stats.get("allocated_bytes.all.current", 0)
        reserved  = mem_stats.get("reserved_bytes.all.current", 0)

        used_pct = (allocated / total * 100) if total > 0 else 0.0
        reserved_pct = (reserved / total * 100) if total > 0 else 0.0

        return GPUStats(
            available=True,
            device_name=name,
            vram_total_gb=to_gb(total),
            vram_used_gb=to_gb(allocated),
            vram_reserved_gb=to_gb(reserved),
            vram_used_pct=round(used_pct, 2),
            vram_reserved_pct=round(reserved_pct, 2),
        )

    def _collect_processes(self) -> list[ProcessInfo]:
        """
        Collects CPU/memory stats for the top 10 processes by CPU usage,
        plus ResearchFlow's own process tree. Applies 2σ anomaly detection
        per PID.
        """
        proc_infos: list[ProcessInfo] = []

        try:
            # Snapshot all processes (fast — one pass)
            procs = [
                p.info for p in psutil.process_iter(
                    ["pid", "name", "cpu_percent", "memory_info",
                     "memory_percent", "status", "num_threads"]
                )
                if p.info["cpu_percent"] is not None
            ]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

        # Sort by CPU% descending, take top 10 + our own process
        own_pids = self._get_own_pids()
        top_procs = sorted(procs, key=lambda p: p["cpu_percent"] or 0, reverse=True)[:10]
        own_procs = [p for p in procs if p["pid"] in own_pids]
        seen_pids: set[int] = set()

        for info in top_procs + own_procs:
            pid = info["pid"]
            if pid in seen_pids or pid is None:
                continue
            seen_pids.add(pid)

            cpu_pct = info["cpu_percent"] or 0.0
            mem_info = info["memory_info"]
            mem_mb = (mem_info.rss / (1024 ** 2)) if mem_info else 0.0
            mem_pct = info["memory_percent"] or 0.0

            # Per-PID anomaly detection
            if pid not in self._process_baselines:
                self._process_baselines[pid] = RollingBaseline()
            is_anomaly = self._process_baselines[pid].update(cpu_pct)

            proc_infos.append(ProcessInfo(
                pid=pid,
                name=(info["name"] or "?")[:30],
                cpu_pct=round(cpu_pct, 2),
                mem_mb=round(mem_mb, 2),
                mem_pct=round(mem_pct, 3),
                status=info["status"] or "?",
                threads=info["num_threads"] or 1,
                is_anomaly=is_anomaly,
            ))

        return proc_infos

    def _get_own_pids(self) -> set[int]:
        """Returns PIDs of own process + all child processes."""
        pids = {self._own_pid}
        try:
            children = self._own_process.children(recursive=True)
            pids.update(c.pid for c in children)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return pids

    def _collect_snapshot(self) -> SystemSnapshot:
        """Assembles a full SystemSnapshot from all collectors."""
        ts = datetime.utcnow().isoformat()

        cpu    = self._collect_cpu()
        memory = self._collect_memory()
        disk   = self._collect_disk()
        gpu    = self._collect_gpu()
        procs  = self._collect_processes()

        # Update rolling baselines
        self._baselines["cpu_overall"].update(cpu.overall_pct)
        self._baselines["ram_used_pct"].update(memory.used_pct)
        self._baselines["disk_read_mb"].update(disk.read_mb_s)
        self._baselines["disk_write_mb"].update(disk.write_mb_s)
        if gpu.available:
            self._baselines["gpu_vram_pct"].update(gpu.vram_used_pct)

        # Sustained CPU high-load streak
        if cpu.overall_pct > CPU_WARN_PCT:
            self._cpu_high_streak += 1
        else:
            self._cpu_high_streak = 0
        sustained_cpu = self._cpu_high_streak >= self._cpu_high_streak_threshold

        # Detect bottleneck
        snapshot_partial = SystemSnapshot(
            timestamp=ts, cpu=cpu, memory=memory, disk=disk, gpu=gpu,
            processes=procs, bottleneck=BottleneckType.NONE, bottleneck_detail="",
        )
        bottleneck, detail = self._detector.classify(snapshot_partial, sustained_cpu)

        snapshot_partial.bottleneck = bottleneck
        snapshot_partial.bottleneck_detail = detail

        return snapshot_partial

    # ── ALERT CHECKING ───────────────────────────────────────────────────────

    def _emit_alert(
        self,
        level: AlertLevel,
        message: str,
        metric: str,
        value: float,
        threshold: float,
        bottleneck: BottleneckType = BottleneckType.NONE,
    ) -> None:
        alert = Alert(
            alert_id=str(uuid.uuid4())[:8],
            level=level,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold,
            timestamp=datetime.utcnow().isoformat(),
            bottleneck=bottleneck,
        )
        try:
            self._alert_queue.put_nowait(alert)
        except queue.Full:
            logger.warning("Alert queue full — dropping alert: %s", message)

        log_fn = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.CRITICAL: logger.critical,
        }[level]
        log_fn("[ALERT %s] %s (%.1f%% > threshold %.1f%%)", level.value, message, value, threshold)

    def _check_alerts(self, snapshot: SystemSnapshot) -> None:
        """Evaluates threshold rules and emits alerts when breached."""
        cpu = snapshot.cpu.overall_pct
        ram = snapshot.memory.used_pct
        swap = snapshot.memory.swap_pct
        gpu = snapshot.gpu.vram_used_pct

        # CPU alerts
        if cpu > CPU_CRIT_PCT:
            self._emit_alert(AlertLevel.CRITICAL, f"CPU critical: {cpu:.1f}%", "cpu", cpu, CPU_CRIT_PCT, BottleneckType.CPU_BOUND)
        elif cpu > CPU_WARN_PCT and self._cpu_high_streak >= self._cpu_high_streak_threshold:
            self._emit_alert(AlertLevel.WARNING, f"CPU sustained high: {cpu:.1f}% for {self._cpu_high_streak * POLL_INTERVAL_S:.0f}s", "cpu", cpu, CPU_WARN_PCT, BottleneckType.CPU_BOUND)

        # Memory alerts
        if ram > RAM_CRIT_PCT:
            self._emit_alert(AlertLevel.CRITICAL, f"RAM critical: {ram:.1f}% used ({snapshot.memory.used_gb:.2f}/{snapshot.memory.total_gb:.2f} GB)", "ram", ram, RAM_CRIT_PCT, BottleneckType.OOM_RISK)
        elif ram > RAM_WARN_PCT:
            self._emit_alert(AlertLevel.WARNING, f"RAM high: {ram:.1f}% used", "ram", ram, RAM_WARN_PCT, BottleneckType.MEM_BOUND)

        # Swap alert
        if swap > SWAP_WARN_PCT:
            self._emit_alert(AlertLevel.WARNING, f"Swap usage: {swap:.1f}% — system under memory pressure", "swap", swap, SWAP_WARN_PCT, BottleneckType.OOM_RISK)

        # GPU alerts
        if snapshot.gpu.available:
            if gpu > GPU_CRIT_PCT:
                self._emit_alert(AlertLevel.CRITICAL, f"GPU VRAM critical: {gpu:.1f}% ({snapshot.gpu.vram_used_gb:.2f}/{snapshot.gpu.vram_total_gb:.2f} GB)", "gpu_vram", gpu, GPU_CRIT_PCT, BottleneckType.GPU_BOUND)
            elif gpu > GPU_WARN_PCT:
                self._emit_alert(AlertLevel.WARNING, f"GPU VRAM high: {gpu:.1f}%", "gpu_vram", gpu, GPU_WARN_PCT, BottleneckType.GPU_BOUND)

        # Anomaly alerts (2σ detection)
        for metric_name, baseline in self._baselines.items():
            if len(baseline._samples) < 10:
                continue
            current_val = {
                "cpu_overall":   cpu,
                "ram_used_pct":  ram,
                "disk_read_mb":  snapshot.disk.read_mb_s,
                "disk_write_mb": snapshot.disk.write_mb_s,
                "gpu_vram_pct":  gpu,
            }.get(metric_name, 0.0)

            if baseline.stdev > 0 and abs(current_val - baseline.mean) > ANOMALY_SIGMA * baseline.stdev:
                self._emit_alert(
                    AlertLevel.INFO,
                    f"Anomaly: {metric_name} = {current_val:.1f} (mean={baseline.mean:.1f}, σ={baseline.stdev:.1f})",
                    metric_name, current_val, baseline.mean + ANOMALY_SIGMA * baseline.stdev,
                )


# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class ExperimentTracker:
    """
    Lightweight experiment management — tracks pipeline runs with:
      - Parameters and configuration
      - Time-series metric logging
      - Resource snapshots at begin/end and on demand
      - Run status lifecycle (running → completed / failed / aborted)
      - JSON persistence to experiment_tracker/

    Designed to be used without any external service (no W&B required).
    """

    def __init__(self) -> None:
        self._active_run: ExperimentRun | None = None
        self._lock = threading.Lock()

    def begin_run(
        self,
        run_name: str,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Start a new experiment run. Returns the run_id."""
        with self._lock:
            if self._active_run and self._active_run.status == RunStatus.RUNNING:
                logger.warning("Closing previous run '%s' before starting new one", self._active_run.run_id)
                self._finalise(RunStatus.ABORTED)

            run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            self._active_run = ExperimentRun(
                run_id=run_id,
                run_name=run_name,
                status=RunStatus.RUNNING,
                started_at=datetime.utcnow().isoformat(),
                ended_at=None,
                duration_s=None,
                params=params or {},
                metrics={},
                resource_snapshots=[],
                bottlenecks_detected=[],
                alerts_fired=[],
                notes="",
            )
            logger.info("ExperimentTracker: run started — %s (%s)", run_id, run_name)
            return run_id

    def record_metric(self, name: str, value: float) -> None:
        """Append a scalar metric value to the active run's time series."""
        with self._lock:
            if self._active_run is None:
                return
            self._active_run.metrics.setdefault(name, []).append(round(value, 6))

    def record_snapshot(self, snapshot: SystemSnapshot) -> None:
        """Capture a resource snapshot at this moment in the active run."""
        with self._lock:
            if self._active_run is None:
                return
            self._active_run.resource_snapshots.append(asdict(snapshot))

    def record_bottleneck(self, bottleneck_type: BottleneckType, detail: str) -> None:
        with self._lock:
            if self._active_run is None:
                return
            entry = f"{datetime.utcnow().isoformat()} | {bottleneck_type.value}: {detail}"
            self._active_run.bottlenecks_detected.append(entry)

    def record_alert(self, alert: Alert) -> None:
        with self._lock:
            if self._active_run is None:
                return
            self._active_run.alerts_fired.append(asdict(alert))

    def end_run(self, notes: str = "") -> ExperimentRun | None:
        """Mark the active run as completed and persist it to disk."""
        with self._lock:
            return self._finalise(RunStatus.COMPLETED, notes)

    def fail_run(self, reason: str = "") -> ExperimentRun | None:
        """Mark the active run as failed and persist it to disk."""
        with self._lock:
            return self._finalise(RunStatus.FAILED, reason)

    def _finalise(self, status: RunStatus, notes: str = "") -> ExperimentRun | None:
        """Internal: sets end time, status, notes, and saves to JSON."""
        if self._active_run is None:
            return None

        end_time = datetime.utcnow()
        start_time = datetime.fromisoformat(self._active_run.started_at)
        self._active_run.ended_at = end_time.isoformat()
        self._active_run.duration_s = round((end_time - start_time).total_seconds(), 3)
        self._active_run.status = status
        self._active_run.notes = notes

        run = self._active_run
        self._active_run = None

        self._persist(run)
        logger.info(
            "ExperimentTracker: run %s — %s in %.1fs",
            run.run_id, status.value, run.duration_s,
        )
        return run

    def _persist(self, run: ExperimentRun) -> None:
        """Write run record to experiment_tracker/<run_id>.json"""
        path = EXPERIMENT_DIR / f"{run.run_id}.json"
        try:
            path.write_text(json.dumps(asdict(run), indent=2, default=str))
            logger.debug("Run persisted to %s", path)
        except Exception as exc:
            logger.error("Failed to persist run %s: %s", run.run_id, exc)

    def list_runs(self) -> list[dict]:
        """Load and return all persisted run summaries from disk."""
        runs = []
        for path in sorted(EXPERIMENT_DIR.glob("run_*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                runs.append({
                    "run_id": data["run_id"],
                    "run_name": data["run_name"],
                    "status": data["status"],
                    "started_at": data["started_at"],
                    "duration_s": data.get("duration_s"),
                    "n_metrics": sum(len(v) for v in data.get("metrics", {}).values()),
                    "n_alerts": len(data.get("alerts_fired", [])),
                    "bottlenecks": len(data.get("bottlenecks_detected", [])),
                })
            except Exception:
                continue
        return runs

    @property
    def active_run_id(self) -> str | None:
        return self._active_run.run_id if self._active_run else None


# ─────────────────────────────────────────────────────────────────────────────
# LAB AGENT (public API)
# ─────────────────────────────────────────────────────────────────────────────

class LabAgent:
    """
    Top-level Lab Assistant Agent — the single entry point for the
    Streamlit dashboard and SchedulerAgent to interact with.

    Composes:
      - SystemMonitor  : background telemetry thread
      - ExperimentTracker : run lifecycle management
      - Alert queue    : thread-safe event bus (monitor → dashboard)

    The LabAgent exposes both a synchronous API (for Streamlit callbacks)
    and an async API (for the SchedulerAgent's resource checks).
    """

    def __init__(self) -> None:
        self._alert_queue: queue.Queue = queue.Queue(maxsize=200)
        self._monitor = SystemMonitor(self._alert_queue)
        self._tracker = ExperimentTracker()
        self._report_log: list[dict] = []

        logger.info("LabAgent initialised")

    # ── LIFECYCLE ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background system monitor. Call once at app startup."""
        self._monitor.start()
        logger.info("LabAgent started — monitor is live")

    def stop(self) -> None:
        """Graceful shutdown of the background monitor thread."""
        self._monitor.stop()
        logger.info("LabAgent stopped")

    # ── TELEMETRY API ────────────────────────────────────────────────────────

    def snapshot(self) -> SystemSnapshot | None:
        """
        Returns the most recent SystemSnapshot (non-blocking).
        Safe to call from Streamlit's main thread.
        """
        return self._monitor.latest_snapshot()

    def drain_alerts(self, max_items: int = 50) -> list[Alert]:
        """
        Drains up to max_items alerts from the alert queue.
        Called by the Streamlit dashboard on each refresh cycle.
        """
        alerts: list[Alert] = []
        for _ in range(max_items):
            try:
                alerts.append(self._alert_queue.get_nowait())
            except queue.Empty:
                break
        return alerts

    def is_safe_to_dispatch(
        self,
        task_cpu_cost: str = "medium",
        task_ram_cost: str = "medium",
        task_gpu_cost: str = "none",
    ) -> tuple[bool, str]:
        """
        Called by SchedulerAgent before dispatching a task.
        Returns (safe: bool, reason: str).

        Cost levels: "low" | "medium" | "high" | "none"

        This is the feedback interface between the OS monitoring layer
        and the task scheduling layer — the mechanism that makes the
        scheduler genuinely resource-aware rather than cosmetically so.
        """
        snap = self.snapshot()
        if snap is None:
            return True, "No telemetry yet — dispatching optimistically"

        cpu = snap.cpu.overall_pct
        ram_avail_pct = 100.0 - snap.memory.used_pct
        gpu = snap.gpu.vram_used_pct if snap.gpu.available else 0.0

        # High-cost CPU task: require at least 20% headroom
        if task_cpu_cost == "high" and cpu > CPU_WARN_PCT:
            return False, f"CPU too high for high-cost task: {cpu:.1f}% > {CPU_WARN_PCT}%"

        # High-cost RAM task: require at least 25% available
        if task_ram_cost == "high" and ram_avail_pct < 25.0:
            return False, f"Insufficient RAM for high-cost task: {ram_avail_pct:.1f}% available < 25%"

        # Medium RAM task: require at least 15% available
        if task_ram_cost == "medium" and ram_avail_pct < 15.0:
            return False, f"RAM too tight for medium task: {ram_avail_pct:.1f}% available < 15%"

        # GPU task: require at least 15% VRAM headroom
        if task_gpu_cost in ("medium", "high") and snap.gpu.available and gpu > GPU_WARN_PCT:
            return False, f"GPU VRAM too high: {gpu:.1f}% > {GPU_WARN_PCT}%"

        # OOM risk: block everything except low-cost tasks
        if snap.bottleneck == BottleneckType.OOM_RISK and task_ram_cost != "low":
            return False, f"OOM risk active — only low-RAM tasks allowed. {snap.bottleneck_detail}"

        return True, "Resources within acceptable thresholds"

    # ── EXPERIMENT TRACKING API ──────────────────────────────────────────────

    def begin_run(self, run_name: str, params: dict[str, Any] | None = None) -> str:
        """Start tracking a pipeline experiment run. Returns run_id."""
        run_id = self._tracker.begin_run(run_name, params)
        snap = self.snapshot()
        if snap:
            self._tracker.record_snapshot(snap)
        return run_id

    def record_metric(self, name: str, value: float) -> None:
        """Log a scalar metric to the active run."""
        self._tracker.record_metric(name, value)

    def checkpoint(self) -> None:
        """
        Capture a resource snapshot at this point in the active run.
        Call at the end of each pipeline stage (e.g. after embedding,
        after clustering) to record per-stage resource consumption.
        """
        snap = self.snapshot()
        if snap:
            self._tracker.record_snapshot(snap)
            if snap.bottleneck != BottleneckType.NONE:
                self._tracker.record_bottleneck(snap.bottleneck, snap.bottleneck_detail)

    def end_run(self, notes: str = "") -> ExperimentRun | None:
        """Finalise the active run as completed."""
        snap = self.snapshot()
        if snap:
            self._tracker.record_snapshot(snap)
        return self._tracker.end_run(notes)

    def fail_run(self, reason: str = "") -> ExperimentRun | None:
        """Finalise the active run as failed."""
        return self._tracker.fail_run(reason)

    def list_runs(self) -> list[dict]:
        """Return all persisted run summaries."""
        return self._tracker.list_runs()

    # ── REPORT GENERATION ────────────────────────────────────────────────────

    def generate_experiment_report(
        self,
        run: ExperimentRun | None = None,
        output_format: str = "markdown",
    ) -> str:
        """
        Generates a structured experiment report for the given run
        (or the most recently completed run if none provided).

        Includes: run metadata, resource utilisation summary,
        bottleneck timeline, alert summary, metric trends.

        Returns the report as a Markdown string (or JSON if requested).
        """
        if run is None:
            runs = self._tracker.list_runs()
            if not runs:
                return "No experiment runs found."
            latest_id = runs[0]["run_id"]
            path = EXPERIMENT_DIR / f"{latest_id}.json"
            if not path.exists():
                return f"Run file not found: {path}"
            run_data = json.loads(path.read_text())
        else:
            run_data = asdict(run)

        if output_format == "json":
            return json.dumps(run_data, indent=2, default=str)

        # ── Markdown report ──────────────────────────────────────────────────
        lines: list[str] = [
            "# ResearchFlow AI — Experiment Report",
            "",
            f"**Run ID:** `{run_data['run_id']}`  ",
            f"**Run Name:** {run_data['run_name']}  ",
            f"**Status:** {run_data['status']}  ",
            f"**Started:** {run_data['started_at']}  ",
            f"**Duration:** {run_data.get('duration_s', 'N/A')}s  ",
            "",
            "---",
            "",
            "## Parameters",
            "",
        ]

        if run_data.get("params"):
            for k, v in run_data["params"].items():
                lines.append(f"- **{k}**: `{v}`")
        else:
            lines.append("_No parameters recorded._")

        lines += ["", "---", "", "## Metrics Summary", ""]
        metrics = run_data.get("metrics", {})
        if metrics:
            lines.append("| Metric | Min | Max | Final | Samples |")
            lines.append("|--------|-----|-----|-------|---------|")
            for name, values in metrics.items():
                if values:
                    lines.append(
                        f"| {name} | {min(values):.4f} | {max(values):.4f} "
                        f"| {values[-1]:.4f} | {len(values)} |"
                    )
        else:
            lines.append("_No metrics recorded._")

        lines += ["", "---", "", "## Resource Utilisation", ""]
        snapshots = run_data.get("resource_snapshots", [])
        if snapshots:
            cpu_vals = [s["cpu"]["overall_pct"] for s in snapshots]
            ram_vals = [s["memory"]["used_pct"] for s in snapshots]

            lines += [
                f"| Resource | Min | Max | Avg |",
                f"|----------|-----|-----|-----|",
                f"| CPU %    | {min(cpu_vals):.1f}% | {max(cpu_vals):.1f}% | {statistics.mean(cpu_vals):.1f}% |",
                f"| RAM %    | {min(ram_vals):.1f}% | {max(ram_vals):.1f}% | {statistics.mean(ram_vals):.1f}% |",
            ]

            gpu_vals = [
                s["gpu"]["vram_used_pct"] for s in snapshots
                if s["gpu"]["available"]
            ]
            if gpu_vals:
                lines.append(
                    f"| GPU VRAM | {min(gpu_vals):.1f}% | {max(gpu_vals):.1f}% | {statistics.mean(gpu_vals):.1f}% |"
                )
        else:
            lines.append("_No resource snapshots recorded._")

        lines += ["", "---", "", "## Bottlenecks Detected", ""]
        bottlenecks = run_data.get("bottlenecks_detected", [])
        if bottlenecks:
            for b in bottlenecks:
                lines.append(f"- {b}")
        else:
            lines.append("_No bottlenecks detected during this run._")

        lines += ["", "---", "", "## Alerts Fired", ""]
        alerts = run_data.get("alerts_fired", [])
        if alerts:
            lines.append("| Time | Level | Metric | Value | Message |")
            lines.append("|------|-------|--------|-------|---------|")
            for a in alerts[:20]:
                ts = a.get("timestamp", "")[:19]
                lines.append(
                    f"| {ts} | {a['level']} | {a['metric']} "
                    f"| {a['value']:.1f}% | {a['message'][:60]} |"
                )
            if len(alerts) > 20:
                lines.append(f"\n_...and {len(alerts) - 20} more alerts._")
        else:
            lines.append("_No alerts fired during this run._")

        if run_data.get("notes"):
            lines += ["", "---", "", "## Notes", "", run_data["notes"]]

        lines += ["", "---", "", "*Generated by ResearchFlow AI Lab Agent*"]

        report_text = "\n".join(lines)

        # Persist report to disk
        report_path = REPORTS_DIR / f"{run_data['run_id']}_lab_report.md"
        try:
            report_path.write_text(report_text)
            logger.info("Lab report written to %s", report_path)
        except Exception as exc:
            logger.warning("Could not write lab report: %s", exc)

        return report_text

    # ── STATUS SUMMARY (for Streamlit sidebar) ───────────────────────────────

    def status_summary(self) -> dict[str, Any]:
        """
        Returns a compact dict suitable for the Streamlit sidebar status panel.
        Non-blocking and safe to call on every Streamlit rerun.
        """
        snap = self.snapshot()
        if snap is None:
            return {"status": "initialising", "monitor_live": False}

        return {
            "status": "live",
            "monitor_live": self._monitor.is_running,
            "timestamp": snap.timestamp,
            "cpu_pct": snap.cpu.overall_pct,
            "ram_pct": snap.memory.used_pct,
            "ram_avail_gb": snap.memory.available_gb,
            "gpu_available": snap.gpu.available,
            "gpu_vram_pct": snap.gpu.vram_used_pct if snap.gpu.available else None,
            "gpu_name": snap.gpu.device_name if snap.gpu.available else None,
            "bottleneck": snap.bottleneck.value,
            "bottleneck_detail": snap.bottleneck_detail,
            "active_run_id": self._tracker.active_run_id,
            "pending_alerts": self._alert_queue.qsize(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / QUICK TEST
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Quick smoke-test — run with: python -m agents.lab_agent
    Starts the monitor, simulates a pipeline run, prints a report.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    agent = LabAgent()
    agent.start()

    print("\nMonitor warming up (4 seconds)...")
    time.sleep(4)

    # Simulate a pipeline run
    run_id = agent.begin_run(
        "demo_pipeline",
        params={"query": "Multimodal LLMs", "max_papers": 50, "embed_model": "SPECTER"},
    )
    print(f"\nRun started: {run_id}")

    for i in range(6):
        agent.record_metric("papers_processed", float(i * 10))
        agent.record_metric("clusters_found", float(i // 2))
        agent.checkpoint()
        time.sleep(1)

    run = agent.end_run(notes="Demo run completed successfully")

    # Print live status
    summary = agent.status_summary()
    print("\n── LIVE STATUS ──")
    for k, v in summary.items():
        print(f"  {k:<22} {v}")

    # Drain and print alerts
    alerts = agent.drain_alerts()
    if alerts:
        print(f"\n── ALERTS ({len(alerts)}) ──")
        for a in alerts:
            print(f"  [{a.level.value}] {a.message}")

    # Generate and print report
    if run:
        report = agent.generate_experiment_report(run)
        print("\n── EXPERIMENT REPORT (first 40 lines) ──")
        for line in report.split("\n")[:40]:
            print(" ", line)

    agent.stop()
    print("\nLabAgent demo complete.")


if __name__ == "__main__":
    _demo()
