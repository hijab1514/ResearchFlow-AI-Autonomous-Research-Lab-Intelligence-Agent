"""
monitoring/process_tracker.py
==============================
ResearchFlow AI — Process Tree Tracker & Anomaly Detector

This module tracks ResearchFlow AI's own process tree (main PID +
all spawned children) and applies 2-sigma statistical anomaly
detection to flag processes that suddenly spike in CPU or memory.

Why process-level tracking matters:
  The SchedulerAgent spawns worker threads (and occasionally subprocesses)
  for embedding generation, FAISS operations, and LLM calls. Without
  process-level accounting, the system-level CPU/RAM metrics in
  SystemMonitor can't distinguish between:
    - ResearchFlow's own embedding worker using 80% CPU (expected)
    - A runaway subprocess leaking memory (anomaly)
    - Another user process competing for resources (external noise)

  ProcessTracker answers: "of the total CPU load, how much is US,
  and is any of our processes behaving unusually?"

OS Concepts demonstrated:
  ┌───────────────────────────────────────────────────────────────────┐
  │  Concept                    │  Implementation                    │
  ├─────────────────────────────┼────────────────────────────────────┤
  │  Process accounting         │  psutil.Process per PID            │
  │  Process tree               │  recursive children() traversal    │
  │  PID namespace              │  os.getpid() + child PID tracking  │
  │  Zombie detection           │  status == "zombie" check          │
  │  Resource limits            │  per-process CPU/RAM thresholds    │
  │  Statistical anomaly det.   │  rolling 2σ baseline per PID       │
  │  Process lifecycle          │  birth / exit detection per tick   │
  │  Context switch accounting  │  voluntary + involuntary ctxsw     │
  └───────────────────────────────────────────────────────────────────┘

Key classes:
  ProcessSnapshot     — single-poll snapshot of one process
  ProcessBaseline     — rolling 2σ baseline for a PID
  ProcessTracker      — orchestrates tree traversal + anomaly detection

Anomaly detection algorithm:
  For each tracked process, maintain a rolling window of the last
  BASELINE_WINDOW CPU% samples. On each new reading:
    z_score = (current - mean) / stdev
    is_anomaly = abs(z_score) > ANOMALY_SIGMA (default 2.0)

  This is the same principle used by Linux's CFS scheduler to detect
  unexpectedly long CPU bursts and in APM tools like Datadog for
  process-level anomaly alerting.

Usage:
    tracker = ProcessTracker()
    tracker.start()

    # Get snapshot of own process tree
    report = tracker.report()
    for proc in report.processes:
        if proc.is_anomaly:
            print(f"Anomaly: {proc.name} CPU={proc.cpu_pct:.1f}%")

    # Check for zombies
    zombies = tracker.zombies()

    tracker.stop()

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import logging
import os
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Deque

import psutil

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL_S:  float = float(os.getenv("PROCESS_POLL_INTERVAL", "3.0"))
BASELINE_WINDOW:  int   = int(os.getenv("PROCESS_BASELINE_WINDOW", "20"))
ANOMALY_SIGMA:    float = float(os.getenv("PROCESS_ANOMALY_SIGMA",  "2.0"))
MIN_SAMPLES:      int   = 5     # minimum samples before anomaly detection fires

# Per-process thresholds for hard alerts (independent of baseline)
CPU_HARD_LIMIT_PCT:  float = 95.0
MEM_HARD_LIMIT_MB:   float = 4096.0   # 4 GB per process
ZOMBIE_CHECK:        bool  = True


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProcessSnapshot:
    """
    Single-poll telemetry snapshot for one process.
    Annotated with anomaly flags from ProcessBaseline.
    """
    pid:               int
    name:              str
    status:            str        # running | sleeping | disk-sleep | zombie | ...
    cpu_pct:           float      # CPU utilisation % (interval-averaged)
    mem_rss_mb:        float      # resident set size (physical RAM) in MB
    mem_vms_mb:        float      # virtual memory size in MB
    mem_pct:           float      # RSS as % of total system RAM
    threads:           int        # thread count
    open_files:        int        # open file descriptor count
    ctx_vol:           int        # voluntary context switches (cumulative)
    ctx_invol:         int        # involuntary context switches (cumulative)
    create_time:       float      # process creation epoch timestamp
    cmdline:           str        # first 80 chars of command line
    is_own:            bool       # True if this is our main PID or a child
    is_zombie:         bool       # True if status == "zombie"

    # Anomaly detection flags (set by ProcessBaseline)
    is_cpu_anomaly:    bool  = False
    is_mem_anomaly:    bool  = False
    cpu_zscore:        float = 0.0
    mem_zscore:        float = 0.0
    cpu_baseline_mean: float = 0.0
    mem_baseline_mean: float = 0.0

    @property
    def is_anomaly(self) -> bool:
        return self.is_cpu_anomaly or self.is_mem_anomaly

    @property
    def age_s(self) -> float:
        return time.time() - self.create_time

    def to_dict(self) -> dict:
        return {
            "pid":               self.pid,
            "name":              self.name,
            "status":            self.status,
            "cpu_pct":           round(self.cpu_pct, 2),
            "mem_rss_mb":        round(self.mem_rss_mb, 2),
            "mem_pct":           round(self.mem_pct, 3),
            "threads":           self.threads,
            "open_files":        self.open_files,
            "is_own":            self.is_own,
            "is_zombie":         self.is_zombie,
            "is_anomaly":        self.is_anomaly,
            "cpu_anomaly":       self.is_cpu_anomaly,
            "mem_anomaly":       self.is_mem_anomaly,
            "cpu_zscore":        round(self.cpu_zscore, 2),
            "mem_zscore":        round(self.mem_zscore, 2),
            "cpu_baseline_mean": round(self.cpu_baseline_mean, 2),
        }


@dataclass
class ProcessTreeReport:
    """
    Aggregated report of the full tracked process tree.
    Published every poll cycle and consumed by LabAgent + dashboard.
    """
    timestamp:          str
    own_pid:            int
    own_cpu_pct:        float       # CPU% of main process only
    own_mem_mb:         float       # RSS of main process only
    tree_cpu_pct:       float       # total CPU% of all own processes
    tree_mem_mb:        float       # total RSS of all own processes
    process_count:      int         # number of tracked processes
    thread_count:       int         # total threads across all own processes
    anomaly_count:      int         # processes flagged as anomalous
    zombie_count:       int         # zombie processes detected
    new_pids:           list[int]   # PIDs that appeared since last poll
    exited_pids:        list[int]   # PIDs that exited since last poll
    processes:          list[ProcessSnapshot]

    @property
    def has_anomalies(self) -> bool:
        return self.anomaly_count > 0

    @property
    def has_zombies(self) -> bool:
        return self.zombie_count > 0

    def to_dict(self) -> dict:
        return {
            "timestamp":       self.timestamp,
            "own_pid":         self.own_pid,
            "own_cpu_pct":     round(self.own_cpu_pct, 2),
            "own_mem_mb":      round(self.own_mem_mb, 2),
            "tree_cpu_pct":    round(self.tree_cpu_pct, 2),
            "tree_mem_mb":     round(self.tree_mem_mb, 2),
            "process_count":   self.process_count,
            "thread_count":    self.thread_count,
            "anomaly_count":   self.anomaly_count,
            "zombie_count":    self.zombie_count,
            "new_pids":        self.new_pids,
            "exited_pids":     self.exited_pids,
            "processes":       [p.to_dict() for p in self.processes],
        }


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING BASELINE (per-PID 2σ anomaly detection)
# ─────────────────────────────────────────────────────────────────────────────

class ProcessBaseline:
    """
    Maintains rolling mean + stdev for CPU% and RSS (MB) per PID.

    On each new observation, computes a z-score:
        z = (value - mean) / stdev

    Returns is_anomaly=True when |z| > ANOMALY_SIGMA (default 2.0)
    AND the window has at least MIN_SAMPLES observations.

    This is the statistical foundation of ResearchFlow's anomaly
    detection — the same technique used by Netflix's Anomaly
    Detection framework and Datadog's Watchdog feature.

    Why 2σ?
      In a normal distribution, 95.4% of values fall within 2σ.
      A value outside 2σ has only a 4.6% chance of being "normal"
      — enough to flag for investigation without excessive noise.
    """

    def __init__(
        self,
        window: int   = BASELINE_WINDOW,
        sigma:  float = ANOMALY_SIGMA,
    ) -> None:
        self._window = window
        self._sigma  = sigma
        self._cpu_samples: Deque[float] = deque(maxlen=window)
        self._mem_samples: Deque[float] = deque(maxlen=window)

    def update(
        self,
        cpu_pct: float,
        mem_mb:  float,
    ) -> tuple[bool, bool, float, float]:
        """
        Update baseline and check for anomalies.

        Returns: (cpu_anomaly, mem_anomaly, cpu_zscore, mem_zscore)
        """
        cpu_z = self._check(self._cpu_samples, cpu_pct)
        mem_z = self._check(self._mem_samples, mem_mb)

        self._cpu_samples.append(cpu_pct)
        self._mem_samples.append(mem_mb)

        cpu_anomaly = abs(cpu_z) > self._sigma and len(self._cpu_samples) >= MIN_SAMPLES
        mem_anomaly = abs(mem_z) > self._sigma and len(self._mem_samples) >= MIN_SAMPLES

        return cpu_anomaly, mem_anomaly, cpu_z, mem_z

    def _check(self, samples: Deque[float], value: float) -> float:
        """Compute z-score against current window. Returns 0.0 if insufficient data."""
        if len(samples) < MIN_SAMPLES:
            return 0.0
        mean = statistics.mean(samples)
        try:
            stdev = statistics.stdev(samples)
        except statistics.StatisticsError:
            return 0.0
        if stdev < 0.001:
            return 0.0
        return (value - mean) / stdev

    @property
    def cpu_mean(self) -> float:
        return statistics.mean(self._cpu_samples) if self._cpu_samples else 0.0

    @property
    def mem_mean(self) -> float:
        return statistics.mean(self._mem_samples) if self._mem_samples else 0.0

    @property
    def sample_count(self) -> int:
        return len(self._cpu_samples)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class ProcessTracker:
    """
    Tracks ResearchFlow AI's own process tree and detects anomalies.

    The tracker polls the process tree every POLL_INTERVAL_S seconds,
    maintains per-PID rolling baselines, and publishes ProcessTreeReport
    objects to an in-memory history buffer.

    "Own process tree" = main process PID + all spawned child processes
    (worker threads, subprocess-based embedding workers, etc.)

    In addition to own-process tracking, it also collects the top-10
    system processes by CPU usage for the dashboard's process table
    (similar to the output of `top` or `htop`).
    """

    def __init__(
        self,
        poll_interval_s: float = POLL_INTERVAL_S,
        history_size:    int   = 60,
    ) -> None:
        self._interval    = poll_interval_s
        self._own_pid     = os.getpid()
        self._own_proc    = psutil.Process(self._own_pid)
        self._history:    Deque[ProcessTreeReport] = deque(maxlen=history_size)
        self._baselines:  dict[int, ProcessBaseline] = {}
        self._prev_pids:  set[int] = set()
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread:     threading.Thread | None = None
        self._poll_seq    = 0

        # Total system RAM for mem_pct calculation
        self._total_ram_mb = psutil.virtual_memory().total / (1024 ** 2)

        logger.info(
            "ProcessTracker init | own_pid=%d | interval=%.1fs",
            self._own_pid, poll_interval_s,
        )

    # ── LIFECYCLE ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("ProcessTracker already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="researchflow-proctrack",
            daemon=True,
        )
        self._thread.start()
        logger.info("ProcessTracker started | thread=%s", self._thread.name)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval * 3)
        logger.info("ProcessTracker stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── REPORT ACCESS ────────────────────────────────────────────────────────

    def report(self) -> ProcessTreeReport | None:
        """Return the most recent ProcessTreeReport. Non-blocking."""
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self, n: int | None = None) -> list[ProcessTreeReport]:
        with self._lock:
            reports = list(self._history)
        return reports[-n:] if n else reports

    def anomalies(self) -> list[ProcessSnapshot]:
        """Return all currently anomalous processes from the latest report."""
        rpt = self.report()
        if rpt is None:
            return []
        return [p for p in rpt.processes if p.is_anomaly]

    def zombies(self) -> list[ProcessSnapshot]:
        """Return all zombie processes from the latest report."""
        rpt = self.report()
        if rpt is None:
            return []
        return [p for p in rpt.processes if p.is_zombie]

    def own_tree_pids(self) -> set[int]:
        """
        Return the current set of PIDs in our process tree.
        Used by SystemMonitor to distinguish own-process load
        from external system load.
        """
        pids: set[int] = {self._own_pid}
        try:
            children = self._own_proc.children(recursive=True)
            pids.update(c.pid for c in children)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return pids

    # ── POLL LOOP ─────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        logger.debug("ProcessTracker poll loop started")
        next_poll = time.monotonic()

        while not self._stop_event.is_set():
            try:
                report = self._collect()
                with self._lock:
                    self._history.append(report)
                self._log_notable(report)
            except Exception as exc:
                logger.error("ProcessTracker poll error: %s", exc, exc_info=True)

            next_poll += self._interval
            sleep_s = next_poll - time.monotonic()
            if sleep_s > 0:
                self._stop_event.wait(timeout=sleep_s)

        logger.debug("ProcessTracker poll loop exited")

    # ── COLLECTION ────────────────────────────────────────────────────────────

    def _collect(self) -> ProcessTreeReport:
        self._poll_seq += 1
        ts = datetime.now(timezone.utc).isoformat()

        # ── Collect own process tree ─────────────────────────────────────────
        own_pids = self.own_tree_pids()
        current_pids: set[int] = set()
        snapshots: list[ProcessSnapshot] = []

        # Process our own tree first
        for pid in own_pids:
            snap = self._snapshot_pid(pid, is_own=True)
            if snap:
                snapshots.append(snap)
                current_pids.add(pid)

        # Top-10 external processes by CPU (for dashboard process table)
        try:
            all_procs = [
                p for p in psutil.process_iter(
                    ["pid", "name", "cpu_percent", "memory_info",
                     "memory_percent", "status", "num_threads",
                     "create_time", "cmdline", "num_ctx_switches"]
                )
                if p.info["pid"] not in own_pids
                and p.info["cpu_percent"] is not None
            ]
            top10 = sorted(
                all_procs,
                key=lambda p: p.info["cpu_percent"] or 0,
                reverse=True,
            )[:10]
            for proc in top10:
                snap = self._snapshot_from_info(proc.info, is_own=False)
                if snap:
                    snapshots.append(snap)
                    current_pids.add(snap.pid)
        except Exception:
            pass

        # ── Lifecycle detection ───────────────────────────────────────────────
        new_pids    = list(current_pids - self._prev_pids)
        exited_pids = list(self._prev_pids - current_pids)
        self._prev_pids = current_pids

        # Clean up baselines for exited processes
        for pid in exited_pids:
            self._baselines.pop(pid, None)

        # ── Aggregate stats ───────────────────────────────────────────────────
        own_snaps  = [s for s in snapshots if s.is_own]
        tree_cpu   = sum(s.cpu_pct for s in own_snaps)
        tree_mem   = sum(s.mem_rss_mb for s in own_snaps)
        own_main   = next((s for s in own_snaps if s.pid == self._own_pid), None)
        own_cpu    = own_main.cpu_pct  if own_main else 0.0
        own_mem    = own_main.mem_rss_mb if own_main else 0.0
        threads    = sum(s.threads for s in own_snaps)
        anomalies  = sum(1 for s in snapshots if s.is_anomaly)
        zombies    = sum(1 for s in snapshots if s.is_zombie)

        return ProcessTreeReport(
            timestamp=ts,
            own_pid=self._own_pid,
            own_cpu_pct=own_cpu,
            own_mem_mb=own_mem,
            tree_cpu_pct=tree_cpu,
            tree_mem_mb=tree_mem,
            process_count=len(own_snaps),
            thread_count=threads,
            anomaly_count=anomalies,
            zombie_count=zombies,
            new_pids=new_pids,
            exited_pids=exited_pids,
            processes=snapshots,
        )

    def _snapshot_pid(
        self,
        pid: int,
        is_own: bool,
    ) -> ProcessSnapshot | None:
        """Collect a ProcessSnapshot for a single PID."""
        try:
            proc = psutil.Process(pid)
            info = proc.as_dict(
                attrs=["name", "status", "cpu_percent", "memory_info",
                       "memory_percent", "num_threads", "create_time",
                       "cmdline", "num_ctx_switches", "open_files"],
                ad_value=None,
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return None

        return self._snapshot_from_info({**info, "pid": pid}, is_own=is_own)

    def _snapshot_from_info(
        self,
        info: dict,
        is_own: bool,
    ) -> ProcessSnapshot | None:
        """Build a ProcessSnapshot from a process info dict."""
        pid = info.get("pid")
        if pid is None:
            return None

        name    = (info.get("name") or "?")[:40]
        status  = info.get("status") or "?"
        cpu_pct = float(info.get("cpu_percent") or 0.0)
        mem_info = info.get("memory_info")
        mem_rss = (mem_info.rss / (1024 ** 2)) if mem_info else 0.0
        mem_vms = (mem_info.vms / (1024 ** 2)) if mem_info else 0.0
        mem_pct = float(info.get("memory_percent") or 0.0)
        threads = int(info.get("num_threads") or 1)
        create  = float(info.get("create_time") or time.time())
        cmdline_list = info.get("cmdline") or []
        cmdline = " ".join(str(c) for c in cmdline_list)[:80]
        ctx     = info.get("num_ctx_switches")
        ctx_vol   = getattr(ctx, "voluntary",   0) if ctx else 0
        ctx_invol = getattr(ctx, "involuntary",  0) if ctx else 0
        open_f  = len(info.get("open_files") or [])

        is_zombie = (status == "zombie")

        # ── Anomaly detection ─────────────────────────────────────────────────
        if pid not in self._baselines:
            self._baselines[pid] = ProcessBaseline()

        baseline = self._baselines[pid]
        cpu_anom, mem_anom, cpu_z, mem_z = baseline.update(cpu_pct, mem_rss)

        # Hard-limit overrides (always flag, regardless of baseline)
        if cpu_pct >= CPU_HARD_LIMIT_PCT:
            cpu_anom = True
        if mem_rss >= MEM_HARD_LIMIT_MB:
            mem_anom = True

        return ProcessSnapshot(
            pid=pid,
            name=name,
            status=status,
            cpu_pct=round(cpu_pct, 2),
            mem_rss_mb=round(mem_rss, 2),
            mem_vms_mb=round(mem_vms, 2),
            mem_pct=round(mem_pct, 3),
            threads=threads,
            open_files=open_f,
            ctx_vol=ctx_vol,
            ctx_invol=ctx_invol,
            create_time=create,
            cmdline=cmdline,
            is_own=is_own,
            is_zombie=is_zombie,
            is_cpu_anomaly=cpu_anom,
            is_mem_anomaly=mem_anom,
            cpu_zscore=round(cpu_z, 2),
            mem_zscore=round(mem_z, 2),
            cpu_baseline_mean=round(baseline.cpu_mean, 2),
            mem_baseline_mean=round(baseline.mem_mean, 2),
        )

    # ── LOGGING ──────────────────────────────────────────────────────────────

    def _log_notable(self, report: ProcessTreeReport) -> None:
        """Log significant events from a new report."""
        for pid in report.new_pids:
            logger.debug("New process appeared: PID %d", pid)

        for pid in report.exited_pids:
            logger.debug("Process exited: PID %d", pid)

        for proc in report.processes:
            if proc.is_zombie:
                logger.warning(
                    "ZOMBIE PROCESS: PID %d '%s'", proc.pid, proc.name
                )
            if proc.is_cpu_anomaly:
                logger.warning(
                    "CPU ANOMALY: PID %d '%s' cpu=%.1f%% z=%.2f (baseline=%.1f%%)",
                    proc.pid, proc.name, proc.cpu_pct,
                    proc.cpu_zscore, proc.cpu_baseline_mean,
                )
            if proc.is_mem_anomaly:
                logger.warning(
                    "MEM ANOMALY: PID %d '%s' rss=%.1fMB z=%.2f",
                    proc.pid, proc.name, proc.mem_rss_mb, proc.mem_zscore,
                )

    # ── STATUS ───────────────────────────────────────────────────────────────

    def status(self) -> dict:
        """Compact status for Streamlit sidebar."""
        rpt = self.report()
        if rpt is None:
            return {"running": self.is_running, "report": None}
        return {
            "running":         self.is_running,
            "own_pid":         rpt.own_pid,
            "own_cpu_pct":     round(rpt.own_cpu_pct, 2),
            "own_mem_mb":      round(rpt.own_mem_mb, 2),
            "tree_cpu_pct":    round(rpt.tree_cpu_pct, 2),
            "tree_mem_mb":     round(rpt.tree_mem_mb, 2),
            "process_count":   rpt.process_count,
            "thread_count":    rpt.thread_count,
            "anomaly_count":   rpt.anomaly_count,
            "zombie_count":    rpt.zombie_count,
        }

    def __repr__(self) -> str:
        rpt = self.report()
        if rpt is None:
            return f"ProcessTracker(pid={self._own_pid}, running={self.is_running})"
        return (
            f"ProcessTracker(pid={self._own_pid}, "
            f"procs={rpt.process_count}, "
            f"threads={rpt.thread_count}, "
            f"anomalies={rpt.anomaly_count})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_tracker: ProcessTracker | None = None


def get_process_tracker() -> ProcessTracker:
    """Return the module-level singleton ProcessTracker."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ProcessTracker()
    return _default_tracker


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Live process tree monitor.
    Run with: python -m monitoring.process_tracker
    Press Ctrl+C to stop.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    tracker = ProcessTracker(poll_interval_s=3.0, history_size=20)
    tracker.start()

    print(f"\nResearchFlow AI — Process Tracker  (Ctrl+C to stop)")
    print(f"  Own PID: {tracker._own_pid} | "
          f"2σ window: {BASELINE_WINDOW} samples | "
          f"σ threshold: {ANOMALY_SIGMA}\n")

    try:
        poll = 0
        while True:
            time.sleep(3.0)
            poll += 1
            rpt = tracker.report()
            if rpt is None:
                print("  Warming up...")
                continue

            print(f"\n  ── Poll #{poll} @ {rpt.timestamp[11:19]} ──")
            print(
                f"  Own process tree: "
                f"CPU={rpt.tree_cpu_pct:.1f}%  "
                f"RAM={rpt.tree_mem_mb:.0f}MB  "
                f"Procs={rpt.process_count}  "
                f"Threads={rpt.thread_count}"
            )

            if rpt.new_pids:
                print(f"  ✦ New PIDs:     {rpt.new_pids}")
            if rpt.exited_pids:
                print(f"  ✦ Exited PIDs:  {rpt.exited_pids}")

            # Own process table
            own_procs = [p for p in rpt.processes if p.is_own]
            if own_procs:
                print(f"\n  {'PID':<8} {'Name':<25} {'CPU%':>6} {'RSS MB':>8} "
                      f"{'Threads':>8} {'Status':<12} {'Flag'}")
                print("  " + "─" * 78)
                for p in sorted(own_procs, key=lambda x: -x.cpu_pct):
                    flag = ""
                    if p.is_cpu_anomaly:
                        flag += f"⚠CPU(z={p.cpu_zscore:.1f}) "
                    if p.is_mem_anomaly:
                        flag += f"⚠MEM(z={p.mem_zscore:.1f}) "
                    if p.is_zombie:
                        flag += "💀ZOMBIE "
                    print(
                        f"  {p.pid:<8} {p.name:<25} {p.cpu_pct:>6.1f} "
                        f"{p.mem_rss_mb:>8.1f} {p.threads:>8} "
                        f"{p.status:<12} {flag or '✓ normal'}"
                    )

            # Top external processes
            ext_procs = [p for p in rpt.processes if not p.is_own][:5]
            if ext_procs:
                print(f"\n  Top external processes:")
                for p in ext_procs:
                    print(
                        f"    PID {p.pid:<8} {p.name:<25} "
                        f"CPU={p.cpu_pct:.1f}%  RSS={p.mem_rss_mb:.0f}MB"
                    )

    except KeyboardInterrupt:
        print("\n\nStopping...\n")

    tracker.stop()
    print(f"\n{tracker}")
    print(f"  History depth: {len(tracker.history())} reports")

    # Anomaly summary
    all_anomalies: list[ProcessSnapshot] = []
    for rpt in tracker.history():
        all_anomalies.extend(p for p in rpt.processes if p.is_anomaly)
    if all_anomalies:
        print(f"\n  Anomalies detected: {len(all_anomalies)}")
        for p in all_anomalies[:5]:
            print(f"    PID {p.pid} '{p.name}' cpu_z={p.cpu_zscore:.2f} mem_z={p.mem_zscore:.2f}")
    else:
        print("\n  No anomalies detected during session.")


if __name__ == "__main__":
    _demo()
