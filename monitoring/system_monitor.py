"""
monitoring/system_monitor.py
=============================
ResearchFlow AI — System Resource Monitor (Background Thread)

This module is the hardware telemetry engine of ResearchFlow AI.
It runs as a daemon thread, polling CPU, RAM, disk, and network
metrics every POLL_INTERVAL_S seconds via psutil, and publishing
snapshots to a thread-safe shared buffer consumed by:

  - SchedulerAgent  → resource-aware task dispatch decisions
  - LabAgent        → experiment checkpointing
  - LabPipeline     → adaptive throttle controller
  - Streamlit UI    → live resource dashboard gauges

OS Concepts demonstrated:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Concept                    │  Implementation                  │
  ├─────────────────────────────┼──────────────────────────────────┤
  │  Daemon process             │  threading.Thread(daemon=True)   │
  │  Process accounting         │  psutil.cpu_percent, memory_info │
  │  Kernel scheduler view      │  per-core utilisation tracking   │
  │  I/O scheduling metrics     │  disk_io_counters delta          │
  │  Context switch accounting  │  cpu_stats().ctx_switches        │
  │  Memory management view     │  virtual_memory + swap           │
  │  Interrupt rate             │  cpu_stats().interrupts          │
  │  Load average               │  getloadavg() (Unix)             │
  │  Clock frequency            │  cpu_freq().current              │
  │  Shared memory              │  threading.Lock + deque buffer   │
  └─────────────────────────────────────────────────────────────────┘

Architecture:
  SystemMonitor (daemon thread)
       │
       ├── _collect_cpu()      → CPUSnapshot
       ├── _collect_memory()   → MemorySnapshot
       ├── _collect_disk()     → DiskSnapshot
       ├── _collect_network()  → NetworkSnapshot
       │
       ▼
  _snapshot_buffer (deque, thread-safe)
       │
       ├── latest()            → most recent snapshot (non-blocking)
       ├── history(n)          → last n snapshots for trend charts
       └── subscribe(cb)       → push-based callback for real-time consumers

Metrics collected every poll cycle:

  CPU:
    - Overall utilisation % (interval-averaged)
    - Per-core utilisation % list
    - Physical and logical core counts
    - Clock frequency (MHz)
    - Load average (1m, 5m, 15m) — Unix only
    - Context switches per second
    - Interrupts per second
    - System call rate estimate

  Memory:
    - Total / used / available / free (bytes + %)
    - Active / inactive / buffers / cached (Linux)
    - Swap total / used / free / %
    - Memory pressure index (used% + swap% combined)

  Disk I/O:
    - Read / write bytes per second (delta from previous poll)
    - Read / write IOPS per second
    - Busy time % (Linux: io_time field)
    - Per-device breakdown (optional)

  Network:
    - Bytes sent / received per second
    - Packets sent / received per second
    - Error and drop counts

Usage:
    monitor = SystemMonitor()
    monitor.start()

    # Non-blocking read from any thread
    snap = monitor.latest()
    print(snap.cpu.overall_pct, snap.memory.used_pct)

    # Historical trend (last 30 samples)
    history = monitor.history(30)

    # Callback-based (called on every new snapshot)
    monitor.subscribe(lambda s: print(s.cpu.overall_pct))

    monitor.stop()

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque

import psutil

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL_S: float = float(os.getenv("MONITOR_POLL_INTERVAL", "2.0"))
HISTORY_SIZE:    int   = int(os.getenv("MONITOR_HISTORY_SIZE",    "150"))  # ~5 min at 2s
MAX_SUBSCRIBERS: int   = 20


# ─────────────────────────────────────────────────────────────────────────────
# SNAPSHOT DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CPUSnapshot:
    """
    CPU utilisation snapshot.

    overall_pct is averaged over the poll interval, giving a smooth
    reading that avoids single-sample spikes.
    per_core_pct allows pinpointing which cores are saturated.
    """
    overall_pct:        float
    per_core_pct:       list[float]
    logical_cores:      int
    physical_cores:     int
    freq_mhz:           float
    freq_min_mhz:       float
    freq_max_mhz:       float
    load_avg_1m:        float        # Unix only; 0.0 on Windows
    load_avg_5m:        float
    load_avg_15m:       float
    ctx_switches_ps:    float        # context switches per second
    interrupts_ps:      float        # hardware interrupts per second
    syscalls_ps:        float        # system calls per second (approx)

    @property
    def is_overloaded(self) -> bool:
        """True when CPU is critically saturated (>= 90%)."""
        return self.overall_pct >= 90.0

    @property
    def headroom_pct(self) -> float:
        return max(0.0, 100.0 - self.overall_pct)

    @property
    def hottest_core(self) -> tuple[int, float]:
        """Return (core_index, utilisation%) for the most loaded core."""
        if not self.per_core_pct:
            return 0, self.overall_pct
        idx = max(range(len(self.per_core_pct)), key=lambda i: self.per_core_pct[i])
        return idx, self.per_core_pct[idx]

    def to_dict(self) -> dict:
        return {
            "overall_pct":     round(self.overall_pct, 2),
            "per_core_pct":    [round(c, 1) for c in self.per_core_pct],
            "logical_cores":   self.logical_cores,
            "physical_cores":  self.physical_cores,
            "freq_mhz":        round(self.freq_mhz, 1),
            "load_avg_1m":     round(self.load_avg_1m, 2),
            "load_avg_5m":     round(self.load_avg_5m, 2),
            "load_avg_15m":    round(self.load_avg_15m, 2),
            "ctx_switches_ps": round(self.ctx_switches_ps, 1),
            "interrupts_ps":   round(self.interrupts_ps, 1),
            "headroom_pct":    round(self.headroom_pct, 2),
        }


@dataclass
class MemorySnapshot:
    """
    RAM and swap utilisation snapshot.

    pressure_index combines RAM% and swap% into a single 0–100 scalar
    that is a better signal for OOM risk than either alone.
    """
    total_bytes:     int
    used_bytes:      int
    available_bytes: int
    free_bytes:      int
    used_pct:        float
    active_bytes:    int     # pages currently in use (Linux)
    inactive_bytes:  int     # pages that can be swapped (Linux)
    buffers_bytes:   int     # kernel buffers (Linux)
    cached_bytes:    int     # disk cache (Linux)
    swap_total:      int
    swap_used:       int
    swap_free:       int
    swap_pct:        float

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024 ** 3)

    @property
    def used_gb(self) -> float:
        return self.used_bytes / (1024 ** 3)

    @property
    def available_gb(self) -> float:
        return self.available_bytes / (1024 ** 3)

    @property
    def headroom_pct(self) -> float:
        return max(0.0, 100.0 - self.used_pct)

    @property
    def pressure_index(self) -> float:
        """
        Combined memory pressure score (0–100).
        High swap usage amplifies the score because it indicates
        the OS is actively paging — a strong OOM precursor.
        """
        return min(100.0, self.used_pct * 0.7 + self.swap_pct * 0.3)

    @property
    def is_oom_risk(self) -> bool:
        return self.used_pct >= 92.0 or self.swap_pct >= 40.0

    def to_dict(self) -> dict:
        return {
            "total_gb":       round(self.total_gb, 3),
            "used_gb":        round(self.used_gb, 3),
            "available_gb":   round(self.available_gb, 3),
            "used_pct":       round(self.used_pct, 2),
            "swap_pct":       round(self.swap_pct, 2),
            "pressure_index": round(self.pressure_index, 2),
            "headroom_pct":   round(self.headroom_pct, 2),
            "is_oom_risk":    self.is_oom_risk,
        }


@dataclass
class DiskSnapshot:
    """
    Disk I/O throughput snapshot (delta from previous poll).

    All rate values are per-second, computed from cumulative
    psutil counters divided by the actual elapsed time between polls.
    """
    read_bytes_ps:   float   # read throughput (bytes/s)
    write_bytes_ps:  float   # write throughput (bytes/s)
    read_iops:       float   # read operations/s
    write_iops:      float   # write operations/s
    busy_pct:        float   # fraction of time disk was busy (Linux)
    read_mb_s:       float   # read_bytes_ps in MB/s (convenience)
    write_mb_s:      float   # write_bytes_ps in MB/s

    @property
    def total_mb_s(self) -> float:
        return self.read_mb_s + self.write_mb_s

    @property
    def is_io_bound(self) -> bool:
        """Heuristic: busy > 70% suggests I/O is a bottleneck."""
        return self.busy_pct >= 70.0

    def to_dict(self) -> dict:
        return {
            "read_mb_s":   round(self.read_mb_s, 3),
            "write_mb_s":  round(self.write_mb_s, 3),
            "total_mb_s":  round(self.total_mb_s, 3),
            "read_iops":   round(self.read_iops, 1),
            "write_iops":  round(self.write_iops, 1),
            "busy_pct":    round(self.busy_pct, 2),
            "is_io_bound": self.is_io_bound,
        }


@dataclass
class NetworkSnapshot:
    """Network I/O throughput snapshot (delta from previous poll)."""
    bytes_sent_ps:     float
    bytes_recv_ps:     float
    packets_sent_ps:   float
    packets_recv_ps:   float
    errin:             int
    errout:            int
    dropin:            int
    dropout:           int
    sent_mb_s:         float
    recv_mb_s:         float

    def to_dict(self) -> dict:
        return {
            "sent_mb_s":   round(self.sent_mb_s, 3),
            "recv_mb_s":   round(self.recv_mb_s, 3),
            "errin":       self.errin,
            "errout":      self.errout,
            "dropin":      self.dropin,
        }


@dataclass
class SystemSnapshot:
    """
    Complete system telemetry snapshot — the primary data structure
    flowing through the entire ResearchFlow AI monitoring stack.

    Published every POLL_INTERVAL_S by SystemMonitor and consumed by
    SchedulerAgent, LabAgent, LabPipeline, and the Streamlit dashboard.
    """
    timestamp:   str
    cpu:         CPUSnapshot
    memory:      MemorySnapshot
    disk:        DiskSnapshot
    network:     NetworkSnapshot
    poll_seq:    int     # monotonic sequence number

    @property
    def bottleneck_hint(self) -> str:
        """
        Quick bottleneck classification based on snapshot values.
        Returns a string hint — the full classification is done
        by bottleneck_detector.py with richer context.
        """
        if self.memory.is_oom_risk:
            return "oom_risk"
        if self.cpu.is_overloaded:
            return "cpu_bound"
        if self.disk.is_io_bound and self.cpu.overall_pct < 50.0:
            return "io_bound"
        if self.memory.used_pct >= 80.0 and self.cpu.overall_pct < 60.0:
            return "mem_bound"
        return "none"

    def to_dict(self) -> dict:
        return {
            "timestamp":   self.timestamp,
            "poll_seq":    self.poll_seq,
            "cpu":         self.cpu.to_dict(),
            "memory":      self.memory.to_dict(),
            "disk":        self.disk.to_dict(),
            "network":     self.network.to_dict(),
            "bottleneck":  self.bottleneck_hint,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DELTA TRACKER (for cumulative counter → rate conversion)
# ─────────────────────────────────────────────────────────────────────────────

class DeltaTracker:
    """
    Converts psutil's cumulative counters into per-second rates.

    psutil counters (bytes_read, ctx_switches, etc.) are monotonically
    increasing totals. We need per-second rates for the dashboard.

    delta_rate(current) = (current - previous) / elapsed_seconds

    This is exactly how OS performance tools like iostat, vmstat,
    and sar compute their displayed rates from kernel counters.
    """

    def __init__(self) -> None:
        self._prev_values: dict[str, float] = {}
        self._prev_time:   dict[str, float] = {}

    def rate(self, key: str, current: float, now: float | None = None) -> float:
        """
        Compute per-second rate for a cumulative counter.
        Returns 0.0 on the first call (no previous value).
        """
        t = now or time.time()
        if key not in self._prev_values:
            self._prev_values[key] = current
            self._prev_time[key]   = t
            return 0.0

        dt = t - self._prev_time[key]
        if dt <= 0:
            return 0.0

        delta = max(0.0, current - self._prev_values[key])
        rate  = delta / dt

        self._prev_values[key] = current
        self._prev_time[key]   = t

        return rate


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class SystemMonitor:
    """
    Background daemon thread that continuously polls system telemetry
    and maintains a thread-safe circular snapshot buffer.

    Key design principles:
      1. Non-blocking reads: latest() and history() never block —
         callers on the Streamlit main thread can read without
         waiting for the next poll cycle.
      2. Precise timing: the poll loop uses wall-clock math to
         compensate for variable collection time, maintaining
         accurate POLL_INTERVAL_S cadence.
      3. Graceful degradation: any collection error is caught and
         logged; the loop continues with a partial snapshot rather
         than crashing the monitoring thread.
      4. Zero-copy reads: history() returns a copy of the deque
         snapshot so callers can iterate without holding the lock.
    """

    def __init__(
        self,
        poll_interval_s: float = POLL_INTERVAL_S,
        history_size:    int   = HISTORY_SIZE,
    ) -> None:
        self._interval   = poll_interval_s
        self._history:   Deque[SystemSnapshot] = deque(maxlen=history_size)
        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._thread:    threading.Thread | None = None
        self._poll_seq   = 0
        self._delta      = DeltaTracker()
        self._subscribers: list[Callable[[SystemSnapshot], None]] = []
        self._own_pid    = os.getpid()
        self._own_proc   = psutil.Process(self._own_pid)

        # Pre-warm psutil (first cpu_percent call returns 0.0; discard it)
        psutil.cpu_percent(interval=None)
        psutil.cpu_percent(percpu=True, interval=None)

        logger.info(
            "SystemMonitor init | PID=%d | interval=%.1fs | history=%d",
            self._own_pid, poll_interval_s, history_size,
        )

    # ── LIFECYCLE ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background polling daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("SystemMonitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="researchflow-sysmon",
            daemon=True,
        )
        self._thread.start()
        logger.info("SystemMonitor started | thread=%s", self._thread.name)

    def stop(self) -> None:
        """Signal the polling thread to exit and wait for it."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval * 3)
        logger.info("SystemMonitor stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── SNAPSHOT ACCESS ──────────────────────────────────────────────────────

    def latest(self) -> SystemSnapshot | None:
        """
        Return the most recent snapshot. Non-blocking.
        Safe to call from any thread, including Streamlit's main thread.
        Returns None during the warm-up period before the first poll.
        """
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self, n: int | None = None) -> list[SystemSnapshot]:
        """
        Return the last n snapshots as a list copy.
        Non-blocking. Returns all history if n is None.
        """
        with self._lock:
            snaps = list(self._history)
        return snaps[-n:] if n else snaps

    def history_dicts(self, n: int | None = None) -> list[dict]:
        """Return history as serialisable dicts — for Streamlit charts."""
        return [s.to_dict() for s in self.history(n)]

    # ── SUBSCRIPTION ────────────────────────────────────────────────────────

    def subscribe(self, callback: Callable[[SystemSnapshot], None]) -> None:
        """
        Register a callback invoked on every new snapshot.
        Callbacks are called from the monitor thread — must be thread-safe
        and must not block (use a queue if heavy work is needed).
        """
        if len(self._subscribers) >= MAX_SUBSCRIBERS:
            logger.warning("SystemMonitor subscriber limit reached (%d)", MAX_SUBSCRIBERS)
            return
        self._subscribers.append(callback)
        logger.debug("Subscriber registered (%d total)", len(self._subscribers))

    def unsubscribe(self, callback: Callable) -> bool:
        try:
            self._subscribers.remove(callback)
            return True
        except ValueError:
            return False

    # ── POLL LOOP ─────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """
        Main background loop.

        Uses wall-clock math to maintain precise interval cadence:
          target_next = last_poll + POLL_INTERVAL_S
          sleep(max(0, target_next - now))

        This prevents drift when collection takes variable time
        (e.g. disk I/O counters are slow on some kernels).
        """
        logger.debug("SystemMonitor poll loop started")
        next_poll = time.monotonic()

        while not self._stop_event.is_set():
            try:
                snap = self._collect()
                with self._lock:
                    self._history.append(snap)
                self._notify_subscribers(snap)
            except Exception as exc:
                logger.error("SystemMonitor poll error: %s", exc, exc_info=True)

            # Precise sleep to next poll time
            next_poll += self._interval
            sleep_s = next_poll - time.monotonic()
            if sleep_s > 0:
                self._stop_event.wait(timeout=sleep_s)

        logger.debug("SystemMonitor poll loop exited")

    # ── COLLECTION ────────────────────────────────────────────────────────────

    def _collect(self) -> SystemSnapshot:
        """
        Collect all system metrics and assemble a SystemSnapshot.
        Each sub-collector is isolated — failure in one does not
        prevent the others from completing.
        """
        now = time.time()
        ts  = datetime.now(timezone.utc).isoformat()
        self._poll_seq += 1

        cpu  = self._collect_cpu(now)
        mem  = self._collect_memory()
        disk = self._collect_disk(now)
        net  = self._collect_network(now)

        return SystemSnapshot(
            timestamp=ts,
            cpu=cpu,
            memory=mem,
            disk=disk,
            network=net,
            poll_seq=self._poll_seq,
        )

    def _collect_cpu(self, now: float) -> CPUSnapshot:
        """
        Collect CPU metrics.

        cpu_percent(interval=None) uses the time since the last call
        as the measurement window — accurate and non-blocking.
        """
        try:
            overall   = psutil.cpu_percent(interval=None)
            per_core  = psutil.cpu_percent(percpu=True, interval=None)
        except Exception:
            overall  = 0.0
            per_core = []

        logical  = psutil.cpu_count(logical=True)  or 1
        physical = psutil.cpu_count(logical=False) or 1

        # Clock frequency
        try:
            freq = psutil.cpu_freq()
            freq_cur = freq.current if freq else 0.0
            freq_min = freq.min     if freq else 0.0
            freq_max = freq.max     if freq else 0.0
        except Exception:
            freq_cur = freq_min = freq_max = 0.0

        # Load average (Unix only)
        try:
            la = psutil.getloadavg()
            la1, la5, la15 = la[0], la[1], la[2]
        except AttributeError:
            la1 = la5 = la15 = 0.0

        # Context switches and interrupts (per-second rates)
        try:
            stats = psutil.cpu_stats()
            ctx_ps  = self._delta.rate("ctx_switches", stats.ctx_switches, now)
            int_ps  = self._delta.rate("interrupts",   stats.interrupts,   now)
            sys_ps  = self._delta.rate("syscalls",
                                       getattr(stats, "syscalls", 0),      now)
        except Exception:
            ctx_ps = int_ps = sys_ps = 0.0

        return CPUSnapshot(
            overall_pct=overall,
            per_core_pct=list(per_core),
            logical_cores=logical,
            physical_cores=physical,
            freq_mhz=freq_cur,
            freq_min_mhz=freq_min,
            freq_max_mhz=freq_max,
            load_avg_1m=la1,
            load_avg_5m=la5,
            load_avg_15m=la15,
            ctx_switches_ps=ctx_ps,
            interrupts_ps=int_ps,
            syscalls_ps=sys_ps,
        )

    def _collect_memory(self) -> MemorySnapshot:
        """Collect RAM and swap metrics."""
        try:
            vm = psutil.virtual_memory()
            sw = psutil.swap_memory()
        except Exception:
            return MemorySnapshot(
                total_bytes=1, used_bytes=0, available_bytes=1,
                free_bytes=1, used_pct=0.0,
                active_bytes=0, inactive_bytes=0,
                buffers_bytes=0, cached_bytes=0,
                swap_total=0, swap_used=0, swap_free=0, swap_pct=0.0,
            )

        return MemorySnapshot(
            total_bytes=vm.total,
            used_bytes=vm.used,
            available_bytes=vm.available,
            free_bytes=vm.free,
            used_pct=vm.percent,
            active_bytes=getattr(vm, "active",   0),
            inactive_bytes=getattr(vm, "inactive", 0),
            buffers_bytes=getattr(vm, "buffers",  0),
            cached_bytes=getattr(vm, "cached",   0),
            swap_total=sw.total,
            swap_used=sw.used,
            swap_free=sw.free,
            swap_pct=sw.percent,
        )

    def _collect_disk(self, now: float) -> DiskSnapshot:
        """
        Collect disk I/O throughput as per-second rates.

        psutil.disk_io_counters() returns monotonically increasing
        byte and operation counts. We divide by elapsed time to get
        per-second rates — the same approach used by iostat.
        """
        try:
            d = psutil.disk_io_counters()
        except Exception:
            d = None

        if d is None:
            return DiskSnapshot(0, 0, 0, 0, 0, 0, 0)

        read_ps   = self._delta.rate("disk_read_bytes",  d.read_bytes,  now)
        write_ps  = self._delta.rate("disk_write_bytes", d.write_bytes, now)
        riops     = self._delta.rate("disk_read_count",  d.read_count,  now)
        wiops     = self._delta.rate("disk_write_count", d.write_count, now)

        # Disk busy% (Linux: busy_time is ms disk was busy in the interval)
        try:
            busy_ms_ps = self._delta.rate("disk_busy_time",
                                          getattr(d, "busy_time", 0), now)
            busy_pct = min(100.0, busy_ms_ps / 10.0)  # ms/s → %
        except Exception:
            busy_pct = 0.0

        def to_mb(bps: float) -> float:
            return bps / (1024 * 1024)

        return DiskSnapshot(
            read_bytes_ps=max(0.0, read_ps),
            write_bytes_ps=max(0.0, write_ps),
            read_iops=max(0.0, riops),
            write_iops=max(0.0, wiops),
            busy_pct=busy_pct,
            read_mb_s=round(to_mb(max(0.0, read_ps)), 3),
            write_mb_s=round(to_mb(max(0.0, write_ps)), 3),
        )

    def _collect_network(self, now: float) -> NetworkSnapshot:
        """Collect network I/O throughput as per-second rates."""
        try:
            n = psutil.net_io_counters()
        except Exception:
            return NetworkSnapshot(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        sent_ps   = self._delta.rate("net_bytes_sent",   n.bytes_sent,   now)
        recv_ps   = self._delta.rate("net_bytes_recv",   n.bytes_recv,   now)
        pkts_s_ps = self._delta.rate("net_pkts_sent",    n.packets_sent, now)
        pkts_r_ps = self._delta.rate("net_pkts_recv",    n.packets_recv, now)

        def to_mb(bps: float) -> float:
            return bps / (1024 * 1024)

        return NetworkSnapshot(
            bytes_sent_ps=max(0.0, sent_ps),
            bytes_recv_ps=max(0.0, recv_ps),
            packets_sent_ps=max(0.0, pkts_s_ps),
            packets_recv_ps=max(0.0, pkts_r_ps),
            errin=n.errin,
            errout=n.errout,
            dropin=n.dropin,
            dropout=n.dropout,
            sent_mb_s=round(to_mb(max(0.0, sent_ps)), 3),
            recv_mb_s=round(to_mb(max(0.0, recv_ps)), 3),
        )

    # ── SUBSCRIBER NOTIFICATION ──────────────────────────────────────────────

    def _notify_subscribers(self, snap: SystemSnapshot) -> None:
        """Call all registered callbacks with the new snapshot."""
        for cb in list(self._subscribers):
            try:
                cb(snap)
            except Exception as exc:
                logger.warning("Subscriber callback error: %s", exc)

    # ── CONVENIENCE STATS ────────────────────────────────────────────────────

    def rolling_averages(self, window: int = 10) -> dict:
        """
        Compute rolling averages over the last `window` snapshots.
        Useful for the LabPipeline adaptive throttle controller.
        """
        snaps = self.history(window)
        if not snaps:
            return {}

        cpu_vals = [s.cpu.overall_pct  for s in snaps]
        ram_vals = [s.memory.used_pct  for s in snaps]
        dsk_vals = [s.disk.total_mb_s  for s in snaps]

        def avg(vals: list) -> float:
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        return {
            "cpu_avg_pct":   avg(cpu_vals),
            "cpu_max_pct":   round(max(cpu_vals), 2),
            "ram_avg_pct":   avg(ram_vals),
            "ram_max_pct":   round(max(ram_vals), 2),
            "disk_avg_mb_s": avg(dsk_vals),
            "window":        len(snaps),
        }

    def status(self) -> dict:
        """Compact status dict for the Streamlit sidebar."""
        snap = self.latest()
        if snap is None:
            return {"running": self.is_running, "snapshot": None}
        return {
            "running":       self.is_running,
            "poll_seq":      self._poll_seq,
            "cpu_pct":       snap.cpu.overall_pct,
            "ram_pct":       snap.memory.used_pct,
            "bottleneck":    snap.bottleneck_hint,
            "history_depth": len(self._history),
        }

    def __repr__(self) -> str:
        snap = self.latest()
        cpu  = f"{snap.cpu.overall_pct:.1f}%" if snap else "—"
        ram  = f"{snap.memory.used_pct:.1f}%" if snap else "—"
        return (
            f"SystemMonitor(running={self.is_running}, "
            f"seq={self._poll_seq}, cpu={cpu}, ram={ram})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_monitor: SystemMonitor | None = None


def get_monitor() -> SystemMonitor:
    """Return the module-level singleton SystemMonitor."""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = SystemMonitor()
    return _default_monitor


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Live terminal monitor — prints a refreshing dashboard every 2 seconds.
    Run with: python -m monitoring.system_monitor
    Press Ctrl+C to stop.
    """
    import sys

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    monitor = SystemMonitor(poll_interval_s=2.0, history_size=30)

    # Subscriber demo: print a brief alert when CPU > 60%
    def cpu_alert(snap: SystemSnapshot) -> None:
        if snap.cpu.overall_pct > 60.0:
            logger.warning("HIGH CPU: %.1f%%", snap.cpu.overall_pct)

    monitor.subscribe(cpu_alert)
    monitor.start()

    print("\nResearchFlow AI — System Monitor  (Ctrl+C to stop)\n")
    print(f"  Polling every {monitor._interval:.1f}s | "
          f"History: {HISTORY_SIZE} samples | "
          f"PID: {os.getpid()}\n")

    try:
        while True:
            time.sleep(2.0)
            snap = monitor.latest()
            if snap is None:
                print("  Warming up...")
                continue

            cpu  = snap.cpu
            mem  = snap.memory
            disk = snap.disk
            net  = snap.network

            # CPU bar
            bar_len   = 30
            filled    = int(cpu.overall_pct / 100 * bar_len)
            cpu_bar   = "█" * filled + "░" * (bar_len - filled)
            ram_filled = int(mem.used_pct / 100 * bar_len)
            ram_bar   = "█" * ram_filled + "░" * (bar_len - ram_filled)

            print(f"\r  CPU  [{cpu_bar}] {cpu.overall_pct:5.1f}%  "
                  f"RAM  [{ram_bar}] {mem.used_pct:5.1f}%  "
                  f"Disk R:{disk.read_mb_s:5.2f}MB/s W:{disk.write_mb_s:5.2f}MB/s  "
                  f"Net ↑{net.sent_mb_s:.2f} ↓{net.recv_mb_s:.2f}MB/s  "
                  f"[{snap.bottleneck_hint}]  seq={snap.poll_seq}",
                  end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopping...\n")

    monitor.stop()

    # Print rolling averages
    avgs = monitor.rolling_averages(window=15)
    print("\n  Rolling averages (last 15 samples):")
    for k, v in avgs.items():
        print(f"    {k:<18} {v}")

    print(f"\n  History buffer: {len(monitor.history())} snapshots")
    print(f"\n{monitor}")


if __name__ == "__main__":
    _demo()
