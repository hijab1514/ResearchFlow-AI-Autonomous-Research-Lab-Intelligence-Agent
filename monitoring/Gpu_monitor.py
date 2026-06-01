"""
monitoring/gpu_monitor.py
==========================
ResearchFlow AI — GPU Memory & Utilisation Monitor

This module provides detailed CUDA GPU telemetry for the ResearchFlow AI
monitoring stack. It wraps torch.cuda memory statistics and (optionally)
nvidia-smi via pynvml to give a complete picture of GPU resource usage.

Why GPU monitoring matters for ResearchFlow AI:
  The most resource-intensive stages of the pipeline run on GPU:
    - SPECTER embedding generation (Sentence Transformer inference)
    - LLM inference via OpenAI API (remote, but local GPU proxy if used)
    - FAISS GPU index operations (if GPU FAISS is available)

  Without GPU monitoring, the scheduler has no way to know when VRAM
  is approaching capacity — leading to CUDA OOM crashes mid-pipeline.
  GPUMonitor feeds real-time VRAM data to ResourcePolicy so the
  scheduler can defer GPU-intensive tasks before they crash workers.

Architecture:
  GPUMonitor
    ├── _collect_torch_stats()    → torch.cuda.memory_stats()
    ├── _collect_nvml_stats()     → pynvml (optional, richer data)
    └── GPUSnapshot               → published to SystemMonitor

Metrics collected per device:
  Memory:
    - VRAM total / allocated / reserved / free (bytes + %)
    - Peak allocated (high-water mark since last reset)
    - Active memory blocks count
    - Memory fragmentation estimate

  Compute (via pynvml, if available):
    - GPU utilisation % (SM occupancy)
    - Memory controller utilisation %
    - Temperature (°C)
    - Power draw (W) and power limit (W)
    - Clock speeds: graphics, memory, SM (MHz)
    - Fan speed %
    - PCIe bandwidth (TX/RX MB/s)

  Process:
    - Per-process VRAM usage (which processes are consuming GPU memory)

OS Concepts demonstrated:
  - Device driver interface  : torch.cuda / pynvml as kernel driver proxies
  - Resource accounting      : per-process VRAM tracking
  - Hardware interrupt proxy : utilisation % from performance counters
  - Memory manager view      : allocated vs reserved (fragmentation)
  - Thermal management       : temperature monitoring

Usage:
    monitor = GPUMonitor()
    monitor.start()

    snap = monitor.latest()
    if snap and snap.is_available:
        print(snap.devices[0].vram_used_pct)
        print(snap.devices[0].utilisation_pct)

    # Check if safe to dispatch GPU task
    safe, reason = monitor.is_safe_for_task("high")
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
from typing import Deque

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
except ImportError:
    torch = None                # type: ignore
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    pynvml = None               # type: ignore
    NVML_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

POLL_INTERVAL_S: float = float(os.getenv("GPU_POLL_INTERVAL",  "2.0"))
HISTORY_SIZE:    int   = int(os.getenv("GPU_HISTORY_SIZE",     "150"))

# VRAM thresholds (mirrors scheduler/resource_policy.py)
VRAM_WARN_PCT:  float = float(os.getenv("SCHEDULER_GPU_THRESHOLD", "85.0"))
VRAM_CRIT_PCT:  float = 95.0
VRAM_EMER_PCT:  float = 97.0

# GPU cost headroom requirements
GPU_HEADROOM: dict[str, float] = {
    "none":   0.0,
    "low":    5.0,
    "medium": 12.0,
    "high":   20.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeviceSnapshot:
    """
    Telemetry snapshot for a single GPU device.

    torch_stats are always available when CUDA is present.
    nvml_stats are richer but require pynvml (nvidia driver).
    """
    device_index:        int
    device_name:         str

    # ── VRAM (always available via torch.cuda) ────────────────────────────
    vram_total_bytes:    int
    vram_allocated_bytes: int   # actively used by tensors
    vram_reserved_bytes: int    # held by PyTorch allocator (pool)
    vram_free_bytes:     int    # truly free (not in pool)
    vram_peak_bytes:     int    # high-water mark since last reset

    vram_total_gb:       float
    vram_used_gb:        float  # = allocated (what the scheduler cares about)
    vram_reserved_gb:    float
    vram_free_gb:        float
    vram_peak_gb:        float

    vram_used_pct:       float  # allocated / total × 100
    vram_reserved_pct:   float  # reserved / total × 100
    vram_headroom_pct:   float  # 100 - vram_used_pct

    # Fragmentation: (reserved - allocated) / reserved
    # High fragmentation means PyTorch is holding memory it isn't using
    fragmentation_pct:   float

    # Active allocation blocks
    active_blocks:       int

    # ── NVML stats (optional) ─────────────────────────────────────────────
    utilisation_pct:     float = 0.0    # SM occupancy %
    mem_util_pct:        float = 0.0    # memory controller busy %
    temperature_c:       float = 0.0    # GPU core temp (°C)
    power_w:             float = 0.0    # current power draw (W)
    power_limit_w:       float = 0.0    # TDP limit (W)
    clock_graphics_mhz:  float = 0.0    # graphics clock (MHz)
    clock_memory_mhz:    float = 0.0    # memory clock (MHz)
    clock_sm_mhz:        float = 0.0    # SM clock (MHz)
    fan_speed_pct:       float = 0.0    # fan speed (%)
    pcie_tx_mb_s:        float = 0.0    # PCIe TX throughput (MB/s)
    pcie_rx_mb_s:        float = 0.0    # PCIe RX throughput (MB/s)
    nvml_available:      bool  = False

    # ── Per-process VRAM usage ────────────────────────────────────────────
    process_vram: list[dict] = field(default_factory=list)
    # [{"pid": int, "name": str, "vram_mb": float}]

    @property
    def is_warning(self) -> bool:
        return self.vram_used_pct >= VRAM_WARN_PCT

    @property
    def is_critical(self) -> bool:
        return self.vram_used_pct >= VRAM_CRIT_PCT

    @property
    def is_emergency(self) -> bool:
        return self.vram_used_pct >= VRAM_EMER_PCT

    @property
    def thermal_throttling(self) -> bool:
        """Heuristic: temp > 85°C may cause throttling on most GPUs."""
        return self.temperature_c > 85.0

    @property
    def power_efficiency(self) -> float:
        """Power utilisation % (current / limit). 0.0 if limit unknown."""
        if self.power_limit_w <= 0:
            return 0.0
        return round(self.power_w / self.power_limit_w * 100, 2)

    def headroom_for(self, gpu_cost: str) -> bool:
        """
        Returns True if this device has enough free VRAM
        to safely run a task with the given gpu_cost level.
        """
        required = GPU_HEADROOM.get(gpu_cost, 0.0)
        return self.vram_headroom_pct >= required

    def to_dict(self) -> dict:
        return {
            "device_index":       self.device_index,
            "device_name":        self.device_name,
            "vram_total_gb":      round(self.vram_total_gb, 3),
            "vram_used_gb":       round(self.vram_used_gb, 3),
            "vram_reserved_gb":   round(self.vram_reserved_gb, 3),
            "vram_free_gb":       round(self.vram_free_gb, 3),
            "vram_peak_gb":       round(self.vram_peak_gb, 3),
            "vram_used_pct":      round(self.vram_used_pct, 2),
            "vram_reserved_pct":  round(self.vram_reserved_pct, 2),
            "vram_headroom_pct":  round(self.vram_headroom_pct, 2),
            "fragmentation_pct":  round(self.fragmentation_pct, 2),
            "active_blocks":      self.active_blocks,
            "utilisation_pct":    round(self.utilisation_pct, 1),
            "temperature_c":      round(self.temperature_c, 1),
            "power_w":            round(self.power_w, 1),
            "clock_graphics_mhz": round(self.clock_graphics_mhz, 0),
            "fan_speed_pct":      round(self.fan_speed_pct, 1),
            "is_warning":         self.is_warning,
            "is_critical":        self.is_critical,
            "thermal_throttling": self.thermal_throttling,
            "nvml_available":     self.nvml_available,
        }


@dataclass
class GPUSnapshot:
    """
    Multi-device GPU telemetry snapshot.

    is_available is False when no CUDA device is detected —
    all consumers check this before reading device metrics.
    """
    timestamp:       str
    is_available:    bool
    device_count:    int
    devices:         list[DeviceSnapshot]
    poll_seq:        int

    @property
    def primary(self) -> DeviceSnapshot | None:
        """Return the first (primary) device, or None if no GPU."""
        return self.devices[0] if self.devices else None

    @property
    def any_critical(self) -> bool:
        return any(d.is_critical for d in self.devices)

    @property
    def any_emergency(self) -> bool:
        return any(d.is_emergency for d in self.devices)

    @property
    def max_vram_used_pct(self) -> float:
        """Highest VRAM utilisation across all devices."""
        return max((d.vram_used_pct for d in self.devices), default=0.0)

    def to_dict(self) -> dict:
        return {
            "timestamp":     self.timestamp,
            "is_available":  self.is_available,
            "device_count":  self.device_count,
            "devices":       [d.to_dict() for d in self.devices],
            "max_vram_pct":  round(self.max_vram_used_pct, 2),
            "any_critical":  self.any_critical,
        }


# ─────────────────────────────────────────────────────────────────────────────
# GPU MONITOR
# ─────────────────────────────────────────────────────────────────────────────

class GPUMonitor:
    """
    Background daemon thread that polls GPU telemetry every
    POLL_INTERVAL_S seconds and maintains a thread-safe snapshot buffer.

    Gracefully degrades:
      - No CUDA / no GPU  → all snapshots have is_available=False
      - pynvml missing    → torch-only stats (VRAM only, no utilisation)
      - Device error      → logs warning, returns partial snapshot

    Thread-safe: latest() and history() never block.
    """

    def __init__(
        self,
        poll_interval_s: float = POLL_INTERVAL_S,
        history_size:    int   = HISTORY_SIZE,
    ) -> None:
        self._interval    = poll_interval_s
        self._history:    Deque[GPUSnapshot] = deque(maxlen=history_size)
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread:     threading.Thread | None = None
        self._poll_seq    = 0
        self._nvml_handles: list = []

        # Detect devices
        if TORCH_AVAILABLE and torch is not None:
            self._device_count = torch.cuda.device_count()
        else:
            self._device_count = 0

        # Initialise NVML handles (one per device)
        if NVML_AVAILABLE and pynvml is not None:
            for i in range(self._device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    self._nvml_handles.append(handle)
                except Exception:
                    self._nvml_handles.append(None)

        logger.info(
            "GPUMonitor init | devices=%d | torch=%s | nvml=%s | interval=%.1fs",
            self._device_count, TORCH_AVAILABLE, NVML_AVAILABLE, poll_interval_s,
        )

    # ── LIFECYCLE ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            logger.warning("GPUMonitor already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="researchflow-gpumon",
            daemon=True,
        )
        self._thread.start()
        logger.info("GPUMonitor started | thread=%s", self._thread.name)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self._interval * 3)
        if NVML_AVAILABLE and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        logger.info("GPUMonitor stopped")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── SNAPSHOT ACCESS ──────────────────────────────────────────────────────

    def latest(self) -> GPUSnapshot | None:
        """Most recent snapshot. Non-blocking. Returns None before first poll."""
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self, n: int | None = None) -> list[GPUSnapshot]:
        with self._lock:
            snaps = list(self._history)
        return snaps[-n:] if n else snaps

    def history_dicts(self, n: int | None = None) -> list[dict]:
        return [s.to_dict() for s in self.history(n)]

    # ── DISPATCH GATE ────────────────────────────────────────────────────────

    def is_safe_for_task(
        self,
        gpu_cost:     str = "medium",
        device_index: int = 0,
    ) -> tuple[bool, str]:
        """
        Check whether it is safe to dispatch a GPU task with the given cost level.
        Returns (safe: bool, reason: str).

        Called by ResourcePolicy before dispatching GPU-intensive tasks.
        """
        if gpu_cost == "none":
            return True, "Task requires no GPU"

        snap = self.latest()
        if snap is None or not snap.is_available:
            return True, "No GPU detected — task will run on CPU"

        if device_index >= len(snap.devices):
            return True, f"Device {device_index} not found — fallback to CPU"

        dev = snap.devices[device_index]

        if dev.is_emergency:
            return False, (
                f"GPU VRAM EMERGENCY: {dev.vram_used_pct:.1f}% "
                f"({dev.vram_used_gb:.2f}/{dev.vram_total_gb:.2f} GB) — "
                f"blocking all GPU tasks"
            )

        if not dev.headroom_for(gpu_cost):
            required = GPU_HEADROOM.get(gpu_cost, 0.0)
            return False, (
                f"GPU VRAM INSUFFICIENT: headroom={dev.vram_headroom_pct:.1f}% "
                f"< required={required:.1f}% for {gpu_cost}-cost task"
            )

        if dev.thermal_throttling:
            return False, (
                f"GPU THERMAL: {dev.temperature_c:.0f}°C — "
                f"dispatch deferred to prevent thermal throttling"
            )

        return True, (
            f"GPU safe: VRAM {dev.vram_used_pct:.1f}% "
            f"headroom={dev.vram_headroom_pct:.1f}%"
        )

    # ── PEAK RESET ────────────────────────────────────────────────────────────

    def reset_peak_stats(self, device_index: int | None = None) -> None:
        """
        Reset PyTorch's VRAM peak-allocated counter.
        Call at the start of each pipeline stage to track per-stage peaks.
        """
        if not TORCH_AVAILABLE or torch is None:
            return
        if device_index is not None:
            torch.cuda.reset_peak_memory_stats(device_index)
        else:
            for i in range(self._device_count):
                torch.cuda.reset_peak_memory_stats(i)

    def empty_cache(self) -> None:
        """
        Release PyTorch's cached (reserved but unused) VRAM back to the OS.
        Call between pipeline stages to reduce fragmentation.
        """
        if TORCH_AVAILABLE and torch is not None:
            torch.cuda.empty_cache()
            logger.info("GPU cache emptied (reserved VRAM released to OS)")

    # ── POLL LOOP ─────────────────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        logger.debug("GPUMonitor poll loop started")
        next_poll = time.monotonic()

        while not self._stop_event.is_set():
            try:
                snap = self._collect()
                with self._lock:
                    self._history.append(snap)
            except Exception as exc:
                logger.error("GPUMonitor poll error: %s", exc, exc_info=True)

            next_poll += self._interval
            sleep_s = next_poll - time.monotonic()
            if sleep_s > 0:
                self._stop_event.wait(timeout=sleep_s)

        logger.debug("GPUMonitor poll loop exited")

    # ── COLLECTION ────────────────────────────────────────────────────────────

    def _collect(self) -> GPUSnapshot:
        self._poll_seq += 1
        ts = datetime.now(timezone.utc).isoformat()

        if not TORCH_AVAILABLE or self._device_count == 0:
            return GPUSnapshot(
                timestamp=ts,
                is_available=False,
                device_count=0,
                devices=[],
                poll_seq=self._poll_seq,
            )

        devices = []
        for i in range(self._device_count):
            try:
                dev = self._collect_device(i)
                devices.append(dev)
            except Exception as exc:
                logger.warning("GPU device %d collection error: %s", i, exc)

        return GPUSnapshot(
            timestamp=ts,
            is_available=True,
            device_count=len(devices),
            devices=devices,
            poll_seq=self._poll_seq,
        )

    def _collect_device(self, index: int) -> DeviceSnapshot:
        """Collect metrics for a single CUDA device."""

        # ── torch.cuda stats ─────────────────────────────────────────────────
        device_name = torch.cuda.get_device_name(index)
        props       = torch.cuda.get_device_properties(index)
        total       = props.total_memory

        mem_stats   = torch.cuda.memory_stats(index)
        allocated   = mem_stats.get("allocated_bytes.all.current", 0)
        reserved    = mem_stats.get("reserved_bytes.all.current",  0)
        peak_alloc  = mem_stats.get("allocated_bytes.all.peak",    0)
        active_blk  = mem_stats.get("active_blocks.all.current",   0)

        free_bytes  = total - reserved   # truly free (outside PyTorch pool)
        used_pct    = (allocated / total * 100) if total > 0 else 0.0
        res_pct     = (reserved  / total * 100) if total > 0 else 0.0
        head_pct    = max(0.0, 100.0 - used_pct)

        # Fragmentation: wasted space in the reserved pool
        frag = 0.0
        if reserved > 0:
            frag = max(0.0, (reserved - allocated) / reserved * 100)

        def to_gb(b: int) -> float:
            return b / (1024 ** 3)

        base = DeviceSnapshot(
            device_index=index,
            device_name=device_name,
            vram_total_bytes=total,
            vram_allocated_bytes=allocated,
            vram_reserved_bytes=reserved,
            vram_free_bytes=max(0, free_bytes),
            vram_peak_bytes=peak_alloc,
            vram_total_gb=round(to_gb(total), 3),
            vram_used_gb=round(to_gb(allocated), 3),
            vram_reserved_gb=round(to_gb(reserved), 3),
            vram_free_gb=round(to_gb(max(0, free_bytes)), 3),
            vram_peak_gb=round(to_gb(peak_alloc), 3),
            vram_used_pct=round(used_pct, 2),
            vram_reserved_pct=round(res_pct, 2),
            vram_headroom_pct=round(head_pct, 2),
            fragmentation_pct=round(frag, 2),
            active_blocks=active_blk,
        )

        # ── pynvml enrichment (optional) ─────────────────────────────────────
        if NVML_AVAILABLE and pynvml is not None and index < len(self._nvml_handles):
            handle = self._nvml_handles[index]
            if handle is not None:
                self._enrich_nvml(base, handle)

        return base

    def _enrich_nvml(self, dev: DeviceSnapshot, handle: object) -> None:
        """
        Enrich a DeviceSnapshot with pynvml data.
        All failures are caught silently — nvml is best-effort.
        """
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            dev.utilisation_pct = float(util.gpu)
            dev.mem_util_pct    = float(util.memory)
        except Exception:
            pass

        try:
            dev.temperature_c = float(
                pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            )
        except Exception:
            pass

        try:
            dev.power_w       = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            dev.power_limit_w = pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
        except Exception:
            pass

        try:
            dev.clock_graphics_mhz = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            )
            dev.clock_memory_mhz   = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            )
            dev.clock_sm_mhz       = float(
                pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            )
        except Exception:
            pass

        try:
            dev.fan_speed_pct = float(pynvml.nvmlDeviceGetFanSpeed(handle))
        except Exception:
            pass

        try:
            pcie = pynvml.nvmlDeviceGetPcieThroughput
            dev.pcie_tx_mb_s = pcie(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024.0
            dev.pcie_rx_mb_s = pcie(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024.0
        except Exception:
            pass

        try:
            procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            dev.process_vram = [
                {
                    "pid":     p.pid,
                    "vram_mb": round(p.usedGpuMemory / (1024 ** 2), 1),
                }
                for p in procs
            ]
        except Exception:
            pass

        dev.nvml_available = True

    # ── UTILITY ──────────────────────────────────────────────────────────────

    def rolling_vram_pct(self, window: int = 10) -> float:
        """
        Rolling average VRAM utilisation over last `window` snapshots.
        Used by the adaptive throttle controller.
        """
        snaps = self.history(window)
        vals  = [s.max_vram_used_pct for s in snaps if s.is_available]
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    def vram_time_series(
        self,
        n: int = 60,
        device_index: int = 0,
    ) -> list[tuple[str, float]]:
        """
        Returns [(timestamp, vram_used_pct)] for the last n snapshots.
        Used by the Streamlit resource dashboard GPU chart.
        """
        snaps = self.history(n)
        result = []
        for s in snaps:
            if s.is_available and device_index < len(s.devices):
                result.append((s.timestamp, s.devices[device_index].vram_used_pct))
        return result

    def status(self) -> dict:
        """Compact status dict for Streamlit sidebar."""
        snap = self.latest()
        if snap is None or not snap.is_available:
            return {"available": False, "device_count": 0}
        dev = snap.primary
        return {
            "available":      True,
            "device_count":   snap.device_count,
            "device_name":    dev.device_name if dev else "?",
            "vram_used_pct":  dev.vram_used_pct if dev else 0.0,
            "vram_used_gb":   dev.vram_used_gb if dev else 0.0,
            "vram_total_gb":  dev.vram_total_gb if dev else 0.0,
            "utilisation_pct": dev.utilisation_pct if dev else 0.0,
            "temperature_c":  dev.temperature_c if dev else 0.0,
            "nvml_available": dev.nvml_available if dev else False,
            "any_critical":   snap.any_critical,
        }

    def __repr__(self) -> str:
        snap = self.latest()
        if snap is None or not snap.is_available:
            return "GPUMonitor(available=False)"
        pct = f"{snap.max_vram_used_pct:.1f}%"
        return (
            f"GPUMonitor(devices={snap.device_count}, "
            f"max_vram={pct}, nvml={NVML_AVAILABLE})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_gpu_monitor: GPUMonitor | None = None


def get_gpu_monitor() -> GPUMonitor:
    """Return the module-level singleton GPUMonitor."""
    global _default_gpu_monitor
    if _default_gpu_monitor is None:
        _default_gpu_monitor = GPUMonitor()
    return _default_gpu_monitor


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Live GPU monitor terminal display.
    Run with: python -m monitoring.gpu_monitor
    Press Ctrl+C to stop.
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    monitor = GPUMonitor(poll_interval_s=2.0, history_size=30)
    monitor.start()

    print(f"\nResearchFlow AI — GPU Monitor  (Ctrl+C to stop)")
    print(f"  CUDA: {TORCH_AVAILABLE} | NVML: {NVML_AVAILABLE}\n")

    try:
        while True:
            time.sleep(2.0)
            snap = monitor.latest()

            if snap is None:
                print("  Warming up...")
                continue

            if not snap.is_available:
                print("  No CUDA GPU detected. Metrics unavailable.")
                print("  (Scheduler will use CPU fallback for all tasks)")
                continue

            for dev in snap.devices:
                bar_len = 30
                filled  = int(dev.vram_used_pct / 100 * bar_len)
                bar     = "█" * filled + "░" * (bar_len - filled)

                status = "⚠ WARN" if dev.is_warning else "✓ OK"
                if dev.is_critical:
                    status = "🔴 CRIT"
                if dev.is_emergency:
                    status = "💀 EMER"

                print(
                    f"\r  GPU{dev.device_index} [{dev.device_name[:20]:<20}] "
                    f"VRAM [{bar}] {dev.vram_used_pct:5.1f}%  "
                    f"({dev.vram_used_gb:.2f}/{dev.vram_total_gb:.2f}GB)  "
                    f"Util:{dev.utilisation_pct:5.1f}%  "
                    f"Temp:{dev.temperature_c:.0f}°C  "
                    f"Pwr:{dev.power_w:.0f}W  "
                    f"Frag:{dev.fragmentation_pct:.1f}%  "
                    f"{status}  seq={snap.poll_seq}",
                    end="", flush=True,
                )

    except KeyboardInterrupt:
        print("\n\nStopping...\n")

    monitor.stop()

    # Final status
    status = monitor.status()
    print("\n  Final GPU status:")
    for k, v in status.items():
        print(f"    {k:<22} {v}")

    print(f"\n  Rolling VRAM avg (last 10): {monitor.rolling_vram_pct(10):.1f}%")
    print(f"\n{monitor}")


if __name__ == "__main__":
    _demo()
