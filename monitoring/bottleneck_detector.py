"""
monitoring/bottleneck_detector.py
==================================
ResearchFlow AI — System Bottleneck Detector

This module classifies the current system state into a bottleneck type
using a multi-signal heuristic rule engine. It sits above the raw
telemetry collectors (SystemMonitor, GPUMonitor, ProcessTracker) and
produces a single, actionable BottleneckReport consumed by:

  - SchedulerAgent   → task dispatch / deferral decisions
  - LabPipeline      → adaptive throttle controller
  - AlertManager     → structured alert generation
  - Streamlit UI     → bottleneck status badge + detail panel

Bottleneck taxonomy:
  ┌──────────────────┬────────────────────────────────────────────────┐
  │ Type             │ Meaning                                        │
  ├──────────────────┼────────────────────────────────────────────────┤
  │ NONE             │ All resources within normal operating range    │
  │ CPU_BOUND        │ CPU saturated; tasks queued waiting for cores  │
  │ IO_BOUND         │ Disk I/O saturated; CPU underutilised         │
  │ MEM_BOUND        │ RAM pressure; allocations slow or failing      │
  │ GPU_BOUND        │ VRAM saturated; GPU tasks queued              │
  │ OOM_RISK         │ Imminent out-of-memory; highest severity       │
  │ THERMAL          │ GPU/CPU thermal throttling detected            │
  │ MIXED            │ Multiple constraints active simultaneously     │
  └──────────────────┴────────────────────────────────────────────────┘

Detection algorithm:
  Each bottleneck type has a Rule — a function that takes the full
  telemetry context and returns a BottleneckSignal(type, confidence, detail)
  or None. Rules are evaluated in priority order. The final classification
  is the highest-confidence non-NONE signal, unless multiple signals fire
  above a MIXED_THRESHOLD — in which case MIXED is returned.

OS Concepts demonstrated:
  ┌─────────────────────────────────────────────────────────────────┐
  │  Concept                  │  Implementation                    │
  ├───────────────────────────┼────────────────────────────────────┤
  │  Performance profiling    │  Multi-signal bottleneck analysis  │
  │  CPU utilisation analysis │  Sustained load detection          │
  │  I/O wait accounting      │  disk_busy% + CPU idle correlation │
  │  Memory pressure          │  Available RAM + swap + pressure   │
  │  Thermal management       │  GPU temperature threshold         │
  │  Workload classification  │  CPU-bound vs I/O-bound heuristics │
  │  Load trending            │  Rolling window trend analysis     │
  └─────────────────────────────────────────────────────────────────┘

Usage:
    detector = BottleneckDetector()

    # One-shot classification from raw telemetry objects
    report = detector.classify(
        cpu_snap=cpu,
        mem_snap=mem,
        disk_snap=disk,
        gpu_snap=gpu,
        proc_report=proc,
    )
    print(report.primary_type, report.confidence, report.detail)

    # Or from a SystemSnapshot + GPUSnapshot directly
    report = detector.classify_from_snapshots(sys_snap, gpu_snap)

    # History-aware classification (uses rolling trend)
    report = detector.classify_with_history(sys_snap, gpu_snap, history)

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import logging
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# CPU thresholds
CPU_BOUND_THRESHOLD:      float = float(os.getenv("BND_CPU_BOUND",    "85.0"))
CPU_SUSTAINED_THRESHOLD:  float = float(os.getenv("BND_CPU_SUSTAIN",  "80.0"))
CPU_SUSTAINED_WINDOW:     int   = int(os.getenv("BND_CPU_WINDOW",      "5"))    # samples

# Memory thresholds
MEM_PRESSURE_THRESHOLD:   float = float(os.getenv("BND_MEM_PRESSURE", "80.0"))
OOM_RISK_THRESHOLD:       float = float(os.getenv("BND_OOM_RAM",       "92.0"))
SWAP_RISK_THRESHOLD:      float = float(os.getenv("BND_OOM_SWAP",      "40.0"))

# Disk I/O thresholds
IO_BUSY_THRESHOLD:        float = float(os.getenv("BND_IO_BUSY",       "70.0"))
IO_MB_THRESHOLD:          float = float(os.getenv("BND_IO_MB_S",       "150.0"))  # MB/s

# GPU thresholds
GPU_BOUND_THRESHOLD:      float = float(os.getenv("BND_GPU_BOUND",    "85.0"))
GPU_THERMAL_THRESHOLD:    float = float(os.getenv("BND_GPU_THERMAL",  "85.0"))   # °C

# CPU underutilisation when I/O is high (I/O-bound signal)
CPU_IDLE_FOR_IO_BOUND:    float = 50.0

# When ≥ MIXED_MIN_SIGNALS types are active above MIN_CONFIDENCE, classify as MIXED
MIXED_MIN_SIGNALS:        int   = 2
MIN_CONFIDENCE:           float = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK TYPE
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckType(str, Enum):
    NONE     = "none"
    CPU_BOUND = "cpu_bound"
    IO_BOUND  = "io_bound"
    MEM_BOUND = "mem_bound"
    GPU_BOUND = "gpu_bound"
    OOM_RISK  = "oom_risk"
    THERMAL   = "thermal"
    MIXED     = "mixed"


# Human-readable labels for Streamlit badge
BOTTLENECK_LABELS: dict[str, str] = {
    "none":      "✅ Normal",
    "cpu_bound": "🔥 CPU Bound",
    "io_bound":  "💽 I/O Bound",
    "mem_bound": "🧠 Memory Bound",
    "gpu_bound": "🎮 GPU Bound",
    "oom_risk":  "💀 OOM Risk",
    "thermal":   "🌡️ Thermal",
    "mixed":     "⚠️ Mixed",
}

# Recommended scheduler action per bottleneck
BOTTLENECK_ACTIONS: dict[str, str] = {
    "none":      "Dispatch all tasks normally",
    "cpu_bound": "Defer high-CPU tasks; reduce concurrency to 1",
    "io_bound":  "Defer disk-intensive tasks; allow CPU-only tasks",
    "mem_bound": "Defer high-RAM tasks; allow low-RAM tasks",
    "gpu_bound": "Defer GPU-intensive tasks; allow CPU-only tasks",
    "oom_risk":  "Block all non-critical tasks; emergency memory recovery",
    "thermal":   "Defer GPU tasks; reduce LLM inference load",
    "mixed":     "Defer all high-cost tasks; emergency concurrency reduction",
}


# ─────────────────────────────────────────────────────────────────────────────
# TELEMETRY CONTEXT (dependency-free view of all monitoring inputs)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TelemetryContext:
    """
    Flat, dependency-free view of all telemetry signals needed for
    bottleneck classification.

    Built from SystemSnapshot + GPUSnapshot + ProcessTreeReport by
    BottleneckDetector.build_context(). Keeping it flat allows
    unit-testing rules without importing the full monitoring stack.
    """
    # CPU
    cpu_pct:             float = 0.0
    cpu_per_core:        list[float] = field(default_factory=list)
    load_avg_1m:         float = 0.0
    logical_cores:       int   = 1

    # Memory
    ram_used_pct:        float = 0.0
    ram_available_gb:    float = 16.0
    swap_pct:            float = 0.0
    mem_pressure_index:  float = 0.0

    # Disk
    disk_busy_pct:       float = 0.0
    disk_read_mb_s:      float = 0.0
    disk_write_mb_s:     float = 0.0

    # GPU
    gpu_available:       bool  = False
    gpu_vram_used_pct:   float = 0.0
    gpu_temperature_c:   float = 0.0
    gpu_utilisation_pct: float = 0.0

    # Process tree
    own_cpu_pct:         float = 0.0
    own_mem_mb:          float = 0.0
    anomaly_count:       int   = 0
    zombie_count:        int   = 0

    # Rolling history (optional — for sustained-load detection)
    cpu_history:         list[float] = field(default_factory=list)
    ram_history:         list[float] = field(default_factory=list)

    @property
    def cpu_headroom(self) -> float:
        return max(0.0, 100.0 - self.cpu_pct)

    @property
    def ram_headroom(self) -> float:
        return max(0.0, 100.0 - self.ram_used_pct)

    @property
    def is_cpu_idle(self) -> bool:
        return self.cpu_pct < CPU_IDLE_FOR_IO_BOUND

    @property
    def cpu_sustained_high(self) -> bool:
        """True if CPU has been consistently high over the history window."""
        if len(self.cpu_history) < CPU_SUSTAINED_WINDOW:
            return self.cpu_pct >= CPU_BOUND_THRESHOLD
        window = self.cpu_history[-CPU_SUSTAINED_WINDOW:]
        return all(v >= CPU_SUSTAINED_THRESHOLD for v in window)

    @property
    def ram_sustained_high(self) -> bool:
        """True if RAM has been consistently high."""
        if len(self.ram_history) < 3:
            return self.ram_used_pct >= MEM_PRESSURE_THRESHOLD
        return all(v >= MEM_PRESSURE_THRESHOLD for v in self.ram_history[-3:])


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK SIGNAL (result of a single rule evaluation)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BottleneckSignal:
    """
    The output of a single detection rule.

    confidence: 0.0–1.0
      0.0–0.3 → weak hint (don't act alone)
      0.3–0.6 → moderate (consider deferring)
      0.6–0.8 → strong (defer affected tasks)
      0.8–1.0 → definitive (immediate action required)

    contributing_metrics: which metrics triggered this signal
    """
    bottleneck:           BottleneckType
    confidence:           float
    detail:               str
    contributing_metrics: dict[str, float] = field(default_factory=dict)
    recommended_action:   str = ""

    def __post_init__(self) -> None:
        if not self.recommended_action:
            self.recommended_action = BOTTLENECK_ACTIONS.get(
                self.bottleneck.value, ""
            )

    @property
    def is_significant(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE

    def to_dict(self) -> dict:
        return {
            "bottleneck":           self.bottleneck.value,
            "confidence":           round(self.confidence, 3),
            "detail":               self.detail,
            "contributing_metrics": {
                k: round(v, 2) for k, v in self.contributing_metrics.items()
            },
            "recommended_action":   self.recommended_action,
        }


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK REPORT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BottleneckReport:
    """
    Final bottleneck classification published by BottleneckDetector.

    primary_type is the dominant bottleneck (or NONE / MIXED).
    all_signals contains every signal that fired above MIN_CONFIDENCE.
    """
    timestamp:       str
    primary_type:    BottleneckType
    confidence:      float
    detail:          str
    label:           str              # human-readable badge text
    action:          str              # recommended scheduler action
    all_signals:     list[BottleneckSignal]
    context_summary: dict             # key metrics that drove the decision

    @property
    def is_critical(self) -> bool:
        return self.primary_type in (
            BottleneckType.OOM_RISK, BottleneckType.MIXED
        )

    @property
    def is_gpu_affected(self) -> bool:
        return any(
            s.bottleneck in (BottleneckType.GPU_BOUND, BottleneckType.THERMAL)
            for s in self.all_signals
        )

    @property
    def is_memory_affected(self) -> bool:
        return any(
            s.bottleneck in (BottleneckType.MEM_BOUND, BottleneckType.OOM_RISK)
            for s in self.all_signals
        )

    def to_dict(self) -> dict:
        return {
            "timestamp":       self.timestamp,
            "primary_type":    self.primary_type.value,
            "confidence":      round(self.confidence, 3),
            "detail":          self.detail,
            "label":           self.label,
            "action":          self.action,
            "all_signals":     [s.to_dict() for s in self.all_signals],
            "is_critical":     self.is_critical,
            "context_summary": self.context_summary,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION RULES
# ─────────────────────────────────────────────────────────────────────────────

# A Rule takes a TelemetryContext and returns a BottleneckSignal or None
Rule = Callable[[TelemetryContext], BottleneckSignal | None]


def rule_oom_risk(ctx: TelemetryContext) -> BottleneckSignal | None:
    """
    OOM Risk: RAM ≥ 92% OR swap ≥ 40%.

    Highest priority rule — OOM kills the entire process.
    Confidence scales with severity above thresholds.

    Analogous to the Linux kernel's OOM killer trigger conditions:
    when available memory falls below a threshold, the kernel
    begins killing processes. We want to act BEFORE that happens.
    """
    ram_over  = max(0.0, ctx.ram_used_pct - OOM_RISK_THRESHOLD)
    swap_over = max(0.0, ctx.swap_pct - SWAP_RISK_THRESHOLD)

    if ctx.ram_used_pct >= OOM_RISK_THRESHOLD:
        confidence = min(1.0, 0.8 + ram_over / 20.0)
        return BottleneckSignal(
            bottleneck=BottleneckType.OOM_RISK,
            confidence=confidence,
            detail=(
                f"RAM critical: {ctx.ram_used_pct:.1f}% used "
                f"({ctx.ram_available_gb:.2f}GB available) — "
                f"OOM kill risk imminent"
            ),
            contributing_metrics={
                "ram_used_pct":   ctx.ram_used_pct,
                "swap_pct":       ctx.swap_pct,
                "ram_available_gb": ctx.ram_available_gb,
            },
        )

    if ctx.swap_pct >= SWAP_RISK_THRESHOLD:
        confidence = min(0.9, 0.6 + swap_over / 30.0)
        return BottleneckSignal(
            bottleneck=BottleneckType.OOM_RISK,
            confidence=confidence,
            detail=(
                f"Swap pressure: {ctx.swap_pct:.1f}% used — "
                f"system is actively paging; OOM risk elevated"
            ),
            contributing_metrics={
                "swap_pct":     ctx.swap_pct,
                "ram_used_pct": ctx.ram_used_pct,
            },
        )

    return None


def rule_gpu_thermal(ctx: TelemetryContext) -> BottleneckSignal | None:
    """
    GPU Thermal throttling: temperature > 85°C.

    Modern GPUs automatically reduce clock speeds when approaching
    thermal limits. This causes GPU tasks to run slower than expected,
    making the pipeline appear GPU-bound when the real cause is thermal.
    """
    if not ctx.gpu_available:
        return None
    if ctx.gpu_temperature_c >= GPU_THERMAL_THRESHOLD:
        severity = min(1.0, (ctx.gpu_temperature_c - GPU_THERMAL_THRESHOLD) / 15.0)
        confidence = min(0.9, 0.6 + severity * 0.3)
        return BottleneckSignal(
            bottleneck=BottleneckType.THERMAL,
            confidence=confidence,
            detail=(
                f"GPU thermal alert: {ctx.gpu_temperature_c:.0f}°C "
                f"(threshold={GPU_THERMAL_THRESHOLD:.0f}°C) — "
                f"thermal throttling likely reducing GPU throughput"
            ),
            contributing_metrics={
                "temperature_c": ctx.gpu_temperature_c,
                "vram_used_pct": ctx.gpu_vram_used_pct,
            },
        )
    return None


def rule_gpu_bound(ctx: TelemetryContext) -> BottleneckSignal | None:
    """
    GPU VRAM saturation: vram_used_pct ≥ 85%.

    When VRAM is saturated, dispatching another GPU task will either:
    a) Cause a CUDA OOM error (crash the worker)
    b) Cause PyTorch to spill to CPU memory (10-100× slower)

    Confidence scales with proximity to total saturation.
    """
    if not ctx.gpu_available:
        return None
    if ctx.gpu_vram_used_pct >= GPU_BOUND_THRESHOLD:
        over = ctx.gpu_vram_used_pct - GPU_BOUND_THRESHOLD
        confidence = min(0.95, 0.65 + over / 20.0)
        return BottleneckSignal(
            bottleneck=BottleneckType.GPU_BOUND,
            confidence=confidence,
            detail=(
                f"GPU VRAM: {ctx.gpu_vram_used_pct:.1f}% used "
                f"(util={ctx.gpu_utilisation_pct:.1f}%) — "
                f"new GPU tasks risk CUDA OOM"
            ),
            contributing_metrics={
                "vram_used_pct":    ctx.gpu_vram_used_pct,
                "gpu_utilisation":  ctx.gpu_utilisation_pct,
            },
        )
    return None


def rule_cpu_bound(ctx: TelemetryContext) -> BottleneckSignal | None:
    """
    CPU saturation: overall CPU ≥ 85%, or sustained CPU ≥ 80%.

    Two sub-cases:
      1. Instantaneous spike: CPU ≥ 85% right now
      2. Sustained load: CPU ≥ 80% for the last CPU_SUSTAINED_WINDOW samples

    Sustained load is a stronger signal (higher confidence) because
    it rules out momentary measurement noise.

    Analogous to Linux's load average: a 1-minute load average above
    the core count indicates sustained saturation.
    """
    # Check load average as a secondary signal (Unix only)
    load_signal = ctx.load_avg_1m > ctx.logical_cores * 0.9 if ctx.logical_cores > 0 else False

    if ctx.cpu_pct >= CPU_BOUND_THRESHOLD:
        over = ctx.cpu_pct - CPU_BOUND_THRESHOLD
        confidence = min(0.95, 0.70 + over / 20.0)
        if load_signal:
            confidence = min(0.98, confidence + 0.05)
        return BottleneckSignal(
            bottleneck=BottleneckType.CPU_BOUND,
            confidence=confidence,
            detail=(
                f"CPU saturated: {ctx.cpu_pct:.1f}% overall "
                f"(load_avg={ctx.load_avg_1m:.2f}, cores={ctx.logical_cores}) — "
                f"high-CPU tasks should be deferred"
            ),
            contributing_metrics={
                "cpu_pct":       ctx.cpu_pct,
                "load_avg_1m":   ctx.load_avg_1m,
                "logical_cores": float(ctx.logical_cores),
            },
        )

    if ctx.cpu_sustained_high:
        # Sustained moderate load — lower confidence than acute spike
        mean_cpu = statistics.mean(ctx.cpu_history[-CPU_SUSTAINED_WINDOW:]) \
                   if ctx.cpu_history else ctx.cpu_pct
        confidence = min(0.80, 0.55 + (mean_cpu - CPU_SUSTAINED_THRESHOLD) / 30.0)
        return BottleneckSignal(
            bottleneck=BottleneckType.CPU_BOUND,
            confidence=confidence,
            detail=(
                f"CPU sustained high: mean={mean_cpu:.1f}% over "
                f"last {CPU_SUSTAINED_WINDOW} samples — "
                f"concurrency reduction recommended"
            ),
            contributing_metrics={"cpu_mean": mean_cpu, "cpu_pct": ctx.cpu_pct},
        )

    return None


def rule_io_bound(ctx: TelemetryContext) -> BottleneckSignal | None:
    """
    I/O bound: disk busy ≥ 70% AND CPU < 50%.

    The CPU < 50% condition is the key diagnostic signal:
    it rules out CPU-bound workloads where high disk I/O is
    a side effect (e.g. model checkpointing during GPU training).
    When CPU is low but disk is busy, the pipeline is WAITING
    for I/O — a classic I/O-bound pattern.

    Analogous to the I/O wait time in /proc/stat — the fraction
    of time the CPU was idle waiting for I/O to complete.
    """
    is_disk_busy = (
        ctx.disk_busy_pct >= IO_BUSY_THRESHOLD
        or ctx.disk_read_mb_s + ctx.disk_write_mb_s >= IO_MB_THRESHOLD
    )

    if is_disk_busy and ctx.is_cpu_idle:
        # Classic I/O wait signature
        io_mb = ctx.disk_read_mb_s + ctx.disk_write_mb_s
        confidence = min(0.85, 0.55 + ctx.disk_busy_pct / 200.0 + io_mb / 500.0)
        return BottleneckSignal(
            bottleneck=BottleneckType.IO_BOUND,
            confidence=confidence,
            detail=(
                f"I/O bottleneck: disk_busy={ctx.disk_busy_pct:.1f}% "
                f"R:{ctx.disk_read_mb_s:.1f}+W:{ctx.disk_write_mb_s:.1f}MB/s "
                f"CPU={ctx.cpu_pct:.1f}% (low) — "
                f"likely embedding cache miss or large model load"
            ),
            contributing_metrics={
                "disk_busy_pct":  ctx.disk_busy_pct,
                "disk_io_mb_s":   io_mb,
                "cpu_pct":        ctx.cpu_pct,
            },
        )
    return None


def rule_mem_bound(ctx: TelemetryContext) -> BottleneckSignal | None:
    """
    Memory pressure: RAM ≥ 80% AND CPU < 60%.

    When RAM is under pressure but CPU is low, the bottleneck is
    memory management — allocations are slow, the kernel is
    spending time on memory compaction, and processes may be
    waiting for pages to be freed or swapped.

    Below OOM_RISK_THRESHOLD but above MEM_PRESSURE_THRESHOLD.
    OOM_RISK takes priority if RAM ≥ 92%.
    """
    # Don't fire if OOM rule will fire (avoids duplicate signals)
    if ctx.ram_used_pct >= OOM_RISK_THRESHOLD:
        return None

    if ctx.ram_used_pct >= MEM_PRESSURE_THRESHOLD and ctx.cpu_pct < 60.0:
        over = ctx.ram_used_pct - MEM_PRESSURE_THRESHOLD
        confidence = min(0.80, 0.50 + over / 30.0)
        if ctx.ram_sustained_high:
            confidence = min(0.85, confidence + 0.08)
        return BottleneckSignal(
            bottleneck=BottleneckType.MEM_BOUND,
            confidence=confidence,
            detail=(
                f"Memory pressure: RAM {ctx.ram_used_pct:.1f}% used "
                f"({ctx.ram_available_gb:.2f}GB free) CPU={ctx.cpu_pct:.1f}% "
                f"— high-RAM tasks should be deferred"
            ),
            contributing_metrics={
                "ram_used_pct":      ctx.ram_used_pct,
                "ram_available_gb":  ctx.ram_available_gb,
                "mem_pressure_idx":  ctx.mem_pressure_index,
                "cpu_pct":           ctx.cpu_pct,
            },
        )
    return None


# Ordered rule chain (evaluated sequentially; first significant signal wins
# unless MIXED threshold is met)
DEFAULT_RULES: list[Rule] = [
    rule_oom_risk,       # 1. OOM first — highest severity
    rule_gpu_thermal,    # 2. Thermal before GPU saturation
    rule_gpu_bound,      # 3. GPU VRAM
    rule_cpu_bound,      # 4. CPU saturation
    rule_io_bound,       # 5. I/O wait
    rule_mem_bound,      # 6. Memory pressure
]


# ─────────────────────────────────────────────────────────────────────────────
# BOTTLENECK DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class BottleneckDetector:
    """
    Multi-signal bottleneck classifier.

    Runs the rule chain against a TelemetryContext and returns a
    BottleneckReport with the primary bottleneck type, confidence,
    detail message, and recommended scheduler action.

    Classification logic:
      1. Evaluate all rules → collect significant signals (conf ≥ MIN_CONFIDENCE)
      2. If 0 signals → NONE
      3. If 1 signal  → that signal's type
      4. If ≥ MIXED_MIN_SIGNALS signals → MIXED (with highest-confidence detail)
      5. Else → highest-confidence signal

    Customisation:
      rules can be replaced or extended via the constructor.
      Individual rule thresholds are set via environment variables.
    """

    def __init__(self, rules: list[Rule] | None = None) -> None:
        self._rules = rules if rules is not None else list(DEFAULT_RULES)
        self._eval_count = 0
        logger.info(
            "BottleneckDetector init | %d rules | thresholds: "
            "cpu=%.0f%% mem=%.0f%% oom=%.0f%% gpu=%.0f%%",
            len(self._rules),
            CPU_BOUND_THRESHOLD, MEM_PRESSURE_THRESHOLD,
            OOM_RISK_THRESHOLD, GPU_BOUND_THRESHOLD,
        )

    # ── MAIN CLASSIFY ────────────────────────────────────────────────────────

    def classify(self, ctx: TelemetryContext) -> BottleneckReport:
        """
        Classify the current system state from a TelemetryContext.
        The primary entry point for the scheduler and lab pipeline.
        """
        self._eval_count += 1
        ts = datetime.now(timezone.utc).isoformat()

        # Evaluate all rules
        signals: list[BottleneckSignal] = []
        for rule in self._rules:
            try:
                signal = rule(ctx)
                if signal is not None and signal.confidence >= MIN_CONFIDENCE:
                    signals.append(signal)
            except Exception as exc:
                logger.error(
                    "Rule '%s' raised: %s",
                    getattr(rule, "__name__", "?"), exc
                )

        # Classify
        primary_type, confidence, detail = self._resolve(signals)
        label  = BOTTLENECK_LABELS.get(primary_type.value, primary_type.value)
        action = BOTTLENECK_ACTIONS.get(primary_type.value, "")

        context_summary = {
            "cpu_pct":         round(ctx.cpu_pct, 1),
            "ram_used_pct":    round(ctx.ram_used_pct, 1),
            "swap_pct":        round(ctx.swap_pct, 1),
            "disk_busy_pct":   round(ctx.disk_busy_pct, 1),
            "gpu_vram_pct":    round(ctx.gpu_vram_used_pct, 1) if ctx.gpu_available else None,
            "gpu_temp_c":      round(ctx.gpu_temperature_c, 1) if ctx.gpu_available else None,
            "own_cpu_pct":     round(ctx.own_cpu_pct, 1),
            "anomaly_count":   ctx.anomaly_count,
        }

        report = BottleneckReport(
            timestamp=ts,
            primary_type=primary_type,
            confidence=round(confidence, 3),
            detail=detail,
            label=label,
            action=action,
            all_signals=signals,
            context_summary=context_summary,
        )

        if primary_type != BottleneckType.NONE:
            logger.info(
                "Bottleneck: %s (conf=%.2f) | %s",
                primary_type.value, confidence, detail[:80],
            )

        return report

    def classify_from_snapshots(
        self,
        sys_snap:  object,
        gpu_snap:  object | None = None,
        proc_rpt:  object | None = None,
    ) -> BottleneckReport:
        """
        Convenience method: build TelemetryContext from raw snapshot objects
        (SystemSnapshot, GPUSnapshot, ProcessTreeReport) and classify.

        Uses getattr with defaults so this module never imports monitoring/*.
        """
        ctx = self._build_context(sys_snap, gpu_snap, proc_rpt)
        return self.classify(ctx)

    def classify_with_history(
        self,
        sys_snap:      object,
        gpu_snap:      object | None       = None,
        proc_rpt:      object | None       = None,
        sys_history:   list[object] | None = None,
    ) -> BottleneckReport:
        """
        History-aware classification. Populates cpu_history and ram_history
        in the TelemetryContext from the last N SystemSnapshots so that
        sustained-load rules have trend data to work with.
        """
        ctx = self._build_context(sys_snap, gpu_snap, proc_rpt)

        if sys_history:
            ctx.cpu_history = [
                getattr(getattr(s, "cpu", None), "overall_pct", 0.0)
                for s in sys_history[-CPU_SUSTAINED_WINDOW * 2:]
            ]
            ctx.ram_history = [
                getattr(getattr(s, "memory", None), "used_pct", 0.0)
                for s in sys_history[-10:]
            ]

        return self.classify(ctx)

    # ── RESOLUTION LOGIC ─────────────────────────────────────────────────────

    def _resolve(
        self,
        signals: list[BottleneckSignal],
    ) -> tuple[BottleneckType, float, str]:
        """
        Resolve a list of signals into a (type, confidence, detail) triple.

        Decision tree:
          0 signals           → NONE
          1 signal            → that signal
          ≥ 2 high-conf sigs  → MIXED (aggregate detail)
          else                → highest confidence signal
        """
        if not signals:
            return BottleneckType.NONE, 0.0, "All resources within normal range"

        if len(signals) == 1:
            s = signals[0]
            return s.bottleneck, s.confidence, s.detail

        # Multiple signals: check for MIXED
        high_conf = [s for s in signals if s.confidence >= 0.6]
        if len(high_conf) >= MIXED_MIN_SIGNALS:
            types_str = " + ".join(s.bottleneck.value for s in high_conf)
            detail = (
                f"MIXED bottleneck ({types_str}): "
                + "; ".join(s.detail[:50] for s in high_conf[:3])
            )
            confidence = min(1.0, max(s.confidence for s in high_conf) * 1.1)
            return BottleneckType.MIXED, round(confidence, 3), detail

        # Return highest-confidence single signal
        best = max(signals, key=lambda s: s.confidence)
        return best.bottleneck, best.confidence, best.detail

    # ── CONTEXT BUILDER ──────────────────────────────────────────────────────

    @staticmethod
    def _build_context(
        sys_snap: object,
        gpu_snap: object | None,
        proc_rpt: object | None,
    ) -> TelemetryContext:
        """
        Extract flat values from rich snapshot objects into TelemetryContext.
        All getattr calls have safe defaults — never raises.
        """
        cpu = getattr(sys_snap, "cpu",    None)
        mem = getattr(sys_snap, "memory", None)
        dsk = getattr(sys_snap, "disk",   None)

        gpu_avail   = False
        gpu_vram    = 0.0
        gpu_temp    = 0.0
        gpu_util    = 0.0

        if gpu_snap:
            primary = getattr(gpu_snap, "primary", None)
            gpu_avail = getattr(gpu_snap, "is_available", False)
            if primary and gpu_avail:
                gpu_vram = getattr(primary, "vram_used_pct",   0.0)
                gpu_temp = getattr(primary, "temperature_c",   0.0)
                gpu_util = getattr(primary, "utilisation_pct", 0.0)

        own_cpu = 0.0
        own_mem = 0.0
        anomaly_count = 0
        zombie_count  = 0
        if proc_rpt:
            own_cpu       = getattr(proc_rpt, "own_cpu_pct",  0.0)
            own_mem       = getattr(proc_rpt, "own_mem_mb",   0.0)
            anomaly_count = getattr(proc_rpt, "anomaly_count", 0)
            zombie_count  = getattr(proc_rpt, "zombie_count",  0)

        vm_mem  = getattr(mem, "available_gb",     16.0)
        vm_pres = getattr(mem, "pressure_index",    0.0)

        return TelemetryContext(
            cpu_pct=             getattr(cpu, "overall_pct",    0.0),
            cpu_per_core=        getattr(cpu, "per_core_pct",   []),
            load_avg_1m=         getattr(cpu, "load_avg_1m",    0.0),
            logical_cores=       getattr(cpu, "logical_cores",  1),
            ram_used_pct=        getattr(mem, "used_pct",       0.0),
            ram_available_gb=    vm_mem,
            swap_pct=            getattr(mem, "swap_pct",       0.0),
            mem_pressure_index=  vm_pres,
            disk_busy_pct=       getattr(dsk, "busy_pct",       0.0),
            disk_read_mb_s=      getattr(dsk, "read_mb_s",      0.0),
            disk_write_mb_s=     getattr(dsk, "write_mb_s",     0.0),
            gpu_available=       gpu_avail,
            gpu_vram_used_pct=   gpu_vram,
            gpu_temperature_c=   gpu_temp,
            gpu_utilisation_pct= gpu_util,
            own_cpu_pct=         own_cpu,
            own_mem_mb=          own_mem,
            anomaly_count=       anomaly_count,
            zombie_count=        zombie_count,
        )

    # ── STATS ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "evaluations": self._eval_count,
            "rule_count":  len(self._rules),
            "rules":       [getattr(r, "__name__", "?") for r in self._rules],
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Demonstrates bottleneck classification under 6 simulated scenarios.
    Run with: python -m monitoring.bottleneck_detector
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    detector = BottleneckDetector()

    scenarios = [
        ("Normal load",        TelemetryContext(cpu_pct=35.0, ram_used_pct=45.0,
            disk_busy_pct=5.0, gpu_available=True, gpu_vram_used_pct=30.0)),
        ("CPU saturated",      TelemetryContext(cpu_pct=91.0, ram_used_pct=50.0,
            load_avg_1m=7.5, logical_cores=8,
            cpu_history=[88.0, 89.0, 90.0, 91.0, 91.0])),
        ("I/O bottleneck",     TelemetryContext(cpu_pct=22.0, ram_used_pct=40.0,
            disk_busy_pct=82.0, disk_read_mb_s=185.0)),
        ("Memory pressure",    TelemetryContext(cpu_pct=30.0, ram_used_pct=84.0,
            ram_available_gb=2.1, swap_pct=12.0,
            ram_history=[82.0, 83.0, 84.0])),
        ("OOM risk",           TelemetryContext(cpu_pct=55.0, ram_used_pct=94.5,
            ram_available_gb=0.8, swap_pct=45.0)),
        ("Mixed: CPU+GPU",     TelemetryContext(cpu_pct=87.0, ram_used_pct=55.0,
            gpu_available=True, gpu_vram_used_pct=92.0,
            gpu_temperature_c=88.0)),
    ]

    print("\n" + "═" * 70)
    print("  BOTTLENECK DETECTION — 6 SCENARIOS")
    print("═" * 70)

    for name, ctx in scenarios:
        report = detector.classify(ctx)
        print(f"\n  Scenario: {name}")
        print(f"  {'─' * 50}")
        print(f"  Primary:    {report.label}")
        print(f"  Confidence: {report.confidence:.3f}")
        print(f"  Detail:     {report.detail[:70]}")
        print(f"  Action:     {report.action[:70]}")
        if len(report.all_signals) > 1:
            print(f"  Signals:    {[s.bottleneck.value for s in report.all_signals]}")

    print("\n" + "═" * 70)
    print("  DETECTOR STATS")
    print("═" * 70)
    for k, v in detector.stats().items():
        print(f"  {k:<20} {v}")


if __name__ == "__main__":
    _demo()
