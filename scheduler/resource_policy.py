"""
scheduler/resource_policy.py
=============================
ResearchFlow AI — Resource-Aware Dispatch Policy Engine

This module implements the policy layer that sits between the
PriorityQueue (which decides *what* to dispatch next) and the
SchedulerAgent (which decides *when* to dispatch it).

Every time the scheduler loop ticks and pops a QueueEntry, it calls
ResourcePolicy.evaluate(entry, snapshot) before dispatching.
The policy returns a PolicyDecision telling the scheduler to:
  - DISPATCH       : resources are sufficient — proceed immediately
  - DEFER          : resources are insufficient — requeue with backoff
  - FORCE_DISPATCH : entry has been deferred too many times (starvation
                     prevention safety valve) — dispatch regardless
  - SKIP           : entry is cancelled or invalid — discard silently

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │                   POLICY EVALUATION CHAIN                        │
  │                                                                  │
  │  QueueEntry + SystemSnapshot                                     │
  │         │                                                        │
  │         ▼                                                        │
  │  [1] CancellationCheck   → SKIP  (fast path)                    │
  │         │                                                        │
  │         ▼                                                        │
  │  [2] CriticalBypass      → DISPATCH (CRITICAL always runs)      │
  │         │                                                        │
  │         ▼                                                        │
  │  [3] StarvationValve     → FORCE_DISPATCH (too many deferrals)  │
  │         │                                                        │
  │         ▼                                                        │
  │  [4] OOMRiskGuard        → DEFER (protect against OOM)          │
  │         │                                                        │
  │         ▼                                                        │
  │  [5] CPUThresholdCheck   → DEFER (CPU overloaded)               │
  │         │                                                        │
  │         ▼                                                        │
  │  [6] RAMThresholdCheck   → DEFER (insufficient free RAM)        │
  │         │                                                        │
  │         ▼                                                        │
  │  [7] GPUThresholdCheck   → DEFER (VRAM saturated)               │
  │         │                                                        │
  │         ▼                                                        │
  │  [8] CostLevelCheck      → DEFER (fine-grained cost matching)   │
  │         │                                                        │
  │         ▼                                                        │
  │  DISPATCH                (all checks passed)                    │
  └──────────────────────────────────────────────────────────────────┘

OS Concepts demonstrated:
  - Admission control   : policies gate task entry into the CPU
  - Resource accounting : per-task cost profiles checked against live state
  - Feedback scheduling : real-time telemetry drives dispatch decisions
  - Starvation prevention : force-dispatch after MAX_DEFER_COUNT
  - Policy separation   : rules are data (ThresholdConfig), not hardcoded

Design:
  ResourcePolicy is configured via ThresholdConfig (loaded from
  configs/scheduler_config.yaml). All thresholds are externally
  configurable with sensible defaults — zero hardcoded magic numbers
  in the policy logic itself.

  PolicyRules is a list of independent Rule callables evaluated in
  chain order. Short-circuits on the first non-DISPATCH decision.
  New rules can be injected without modifying existing logic.

Usage:
    config  = ThresholdConfig()
    policy  = ResourcePolicy(config)
    snapshot = lab_agent.snapshot()

    entry   = pq.pop()
    decision = policy.evaluate(entry, snapshot)

    if decision.action == PolicyAction.DISPATCH:
        dispatcher.dispatch(entry)
    elif decision.action == PolicyAction.DEFER:
        pq.requeue(entry)
    elif decision.action == PolicyAction.FORCE_DISPATCH:
        dispatcher.dispatch(entry)   # bypass but log warning
    else:   # SKIP
        pass   # discard silently

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, NamedTuple

from scheduler.priority_queue import Priority, QueueEntry

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# POLICY ACTION ENUM
# ─────────────────────────────────────────────────────────────────────────────

class PolicyAction(str, Enum):
    """
    The four possible outcomes of a policy evaluation.

    DISPATCH       — conditions met; submit task to worker thread
    DEFER          — conditions not met; requeue with incremented defer_count
    FORCE_DISPATCH — deferred too many times; dispatch despite conditions
                     (starvation prevention)
    SKIP           — entry is cancelled or invalid; discard without action
    """
    DISPATCH       = "dispatch"
    DEFER          = "defer"
    FORCE_DISPATCH = "force_dispatch"
    SKIP           = "skip"


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE COST LEVELS (mirrors ResourceProfile in scheduler_agent.py)
# ─────────────────────────────────────────────────────────────────────────────

# Minimum required headroom per cost level (as % of total resource)
# e.g. a "high" CPU task needs at least 20% CPU headroom to dispatch
CPU_HEADROOM_REQUIRED: dict[str, float] = {
    "none":   0.0,
    "low":    5.0,
    "medium": 15.0,
    "high":   25.0,
}

RAM_HEADROOM_REQUIRED: dict[str, float] = {
    "none":   0.0,
    "low":    8.0,
    "medium": 15.0,
    "high":   25.0,
}

GPU_HEADROOM_REQUIRED: dict[str, float] = {
    "none":   0.0,
    "low":    5.0,
    "medium": 12.0,
    "high":   20.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# THRESHOLD CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ThresholdConfig:
    """
    All resource thresholds for policy decisions.

    Loaded from configs/scheduler_config.yaml at startup (defaults
    shown here are also the fallback values if the file is absent).

    Separation of thresholds from logic means:
      - Operators can tune the system without changing code
      - Different environments (laptop vs server) use different configs
      - Thresholds can be A/B tested across runs
    """

    # ── CPU thresholds ────────────────────────────────────────────────────────
    # Dispatch blocked when overall CPU% exceeds this for high-cost tasks
    cpu_high_task_limit:    float = float(os.getenv("SCHED_CPU_HIGH_LIMIT",   "80.0"))
    # Dispatch blocked when overall CPU% exceeds this for medium-cost tasks
    cpu_medium_task_limit:  float = float(os.getenv("SCHED_CPU_MED_LIMIT",    "88.0"))
    # All non-critical tasks blocked above this CPU%
    cpu_emergency_limit:    float = float(os.getenv("SCHED_CPU_EMERGENCY",    "95.0"))
    # How many consecutive high-CPU ticks before throttle kicks in
    cpu_sustained_ticks:    int   = int(os.getenv("SCHED_CPU_SUSTAINED_TICKS", "3"))

    # ── RAM thresholds ────────────────────────────────────────────────────────
    # % RAM used above which high-cost tasks are deferred
    ram_high_task_limit:    float = float(os.getenv("SCHED_RAM_HIGH_LIMIT",   "75.0"))
    # % RAM used above which medium tasks are deferred
    ram_medium_task_limit:  float = float(os.getenv("SCHED_RAM_MED_LIMIT",    "85.0"))
    # % RAM used above which OOM risk guard fires (blocks everything non-critical)
    ram_oom_risk_limit:     float = float(os.getenv("SCHED_RAM_OOM_LIMIT",    "92.0"))
    # Swap % above which OOM risk guard fires regardless of RAM%
    swap_oom_risk_limit:    float = float(os.getenv("SCHED_SWAP_OOM_LIMIT",   "40.0"))

    # ── GPU thresholds ────────────────────────────────────────────────────────
    # VRAM % used above which GPU-heavy tasks are deferred
    gpu_high_task_limit:    float = float(os.getenv("SCHED_GPU_HIGH_LIMIT",   "85.0"))
    # VRAM % used above which medium GPU tasks are deferred
    gpu_medium_task_limit:  float = float(os.getenv("SCHED_GPU_MED_LIMIT",    "92.0"))
    # VRAM % used above which all GPU tasks are blocked
    gpu_emergency_limit:    float = float(os.getenv("SCHED_GPU_EMERGENCY",    "97.0"))

    # ── Starvation prevention ─────────────────────────────────────────────────
    # Force-dispatch after this many deferrals regardless of resource state
    max_defer_count:        int   = int(os.getenv("SCHED_MAX_DEFER",          "30"))
    # Force-dispatch CRITICAL tasks after this many deferrals (should be 0)
    critical_max_defer:     int   = int(os.getenv("SCHED_CRITICAL_DEFER",     "0"))

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ─────────────────────────────────────────────────────────────────────────────
# LIVE RESOURCE SNAPSHOT (lightweight, dependency-free view)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResourceSnapshot:
    """
    A slim, self-contained resource snapshot used by the policy engine.
    Decoupled from LabAgent's SystemSnapshot so this module stays
    import-free from the agents/ package.

    Constructed by ResourcePolicy.from_system_snapshot() or
    built directly in tests.
    """
    cpu_pct:          float = 0.0    # overall CPU utilisation %
    ram_used_pct:     float = 0.0    # RAM used %
    ram_available_gb: float = 16.0   # free RAM in GB
    swap_pct:         float = 0.0    # swap used %
    gpu_available:    bool  = False
    gpu_vram_pct:     float = 0.0    # GPU VRAM used %
    bottleneck:       str   = "none" # from BottleneckType.value
    timestamp:        str   = field(default_factory=lambda: datetime.utcnow().isoformat())

    @classmethod
    def from_system_snapshot(cls, snap: object) -> ResourceSnapshot:
        """
        Build a ResourceSnapshot from a LabAgent SystemSnapshot.
        Uses getattr with defaults so this module never imports LabAgent.
        """
        if snap is None:
            return cls()

        cpu   = getattr(snap, "cpu",    None)
        mem   = getattr(snap, "memory", None)
        gpu   = getattr(snap, "gpu",    None)
        bt    = getattr(snap, "bottleneck", None)

        return cls(
            cpu_pct=          getattr(cpu, "overall_pct",    0.0),
            ram_used_pct=     getattr(mem, "used_pct",       0.0),
            ram_available_gb= getattr(mem, "available_gb",  16.0),
            swap_pct=         getattr(mem, "swap_pct",       0.0),
            gpu_available=    getattr(gpu, "available",     False),
            gpu_vram_pct=     getattr(gpu, "vram_used_pct",  0.0),
            bottleneck=       getattr(bt, "value", "none") if bt else "none",
        )

    @property
    def cpu_headroom(self) -> float:
        """Free CPU headroom (100 - cpu_pct)."""
        return max(0.0, 100.0 - self.cpu_pct)

    @property
    def ram_headroom(self) -> float:
        """Free RAM headroom (100 - ram_used_pct)."""
        return max(0.0, 100.0 - self.ram_used_pct)

    @property
    def gpu_headroom(self) -> float:
        """Free GPU VRAM headroom (100 - gpu_vram_pct)."""
        return max(0.0, 100.0 - self.gpu_vram_pct)

    @property
    def is_oom_risk(self) -> bool:
        return self.bottleneck == "oom_risk"

    def to_dict(self) -> dict:
        return {
            "cpu_pct":          self.cpu_pct,
            "ram_used_pct":     self.ram_used_pct,
            "ram_available_gb": self.ram_available_gb,
            "swap_pct":         self.swap_pct,
            "gpu_available":    self.gpu_available,
            "gpu_vram_pct":     self.gpu_vram_pct,
            "bottleneck":       self.bottleneck,
        }


# ─────────────────────────────────────────────────────────────────────────────
# POLICY DECISION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyDecision:
    """
    The result of a ResourcePolicy.evaluate() call.

    action        — what the scheduler should do with the entry
    reason        — human-readable explanation (logged + shown in Gantt)
    rule_name     — which rule triggered this decision
    resource_used — snapshot at decision time (for Gantt log)
    suggested_retry_s — how long to wait before trying again (for DEFER)
    """
    action:             PolicyAction
    reason:             str
    rule_name:          str           = ""
    resource_snapshot:  dict          = field(default_factory=dict)
    suggested_retry_s:  float         = 0.0

    @property
    def should_dispatch(self) -> bool:
        return self.action in (PolicyAction.DISPATCH, PolicyAction.FORCE_DISPATCH)

    @property
    def should_defer(self) -> bool:
        return self.action == PolicyAction.DEFER

    @property
    def should_skip(self) -> bool:
        return self.action == PolicyAction.SKIP

    def __str__(self) -> str:
        return f"PolicyDecision({self.action.value}) [{self.rule_name}]: {self.reason}"

    def to_dict(self) -> dict:
        return {
            "action":            self.action.value,
            "reason":            self.reason,
            "rule_name":         self.rule_name,
            "suggested_retry_s": self.suggested_retry_s,
            "resource_snapshot": self.resource_snapshot,
        }


# ─────────────────────────────────────────────────────────────────────────────
# POLICY RULE TYPE
# ─────────────────────────────────────────────────────────────────────────────

# A Rule is a callable that returns a PolicyDecision or None.
# None means "no opinion — continue to next rule".
# Returning a decision short-circuits the rule chain.
Rule = Callable[[QueueEntry, ResourceSnapshot, ThresholdConfig], PolicyDecision | None]


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL RULES
# ─────────────────────────────────────────────────────────────────────────────

def rule_cancellation_check(
    entry: QueueEntry,
    snap: ResourceSnapshot,
    cfg: ThresholdConfig,
) -> PolicyDecision | None:
    """
    Rule 1 — Fast path: skip cancelled or invalid entries immediately.
    No resource check needed.
    """
    if entry.state == "cancelled":
        return PolicyDecision(
            action=PolicyAction.SKIP,
            reason=f"Entry '{entry.task_name}' is cancelled",
            rule_name="cancellation_check",
        )
    if not entry.task_id or not entry.task_name:
        return PolicyDecision(
            action=PolicyAction.SKIP,
            reason="Entry has missing task_id or task_name",
            rule_name="cancellation_check",
        )
    return None


def rule_critical_bypass(
    entry: QueueEntry,
    snap: ResourceSnapshot,
    cfg: ThresholdConfig,
) -> PolicyDecision | None:
    """
    Rule 2 — CRITICAL priority tasks bypass ALL resource checks.

    Rationale: CRITICAL tasks are system-level operations (e.g. monitor
    heartbeats, emergency shutdowns). Blocking them on resource limits
    could cause the monitoring layer itself to stop, making the situation
    worse. This mirrors the OS concept of interrupt handlers and kernel
    threads that preempt all user-space work regardless of CPU load.
    """
    if entry.priority == Priority.CRITICAL:
        return PolicyDecision(
            action=PolicyAction.DISPATCH,
            reason=f"CRITICAL priority bypasses all resource checks",
            rule_name="critical_bypass",
            resource_snapshot=snap.to_dict(),
        )
    return None


def rule_starvation_valve(
    entry: QueueEntry,
    snap: ResourceSnapshot,
    cfg: ThresholdConfig,
) -> PolicyDecision | None:
    """
    Rule 3 — Force-dispatch entries that have been deferred too many times.

    Implements starvation prevention: no task may be deferred more than
    cfg.max_defer_count times. After that threshold, it is force-dispatched
    regardless of resource state.

    This mirrors Linux's SCHED_DEADLINE watchdog and the classical
    'aging' starvation prevention technique from OS theory.

    The Gantt logger marks these events distinctly so they are visible
    in the dashboard as anomalies worth investigating.
    """
    limit = (
        cfg.critical_max_defer
        if entry.priority == Priority.CRITICAL
        else cfg.max_defer_count
    )
    if entry.defer_count >= limit and limit > 0:
        return PolicyDecision(
            action=PolicyAction.FORCE_DISPATCH,
            reason=(
                f"STARVATION PREVENTION: '{entry.task_name}' deferred "
                f"{entry.defer_count} times (limit={limit}) — "
                f"force-dispatching regardless of resource state"
            ),
            rule_name="starvation_valve",
            resource_snapshot=snap.to_dict(),
        )
    return None


def rule_oom_risk_guard(
    entry: QueueEntry,
    snap: ResourceSnapshot,
    cfg: ThresholdConfig,
) -> PolicyDecision | None:
    """
    Rule 4 — Block non-critical, non-low-RAM tasks when OOM risk is active.

    OOM risk = RAM > 92% OR swap > 40%. In this state, launching any
    memory-intensive work could push the system into an OOM kill, which
    would terminate the entire process. We block medium and high RAM tasks
    and only allow low-RAM work to proceed.

    Retry suggestion: 5 seconds (wait for a memory-intensive stage to finish).

    Analogous to the OS kernel's OOM killer avoidance logic — the kernel
    defers new allocations and runs compaction before killing processes.
    """
    if snap.is_oom_risk or snap.ram_used_pct >= cfg.ram_oom_risk_limit:
        if entry.ram_cost not in ("none", "low"):
            return PolicyDecision(
                action=PolicyAction.DEFER,
                reason=(
                    f"OOM RISK GUARD: RAM {snap.ram_used_pct:.1f}% "
                    f"(limit={cfg.ram_oom_risk_limit}%) — "
                    f"blocking {entry.ram_cost}-RAM task '{entry.task_name}'"
                ),
                rule_name="oom_risk_guard",
                resource_snapshot=snap.to_dict(),
                suggested_retry_s=5.0,
            )

    if snap.swap_pct >= cfg.swap_oom_risk_limit and entry.ram_cost == "high":
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"SWAP PRESSURE: swap {snap.swap_pct:.1f}% "
                f"(limit={cfg.swap_oom_risk_limit}%) — "
                f"deferring high-RAM task '{entry.task_name}'"
            ),
            rule_name="oom_risk_guard",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=8.0,
        )
    return None


def rule_cpu_threshold(
    entry: QueueEntry,
    snap: ResourceSnapshot,
    cfg: ThresholdConfig,
) -> PolicyDecision | None:
    """
    Rule 5 — Defer tasks when CPU utilisation exceeds cost-adjusted limits.

    Threshold matrix:
      high-CPU task   : blocked when CPU > cpu_high_task_limit (80%)
      medium-CPU task : blocked when CPU > cpu_medium_task_limit (88%)
      any task        : blocked when CPU > cpu_emergency_limit (95%)

    Low and none cost tasks are not blocked by CPU (they have negligible
    CPU impact and blocking them wastes scheduling opportunities).

    Retry suggestion scales with how far over the threshold we are:
      < 5% over → retry in 2s
      5–15% over → retry in 4s
      > 15% over → retry in 8s

    Analogous to CPU admission control in real-time OS schedulers.
    """
    cpu = snap.cpu_pct

    # Emergency: block everything except low/none regardless of cost
    if cpu >= cfg.cpu_emergency_limit and entry.cpu_cost not in ("none", "low"):
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"CPU EMERGENCY: {cpu:.1f}% ≥ {cfg.cpu_emergency_limit}% — "
                f"deferring all non-trivial tasks"
            ),
            rule_name="cpu_threshold",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=_retry_s(cpu, cfg.cpu_emergency_limit),
        )

    if entry.cpu_cost == "high" and cpu >= cfg.cpu_high_task_limit:
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"CPU HIGH LIMIT: {cpu:.1f}% ≥ {cfg.cpu_high_task_limit}% — "
                f"deferring high-CPU task '{entry.task_name}' "
                f"(needs {CPU_HEADROOM_REQUIRED['high']}% headroom, "
                f"available={snap.cpu_headroom:.1f}%)"
            ),
            rule_name="cpu_threshold",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=_retry_s(cpu, cfg.cpu_high_task_limit),
        )

    if entry.cpu_cost == "medium" and cpu >= cfg.cpu_medium_task_limit:
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"CPU MEDIUM LIMIT: {cpu:.1f}% ≥ {cfg.cpu_medium_task_limit}% — "
                f"deferring medium-CPU task '{entry.task_name}'"
            ),
            rule_name="cpu_threshold",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=_retry_s(cpu, cfg.cpu_medium_task_limit),
        )

    return None


def rule_ram_threshold(
    entry: QueueEntry,
    snap: ResourceSnapshot,
    cfg: ThresholdConfig,
) -> PolicyDecision | None:
    """
    Rule 6 — Defer tasks when RAM headroom is insufficient for their cost.

    Unlike CPU (which recovers quickly), RAM pressure persists until a
    consuming process frees memory. Retry suggestions are therefore longer.

    The headroom check is more granular than a simple threshold:
    we compare the required headroom (from RAM_HEADROOM_REQUIRED) against
    the actual available headroom, giving smoother admission control.
    """
    ram_used = snap.ram_used_pct
    required_headroom = RAM_HEADROOM_REQUIRED.get(entry.ram_cost, 0.0)

    if snap.ram_headroom < required_headroom:
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"RAM INSUFFICIENT: available={snap.ram_headroom:.1f}% "
                f"< required={required_headroom:.1f}% for "
                f"{entry.ram_cost}-RAM task '{entry.task_name}' "
                f"({snap.ram_available_gb:.2f} GB free)"
            ),
            rule_name="ram_threshold",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=_retry_s(ram_used, cfg.ram_high_task_limit, base_s=4.0),
        )

    # Belt-and-suspenders: hard percent limits
    if entry.ram_cost == "high" and ram_used >= cfg.ram_high_task_limit:
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"RAM HIGH LIMIT: {ram_used:.1f}% ≥ {cfg.ram_high_task_limit}% — "
                f"deferring high-RAM task '{entry.task_name}'"
            ),
            rule_name="ram_threshold",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=5.0,
        )

    if entry.ram_cost == "medium" and ram_used >= cfg.ram_medium_task_limit:
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"RAM MEDIUM LIMIT: {ram_used:.1f}% ≥ {cfg.ram_medium_task_limit}% — "
                f"deferring medium-RAM task '{entry.task_name}'"
            ),
            rule_name="ram_threshold",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=4.0,
        )

    return None


def rule_gpu_threshold(
    entry: QueueEntry,
    snap: ResourceSnapshot,
    cfg: ThresholdConfig,
) -> PolicyDecision | None:
    """
    Rule 7 — Defer GPU-intensive tasks when VRAM is saturated.

    GPU VRAM is a hard resource — unlike CPU, you cannot over-subscribe it.
    A CUDA out-of-memory error will crash the worker process, not just slow
    it down. We therefore apply strict gating.

    If no GPU is available (snap.gpu_available = False), GPU cost tasks
    are allowed through — they will run on CPU (slower but not fatal).
    """
    if not snap.gpu_available:
        return None   # no GPU — let it through to CPU fallback

    gpu = snap.gpu_vram_pct
    required_headroom = GPU_HEADROOM_REQUIRED.get(entry.gpu_cost, 0.0)

    if entry.gpu_cost == "none":
        return None

    # Emergency: block all GPU tasks when VRAM is critically full
    if gpu >= cfg.gpu_emergency_limit:
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"GPU VRAM EMERGENCY: {gpu:.1f}% ≥ {cfg.gpu_emergency_limit}% — "
                f"blocking all GPU tasks to prevent CUDA OOM"
            ),
            rule_name="gpu_threshold",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=10.0,
        )

    if entry.gpu_cost == "high" and gpu >= cfg.gpu_high_task_limit:
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"GPU VRAM HIGH LIMIT: {gpu:.1f}% ≥ {cfg.gpu_high_task_limit}% — "
                f"deferring high-VRAM task '{entry.task_name}' "
                f"(headroom={snap.gpu_headroom:.1f}%, required={required_headroom:.1f}%)"
            ),
            rule_name="gpu_threshold",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=_retry_s(gpu, cfg.gpu_high_task_limit, base_s=5.0),
        )

    if entry.gpu_cost == "medium" and gpu >= cfg.gpu_medium_task_limit:
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"GPU VRAM MEDIUM LIMIT: {gpu:.1f}% ≥ {cfg.gpu_medium_task_limit}% — "
                f"deferring medium-VRAM task '{entry.task_name}'"
            ),
            rule_name="gpu_threshold",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=8.0,
        )

    return None


def rule_cost_level_check(
    entry: QueueEntry,
    snap: ResourceSnapshot,
    cfg: ThresholdConfig,
) -> PolicyDecision | None:
    """
    Rule 8 — Fine-grained headroom check across all three resources.

    After the threshold rules have passed, this rule does a final
    composite check: does the entry's combined cost profile fit
    within the available headroom on ALL resources simultaneously?

    A task with medium CPU + medium RAM + medium GPU cost should only
    run if all three resources have sufficient headroom — not just
    each one in isolation (rules 5–7 check them independently).

    This rule prevents the scheduler from dispatching a task that
    individually passes each threshold check but collectively would
    push the system over capacity.
    """
    violations: list[str] = []

    cpu_req = CPU_HEADROOM_REQUIRED.get(entry.cpu_cost, 0.0)
    if snap.cpu_headroom < cpu_req:
        violations.append(
            f"CPU headroom {snap.cpu_headroom:.1f}% < required {cpu_req:.1f}%"
        )

    ram_req = RAM_HEADROOM_REQUIRED.get(entry.ram_cost, 0.0)
    if snap.ram_headroom < ram_req:
        violations.append(
            f"RAM headroom {snap.ram_headroom:.1f}% < required {ram_req:.1f}%"
        )

    if snap.gpu_available and entry.gpu_cost != "none":
        gpu_req = GPU_HEADROOM_REQUIRED.get(entry.gpu_cost, 0.0)
        if snap.gpu_headroom < gpu_req:
            violations.append(
                f"GPU headroom {snap.gpu_headroom:.1f}% < required {gpu_req:.1f}%"
            )

    if violations:
        return PolicyDecision(
            action=PolicyAction.DEFER,
            reason=(
                f"COMPOSITE HEADROOM FAIL for '{entry.task_name}': "
                f"{'; '.join(violations)}"
            ),
            rule_name="cost_level_check",
            resource_snapshot=snap.to_dict(),
            suggested_retry_s=3.0,
        )

    return None


# Default rule chain (evaluated in order)
DEFAULT_RULES: list[Rule] = [
    rule_cancellation_check,
    rule_critical_bypass,
    rule_starvation_valve,
    rule_oom_risk_guard,
    rule_cpu_threshold,
    rule_ram_threshold,
    rule_gpu_threshold,
    rule_cost_level_check,
]


# ─────────────────────────────────────────────────────────────────────────────
# POLICY STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PolicyStats:
    """
    Cumulative counters per policy action and rule.
    Exposed to the Streamlit dashboard for live scheduling analysis.
    """
    total_evaluated:      int = 0
    total_dispatched:     int = 0
    total_deferred:       int = 0
    total_force_dispatched: int = 0
    total_skipped:        int = 0

    # Per-rule defer counts (rule_name → count)
    defer_by_rule:        dict[str, int] = field(default_factory=dict)

    # Resource condition at time of deferral (for trend analysis)
    defer_cpu_values:     list[float] = field(default_factory=list)
    defer_ram_values:     list[float] = field(default_factory=list)

    def record(self, decision: PolicyDecision, snap: ResourceSnapshot) -> None:
        self.total_evaluated += 1
        if decision.action == PolicyAction.DISPATCH:
            self.total_dispatched += 1
        elif decision.action == PolicyAction.DEFER:
            self.total_deferred += 1
            rule = decision.rule_name or "unknown"
            self.defer_by_rule[rule] = self.defer_by_rule.get(rule, 0) + 1
            self.defer_cpu_values.append(snap.cpu_pct)
            self.defer_ram_values.append(snap.ram_used_pct)
        elif decision.action == PolicyAction.FORCE_DISPATCH:
            self.total_force_dispatched += 1
        elif decision.action == PolicyAction.SKIP:
            self.total_skipped += 1

    @property
    def dispatch_rate(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return round(
            (self.total_dispatched + self.total_force_dispatched) / self.total_evaluated, 4
        )

    @property
    def defer_rate(self) -> float:
        if self.total_evaluated == 0:
            return 0.0
        return round(self.total_deferred / self.total_evaluated, 4)

    @property
    def avg_cpu_at_defer(self) -> float:
        return round(
            sum(self.defer_cpu_values) / len(self.defer_cpu_values), 2
        ) if self.defer_cpu_values else 0.0

    @property
    def avg_ram_at_defer(self) -> float:
        return round(
            sum(self.defer_ram_values) / len(self.defer_ram_values), 2
        ) if self.defer_ram_values else 0.0

    def to_dict(self) -> dict:
        return {
            "total_evaluated":       self.total_evaluated,
            "total_dispatched":      self.total_dispatched,
            "total_deferred":        self.total_deferred,
            "total_force_dispatched": self.total_force_dispatched,
            "total_skipped":         self.total_skipped,
            "dispatch_rate":         self.dispatch_rate,
            "defer_rate":            self.defer_rate,
            "defer_by_rule":         self.defer_by_rule,
            "avg_cpu_at_defer":      self.avg_cpu_at_defer,
            "avg_ram_at_defer":      self.avg_ram_at_defer,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE POLICY
# ─────────────────────────────────────────────────────────────────────────────

class ResourcePolicy:
    """
    The policy engine — evaluates a QueueEntry against a ResourceSnapshot
    by running it through the configured rule chain.

    Rules are evaluated in order; the first rule to return a non-None
    PolicyDecision short-circuits the chain. If no rule fires, the
    default outcome is DISPATCH.

    Custom rules can be prepended or appended via add_rule() for
    experiment-specific admission control (e.g. rate-limit arXiv calls).
    """

    def __init__(
        self,
        config: ThresholdConfig | None = None,
        rules: list[Rule] | None = None,
    ) -> None:
        self._config = config or ThresholdConfig()
        self._rules: list[Rule] = rules if rules is not None else list(DEFAULT_RULES)
        self._stats = PolicyStats()

        logger.info(
            "ResourcePolicy initialised | %d rules | "
            "cpu_high=%.0f%% ram_oom=%.0f%% gpu_high=%.0f%% max_defer=%d",
            len(self._rules),
            self._config.cpu_high_task_limit,
            self._config.ram_oom_risk_limit,
            self._config.gpu_high_task_limit,
            self._config.max_defer_count,
        )

    # ── MAIN EVALUATE ────────────────────────────────────────────────────────

    def evaluate(
        self,
        entry: QueueEntry,
        snapshot: ResourceSnapshot | None = None,
    ) -> PolicyDecision:
        """
        Evaluate a QueueEntry against the current resource state.

        If snapshot is None, a permissive default snapshot is used
        (all resources at 0% — dispatch everything). This covers the
        warm-up period before the first psutil poll completes.

        Returns a PolicyDecision; never raises.
        """
        snap = snapshot or ResourceSnapshot()

        for rule in self._rules:
            try:
                decision = rule(entry, snap, self._config)
                if decision is not None:
                    self._stats.record(decision, snap)
                    self._log_decision(entry, decision)
                    return decision
            except Exception as exc:
                logger.error(
                    "Rule '%s' raised an exception for task '%s': %s",
                    getattr(rule, "__name__", "?"), entry.task_name, exc,
                )

        # All rules returned None → default DISPATCH
        decision = PolicyDecision(
            action=PolicyAction.DISPATCH,
            reason="All policy rules passed — resources sufficient",
            rule_name="default_dispatch",
            resource_snapshot=snap.to_dict(),
        )
        self._stats.record(decision, snap)
        self._log_decision(entry, decision)
        return decision

    # ── BATCH EVALUATE ───────────────────────────────────────────────────────

    def evaluate_batch(
        self,
        entries: list[QueueEntry],
        snapshot: ResourceSnapshot | None = None,
    ) -> list[tuple[QueueEntry, PolicyDecision]]:
        """
        Evaluate multiple entries against the same snapshot.
        Useful for the scheduler to pre-screen a batch of candidates.
        Returns a list of (entry, decision) pairs in input order.
        """
        snap = snapshot or ResourceSnapshot()
        return [(entry, self.evaluate(entry, snap)) for entry in entries]

    # ── RULE MANAGEMENT ──────────────────────────────────────────────────────

    def prepend_rule(self, rule: Rule) -> None:
        """Add a rule at the front of the chain (highest priority evaluation)."""
        self._rules.insert(0, rule)
        logger.info("Rule prepended: %s", getattr(rule, "__name__", "?"))

    def append_rule(self, rule: Rule) -> None:
        """Add a rule at the end of the chain (lowest priority evaluation)."""
        self._rules.append(rule)
        logger.info("Rule appended: %s", getattr(rule, "__name__", "?"))

    def remove_rule(self, rule_name: str) -> bool:
        """Remove a rule by function name. Returns True if found and removed."""
        for i, rule in enumerate(self._rules):
            if getattr(rule, "__name__", "") == rule_name:
                self._rules.pop(i)
                logger.info("Rule removed: %s", rule_name)
                return True
        return False

    def list_rules(self) -> list[str]:
        """Return the names of all configured rules in evaluation order."""
        return [getattr(r, "__name__", "?") for r in self._rules]

    # ── THRESHOLD UPDATING ───────────────────────────────────────────────────

    def update_config(self, **kwargs) -> None:
        """
        Update threshold configuration at runtime.
        Useful for adaptive threshold tuning or YAML config hot-reload.
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info("ThresholdConfig updated: %s = %s", key, value)
            else:
                logger.warning("Unknown config key: %s", key)

    # ── STATS & INTROSPECTION ────────────────────────────────────────────────

    @property
    def stats(self) -> PolicyStats:
        return self._stats

    def stats_dict(self) -> dict:
        return self._stats.to_dict()

    def config_dict(self) -> dict:
        return self._config.to_dict()

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _log_decision(self, entry: QueueEntry, decision: PolicyDecision) -> None:
        if decision.action == PolicyAction.DISPATCH:
            logger.debug(
                "DISPATCH: '%s' [%s] | %s",
                entry.task_name, entry.task_id, decision.reason,
            )
        elif decision.action == PolicyAction.DEFER:
            logger.info(
                "DEFER: '%s' [%s] defer=%d retry=%.1fs | %s",
                entry.task_name, entry.task_id,
                entry.defer_count, decision.suggested_retry_s,
                decision.reason,
            )
        elif decision.action == PolicyAction.FORCE_DISPATCH:
            logger.warning(
                "FORCE_DISPATCH: '%s' [%s] defer=%d | %s",
                entry.task_name, entry.task_id,
                entry.defer_count, decision.reason,
            )
        elif decision.action == PolicyAction.SKIP:
            logger.debug(
                "SKIP: '%s' [%s] | %s",
                entry.task_name, entry.task_id, decision.reason,
            )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: compute retry backoff
# ─────────────────────────────────────────────────────────────────────────────

def _retry_s(current: float, limit: float, base_s: float = 2.0) -> float:
    """
    Compute a suggested retry delay based on how far over the limit
    the current value is.

    Over by < 5%  → base_s
    Over by 5–15% → base_s * 2
    Over by > 15% → base_s * 4
    """
    overshoot = current - limit
    if overshoot < 5.0:
        return base_s
    elif overshoot < 15.0:
        return base_s * 2.0
    return base_s * 4.0


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Demonstrates the full policy evaluation chain under different
    simulated resource conditions.

    Run with: python -m scheduler.resource_policy
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    policy = ResourcePolicy()

    # ── Test entries ─────────────────────────────────────────────────────────
    entries = [
        QueueEntry("t1", "system_monitor",       Priority.CRITICAL, "low",    "low",    "none"),
        QueueEntry("t2", "abstract_embedding",   Priority.MEDIUM,   "high",   "medium", "medium"),
        QueueEntry("t3", "gap_analysis_react",   Priority.HIGH,     "medium", "medium", "high"),
        QueueEntry("t4", "umap_reduction",        Priority.LOW,      "high",   "high",   "none"),
        QueueEntry("t5", "report_generation",    Priority.LOW,      "low",    "low",    "none"),
        QueueEntry("t6", "cluster_labeling",     Priority.MEDIUM,   "medium", "low",    "high"),
    ]

    # Set up t3 with lots of deferrals to trigger starvation valve
    entries[2].defer_count = 31

    # ── Scenario 1: Normal load ──────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SCENARIO 1: Normal load (CPU=40%, RAM=50%, GPU=30%)")
    print("═" * 60)
    normal = ResourceSnapshot(
        cpu_pct=40.0, ram_used_pct=50.0,
        gpu_available=True, gpu_vram_pct=30.0,
        bottleneck="none"
    )
    for e in entries:
        d = policy.evaluate(e, normal)
        icon = {"dispatch": "✓", "defer": "✗", "force_dispatch": "⚡", "skip": "○"}[d.action.value]
        print(f"  {icon} [{d.action.value:<14}] '{e.task_name}' | {d.rule_name} | {d.reason[:60]}")

    # ── Scenario 2: High CPU load ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SCENARIO 2: High CPU (CPU=87%, RAM=50%, GPU=30%)")
    print("═" * 60)
    high_cpu = ResourceSnapshot(
        cpu_pct=87.0, ram_used_pct=50.0,
        gpu_available=True, gpu_vram_pct=30.0,
        bottleneck="cpu_bound"
    )
    for e in entries:
        d = policy.evaluate(e, high_cpu)
        icon = {"dispatch": "✓", "defer": "✗", "force_dispatch": "⚡", "skip": "○"}[d.action.value]
        print(f"  {icon} [{d.action.value:<14}] '{e.task_name}' | {d.rule_name}")

    # ── Scenario 3: OOM risk ─────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SCENARIO 3: OOM risk (CPU=60%, RAM=94%, GPU=20%)")
    print("═" * 60)
    oom = ResourceSnapshot(
        cpu_pct=60.0, ram_used_pct=94.0,
        gpu_available=True, gpu_vram_pct=20.0,
        bottleneck="oom_risk"
    )
    for e in entries:
        d = policy.evaluate(e, oom)
        icon = {"dispatch": "✓", "defer": "✗", "force_dispatch": "⚡", "skip": "○"}[d.action.value]
        print(f"  {icon} [{d.action.value:<14}] '{e.task_name}' | {d.rule_name} | retry={d.suggested_retry_s:.1f}s")

    # ── Scenario 4: GPU saturated ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SCENARIO 4: GPU saturated (CPU=40%, RAM=50%, GPU=93%)")
    print("═" * 60)
    gpu_sat = ResourceSnapshot(
        cpu_pct=40.0, ram_used_pct=50.0,
        gpu_available=True, gpu_vram_pct=93.0,
        bottleneck="gpu_bound"
    )
    for e in entries:
        d = policy.evaluate(e, gpu_sat)
        icon = {"dispatch": "✓", "defer": "✗", "force_dispatch": "⚡", "skip": "○"}[d.action.value]
        print(f"  {icon} [{d.action.value:<14}] '{e.task_name}' | {d.rule_name}")

    # ── Final stats ──────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  POLICY STATS")
    print("═" * 60)
    for k, v in policy.stats_dict().items():
        print(f"  {k:<30} {v}")

    print(f"\n  Rule chain: {policy.list_rules()}")


if __name__ == "__main__":
    _demo()
