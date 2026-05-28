"""
scheduler/preemption_handler.py
================================
ResearchFlow AI — Deferral, Retry & Backoff Logic

This module is the last piece of the scheduler/ layer. It handles
everything that happens when a task CANNOT be dispatched:

  1. DEFERRAL    — task requeued with updated backoff state
  2. RETRY       — failed task re-submitted after exponential backoff
  3. ESCALATION  — repeated deferrals trigger priority escalation
  4. EXPIRY      — tasks deferred beyond the deadline are expired/cancelled
  5. HISTORY     — full deferral + retry audit trail per task

OS Concepts demonstrated:
  ┌────────────────────────────────────────────────────────────────┐
  │  Concept                   │  Implementation                  │
  ├────────────────────────────┼──────────────────────────────────┤
  │  Preemption / Deferral     │  DeferralRecord per task         │
  │  Exponential Backoff       │  backoff_s × 2^(attempt-1)       │
  │  Jitter (anti-thundering)  │  ±20% random jitter on backoff   │
  │  Priority Escalation       │  boost after N consecutive defs  │
  │  Admission Deadline        │  max_wall_time_s per task        │
  │  Retry Budget              │  max_retries with budget tracking │
  │  Audit Trail               │  full history per task_id        │
  │  Backpressure signalling   │  BackpressureSignal to scheduler │
  └────────────────────────────────────────────────────────────────┘

Key classes:
  DeferralRecord      — per-task deferral state (immutable snapshots)
  RetryRecord         — per-task retry state after execution failure
  BackoffStrategy     — pluggable backoff algorithms (exp, linear, const)
  PreemptionHandler   — orchestrates deferral + retry + escalation
  BackpressureSignal  — emitted when the system is chronically overloaded

Flow when a task is deferred:
  ┌──────────────┐   policy=DEFER    ┌──────────────────────┐
  │SchedulerAgent│ ────────────────► │ PreemptionHandler    │
  │              │                   │  .handle_deferral()  │
  └──────────────┘                   └──────┬───────────────┘
                                            │
                    ┌───────────────────────┼──────────────────┐
                    ▼                       ▼                  ▼
             compute backoff         escalate priority    check deadline
                    │                       │                  │
                    ▼                       ▼                  ▼
             sleep(backoff_s)         entry.tick()       expire if past
                    │                       │              deadline
                    ▼                       ▼
             pq.requeue(entry)       gantt.on_deferred()

Flow when a task FAILS (execution error, not resource deferral):
  ┌──────────────┐   exception       ┌──────────────────────┐
  │WorkerThread  │ ────────────────► │ PreemptionHandler    │
  │              │                   │  .handle_failure()   │
  └──────────────┘                   └──────┬───────────────┘
                                            │
                    ┌───────────────────────┼──────────────────┐
                    ▼                       ▼                  ▼
             within budget?         compute backoff      classify error
                    │                       │                  │
                    ▼                       ▼                  ▼
            re-submit to queue      sleep(backoff_s)    retryable?
                    │                                         │
                    ▼                                         ▼
            gantt.on_completed()                    expire + propagate

Usage:
    handler = PreemptionHandler(
        priority_queue=pq,
        gantt_logger=gantt,
    )

    # When scheduler decides to defer:
    result = handler.handle_deferral(entry, reason, snapshot)
    if result.action == DeferralAction.EXPIRED:
        future.set_exception(TaskExpiredError(entry.task_name))

    # When a worker thread raises an exception:
    result = handler.handle_failure(entry, exception, future)
    # handler automatically re-submits to queue if within retry budget

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import logging
import math
import random
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from scheduler.priority_queue import Priority, PriorityQueue, QueueEntry
from scheduler.gantt_logger import GanttLogger

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Jitter fraction applied to computed backoff (±20%)
JITTER_FRACTION: float = 0.20

# After this many consecutive deferrals, emit a BackpressureSignal
BACKPRESSURE_DEFERRAL_THRESHOLD: int = 10

# After this many DIFFERENT tasks are deferred simultaneously,
# emit a SystemOverloadSignal
OVERLOAD_TASK_THRESHOLD: int = 5

# Default max wall-clock seconds a task may remain in the system
# (queued + deferred time) before being expired
DEFAULT_DEADLINE_S: float = 600.0  # 10 minutes


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class DeferralAction(str, Enum):
    """
    Outcome of a handle_deferral() call.

    REQUEUED   — task returned to priority queue; will retry after backoff
    ESCALATED  — task requeued with boosted priority due to repeated deferral
    EXPIRED    — task has exceeded its deadline; removed and future failed
    CANCELLED  — task was externally cancelled during deferral handling
    """
    REQUEUED  = "requeued"
    ESCALATED = "escalated"
    EXPIRED   = "expired"
    CANCELLED = "cancelled"


class RetryAction(str, Enum):
    """
    Outcome of a handle_failure() call.

    RETRYING   — task re-submitted to priority queue with backoff delay
    EXHAUSTED  — retry budget used up; future set with exception
    SKIPPED    — error classified as non-retryable; immediate failure
    """
    RETRYING  = "retrying"
    EXHAUSTED = "exhausted"
    SKIPPED   = "skipped"


class BackoffAlgorithm(str, Enum):
    """Pluggable backoff strategies."""
    EXPONENTIAL = "exponential"   # base × 2^attempt  (default)
    LINEAR      = "linear"        # base × attempt
    CONSTANT    = "constant"      # base always
    FIBONACCI   = "fibonacci"     # base × fib(attempt)


# ─────────────────────────────────────────────────────────────────────────────
# ERRORS
# ─────────────────────────────────────────────────────────────────────────────

class TaskExpiredError(Exception):
    """Raised when a task is cancelled because it exceeded its deadline."""
    def __init__(self, task_name: str, deadline_s: float, elapsed_s: float) -> None:
        self.task_name  = task_name
        self.deadline_s = deadline_s
        self.elapsed_s  = elapsed_s
        super().__init__(
            f"Task '{task_name}' expired after {elapsed_s:.1f}s "
            f"(deadline={deadline_s:.1f}s)"
        )


class RetryBudgetExhaustedError(Exception):
    """Raised when a task has used all its retry attempts."""
    def __init__(self, task_name: str, attempts: int, last_error: str) -> None:
        self.task_name  = task_name
        self.attempts   = attempts
        self.last_error = last_error
        super().__init__(
            f"Task '{task_name}' failed after {attempts} attempts. "
            f"Last error: {last_error}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BACKOFF STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

class BackoffStrategy:
    """
    Computes wait times between retry/deferral attempts.

    Supports exponential, linear, constant, and Fibonacci algorithms.
    All algorithms apply ±JITTER_FRACTION random jitter to prevent
    the thundering herd problem — multiple deferred tasks all
    retrying at exactly the same instant.

    This is the same principle used in TCP congestion control
    (exponential backoff with jitter) and distributed systems
    retry logic (AWS SDK, gRPC).
    """

    _FIB_CACHE: dict[int, int] = {0: 0, 1: 1}

    def __init__(
        self,
        algorithm:  BackoffAlgorithm = BackoffAlgorithm.EXPONENTIAL,
        base_s:     float            = 2.0,
        max_s:      float            = 60.0,
        jitter:     bool             = True,
    ) -> None:
        self._algorithm = algorithm
        self._base      = base_s
        self._max       = max_s
        self._jitter    = jitter

    def compute(self, attempt: int) -> float:
        """
        Compute wait time for the given attempt number (1-based).

        attempt=1 → first retry/deferral
        attempt=2 → second retry/deferral
        etc.
        """
        attempt = max(1, attempt)

        if self._algorithm == BackoffAlgorithm.EXPONENTIAL:
            raw = self._base * (2 ** (attempt - 1))

        elif self._algorithm == BackoffAlgorithm.LINEAR:
            raw = self._base * attempt

        elif self._algorithm == BackoffAlgorithm.CONSTANT:
            raw = self._base

        elif self._algorithm == BackoffAlgorithm.FIBONACCI:
            raw = self._base * self._fib(attempt)

        else:
            raw = self._base

        # Apply ceiling
        raw = min(raw, self._max)

        # Apply jitter: ±JITTER_FRACTION × raw
        if self._jitter:
            jitter_range = raw * JITTER_FRACTION
            raw += random.uniform(-jitter_range, jitter_range)
            raw = max(0.1, raw)   # never negative or zero

        return round(raw, 3)

    def sequence(self, n: int) -> list[float]:
        """Return the first n backoff values (useful for display/testing)."""
        return [self.compute(i) for i in range(1, n + 1)]

    @classmethod
    def _fib(cls, n: int) -> int:
        if n not in cls._FIB_CACHE:
            cls._FIB_CACHE[n] = cls._fib(n - 1) + cls._fib(n - 2)
        return cls._FIB_CACHE[n]


# ─────────────────────────────────────────────────────────────────────────────
# DEFERRAL RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeferralRecord:
    """
    Immutable snapshot of a single deferral event for a task.
    Stored in PreemptionHandler._deferral_history for audit.
    """
    task_id:          str
    task_name:        str
    defer_number:     int          # 1 = first deferral, 2 = second, etc.
    timestamp:        str
    reason:           str
    rule_name:        str          # which ResourcePolicy rule triggered
    backoff_s:        float
    cpu_pct:          float = 0.0
    ram_pct:          float = 0.0
    gpu_pct:          float = 0.0
    priority_before:  str   = "MEDIUM"
    priority_after:   str   = "MEDIUM"
    escalated:        bool  = False


@dataclass
class DeferralResult:
    """Result returned by handle_deferral()."""
    action:       DeferralAction
    backoff_s:    float
    defer_number: int
    escalated:    bool
    reason:       str
    expired:      bool = False


# ─────────────────────────────────────────────────────────────────────────────
# RETRY RECORD
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetryRecord:
    """
    Tracks the full retry state for a task that failed during execution
    (as opposed to being deferred by the resource policy).
    """
    task_id:          str
    task_name:        str
    max_retries:      int
    attempts:         int   = 0
    last_error:       str   = ""
    last_error_type:  str   = ""
    first_failure_at: str   = ""
    last_failure_at:  str   = ""
    total_retry_s:    float = 0.0   # total time spent in backoff

    @property
    def remaining(self) -> int:
        return max(0, self.max_retries - self.attempts + 1)

    @property
    def is_exhausted(self) -> bool:
        return self.attempts > self.max_retries

    def to_dict(self) -> dict:
        return {
            "task_id":         self.task_id,
            "task_name":       self.task_name,
            "max_retries":     self.max_retries,
            "attempts":        self.attempts,
            "remaining":       self.remaining,
            "last_error":      self.last_error,
            "first_failure_at": self.first_failure_at,
            "last_failure_at": self.last_failure_at,
            "total_retry_s":   self.total_retry_s,
            "exhausted":       self.is_exhausted,
        }


@dataclass
class RetryResult:
    """Result returned by handle_failure()."""
    action:       RetryAction
    attempt:      int
    backoff_s:    float
    retryable:    bool
    reason:       str


# ─────────────────────────────────────────────────────────────────────────────
# BACKPRESSURE SIGNAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BackpressureSignal:
    """
    Emitted by PreemptionHandler when it detects chronic overload.

    The SchedulerAgent uses this signal to:
      - Reduce max_workers further (beyond the standard throttle)
      - Pause new task submissions from the research pipeline
      - Alert the LabAgent's alert queue

    Analogous to TCP's explicit congestion notification (ECN) —
    a signal from the network layer that the receiver is overwhelmed,
    prompting the sender to reduce transmission rate.
    """
    signal_type:     str    # "task_backpressure" | "system_overload"
    triggered_by:    str    # task_name that triggered it
    defer_count:     int
    active_deferrals: int   # how many tasks are currently deferred
    timestamp:       str    = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    suggestion:      str    = ""

    def __str__(self) -> str:
        return (
            f"BackpressureSignal({self.signal_type}) "
            f"task='{self.triggered_by}' "
            f"defers={self.defer_count} "
            f"active={self.active_deferrals}: "
            f"{self.suggestion}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# NON-RETRYABLE ERROR CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

# Exception types that should NOT be retried (immediate failure)
NON_RETRYABLE_TYPES: set[str] = {
    "KeyboardInterrupt",
    "SystemExit",
    "ValueError",           # bad input — retrying won't fix it
    "NotImplementedError",  # code bug — retrying won't fix it
    "ImportError",          # missing dependency — retrying won't fix it
    "TaskExpiredError",     # already expired — no point retrying
    "RetryBudgetExhaustedError",
}

# Exception type name prefixes that ARE retryable
RETRYABLE_PREFIXES: set[str] = {
    "ConnectionError",
    "TimeoutError",
    "OSError",
    "IOError",
    "httpx",
    "openai",
    "RateLimitError",
    "APIError",
    "arxiv",
}


def is_retryable(exc: Exception) -> bool:
    """
    Classify whether an exception warrants a retry.

    Returns True for transient errors (network, timeout, rate limits).
    Returns False for logic errors (bad input, missing code, expired tasks).
    """
    exc_type = type(exc).__name__
    if exc_type in NON_RETRYABLE_TYPES:
        return False
    for prefix in RETRYABLE_PREFIXES:
        if exc_type.startswith(prefix) or prefix.lower() in exc_type.lower():
            return True
    # Default: assume retryable for unknown exceptions
    return True


# ─────────────────────────────────────────────────────────────────────────────
# PREEMPTION HANDLER
# ─────────────────────────────────────────────────────────────────────────────

class PreemptionHandler:
    """
    Orchestrates deferral, retry, backoff, escalation, and expiry logic
    for the SchedulerAgent.

    Responsibilities:
      handle_deferral()  — called when ResourcePolicy returns DEFER
      handle_failure()   — called when a worker thread raises an exception
      drain_signals()    — returns pending BackpressureSignals
      audit_trail()      — full deferral + retry history for a task
      stats()            — aggregate counters for dashboard

    The handler is the only place in the system where tasks sleep
    (via backoff). It runs blocking sleeps in the calling thread —
    callers that want non-blocking behaviour should use a thread pool.

    Thread safety: all public methods acquire self._lock.
    """

    def __init__(
        self,
        priority_queue:    PriorityQueue,
        gantt_logger:      GanttLogger,
        deferral_backoff:  BackoffStrategy | None = None,
        retry_backoff:     BackoffStrategy | None = None,
        default_deadline_s: float = DEFAULT_DEADLINE_S,
        escalation_threshold: int = 5,   # deferrals before priority boost
        backpressure_cb: Callable[[BackpressureSignal], None] | None = None,
    ) -> None:
        self._pq         = priority_queue
        self._gantt      = gantt_logger
        self._deadline_s = default_deadline_s
        self._escalation_threshold = escalation_threshold
        self._backpressure_cb = backpressure_cb

        # Backoff strategies
        self._deferral_backoff = deferral_backoff or BackoffStrategy(
            algorithm=BackoffAlgorithm.EXPONENTIAL,
            base_s=1.5,
            max_s=30.0,
            jitter=True,
        )
        self._retry_backoff = retry_backoff or BackoffStrategy(
            algorithm=BackoffAlgorithm.EXPONENTIAL,
            base_s=2.0,
            max_s=60.0,
            jitter=True,
        )

        # Per-task state
        self._deferral_history: dict[str, list[DeferralRecord]] = {}
        self._retry_records:    dict[str, RetryRecord]          = {}
        self._task_enqueue_times: dict[str, float] = {}   # task_id → enqueue epoch

        # Backpressure signal queue
        self._signals: list[BackpressureSignal] = []

        # Stats
        self._total_deferrals:   int = 0
        self._total_escalations: int = 0
        self._total_expirations: int = 0
        self._total_retries:     int = 0
        self._total_exhausted:   int = 0

        self._lock = threading.Lock()
        logger.info(
            "PreemptionHandler initialised | "
            "deferral_backoff=%s escalation_at=%d deadline=%.0fs",
            self._deferral_backoff._algorithm.value,
            escalation_threshold,
            default_deadline_s,
        )

    # ── DEFERRAL HANDLING ────────────────────────────────────────────────────

    def handle_deferral(
        self,
        entry:            QueueEntry,
        reason:           str         = "",
        rule_name:        str         = "",
        cpu_pct:          float       = 0.0,
        ram_pct:          float       = 0.0,
        gpu_pct:          float       = 0.0,
        bottleneck:       str         = "none",
        scheduler_state:  str         = "running",
        deadline_s:       float | None = None,
    ) -> DeferralResult:
        """
        Handle a task being deferred due to resource constraints.

        Steps:
          1. Record deferral time and check deadline
          2. If deadline exceeded → expire the task
          3. Compute backoff wait time
          4. Check if escalation threshold reached → boost priority
          5. Sleep for backoff duration
          6. Requeue entry in priority queue
          7. Log to GanttLogger
          8. Check backpressure thresholds → emit signal if needed

        Returns a DeferralResult describing what happened.
        """
        task_id   = entry.task_id
        task_name = entry.task_name
        eff_deadline = deadline_s if deadline_s is not None else self._deadline_s

        with self._lock:
            # Track first-seen time
            if task_id not in self._task_enqueue_times:
                self._task_enqueue_times[task_id] = time.time()

            elapsed_s = time.time() - self._task_enqueue_times[task_id]
            defer_num = entry.defer_count + 1   # this will be the new defer count

            # ── 1: Deadline check ──────────────────────────────────────────
            if elapsed_s >= eff_deadline:
                self._total_expirations += 1
                self._cleanup_task(task_id)
                logger.warning(
                    "EXPIRED: '%s' [%s] elapsed=%.1fs deadline=%.1fs",
                    task_name, task_id, elapsed_s, eff_deadline,
                )
                return DeferralResult(
                    action=DeferralAction.EXPIRED,
                    backoff_s=0.0,
                    defer_number=defer_num,
                    escalated=False,
                    reason=f"Deadline exceeded: {elapsed_s:.1f}s ≥ {eff_deadline:.1f}s",
                    expired=True,
                )

            # ── 2: Compute backoff ─────────────────────────────────────────
            backoff_s = self._deferral_backoff.compute(defer_num)

            # ── 3: Check escalation threshold ─────────────────────────────
            priority_before = entry.effective_priority_name
            escalated       = False

            if defer_num >= self._escalation_threshold:
                old_eff = entry.effective_priority
                entry.effective_priority = max(
                    int(Priority.CRITICAL),
                    entry.effective_priority - 1,
                )
                escalated = old_eff != entry.effective_priority
                if escalated:
                    self._total_escalations += 1
                    logger.info(
                        "ESCALATED: '%s' %s → %s after %d deferrals",
                        task_name,
                        entry.effective_priority_name,
                        Priority(entry.effective_priority).name,
                        defer_num,
                    )

            priority_after = entry.effective_priority_name

            # ── 4: Record deferral ────────────────────────────────────────
            record = DeferralRecord(
                task_id=task_id,
                task_name=task_name,
                defer_number=defer_num,
                timestamp=datetime.now(timezone.utc).isoformat(),
                reason=reason,
                rule_name=rule_name,
                backoff_s=backoff_s,
                cpu_pct=cpu_pct,
                ram_pct=ram_pct,
                gpu_pct=gpu_pct,
                priority_before=priority_before,
                priority_after=priority_after,
                escalated=escalated,
            )
            self._deferral_history.setdefault(task_id, []).append(record)
            self._total_deferrals += 1

            # ── 5: Check backpressure ─────────────────────────────────────
            signal = self._check_backpressure(entry, defer_num)

        # ── 6: Sleep (outside lock — no need to hold it during sleep) ────
        if backoff_s > 0:
            logger.debug(
                "Deferral backoff: '%s' sleeping %.2fs (defer #%d)",
                task_name, backoff_s, defer_num,
            )
            time.sleep(backoff_s)

        with self._lock:
            # ── 7: Requeue ────────────────────────────────────────────────
            self._pq.requeue(entry)

        # ── 8: Log to Gantt ───────────────────────────────────────────────
        self._gantt.on_deferred(
            task_id=task_id,
            defer_count=defer_num,
            reason=reason,
            cpu_pct=cpu_pct,
            ram_pct=ram_pct,
            gpu_pct=gpu_pct,
            bottleneck=bottleneck,
            scheduler_state=scheduler_state,
        )

        # ── 9: Fire backpressure signal ───────────────────────────────────
        if signal and self._backpressure_cb:
            try:
                self._backpressure_cb(signal)
            except Exception as exc:
                logger.warning("Backpressure callback error: %s", exc)

        action = DeferralAction.ESCALATED if escalated else DeferralAction.REQUEUED
        result = DeferralResult(
            action=action,
            backoff_s=backoff_s,
            defer_number=defer_num,
            escalated=escalated,
            reason=reason,
        )

        logger.info(
            "DEFERRED: '%s' [%s] #%d backoff=%.2fs escalated=%s | %s",
            task_name, task_id, defer_num, backoff_s, escalated,
            reason[:60],
        )
        return result

    # ── FAILURE / RETRY HANDLING ─────────────────────────────────────────────

    def handle_failure(
        self,
        entry:        QueueEntry,
        exception:    Exception,
        resubmit_fn:  Callable[[QueueEntry], Any] | None = None,
        max_retries:  int = 2,
    ) -> RetryResult:
        """
        Handle a task that raised an exception during execution.

        Steps:
          1. Classify exception as retryable or non-retryable
          2. Look up or create RetryRecord for this task
          3. If exhausted → set exception on future, log FAILED
          4. If non-retryable → immediate failure, log FAILED
          5. If retryable + within budget:
               - compute backoff
               - sleep
               - re-submit to priority queue via resubmit_fn
               - log retry attempt

        Returns a RetryResult describing the outcome.
        """
        task_id   = entry.task_id
        task_name = entry.task_name
        exc_type  = type(exception).__name__
        exc_msg   = str(exception)[:200]
        now_iso   = datetime.now(timezone.utc).isoformat()

        with self._lock:
            # ── 1: Get or create RetryRecord ──────────────────────────────
            if task_id not in self._retry_records:
                self._retry_records[task_id] = RetryRecord(
                    task_id=task_id,
                    task_name=task_name,
                    max_retries=max_retries,
                    first_failure_at=now_iso,
                )
            record = self._retry_records[task_id]
            record.attempts += 1
            record.last_error      = exc_msg
            record.last_error_type = exc_type
            record.last_failure_at = now_iso

            # ── 2: Non-retryable check ────────────────────────────────────
            if not is_retryable(exception):
                self._total_exhausted += 1
                self._cleanup_task(task_id)
                self._gantt.on_failed(task_id, error=f"[NON-RETRYABLE] {exc_msg}")
                logger.error(
                    "NON-RETRYABLE: '%s' %s: %s",
                    task_name, exc_type, exc_msg,
                )
                return RetryResult(
                    action=RetryAction.SKIPPED,
                    attempt=record.attempts,
                    backoff_s=0.0,
                    retryable=False,
                    reason=f"Non-retryable exception type: {exc_type}",
                )

            # ── 3: Budget exhausted check ─────────────────────────────────
            if record.is_exhausted:
                self._total_exhausted += 1
                self._cleanup_task(task_id)
                self._gantt.on_failed(task_id, error=f"[EXHAUSTED] {exc_msg}")
                logger.error(
                    "EXHAUSTED: '%s' failed after %d attempts. Last: %s",
                    task_name, record.attempts, exc_msg,
                )
                return RetryResult(
                    action=RetryAction.EXHAUSTED,
                    attempt=record.attempts,
                    backoff_s=0.0,
                    retryable=True,
                    reason=(
                        f"Retry budget exhausted after {record.attempts} attempts. "
                        f"Last error: {exc_msg}"
                    ),
                )

            # ── 4: Compute retry backoff ──────────────────────────────────
            backoff_s = self._retry_backoff.compute(record.attempts)
            record.total_retry_s += backoff_s
            self._total_retries += 1

        # ── 5: Sleep outside lock ─────────────────────────────────────────
        logger.info(
            "RETRY: '%s' attempt %d/%d backoff=%.2fs | %s: %s",
            task_name,
            record.attempts,
            record.max_retries + 1,
            backoff_s,
            exc_type, exc_msg[:60],
        )
        time.sleep(backoff_s)

        # ── 6: Re-submit to queue ─────────────────────────────────────────
        if resubmit_fn:
            try:
                entry.state = "pending"
                entry.defer_count = 0       # reset deferral count for fresh run
                resubmit_fn(entry)
                logger.info(
                    "RE-SUBMITTED: '%s' [%s] (attempt %d)",
                    task_name, task_id, record.attempts,
                )
            except Exception as exc:
                logger.error("Re-submit failed for '%s': %s", task_name, exc)

        return RetryResult(
            action=RetryAction.RETRYING,
            attempt=record.attempts,
            backoff_s=backoff_s,
            retryable=True,
            reason=(
                f"Retrying attempt {record.attempts}/{record.max_retries + 1} "
                f"after {backoff_s:.2f}s backoff. Error: {exc_msg[:60]}"
            ),
        )

    # ── BACKPRESSURE ─────────────────────────────────────────────────────────

    def _check_backpressure(
        self,
        entry: QueueEntry,
        defer_num: int,
    ) -> BackpressureSignal | None:
        """
        Emit a BackpressureSignal if:
          - A single task has been deferred ≥ BACKPRESSURE_DEFERRAL_THRESHOLD
          - OR ≥ OVERLOAD_TASK_THRESHOLD different tasks are actively deferred

        Must be called with self._lock held.
        """
        # Check single-task threshold
        if defer_num >= BACKPRESSURE_DEFERRAL_THRESHOLD:
            signal = BackpressureSignal(
                signal_type="task_backpressure",
                triggered_by=entry.task_name,
                defer_count=defer_num,
                active_deferrals=len(self._deferral_history),
                suggestion=(
                    f"Task '{entry.task_name}' has been deferred {defer_num} times. "
                    f"Consider reducing pipeline concurrency or "
                    f"increasing resource thresholds in scheduler_config.yaml."
                ),
            )
            self._signals.append(signal)
            logger.warning("BACKPRESSURE: %s", signal)
            return signal

        # Check system-wide overload threshold
        active_count = len([
            tid for tid, hist in self._deferral_history.items()
            if hist and hist[-1].defer_number > 0
        ])
        if active_count >= OVERLOAD_TASK_THRESHOLD:
            signal = BackpressureSignal(
                signal_type="system_overload",
                triggered_by=entry.task_name,
                defer_count=defer_num,
                active_deferrals=active_count,
                suggestion=(
                    f"{active_count} tasks are simultaneously deferred. "
                    f"System is chronically overloaded. "
                    f"Recommended: reduce max_workers in scheduler_config.yaml "
                    f"or upgrade compute resources."
                ),
            )
            self._signals.append(signal)
            logger.warning("SYSTEM OVERLOAD: %s", signal)
            return signal

        return None

    def drain_signals(self) -> list[BackpressureSignal]:
        """
        Return and clear pending BackpressureSignals.
        Called by the SchedulerAgent on each tick.
        """
        with self._lock:
            signals = list(self._signals)
            self._signals.clear()
        return signals

    # ── AUDIT TRAIL ──────────────────────────────────────────────────────────

    def deferral_history(self, task_id: str) -> list[dict]:
        """Return the full deferral history for a task as a list of dicts."""
        with self._lock:
            records = self._deferral_history.get(task_id, [])
        return [
            {
                "defer_number":    r.defer_number,
                "timestamp":       r.timestamp,
                "reason":          r.reason,
                "rule_name":       r.rule_name,
                "backoff_s":       r.backoff_s,
                "cpu_pct":         r.cpu_pct,
                "ram_pct":         r.ram_pct,
                "priority_before": r.priority_before,
                "priority_after":  r.priority_after,
                "escalated":       r.escalated,
            }
            for r in records
        ]

    def retry_history(self, task_id: str) -> dict:
        """Return the retry record for a task as a dict."""
        with self._lock:
            record = self._retry_records.get(task_id)
        return record.to_dict() if record else {}

    def all_active_deferrals(self) -> list[dict]:
        """
        Returns a summary of all tasks currently in a deferred state.
        Used by the Streamlit scheduler panel to highlight stuck tasks.
        """
        with self._lock:
            result = []
            for task_id, history in self._deferral_history.items():
                if history:
                    last = history[-1]
                    elapsed = time.time() - self._task_enqueue_times.get(task_id, time.time())
                    result.append({
                        "task_id":      task_id,
                        "task_name":    last.task_name,
                        "defer_count":  last.defer_number,
                        "last_reason":  last.reason[:80],
                        "last_rule":    last.rule_name,
                        "escalated":    last.escalated,
                        "elapsed_s":    round(elapsed, 1),
                        "deadline_s":   self._deadline_s,
                        "pct_deadline": round(elapsed / self._deadline_s * 100, 1),
                    })
        return sorted(result, key=lambda x: -x["defer_count"])

    # ── STATS ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Aggregate counters for the Streamlit dashboard."""
        with self._lock:
            active = len(self._deferral_history)
            pending_signals = len(self._signals)
        return {
            "total_deferrals":   self._total_deferrals,
            "total_escalations": self._total_escalations,
            "total_expirations": self._total_expirations,
            "total_retries":     self._total_retries,
            "total_exhausted":   self._total_exhausted,
            "active_deferred":   active,
            "pending_signals":   pending_signals,
        }

    def reset(self) -> None:
        """Clear all state (used between pipeline runs)."""
        with self._lock:
            self._deferral_history.clear()
            self._retry_records.clear()
            self._task_enqueue_times.clear()
            self._signals.clear()
            self._total_deferrals   = 0
            self._total_escalations = 0
            self._total_expirations = 0
            self._total_retries     = 0
            self._total_exhausted   = 0
        logger.info("PreemptionHandler reset")

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _cleanup_task(self, task_id: str) -> None:
        """Remove per-task state after expiry or exhaustion. Lock must be held."""
        self._deferral_history.pop(task_id, None)
        self._retry_records.pop(task_id, None)
        self._task_enqueue_times.pop(task_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Demonstrates deferral backoff sequences, escalation, expiry,
    retry with exponential backoff, and backpressure signals.

    Run with: python -m scheduler.preemption_handler
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # ── Backoff sequence visualisation ───────────────────────────────────────
    print("\n═══ BACKOFF SEQUENCES (no jitter for clarity) ═══\n")

    algorithms = [
        BackoffAlgorithm.EXPONENTIAL,
        BackoffAlgorithm.LINEAR,
        BackoffAlgorithm.CONSTANT,
        BackoffAlgorithm.FIBONACCI,
    ]
    for algo in algorithms:
        strat = BackoffStrategy(algorithm=algo, base_s=2.0, max_s=30.0, jitter=False)
        seq   = strat.sequence(8)
        bars  = " ".join(f"{s:5.1f}" for s in seq)
        print(f"  {algo.value:<14}: {bars}")

    print("\n  (With ±20% jitter applied in production)")

    # ── Deferral + escalation demo ───────────────────────────────────────────
    print("\n═══ DEFERRAL + ESCALATION DEMO ═══\n")

    pq    = PriorityQueue()
    gantt = GanttLogger(log_path=Path("logs/demo_preemption.jsonl"))

    signals_received: list[BackpressureSignal] = []

    handler = PreemptionHandler(
        priority_queue=pq,
        gantt_logger=gantt,
        deferral_backoff=BackoffStrategy(
            algorithm=BackoffAlgorithm.EXPONENTIAL,
            base_s=0.05,    # very short for demo
            max_s=0.5,
            jitter=False,
        ),
        retry_backoff=BackoffStrategy(
            algorithm=BackoffAlgorithm.EXPONENTIAL,
            base_s=0.05,
            max_s=0.5,
            jitter=False,
        ),
        default_deadline_s=60.0,
        escalation_threshold=3,
        backpressure_cb=signals_received.append,
    )

    entry = QueueEntry(
        task_id="t_embed",
        task_name="abstract_embedding",
        priority=Priority.MEDIUM,
        cpu_cost="high",
        ram_cost="medium",
        gpu_cost="medium",
    )
    pq.push(entry)
    entry = pq.pop()
    gantt.on_queued(
        entry.task_id, entry.task_name,
        "MEDIUM", 2, "high", "medium", "medium", "embedding"
    )

    print(f"  Task: '{entry.task_name}' | priority={entry.priority.name}\n")

    for i in range(6):
        result = handler.handle_deferral(
            entry,
            reason=f"CPU 85% > limit (sim defer #{i+1})",
            rule_name="cpu_threshold",
            cpu_pct=85.0 + i,
            ram_pct=50.0,
        )
        icon = "⬆ ESCALATED" if result.escalated else "↩ requeued"
        print(
            f"  Defer #{result.defer_number:2d} | "
            f"backoff={result.backoff_s:.3f}s | "
            f"eff_priority={entry.effective_priority_name} | "
            f"{icon}"
        )
        # Pop back out for next iteration
        entry = pq.pop() or entry

    # ── Expiry demo ───────────────────────────────────────────────────────────
    print("\n═══ EXPIRY DEMO ═══\n")
    expired_entry = QueueEntry("t_expire", "umap_reduction", Priority.LOW)
    pq.push(expired_entry)
    expired_entry = pq.pop()
    gantt.on_queued("t_expire", "umap_reduction", "LOW", 3)

    # Manually backdate the enqueue time to simulate elapsed deadline
    with handler._lock:
        handler._task_enqueue_times["t_expire"] = time.time() - 700  # 700s ago

    result = handler.handle_deferral(
        expired_entry,
        reason="Simulated long wait",
        rule_name="ram_threshold",
    )
    print(f"  Action: {result.action} | expired={result.expired}")
    print(f"  Reason: {result.reason}")

    # ── Retry demo ────────────────────────────────────────────────────────────
    print("\n═══ RETRY DEMO ═══\n")
    retry_entry = QueueEntry("t_retry", "gap_analysis_react", Priority.HIGH)
    pq.push(retry_entry)
    retry_entry = pq.pop()
    gantt.on_queued("t_retry", "gap_analysis_react", "HIGH", 1)
    gantt.on_dispatched("t_retry", cpu_pct=45.0, ram_pct=50.0)

    resubmitted: list[str] = []

    def fake_resubmit(e: QueueEntry) -> None:
        resubmitted.append(e.task_id)
        pq.push(e)

    for attempt in range(1, 4):
        exc = ConnectionError(f"OpenAI API timeout (attempt {attempt})")
        result = handler.handle_failure(
            retry_entry, exc,
            resubmit_fn=fake_resubmit,
            max_retries=2,
        )
        print(
            f"  Attempt {result.attempt} | "
            f"action={result.action} | "
            f"backoff={result.backoff_s:.3f}s | "
            f"retryable={result.retryable}"
        )
        if result.action == RetryAction.EXHAUSTED:
            break

    print(f"  Re-submitted to queue: {resubmitted}")

    # ── Backpressure signals ───────────────────────────────────────────────────
    print("\n═══ BACKPRESSURE SIGNALS ═══\n")
    for sig in signals_received:
        print(f"  [{sig.signal_type}] {sig.suggestion[:80]}")

    # ── Stats ─────────────────────────────────────────────────────────────────
    print("\n═══ PREEMPTION HANDLER STATS ═══\n")
    for k, v in handler.stats().items():
        print(f"  {k:<26} {v}")

    print("\n═══ GANTT METRICS ═══\n")
    met = gantt.scheduling_metrics()
    for line in met.summary_lines():
        print(line)


if __name__ == "__main__":
    from pathlib import Path
    _demo()
