"""
scheduler/priority_queue.py
============================
ResearchFlow AI — Heapq-Backed Priority Queue with Aging

This module implements the core scheduling data structure used by the
SchedulerAgent. It is intentionally kept as a standalone, dependency-free
module so it can be unit-tested and reasoned about in isolation.

OS Scheduling Concepts implemented here:
  ┌───────────────────────────────────────────────────────────────────┐
  │  Concept                  │  Implementation                       │
  ├───────────────────────────┼───────────────────────────────────────┤
  │  Priority Scheduling      │  heapq min-heap on (priority, seq)   │
  │  Priority Aging           │  age_ticks → effective_priority boost │
  │  Starvation Prevention    │  MAX_AGE_TICKS force-boost to HIGH    │
  │  FIFO tiebreak            │  monotonic arrival_seq counter        │
  │  Ready Queue              │  PriorityQueue.push / pop             │
  │  Process Control Block    │  QueueEntry dataclass                 │
  │  Preemption flag          │  QueueEntry.preempted field           │
  │  Batch submission         │  push_many()                         │
  │  Queue introspection      │  peek / snapshot / depth             │
  └───────────────────────────────────────────────────────────────────┘

Key design decisions:
  1. The heap stores (sort_key, QueueEntry) tuples — sort_key is
     recomputed on every heapify() call to reflect aging.
  2. Aging never lets a task starve below CRITICAL priority (0).
  3. Thread-safe: all public methods acquire self._lock.
  4. QueueEntry is immutable after construction except for the mutable
     fields annotated below — all mutations go through queue methods.
  5. The module is pure Python stdlib — no agents/ imports.

Usage:
    pq = PriorityQueue()

    entry = QueueEntry(
        task_id="abc123",
        task_name="abstract_embedding",
        priority=Priority.MEDIUM,
        cpu_cost="high",
        ram_cost="medium",
        gpu_cost="medium",
    )
    pq.push(entry)

    next_entry = pq.pop()         # highest-priority ready entry
    snapshot   = pq.snapshot()    # full queue state (non-destructive)
    stats      = pq.stats()       # counters for dashboard

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Iterator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# How many ticks before a task's effective priority is boosted by 1 level
AGING_BOOST_INTERVAL: int = 10

# How many total ticks before a task is force-elevated to HIGH priority
# (starvation prevention safety valve)
MAX_AGE_TICKS: int = 50

# Priority level at which aging stops (can't go higher than CRITICAL)
MIN_PRIORITY_VALUE: int = 0


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY ENUM
# ─────────────────────────────────────────────────────────────────────────────

class Priority(IntEnum):
    """
    Task priority levels.

    Lower integer = higher scheduling priority (heapq is a min-heap,
    so 0 is popped first). This mirrors OS nice-value inversion:
    CRITICAL (0) is scheduled before IDLE (4).

    Maps directly to agents/scheduler_agent.Priority — kept separate
    here so this module has zero cross-module dependencies.
    """
    CRITICAL = 0    # system monitor, safety operations
    HIGH     = 1    # gap detection, query reformulation
    MEDIUM   = 2    # embedding, clustering, labeling
    LOW      = 3    # report generation, roadmap building
    IDLE     = 4    # background cleanup, optional enrichment


PRIORITY_NAMES: dict[int, str] = {
    0: "CRITICAL",
    1: "HIGH",
    2: "MEDIUM",
    3: "LOW",
    4: "IDLE",
}


# ─────────────────────────────────────────────────────────────────────────────
# QUEUE ENTRY (Process Control Block analogue)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueueEntry:
    """
    Represents a single schedulable item in the priority queue.

    Analogous to an OS Process Control Block (PCB) — it carries all
    the metadata the scheduler needs to make dispatch decisions without
    executing any work itself.

    Immutable fields (set at construction, never changed):
      task_id, task_name, priority, cpu_cost, ram_cost, gpu_cost,
      arrival_seq, enqueue_time, submit_time_str

    Mutable scheduling fields (updated by the queue):
      effective_priority  — starts == priority, decremented by aging
      age_ticks           — incremented on every scheduler tick
      defer_count         — incremented each time the entry is requeued
      preempted           — set True if entry was removed by preemption
      state               — PENDING → RUNNING / CANCELLED
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    task_id:            str
    task_name:          str

    # ── Scheduling policy ────────────────────────────────────────────────────
    priority:           Priority = Priority.MEDIUM
    cpu_cost:           str      = "medium"   # "none"|"low"|"medium"|"high"
    ram_cost:           str      = "medium"
    gpu_cost:           str      = "none"
    estimated_s:        float    = 0.0        # hint for future SJF variant

    # ── Queue metadata (set by PriorityQueue.push) ───────────────────────────
    arrival_seq:        int   = 0             # monotonic; FIFO tiebreak
    enqueue_time:       float = field(default_factory=time.time)
    submit_time_str:    str   = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── Mutable scheduling state ─────────────────────────────────────────────
    effective_priority: int   = field(init=False)
    age_ticks:          int   = 0
    defer_count:        int   = 0
    preempted:          bool  = False
    state:              str   = "pending"     # "pending"|"running"|"cancelled"

    def __post_init__(self) -> None:
        self.effective_priority = int(self.priority)

    # ── Comparison (heapq ordering) ──────────────────────────────────────────

    def sort_key(self) -> tuple[int, int]:
        """
        Primary sort key: (effective_priority, arrival_seq).
        Lower effective_priority = dispatched first.
        arrival_seq breaks ties within the same priority → FIFO.
        """
        return (self.effective_priority, self.arrival_seq)

    def __lt__(self, other: QueueEntry) -> bool:
        return self.sort_key() < other.sort_key()

    # ── Aging ────────────────────────────────────────────────────────────────

    def tick(self) -> bool:
        """
        Age this entry by one scheduler tick.

        Returns True if effective_priority was boosted (useful for
        logging / dashboard highlighting of boosted tasks).

        Aging logic:
          Every AGING_BOOST_INTERVAL ticks → boost effective_priority by 1
          After MAX_AGE_TICKS total ticks → force-boost to HIGH (1)
          Never below CRITICAL (0)

        This mirrors Linux's priority boosting mechanism for interactive
        tasks and prevents starvation of LOW/IDLE-priority work.
        """
        self.age_ticks += 1
        boosted = False

        # Safety valve: force to HIGH after MAX_AGE_TICKS regardless
        if self.age_ticks >= MAX_AGE_TICKS and self.effective_priority > int(Priority.HIGH):
            old = self.effective_priority
            self.effective_priority = int(Priority.HIGH)
            logger.warning(
                "Starvation prevention: '%s' force-boosted %s → HIGH after %d ticks",
                self.task_name,
                PRIORITY_NAMES.get(old, str(old)),
                self.age_ticks,
            )
            return True

        # Normal aging: boost every AGING_BOOST_INTERVAL ticks
        if (
            self.age_ticks % AGING_BOOST_INTERVAL == 0
            and self.effective_priority > MIN_PRIORITY_VALUE
        ):
            self.effective_priority -= 1
            boosted = True
            logger.debug(
                "Aging boost: '%s' %s → %s (tick %d)",
                self.task_name,
                PRIORITY_NAMES.get(self.effective_priority + 1, "?"),
                PRIORITY_NAMES.get(self.effective_priority, "?"),
                self.age_ticks,
            )

        return boosted

    # ── Convenience properties ───────────────────────────────────────────────

    @property
    def wait_time_s(self) -> float:
        """Seconds since this entry was enqueued."""
        return round(time.time() - self.enqueue_time, 3)

    @property
    def priority_name(self) -> str:
        return PRIORITY_NAMES.get(int(self.priority), "UNKNOWN")

    @property
    def effective_priority_name(self) -> str:
        return PRIORITY_NAMES.get(self.effective_priority, "UNKNOWN")

    @property
    def has_been_aged(self) -> bool:
        """True if aging has boosted this entry above its original priority."""
        return self.effective_priority < int(self.priority)

    def to_dict(self) -> dict:
        """Serialisable snapshot for dashboard / logging."""
        return {
            "task_id":              self.task_id,
            "task_name":            self.task_name,
            "priority":             self.priority_name,
            "effective_priority":   self.effective_priority_name,
            "cpu_cost":             self.cpu_cost,
            "ram_cost":             self.ram_cost,
            "gpu_cost":             self.gpu_cost,
            "estimated_s":          self.estimated_s,
            "arrival_seq":          self.arrival_seq,
            "submit_time":          self.submit_time_str,
            "wait_time_s":          self.wait_time_s,
            "age_ticks":            self.age_ticks,
            "defer_count":          self.defer_count,
            "has_been_aged":        self.has_been_aged,
            "state":                self.state,
        }


# ─────────────────────────────────────────────────────────────────────────────
# QUEUE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueueStats:
    """Cumulative counters maintained by PriorityQueue."""
    total_pushed:      int   = 0
    total_popped:      int   = 0
    total_cancelled:   int   = 0
    total_requeued:    int   = 0    # defer_count increments
    total_age_boosts:  int   = 0
    total_starvation_prevented: int = 0
    peak_depth:        int   = 0
    avg_wait_time_s:   float = 0.0

    # Running totals for avg computation
    _wait_samples:     list[float] = field(default_factory=list)

    def record_wait(self, t: float) -> None:
        self._wait_samples.append(t)
        self.avg_wait_time_s = round(
            sum(self._wait_samples) / len(self._wait_samples), 3
        )

    def to_dict(self) -> dict:
        return {
            "total_pushed":              self.total_pushed,
            "total_popped":              self.total_popped,
            "total_cancelled":           self.total_cancelled,
            "total_requeued":            self.total_requeued,
            "total_age_boosts":          self.total_age_boosts,
            "total_starvation_prevented": self.total_starvation_prevented,
            "peak_depth":                self.peak_depth,
            "avg_wait_time_s":           self.avg_wait_time_s,
        }


# ─────────────────────────────────────────────────────────────────────────────
# PRIORITY QUEUE
# ─────────────────────────────────────────────────────────────────────────────

class PriorityQueue:
    """
    Thread-safe, heapq-backed priority queue with:

      - O(log n) push and pop
      - O(n) age_all() (called once per scheduler tick — acceptable)
      - O(1) peek (non-destructive highest-priority read)
      - O(n) cancel by task_id
      - O(n) snapshot (returns sorted copy without modifying heap)
      - Priority aging with starvation prevention

    Internal representation:
      self._heap stores (sort_key_tuple, QueueEntry) pairs.
      heapq operates on sort_key_tuple directly (tuple comparison).
      We re-heapify after age_all() because entries' sort keys change.

    Thread safety:
      All public methods acquire self._lock.
      Callers must NOT hold the lock when calling public methods —
      the lock is non-reentrant (threading.Lock, not RLock).
    """

    def __init__(self) -> None:
        self._heap: list[tuple[tuple[int, int], QueueEntry]] = []
        self._lock = threading.Lock()
        self._seq_counter: int = 0
        self._stats = QueueStats()
        self._cancelled_ids: set[str] = set()   # fast O(1) cancellation check

        logger.debug("PriorityQueue initialised")

    # ── PUSH ─────────────────────────────────────────────────────────────────

    def push(self, entry: QueueEntry) -> int:
        """
        Add an entry to the priority queue.

        Sets entry.arrival_seq (monotonic FIFO counter).
        Returns the assigned arrival_seq.

        Analogous to a process arriving in the OS ready queue.
        """
        with self._lock:
            entry.arrival_seq = self._seq_counter
            self._seq_counter += 1
            entry.enqueue_time = time.time()
            entry.submit_time_str = datetime.utcnow().isoformat()
            entry.effective_priority = int(entry.priority)
            entry.state = "pending"

            heapq.heappush(self._heap, (entry.sort_key(), entry))

            depth = len(self._heap)
            self._stats.total_pushed += 1
            if depth > self._stats.peak_depth:
                self._stats.peak_depth = depth

            logger.debug(
                "PUSH: '%s' [%s] pri=%s seq=%d depth=%d",
                entry.task_name, entry.task_id,
                entry.priority_name, entry.arrival_seq, depth,
            )
            return entry.arrival_seq

    def push_many(self, entries: list[QueueEntry]) -> list[int]:
        """
        Batch-push multiple entries atomically (single lock acquisition).
        More efficient than calling push() in a loop for bulk submissions.
        Returns list of assigned arrival_seqs.
        """
        with self._lock:
            seqs: list[int] = []
            for entry in entries:
                entry.arrival_seq = self._seq_counter
                self._seq_counter += 1
                entry.enqueue_time = time.time()
                entry.submit_time_str = datetime.utcnow().isoformat()
                entry.effective_priority = int(entry.priority)
                entry.state = "pending"
                heapq.heappush(self._heap, (entry.sort_key(), entry))
                seqs.append(entry.arrival_seq)

            depth = len(self._heap)
            self._stats.total_pushed += len(entries)
            if depth > self._stats.peak_depth:
                self._stats.peak_depth = depth

            logger.debug("PUSH_MANY: %d entries, depth now %d", len(entries), depth)
            return seqs

    # ── POP ──────────────────────────────────────────────────────────────────

    def pop(self) -> QueueEntry | None:
        """
        Remove and return the highest-priority non-cancelled entry.

        Silently skips and discards cancelled entries (lazy deletion).
        Returns None if the queue is empty (after skipping cancelled).

        Analogous to the OS dispatcher selecting the next process
        to run from the ready queue.
        """
        with self._lock:
            while self._heap:
                _, entry = heapq.heappop(self._heap)

                # Lazy deletion: skip cancelled entries
                if entry.task_id in self._cancelled_ids:
                    self._cancelled_ids.discard(entry.task_id)
                    continue

                entry.state = "running"
                self._stats.total_popped += 1
                self._stats.record_wait(entry.wait_time_s)

                logger.debug(
                    "POP: '%s' [%s] eff_pri=%s wait=%.2fs defer=%d",
                    entry.task_name, entry.task_id,
                    entry.effective_priority_name,
                    entry.wait_time_s, entry.defer_count,
                )
                return entry

        return None

    def pop_batch(self, n: int) -> list[QueueEntry]:
        """
        Pop up to n entries in one call.
        Used when the scheduler has multiple available worker slots.
        """
        results: list[QueueEntry] = []
        for _ in range(n):
            entry = self.pop()
            if entry is None:
                break
            results.append(entry)
        return results

    # ── PEEK ─────────────────────────────────────────────────────────────────

    def peek(self) -> QueueEntry | None:
        """
        Return the highest-priority entry WITHOUT removing it.
        Thread-safe. Returns None if queue is empty.

        Used by the scheduler to inspect the next candidate before
        calling ResourcePolicy.evaluate() — avoids popping a task
        only to requeue it if resources are unavailable.
        """
        with self._lock:
            for _, entry in sorted(self._heap):
                if entry.task_id not in self._cancelled_ids:
                    return entry
        return None

    # ── REQUEUE ───────────────────────────────────────────────────────────────

    def requeue(self, entry: QueueEntry) -> None:
        """
        Return a previously popped entry to the queue (deferral).

        Increments defer_count and preserves the current effective_priority
        (aging already applied before the pop attempt).

        Analogous to a task being moved back from RUNNING to READY
        due to a resource constraint — not a new submission.
        """
        with self._lock:
            entry.defer_count += 1
            entry.state = "pending"
            entry.enqueue_time = time.time()   # reset wait timer

            heapq.heappush(self._heap, (entry.sort_key(), entry))
            self._stats.total_requeued += 1

            logger.debug(
                "REQUEUE: '%s' defer_count=%d eff_pri=%s",
                entry.task_name, entry.defer_count, entry.effective_priority_name,
            )

    # ── CANCEL ───────────────────────────────────────────────────────────────

    def cancel(self, task_id: str) -> bool:
        """
        Mark a task as cancelled (lazy deletion — removed on next pop).

        Returns True if the task was found in the queue, False if it
        was not present (already popped or never pushed).

        O(n) scan to verify presence; O(1) deletion via cancelled_ids set.
        """
        with self._lock:
            found = any(e.task_id == task_id for _, e in self._heap)
            if found:
                self._cancelled_ids.add(task_id)
                self._stats.total_cancelled += 1
                logger.info("CANCEL: task_id=%s marked for lazy deletion", task_id)
            return found

    def cancel_by_name(self, task_name: str) -> int:
        """
        Cancel all pending tasks with the given name.
        Returns the number of tasks cancelled.
        """
        with self._lock:
            count = 0
            for _, entry in self._heap:
                if entry.task_name == task_name and entry.task_id not in self._cancelled_ids:
                    self._cancelled_ids.add(entry.task_id)
                    count += 1
            self._stats.total_cancelled += count
            if count:
                logger.info("CANCEL_BY_NAME: '%s' → %d tasks cancelled", task_name, count)
            return count

    # ── AGING ────────────────────────────────────────────────────────────────

    def age_all(self) -> int:
        """
        Age every pending entry by one tick.

        Called once per scheduler tick (e.g. every 1 second).
        After aging, re-heapifies in O(n) to restore heap invariant
        (necessary because sort keys have changed).

        Returns the number of entries whose priority was boosted.

        This is the direct implementation of priority aging —
        the OS technique that prevents indefinite starvation of
        low-priority processes in a preemptive priority scheduler.
        """
        with self._lock:
            boosts = 0
            for _, entry in self._heap:
                if entry.task_id in self._cancelled_ids:
                    continue
                if entry.state != "pending":
                    continue

                was_boosted = entry.tick()
                if was_boosted:
                    boosts += 1
                    self._stats.total_age_boosts += 1
                    if entry.age_ticks >= MAX_AGE_TICKS:
                        self._stats.total_starvation_prevented += 1

            if boosts:
                # Re-heapify: sort keys have changed due to priority boosts
                self._heap = [
                    (entry.sort_key(), entry)
                    for _, entry in self._heap
                ]
                heapq.heapify(self._heap)
                logger.debug("age_all: %d entries boosted, heap re-heapified", boosts)

            return boosts

    # ── INTROSPECTION ────────────────────────────────────────────────────────

    def snapshot(self) -> list[QueueEntry]:
        """
        Return a sorted copy of all pending (non-cancelled) entries.
        Non-destructive. Sorted by (effective_priority, arrival_seq).

        Used by the Streamlit scheduler panel to render the queue state.
        """
        with self._lock:
            entries = [
                entry for _, entry in self._heap
                if entry.task_id not in self._cancelled_ids
            ]
        return sorted(entries)

    def snapshot_dicts(self) -> list[dict]:
        """Snapshot as a list of serialisable dicts — for dashboard."""
        return [e.to_dict() for e in self.snapshot()]

    def peek_by_priority(self, priority: Priority) -> list[QueueEntry]:
        """
        Return all pending entries at exactly the given priority level.
        Non-destructive. Useful for targeted dashboard filtering.
        """
        with self._lock:
            return [
                entry for _, entry in self._heap
                if entry.priority == priority
                and entry.task_id not in self._cancelled_ids
            ]

    def iter_pending(self) -> Iterator[QueueEntry]:
        """
        Iterate over all pending entries in priority order.
        Non-destructive. Returns a generator over a snapshot copy.
        """
        yield from self.snapshot()

    # ── PROPERTIES ───────────────────────────────────────────────────────────

    @property
    def depth(self) -> int:
        """Current number of entries (including cancelled, pre-cleanup)."""
        with self._lock:
            return len(self._heap)

    @property
    def pending_count(self) -> int:
        """Number of non-cancelled pending entries."""
        with self._lock:
            return sum(
                1 for _, e in self._heap
                if e.task_id not in self._cancelled_ids
            )

    @property
    def is_empty(self) -> bool:
        return self.pending_count == 0

    @property
    def stats(self) -> QueueStats:
        return self._stats

    def stats_dict(self) -> dict:
        return self._stats.to_dict()

    # ── PRIORITY DISTRIBUTION ────────────────────────────────────────────────

    def priority_distribution(self) -> dict[str, int]:
        """
        Returns a count of pending entries per effective priority level.
        Used by the Streamlit dashboard to render priority distribution bars.
        """
        dist: dict[str, int] = {name: 0 for name in PRIORITY_NAMES.values()}
        with self._lock:
            for _, entry in self._heap:
                if entry.task_id not in self._cancelled_ids:
                    name = PRIORITY_NAMES.get(entry.effective_priority, "UNKNOWN")
                    dist[name] = dist.get(name, 0) + 1
        return dist

    # ── DRAIN & CLEAR ────────────────────────────────────────────────────────

    def drain(self) -> list[QueueEntry]:
        """
        Pop all pending entries and return them in priority order.
        Destructive — queue is empty after this call.
        Used for graceful shutdown or queue inspection.
        """
        results: list[QueueEntry] = []
        while not self.is_empty:
            entry = self.pop()
            if entry:
                results.append(entry)
        return results

    def clear(self) -> int:
        """
        Cancel all pending entries and empty the heap.
        Returns the number of entries cleared.
        """
        with self._lock:
            count = len(self._heap)
            for _, entry in self._heap:
                self._cancelled_ids.add(entry.task_id)
            self._heap.clear()
            self._stats.total_cancelled += count
            logger.info("Queue cleared: %d entries removed", count)
            return count

    # ── REPR ─────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self.pending_count

    def __repr__(self) -> str:
        return (
            f"PriorityQueue(depth={self.pending_count}, "
            f"pushed={self._stats.total_pushed}, "
            f"popped={self._stats.total_popped}, "
            f"boosts={self._stats.total_age_boosts})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: build QueueEntry from a Task-like object
# ─────────────────────────────────────────────────────────────────────────────

def entry_from_task(task: object) -> QueueEntry:
    """
    Construct a QueueEntry from a SchedulerAgent Task object.
    Allows the priority_queue module to remain import-free from agents/.

    Expected task attributes:
      task_id, name, priority (Priority or int), resource (ResourceProfile)
    """
    resource = getattr(task, "resource", None)
    return QueueEntry(
        task_id=getattr(task, "task_id", str(id(task))),
        task_name=getattr(task, "name", "unknown"),
        priority=Priority(int(getattr(task, "priority", Priority.MEDIUM))),
        cpu_cost=getattr(resource, "cpu_cost", "medium") if resource else "medium",
        ram_cost=getattr(resource, "ram_cost", "medium") if resource else "medium",
        gpu_cost=getattr(resource, "gpu_cost", "none")   if resource else "none",
        estimated_s=getattr(resource, "estimated_duration_s", 0.0) if resource else 0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Visual demonstration of the priority queue's scheduling behaviour.

    Shows:
      1. Push 8 tasks with mixed priorities
      2. Run 15 aging ticks and observe priority boosts
      3. Pop all tasks and verify dispatch order
      4. Demonstrate cancel, requeue, and batch operations
      5. Print full stats

    Run with: python -m scheduler.priority_queue
    """
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    pq = PriorityQueue()

    # ── Push diverse tasks ───────────────────────────────────────────────────
    tasks = [
        QueueEntry("t1", "report_generation",    Priority.LOW,      "low",    "low",    "none"),
        QueueEntry("t2", "umap_reduction",        Priority.LOW,      "high",   "high",   "none"),
        QueueEntry("t3", "gap_analysis_react",    Priority.HIGH,     "medium", "medium", "high"),
        QueueEntry("t4", "abstract_embedding",    Priority.MEDIUM,   "high",   "medium", "medium"),
        QueueEntry("t5", "cluster_labeling",      Priority.MEDIUM,   "medium", "low",    "high"),
        QueueEntry("t6", "arxiv_retrieval",       Priority.HIGH,     "low",    "low",    "none"),
        QueueEntry("t7", "hypothesis_generation", Priority.HIGH,     "low",    "low",    "medium"),
        QueueEntry("t8", "system_monitor_tick",   Priority.CRITICAL, "low",    "low",    "none"),
    ]

    print("\n═══ PUSH PHASE ═══")
    for t in tasks:
        seq = pq.push(t)
        print(f"  PUSHED  seq={seq:02d}  pri={t.priority_name:<8}  '{t.task_name}'")

    print(f"\nQueue depth: {pq.depth}")
    print(f"Priority distribution: {pq.priority_distribution()}")

    # ── Aging simulation ─────────────────────────────────────────────────────
    print("\n═══ AGING SIMULATION (15 ticks) ═══")
    for tick in range(1, 16):
        boosts = pq.age_all()
        if boosts:
            print(f"  Tick {tick:02d}: {boosts} entries boosted")
            for entry in pq.snapshot():
                if entry.has_been_aged:
                    print(
                        f"           '{entry.task_name}' "
                        f"{entry.priority_name} → {entry.effective_priority_name} "
                        f"(age={entry.age_ticks})"
                    )

    print(f"\nPost-aging distribution: {pq.priority_distribution()}")

    # ── Cancel one task ───────────────────────────────────────────────────────
    print("\n═══ CANCEL t2 (umap_reduction) ═══")
    cancelled = pq.cancel("t2")
    print(f"  Cancelled: {cancelled}")

    # ── Peek ────────────────────────────────────────────────────────────────
    top = pq.peek()
    print(f"\n═══ PEEK ═══\n  Next: '{top.task_name}' [{top.effective_priority_name}]")

    # ── Pop all and show dispatch order ──────────────────────────────────────
    print("\n═══ DISPATCH ORDER (pop all) ═══")
    position = 1
    while not pq.is_empty:
        entry = pq.pop()
        if entry:
            aged_marker = " ★ AGED" if entry.has_been_aged else ""
            print(
                f"  [{position:02d}] pri={entry.effective_priority_name:<8}  "
                f"seq={entry.arrival_seq:02d}  "
                f"wait={entry.wait_time_s:.3f}s  "
                f"'{entry.task_name}'{aged_marker}"
            )
            position += 1

    # ── Requeue demo ─────────────────────────────────────────────────────────
    print("\n═══ REQUEUE DEMO ═══")
    deferred = QueueEntry("t9", "deferred_task", Priority.MEDIUM, "medium", "medium", "high")
    pq.push(deferred)
    entry = pq.pop()
    print(f"  Popped: '{entry.task_name}' defer_count={entry.defer_count}")
    pq.requeue(entry)
    entry2 = pq.pop()
    print(f"  Requeued + popped: '{entry2.task_name}' defer_count={entry2.defer_count}")

    # ── Batch push ───────────────────────────────────────────────────────────
    print("\n═══ BATCH PUSH ═══")
    batch = [
        QueueEntry(f"b{i}", f"batch_task_{i}", Priority.IDLE, "low", "low", "none")
        for i in range(5)
    ]
    seqs = pq.push_many(batch)
    print(f"  Pushed {len(seqs)} tasks in one call: seqs={seqs}")
    cleared = pq.clear()
    print(f"  Cleared {cleared} entries")

    # ── Final stats ──────────────────────────────────────────────────────────
    print("\n═══ QUEUE STATS ═══")
    for k, v in pq.stats_dict().items():
        print(f"  {k:<30} {v}")

    print(f"\n{pq}")


if __name__ == "__main__":
    _demo()
