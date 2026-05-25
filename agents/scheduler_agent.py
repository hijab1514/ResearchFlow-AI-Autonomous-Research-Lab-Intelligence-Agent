"""
agents/scheduler_agent.py
=========================
ResearchFlow AI — OS-Aware Priority Queue Task Scheduler Agent

This module is the architectural centrepiece of ResearchFlow AI's OS integration.
It implements a genuine feedback-controlled task scheduling system inspired by
classical OS scheduling algorithms (Priority Scheduling, CFS, Rate-Monotonic).

Core design:
  ┌─────────────────────────────────────────────────────────────────────┐
  │                     SCHEDULER FEEDBACK LOOP                        │
  │                                                                     │
  │   psutil / torch.cuda  →  LabAgent.is_safe_to_dispatch()          │
  │           ↓                          ↓                             │
  │   ResourcePolicy.evaluate()  →  PriorityQueue (heapq)             │
  │           ↓                          ↓                             │
  │   DISPATCH / DEFER / THROTTLE  →  Worker ThreadPool               │
  │           ↓                          ↓                             │
  │   GanttLogger.record()        →  Streamlit Gantt Chart            │
  └─────────────────────────────────────────────────────────────────────┘

OS Concepts demonstrated:
  - Priority Scheduling   : heapq with dynamic priority adjustment
  - Priority Aging        : starvation prevention via age-based boost
  - Preemption / Deferral : tasks deferred when resource thresholds breached
  - Concurrency control   : ThreadPoolExecutor with dynamic worker count
  - Producer-consumer     : submission queue → scheduler loop → worker pool
  - Resource-aware policy : CPU/RAM/GPU thresholds gate task dispatch
  - Gantt chart logging   : burst time recording for scheduling analysis

Task resource profiles (mirrors OS process resource accounting):
  Each Task carries a ResourceProfile describing its expected CPU, RAM,
  and GPU cost (low / medium / high). The scheduler checks live telemetry
  against these profiles before dispatch — deferring tasks whose cost
  cannot be safely absorbed by the current system state.

Usage:
    lab   = LabAgent()
    lab.start()

    sched = SchedulerAgent(lab_agent=lab)
    sched.start()

    future = sched.submit(ResearchTask(
        name="embed_papers",
        fn=agent.embed_papers,
        args=(papers,),
        priority=Priority.MEDIUM,
        cpu_cost="high",
        ram_cost="medium",
    ))
    result = await future   # or future.result() from sync code

    sched.stop()

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import os
import threading
import time
import traceback
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Coroutine

from agents.lab_agent import BottleneckType, LabAgent

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# How often the scheduler loop ticks (seconds)
SCHEDULER_TICK_S: float = float(os.getenv("SCHEDULER_TICK_S", "1.0"))

# Default maximum concurrent worker threads
DEFAULT_MAX_WORKERS: int = int(os.getenv("SCHEDULER_MAX_WORKERS", "3"))

# Reduced worker count when CPU is under sustained load
THROTTLED_MAX_WORKERS: int = 1

# How many ticks a CRITICAL-priority task may be deferred before it is
# force-dispatched regardless of resource state (safety valve)
MAX_DEFER_TICKS: int = int(os.getenv("SCHEDULER_MAX_DEFER_TICKS", "30"))

# Priority aging: boost a task's effective priority by 1 level every N ticks
AGING_BOOST_TICKS: int = int(os.getenv("SCHEDULER_AGING_TICKS", "10"))

# Gantt log path
GANTT_LOG_PATH = Path("logs/gantt_timeline.jsonl")
GANTT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class Priority(IntEnum):
    """
    Task priority levels — lower integer = higher priority (heapq is min-heap).
    Mirrors OS nice-value inversion: CRITICAL dispatched first.
    """
    CRITICAL = 0
    HIGH     = 1
    MEDIUM   = 2
    LOW      = 3
    IDLE     = 4


class TaskStatus(str):
    PENDING    = "pending"
    RUNNING    = "running"
    COMPLETED  = "completed"
    FAILED     = "failed"
    DEFERRED   = "deferred"
    CANCELLED  = "cancelled"


class SchedulerState(str):
    IDLE      = "idle"
    RUNNING   = "running"
    THROTTLED = "throttled"
    STOPPING  = "stopping"
    STOPPED   = "stopped"


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE PROFILE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResourceProfile:
    """
    Describes the expected resource cost of a task.
    Used by the ResourcePolicy to decide whether to dispatch or defer.

    Cost levels: "none" | "low" | "medium" | "high"

    Reference table (from task_registry.py):
    ┌──────────────────────────┬─────┬────────┬──────┬──────────┐
    │ Task                     │ CPU │ RAM    │ GPU  │ Priority │
    ├──────────────────────────┼─────┼────────┼──────┼──────────┤
    │ arXiv API retrieval      │ low │ low    │ none │ HIGH     │
    │ Abstract embedding       │ high│ medium │ opt. │ MEDIUM   │
    │ LLM summarization        │ med │ low    │ high │ MEDIUM   │
    │ UMAP reduction           │ high│ high   │ none │ LOW      │
    │ Gap analysis (ReAct)     │ med │ medium │ high │ HIGH     │
    │ Hypothesis generation    │ low │ low    │ med  │ HIGH     │
    │ Report generation        │ low │ low    │ none │ LOW      │
    │ System monitor poll      │ low │ low    │ none │ CRITICAL │
    └──────────────────────────┴─────┴────────┴──────┴──────────┘
    """
    cpu_cost: str = "medium"     # "none" | "low" | "medium" | "high"
    ram_cost: str = "medium"     # "none" | "low" | "medium" | "high"
    gpu_cost: str = "none"       # "none" | "low" | "medium" | "high"
    estimated_duration_s: float = 0.0   # 0 = unknown


# ─────────────────────────────────────────────────────────────────────────────
# TASK
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Task:
    """
    A unit of schedulable work — analogous to an OS process control block (PCB).

    Holds:
      - The callable to execute (sync or async)
      - Its arguments
      - Priority and resource profile
      - Scheduling metadata (arrival time, defer count, age ticks)
      - A Future for the caller to await the result

    The task is placed in the heapq by (effective_priority, arrival_seq).
    arrival_seq breaks ties in FIFO order within the same priority level.
    """
    name: str
    fn: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: Priority = Priority.MEDIUM
    resource: ResourceProfile = field(default_factory=ResourceProfile)

    # Populated by the scheduler on submission
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    arrival_seq: int = 0                 # monotonic counter for FIFO tiebreak
    arrival_time: float = field(default_factory=time.time)
    submit_time_str: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # Scheduling state
    status: str = TaskStatus.PENDING
    defer_count: int = 0                 # how many ticks this task was deferred
    age_ticks: int = 0                   # ticks since last priority boost
    effective_priority: int = field(init=False)

    # Result plumbing — Future is set by scheduler at submission
    future: Future = field(default_factory=Future)

    # Timing (set by scheduler during execution)
    start_time: float | None = None
    end_time: float | None = None
    burst_time_s: float | None = None

    def __post_init__(self) -> None:
        self.effective_priority = int(self.priority)

    def __lt__(self, other: Task) -> bool:
        """heapq ordering: lower effective_priority first; FIFO tiebreak."""
        if self.effective_priority != other.effective_priority:
            return self.effective_priority < other.effective_priority
        return self.arrival_seq < other.arrival_seq

    def age(self) -> None:
        """
        Priority aging: increment age_ticks and boost effective_priority
        every AGING_BOOST_TICKS to prevent starvation of low-priority tasks.
        Mirrors Linux's priority boosting for starved tasks.
        """
        self.age_ticks += 1
        if self.age_ticks >= AGING_BOOST_TICKS and self.effective_priority > int(Priority.CRITICAL):
            self.effective_priority -= 1
            self.age_ticks = 0
            logger.debug(
                "Task '%s' aged: priority boosted to %d",
                self.name, self.effective_priority,
            )

    @property
    def wait_time_s(self) -> float:
        return time.time() - self.arrival_time


# ─────────────────────────────────────────────────────────────────────────────
# GANTT LOGGER
# ─────────────────────────────────────────────────────────────────────────────

class GanttLogger:
    """
    Records task execution events to a JSONL file for Gantt chart rendering
    in the Streamlit dashboard.

    Each record captures: task name, priority, resource profile, start/end
    times, burst time, defer count, status, and a resource snapshot.

    This is the direct implementation of the Gantt chart analysis used in
    OS coursework for visualising CPU burst scheduling (SJF, RR, Priority).

    The Streamlit component (app/components/gantt_chart.py) reads this file
    and renders it as an interactive Plotly timeline.
    """

    def __init__(self, path: Path = GANTT_LOG_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._events: list[dict] = []    # in-memory mirror for fast dashboard reads

    def record(
        self,
        task: Task,
        status: str,
        resource_snapshot: dict | None = None,
        scheduler_state: str = SchedulerState.RUNNING,
        note: str = "",
    ) -> None:
        """Append a Gantt event record."""
        event = {
            "event_id": uuid.uuid4().hex[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": task.task_id,
            "task_name": task.name,
            "priority": task.priority.name,
            "effective_priority": task.effective_priority,
            "cpu_cost": task.resource.cpu_cost,
            "ram_cost": task.resource.ram_cost,
            "gpu_cost": task.resource.gpu_cost,
            "status": status,
            "submit_time": task.submit_time_str,
            "start_time": datetime.utcfromtimestamp(task.start_time).isoformat() if task.start_time else None,
            "end_time": datetime.utcfromtimestamp(task.end_time).isoformat() if task.end_time else None,
            "burst_time_s": task.burst_time_s,
            "wait_time_s": round(task.wait_time_s, 3),
            "defer_count": task.defer_count,
            "scheduler_state": scheduler_state,
            "resource_snapshot": resource_snapshot,
            "note": note,
        }

        with self._lock:
            self._events.append(event)
            try:
                with open(self._path, "a") as f:
                    import json
                    f.write(json.dumps(event) + "\n")
            except Exception as exc:
                logger.warning("GanttLogger write failed: %s", exc)

    def get_events(self, last_n: int | None = None) -> list[dict]:
        """Return in-memory events (thread-safe). Used by Streamlit component."""
        with self._lock:
            events = list(self._events)
        return events[-last_n:] if last_n else events

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
        try:
            self._path.write_text("")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE POLICY
# ─────────────────────────────────────────────────────────────────────────────

class ResourcePolicy:
    """
    Evaluates whether a task should be dispatched, deferred, or force-
    dispatched (safety valve for over-deferred critical tasks).

    Decision matrix:
    ┌─────────────────────┬──────────────────────────────────────────┐
    │ Condition           │ Action                                   │
    ├─────────────────────┼──────────────────────────────────────────┤
    │ defer_count ≥ MAX   │ FORCE DISPATCH (prevent starvation)      │
    │ CRITICAL priority   │ DISPATCH (bypass resource checks)        │
    │ is_safe → False     │ DEFER (requeue, increment defer_count)   │
    │ is_safe → True      │ DISPATCH                                 │
    └─────────────────────┴──────────────────────────────────────────┘
    """

    def __init__(self, lab_agent: LabAgent) -> None:
        self._lab = lab_agent

    def evaluate(self, task: Task) -> tuple[str, str]:
        """
        Returns ("dispatch" | "defer" | "force_dispatch", reason: str).

        "force_dispatch" is returned when a task has been deferred too many
        times — analogous to an OS scheduler's starvation prevention
        mechanism (e.g. Linux's SCHED_DEADLINE watchdog).
        """
        # Safety valve: prevent infinite deferral
        if task.defer_count >= MAX_DEFER_TICKS:
            return (
                "force_dispatch",
                f"Force-dispatching '{task.name}' after {task.defer_count} deferrals "
                f"(starvation prevention)",
            )

        # CRITICAL tasks bypass all resource checks — must always run
        if task.priority == Priority.CRITICAL:
            return "dispatch", f"CRITICAL priority — bypassing resource checks"

        # Consult LabAgent's live telemetry
        safe, reason = self._lab.is_safe_to_dispatch(
            task_cpu_cost=task.resource.cpu_cost,
            task_ram_cost=task.resource.ram_cost,
            task_gpu_cost=task.resource.gpu_cost,
        )

        if safe:
            return "dispatch", reason
        return "defer", reason


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SchedulerStats:
    """Cumulative scheduler performance counters — exposed to dashboard."""
    total_submitted: int = 0
    total_dispatched: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_deferred: int = 0
    total_force_dispatched: int = 0
    total_cancelled: int = 0
    total_throttle_events: int = 0
    avg_wait_time_s: float = 0.0
    avg_burst_time_s: float = 0.0
    peak_queue_depth: int = 0

    # Running totals for avg computation
    _wait_times: list[float] = field(default_factory=list)
    _burst_times: list[float] = field(default_factory=list)

    def record_wait(self, t: float) -> None:
        self._wait_times.append(t)
        self.avg_wait_time_s = sum(self._wait_times) / len(self._wait_times)

    def record_burst(self, t: float) -> None:
        self._burst_times.append(t)
        self.avg_burst_time_s = sum(self._burst_times) / len(self._burst_times)


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER AGENT
# ─────────────────────────────────────────────────────────────────────────────

class SchedulerAgent:
    """
    OS-aware priority queue task scheduler.

    Architecture:
      - A heapq-based priority queue holds pending Task objects
      - A background daemon thread runs the scheduling loop every SCHEDULER_TICK_S
      - A ThreadPoolExecutor executes dispatched tasks concurrently
      - The ResourcePolicy gate (backed by live LabAgent telemetry) decides
        dispatch vs deferral on every tick
      - Priority aging prevents low-priority task starvation
      - GanttLogger records every scheduling event for dashboard rendering

    The scheduler thread (analogous to an OS dispatcher) never blocks on
    task execution — it submits work to the executor and immediately returns
    to polling the queue. This is the non-preemptive cooperative scheduling
    model: tasks run to completion once dispatched.

    Thread safety:
      - The heap is protected by self._heap_lock
      - Task submission is thread-safe (external callers may submit from
        any thread or async context)
      - The scheduler loop is the sole consumer of the heap
    """

    def __init__(
        self,
        lab_agent: LabAgent,
        max_workers: int = DEFAULT_MAX_WORKERS,
        tick_interval_s: float = SCHEDULER_TICK_S,
    ) -> None:
        self._lab = lab_agent
        self._max_workers = max_workers
        self._tick_s = tick_interval_s

        # Priority queue (min-heap)
        self._heap: list[Task] = []
        self._heap_lock = threading.Lock()

        # Monotonic arrival sequence counter (FIFO tiebreak within same priority)
        self._arrival_counter = 0
        self._counter_lock = threading.Lock()

        # Worker thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="researchflow-worker",
        )

        # Currently executing tasks {task_id: Task}
        self._running: dict[str, Task] = {}
        self._running_lock = threading.Lock()

        # Scheduler control
        self._stop_event = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._state: str = SchedulerState.IDLE

        # Sub-components
        self._policy = ResourcePolicy(lab_agent)
        self._gantt = GanttLogger()
        self._stats = SchedulerStats()

        logger.info(
            "SchedulerAgent initialised | max_workers=%d | tick=%.1fs",
            max_workers, tick_interval_s,
        )

    # ── LIFECYCLE ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Launch the background scheduling loop thread."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            logger.warning("SchedulerAgent already running")
            return

        self._stop_event.clear()
        self._state = SchedulerState.RUNNING
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="researchflow-scheduler",
            daemon=True,
        )
        self._scheduler_thread.start()
        logger.info("SchedulerAgent started (thread: %s)", self._scheduler_thread.name)

    def stop(self, wait: bool = True, timeout_s: float = 30.0) -> None:
        """
        Signal the scheduler to stop.
        If wait=True, blocks until the thread exits or timeout is reached.
        """
        logger.info("SchedulerAgent stopping...")
        self._state = SchedulerState.STOPPING
        self._stop_event.set()

        if wait and self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=timeout_s)

        self._executor.shutdown(wait=wait, cancel_futures=False)
        self._state = SchedulerState.STOPPED
        logger.info("SchedulerAgent stopped | stats: %s", self._format_stats())

    @property
    def is_running(self) -> bool:
        return (
            self._scheduler_thread is not None
            and self._scheduler_thread.is_alive()
            and self._state not in (SchedulerState.STOPPING, SchedulerState.STOPPED)
        )

    # ── TASK SUBMISSION ──────────────────────────────────────────────────────

    def submit(self, task: Task) -> Future:
        """
        Submit a task to the priority queue.
        Thread-safe — callable from any thread or async context.

        Returns a Future that will hold the task's return value
        (or exception) when it completes.

        Analogous to an OS process arriving and entering the ready queue.
        """
        with self._counter_lock:
            task.arrival_seq = self._arrival_counter
            self._arrival_counter += 1

        task.status = TaskStatus.PENDING
        task.arrival_time = time.time()
        task.submit_time_str = datetime.utcnow().isoformat()
        task.effective_priority = int(task.priority)

        with self._heap_lock:
            heapq.heappush(self._heap, task)
            depth = len(self._heap)

        self._stats.total_submitted += 1
        if depth > self._stats.peak_queue_depth:
            self._stats.peak_queue_depth = depth

        self._gantt.record(task, TaskStatus.PENDING, note=f"Queued | depth={depth}")
        logger.info(
            "Task submitted: '%s' [%s] | priority=%s | queue_depth=%d",
            task.name, task.task_id, task.priority.name, depth,
        )

        return task.future

    def submit_coroutine(
        self,
        name: str,
        coro_fn: Callable[..., Coroutine],
        args: tuple = (),
        kwargs: dict | None = None,
        priority: Priority = Priority.MEDIUM,
        resource: ResourceProfile | None = None,
    ) -> Future:
        """
        Convenience wrapper: submit an async coroutine function as a task.
        The scheduler runs it in a new event loop on the worker thread.
        """
        def _run_coro(*a, **kw) -> Any:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro_fn(*a, **kw))
            finally:
                loop.close()

        task = Task(
            name=name,
            fn=_run_coro,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            resource=resource or ResourceProfile(),
        )
        return self.submit(task)

    def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending task by ID (cannot cancel already-running tasks).
        Returns True if the task was found and cancelled.
        """
        with self._heap_lock:
            for i, task in enumerate(self._heap):
                if task.task_id == task_id:
                    task.status = TaskStatus.CANCELLED
                    task.future.cancel()
                    self._heap.pop(i)
                    heapq.heapify(self._heap)
                    self._stats.total_cancelled += 1
                    self._gantt.record(task, TaskStatus.CANCELLED)
                    logger.info("Task cancelled: '%s' [%s]", task.name, task_id)
                    return True
        return False

    # ── SCHEDULER LOOP (background thread) ───────────────────────────────────

    def _scheduler_loop(self) -> None:
        """
        Main scheduling loop — ticks every SCHEDULER_TICK_S.

        Each tick:
          1. Age all pending tasks (starvation prevention)
          2. Check live resource state → adjust concurrency (throttle/unthrottle)
          3. Peek at the highest-priority pending task
          4. Evaluate ResourcePolicy (dispatch / defer / force_dispatch)
          5. Dispatch or requeue accordingly
          6. Log to Gantt

        This loop is the 'OS dispatcher' in our system — it runs continuously,
        makes fast scheduling decisions, and never blocks on task execution.
        """
        logger.debug("Scheduler loop started")

        while not self._stop_event.is_set():
            tick_start = time.time()

            try:
                self._tick()
            except Exception as exc:
                logger.error("Scheduler tick error: %s\n%s", exc, traceback.format_exc())

            # Sleep for remainder of tick interval
            elapsed = time.time() - tick_start
            sleep_s = max(0.0, self._tick_s - elapsed)
            self._stop_event.wait(timeout=sleep_s)

        logger.debug("Scheduler loop exited")

    def _tick(self) -> None:
        """Execute one scheduling tick."""

        # ── Step 1: Adjust concurrency based on resource state ─────────────
        self._adjust_concurrency()

        # ── Step 2: Age all pending tasks (starvation prevention) ──────────
        with self._heap_lock:
            for task in self._heap:
                if task.status == TaskStatus.PENDING:
                    task.age()
            # Re-heapify after priority mutations
            heapq.heapify(self._heap)

        # ── Step 3: Determine available worker slots ────────────────────────
        with self._running_lock:
            running_count = len(self._running)

        current_max = (
            THROTTLED_MAX_WORKERS
            if self._state == SchedulerState.THROTTLED
            else self._max_workers
        )
        available_slots = current_max - running_count

        if available_slots <= 0:
            logger.debug("No worker slots available (%d/%d busy)", running_count, current_max)
            return

        # ── Step 4: Dispatch up to available_slots tasks ────────────────────
        dispatched_this_tick = 0

        while dispatched_this_tick < available_slots:
            task = self._pop_next_pending()
            if task is None:
                break   # queue empty

            decision, reason = self._policy.evaluate(task)

            if decision in ("dispatch", "force_dispatch"):
                if decision == "force_dispatch":
                    self._stats.total_force_dispatched += 1
                    logger.warning("Force-dispatching '%s': %s", task.name, reason)
                    self._gantt.record(
                        task, TaskStatus.PENDING,
                        note=f"FORCE_DISPATCH after {task.defer_count} deferrals",
                    )
                self._dispatch(task, reason)
                dispatched_this_tick += 1

            else:  # defer
                task.defer_count += 1
                task.status = TaskStatus.DEFERRED
                self._stats.total_deferred += 1

                # Requeue with effective_priority preserved (aging already applied)
                with self._heap_lock:
                    heapq.heappush(self._heap, task)

                self._gantt.record(
                    task, TaskStatus.DEFERRED,
                    resource_snapshot=self._resource_snapshot(),
                    note=f"Deferred #{task.defer_count}: {reason}",
                )
                logger.debug(
                    "Task deferred (%d): '%s' — %s",
                    task.defer_count, task.name, reason,
                )
                break   # if the top task was deferred, lower ones likely will be too

    def _adjust_concurrency(self) -> None:
        """
        Dynamically adjusts the scheduler state between RUNNING (full
        concurrency) and THROTTLED (single-worker) based on live telemetry.

        Mirrors the Linux CFS concept of load-based thread count management.
        """
        snap = self._lab.snapshot()
        if snap is None:
            return

        cpu = snap.cpu.overall_pct
        bottleneck = snap.bottleneck

        should_throttle = (
            cpu > 85.0
            or bottleneck == BottleneckType.OOM_RISK
            or bottleneck == BottleneckType.CPU_BOUND
        )

        if should_throttle and self._state == SchedulerState.RUNNING:
            self._state = SchedulerState.THROTTLED
            self._stats.total_throttle_events += 1
            logger.warning(
                "THROTTLING: concurrency reduced %d→%d | CPU=%.1f%% | bottleneck=%s",
                self._max_workers, THROTTLED_MAX_WORKERS, cpu, bottleneck.value,
            )

        elif not should_throttle and self._state == SchedulerState.THROTTLED:
            self._state = SchedulerState.RUNNING
            logger.info(
                "UNTHROTTLING: concurrency restored to %d | CPU=%.1f%%",
                self._max_workers, cpu,
            )

    def _pop_next_pending(self) -> Task | None:
        """
        Pop the highest-priority pending task from the heap.
        Skips CANCELLED tasks silently.
        Returns None if queue is empty.
        """
        with self._heap_lock:
            while self._heap:
                task = heapq.heappop(self._heap)
                if task.status == TaskStatus.CANCELLED:
                    continue   # silently discard cancelled tasks
                return task
        return None

    # ── TASK DISPATCH ────────────────────────────────────────────────────────

    def _dispatch(self, task: Task, reason: str) -> None:
        """
        Dispatch a task to the ThreadPoolExecutor.

        Records arrival→dispatch wait time for scheduling analysis
        (equivalent to OS scheduling latency measurement).
        """
        task.status = TaskStatus.RUNNING
        task.start_time = time.time()
        wait_s = task.start_time - task.arrival_time
        self._stats.record_wait(wait_s)
        self._stats.total_dispatched += 1

        with self._running_lock:
            self._running[task.task_id] = task

        self._gantt.record(
            task, TaskStatus.RUNNING,
            resource_snapshot=self._resource_snapshot(),
            scheduler_state=self._state,
            note=f"Dispatched | wait={wait_s:.2f}s | {reason}",
        )

        logger.info(
            "DISPATCH: '%s' [%s] | priority=%s | wait=%.2fs | workers=%d/%d",
            task.name, task.task_id, task.priority.name,
            wait_s, len(self._running), self._max_workers,
        )

        # Submit to thread pool — non-blocking
        exec_future = self._executor.submit(self._execute_task, task)
        exec_future.add_done_callback(lambda f: self._on_executor_done(f, task))

    def _execute_task(self, task: Task) -> Any:
        """
        Runs the task's callable in a worker thread.
        Wraps execution with timing and exception handling.
        """
        try:
            result = task.fn(*task.args, **task.kwargs)
            return result
        except Exception as exc:
            logger.error("Task '%s' raised: %s\n%s", task.name, exc, traceback.format_exc())
            raise

    def _on_executor_done(self, exec_future: Any, task: Task) -> None:
        """
        Callback fired when the executor future completes (success or failure).
        Updates task state, records burst time, sets the caller's Future.
        """
        task.end_time = time.time()
        task.burst_time_s = round(task.end_time - (task.start_time or task.end_time), 3)
        self._stats.record_burst(task.burst_time_s)

        with self._running_lock:
            self._running.pop(task.task_id, None)

        exc = exec_future.exception()

        if exc is None:
            task.status = TaskStatus.COMPLETED
            self._stats.total_completed += 1
            result = exec_future.result()
            task.future.set_result(result)
            self._gantt.record(
                task, TaskStatus.COMPLETED,
                resource_snapshot=self._resource_snapshot(),
                note=f"burst={task.burst_time_s}s",
            )
            logger.info(
                "COMPLETED: '%s' [%s] | burst=%.2fs | total_done=%d",
                task.name, task.task_id, task.burst_time_s, self._stats.total_completed,
            )
        else:
            task.status = TaskStatus.FAILED
            self._stats.total_failed += 1
            task.future.set_exception(exc)
            self._gantt.record(
                task, TaskStatus.FAILED,
                note=f"Exception: {type(exc).__name__}: {exc}",
            )
            logger.error(
                "FAILED: '%s' [%s] | burst=%.2fs | error=%s",
                task.name, task.task_id, task.burst_time_s, exc,
            )

        # Notify LabAgent experiment tracker about bottleneck if one is active
        snap = self._lab.snapshot()
        if snap and snap.bottleneck != BottleneckType.NONE:
            self._lab._tracker.record_bottleneck(snap.bottleneck, snap.bottleneck_detail)

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _resource_snapshot(self) -> dict | None:
        """Return a compact resource dict for Gantt log embedding."""
        snap = self._lab.snapshot()
        if snap is None:
            return None
        return {
            "cpu_pct":   snap.cpu.overall_pct,
            "ram_pct":   snap.memory.used_pct,
            "gpu_pct":   snap.gpu.vram_used_pct if snap.gpu.available else None,
            "bottleneck": snap.bottleneck.value,
        }

    def _format_stats(self) -> str:
        s = self._stats
        return (
            f"submitted={s.total_submitted} dispatched={s.total_dispatched} "
            f"completed={s.total_completed} failed={s.total_failed} "
            f"deferred={s.total_deferred} force={s.total_force_dispatched} "
            f"throttle_events={s.total_throttle_events} "
            f"avg_wait={s.avg_wait_time_s:.2f}s avg_burst={s.avg_burst_time_s:.2f}s"
        )

    # ── PUBLIC STATUS API ────────────────────────────────────────────────────

    def status(self) -> dict:
        """
        Returns a complete scheduler status dict — consumed by the
        Streamlit dashboard's live scheduler panel.
        """
        with self._heap_lock:
            queue_snapshot = [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "priority": t.priority.name,
                    "effective_priority": t.effective_priority,
                    "status": t.status,
                    "defer_count": t.defer_count,
                    "wait_s": round(t.wait_time_s, 2),
                    "cpu_cost": t.resource.cpu_cost,
                    "ram_cost": t.resource.ram_cost,
                    "gpu_cost": t.resource.gpu_cost,
                }
                for t in sorted(self._heap)
            ]

        with self._running_lock:
            running_snapshot = [
                {
                    "task_id": t.task_id,
                    "name": t.name,
                    "priority": t.priority.name,
                    "running_for_s": round(time.time() - (t.start_time or time.time()), 2),
                }
                for t in self._running.values()
            ]

        s = self._stats
        return {
            "scheduler_state": self._state,
            "is_running": self.is_running,
            "queue_depth": len(self._heap),
            "running_count": len(self._running),
            "max_workers": (
                THROTTLED_MAX_WORKERS
                if self._state == SchedulerState.THROTTLED
                else self._max_workers
            ),
            "queue": queue_snapshot,
            "running": running_snapshot,
            "stats": {
                "total_submitted":     s.total_submitted,
                "total_dispatched":    s.total_dispatched,
                "total_completed":     s.total_completed,
                "total_failed":        s.total_failed,
                "total_deferred":      s.total_deferred,
                "total_force_dispatched": s.total_force_dispatched,
                "total_cancelled":     s.total_cancelled,
                "total_throttle_events": s.total_throttle_events,
                "avg_wait_time_s":     round(s.avg_wait_time_s, 3),
                "avg_burst_time_s":    round(s.avg_burst_time_s, 3),
                "peak_queue_depth":    s.peak_queue_depth,
            },
        }

    def gantt_events(self, last_n: int | None = None) -> list[dict]:
        """Return Gantt log events for the Streamlit Gantt chart component."""
        return self._gantt.get_events(last_n)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE TASK FACTORY
# ─────────────────────────────────────────────────────────────────────────────

class PipelineTasks:
    """
    Factory that constructs the standard ResearchFlow pipeline tasks
    with their pre-assigned ResourceProfiles and Priorities.

    Keeps task definitions co-located and makes the scheduler integration
    trivially readable in research_pipeline.py:

        sched.submit(PipelineTasks.arxiv_retrieval(agent.retrieve_papers, (queries,)))
        sched.submit(PipelineTasks.embedding(agent.embed_papers, (papers,)))
        ...
    """

    @staticmethod
    def arxiv_retrieval(fn: Callable, args: tuple = ()) -> Task:
        return Task(
            name="arxiv_retrieval",
            fn=fn, args=args,
            priority=Priority.HIGH,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="none",
                estimated_duration_s=15.0,
            ),
        )

    @staticmethod
    def query_reformulation(fn: Callable, args: tuple = ()) -> Task:
        return Task(
            name="query_reformulation",
            fn=fn, args=args,
            priority=Priority.HIGH,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="low",
                estimated_duration_s=5.0,
            ),
        )

    @staticmethod
    def embedding(fn: Callable, args: tuple = ()) -> Task:
        return Task(
            name="abstract_embedding",
            fn=fn, args=args,
            priority=Priority.MEDIUM,
            resource=ResourceProfile(
                cpu_cost="high", ram_cost="medium", gpu_cost="medium",
                estimated_duration_s=60.0,
            ),
        )

    @staticmethod
    def clustering(fn: Callable, args: tuple = ()) -> Task:
        return Task(
            name="hdbscan_clustering",
            fn=fn, args=args,
            priority=Priority.MEDIUM,
            resource=ResourceProfile(
                cpu_cost="high", ram_cost="medium", gpu_cost="none",
                estimated_duration_s=10.0,
            ),
        )

    @staticmethod
    def umap_reduction(fn: Callable, args: tuple = ()) -> Task:
        return Task(
            name="umap_reduction",
            fn=fn, args=args,
            priority=Priority.LOW,
            resource=ResourceProfile(
                cpu_cost="high", ram_cost="high", gpu_cost="none",
                estimated_duration_s=20.0,
            ),
        )

    @staticmethod
    def cluster_labeling(fn: Callable, args: tuple = ()) -> Task:
        return Task(
            name="cluster_labeling",
            fn=fn, args=args,
            priority=Priority.MEDIUM,
            resource=ResourceProfile(
                cpu_cost="medium", ram_cost="low", gpu_cost="high",
                estimated_duration_s=30.0,
            ),
        )

    @staticmethod
    def gap_analysis(fn: Callable, args: tuple = ()) -> Task:
        return Task(
            name="gap_analysis_react",
            fn=fn, args=args,
            priority=Priority.HIGH,
            resource=ResourceProfile(
                cpu_cost="medium", ram_cost="medium", gpu_cost="high",
                estimated_duration_s=45.0,
            ),
        )

    @staticmethod
    def hypothesis_generation(fn: Callable, args: tuple = ()) -> Task:
        return Task(
            name="hypothesis_generation",
            fn=fn, args=args,
            priority=Priority.HIGH,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="medium",
                estimated_duration_s=15.0,
            ),
        )

    @staticmethod
    def report_generation(fn: Callable, args: tuple = ()) -> Task:
        return Task(
            name="report_generation",
            fn=fn, args=args,
            priority=Priority.LOW,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="none",
                estimated_duration_s=10.0,
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Smoke-test: demonstrates the full scheduling feedback loop.
    Submits 8 tasks with mixed priorities and resource profiles,
    watches the scheduler dispatch/defer/throttle in real time,
    then prints the Gantt log and final stats.

    Run with: python -m agents.scheduler_agent
    """
    import json
    import random

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    # ── Boot the stack ───────────────────────────────────────────────────────
    lab = LabAgent()
    lab.start()

    time.sleep(2)   # let monitor warm up

    sched = SchedulerAgent(lab_agent=lab, max_workers=3, tick_interval_s=1.0)
    sched.start()

    # ── Define dummy work functions ──────────────────────────────────────────
    def fake_embed(n: int) -> str:
        time.sleep(random.uniform(1.5, 3.0))
        return f"embedded {n} abstracts"

    def fake_cluster() -> str:
        time.sleep(random.uniform(0.5, 1.5))
        return "clustered OK"

    def fake_llm_call(step: str) -> str:
        time.sleep(random.uniform(1.0, 2.0))
        return f"LLM done: {step}"

    def fake_report() -> str:
        time.sleep(0.5)
        return "report generated"

    # ── Submit pipeline tasks ────────────────────────────────────────────────
    futures = []

    futures.append(sched.submit(PipelineTasks.query_reformulation(fake_llm_call, ("query_reform",))))
    futures.append(sched.submit(PipelineTasks.arxiv_retrieval(fake_cluster)))
    futures.append(sched.submit(PipelineTasks.embedding(fake_embed, (80,))))
    futures.append(sched.submit(PipelineTasks.clustering(fake_cluster)))
    futures.append(sched.submit(PipelineTasks.umap_reduction(fake_cluster)))
    futures.append(sched.submit(PipelineTasks.cluster_labeling(fake_llm_call, ("cluster_label",))))
    futures.append(sched.submit(PipelineTasks.gap_analysis(fake_llm_call, ("gap_analysis",))))
    futures.append(sched.submit(PipelineTasks.report_generation(fake_report)))

    print(f"\nSubmitted {len(futures)} tasks — scheduler running...\n")

    # ── Wait for all tasks ───────────────────────────────────────────────────
    for f in futures:
        try:
            result = f.result(timeout=60)
            print(f"  ✓ {result}")
        except Exception as exc:
            print(f"  ✗ Task failed: {exc}")

    # ── Print final status ───────────────────────────────────────────────────
    status = sched.status()
    print("\n── SCHEDULER STATS ──")
    for k, v in status["stats"].items():
        print(f"  {k:<28} {v}")

    # ── Print Gantt events ───────────────────────────────────────────────────
    events = sched.gantt_events()
    print(f"\n── GANTT LOG ({len(events)} events) ──")
    for ev in events:
        start = ev.get("start_time", "")[:19] if ev.get("start_time") else "         —         "
        burst = f"{ev['burst_time_s']:.2f}s" if ev.get("burst_time_s") else "  —   "
        print(
            f"  [{ev['status']:<10}] {ev['task_name']:<28} "
            f"pri={ev['priority']:<8} start={start} burst={burst} "
            f"defer={ev['defer_count']} | {ev.get('note','')[:60]}"
        )

    sched.stop()
    lab.stop()
    print("\nSchedulerAgent demo complete.")


if __name__ == "__main__":
    _demo()
