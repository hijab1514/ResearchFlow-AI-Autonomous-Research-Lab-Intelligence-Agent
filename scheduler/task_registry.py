"""
scheduler/task_registry.py
==========================
ResearchFlow AI — Task Registry, Resource Profiles & Dependency DAG

This module is the single source of truth for every schedulable task
in ResearchFlow AI. It answers three questions:

  1. WHAT tasks exist?       → TaskDefinition catalogue
  2. HOW much do they cost?  → ResourceProfile per task
  3. IN WHAT ORDER?          → TaskDAG dependency graph

Architecture:
  ┌──────────────────────────────────────────────────────────────────┐
  │                     TASK REGISTRY                                │
  │                                                                  │
  │   TaskDefinition          ResourceProfile     TaskDAG            │
  │   ─────────────           ───────────────     ───────            │
  │   name                    cpu_cost            nodes (tasks)      │
  │   description             ram_cost            edges (deps)       │
  │   priority                gpu_cost            topological sort   │
  │   tags                    estimated_s         critical path      │
  │   retry_policy            parallelisable      cycle detection    │
  │   timeout_s               burst_phase                            │
  │                                                                  │
  │   TaskRegistry                                                   │
  │   ─────────────                                                  │
  │   register() / get()      validate_dag()                         │
  │   build_execution_plan()  get_ready_tasks()                      │
  │   topological_sort()      critical_path_analysis()               │
  └──────────────────────────────────────────────────────────────────┘

OS Concepts demonstrated:
  - Process resource accounting  : ResourceProfile per task
  - DAG-based job scheduling     : dependency graph → topological order
  - Critical path analysis       : longest path = pipeline bottleneck
  - Parallelism detection        : tasks with no mutual deps can run together
  - Admission control metadata   : cost profiles fed to ResourcePolicy
  - Task lifecycle tracking      : PENDING → READY → RUNNING → DONE/FAILED

Pipeline task catalogue (full ResearchFlow AI pipeline):
  ┌──────────────────────────────────┬──────┬────────┬──────┬──────────┐
  │ Task                             │ CPU  │ RAM    │ GPU  │ Priority │
  ├──────────────────────────────────┼──────┼────────┼──────┼──────────┤
  │ query_reformulation              │ low  │ low    │ low  │ HIGH     │
  │ arxiv_retrieval                  │ low  │ low    │ none │ HIGH     │
  │ cache_check                      │ low  │ low    │ none │ HIGH     │
  │ abstract_embedding               │ high │ medium │ med  │ MEDIUM   │
  │ faiss_index_build                │ med  │ medium │ med  │ MEDIUM   │
  │ hdbscan_clustering               │ high │ medium │ none │ MEDIUM   │
  │ umap_reduction                   │ high │ high   │ none │ LOW      │
  │ cluster_labeling                 │ med  │ low    │ high │ MEDIUM   │
  │ gap_analysis_react               │ med  │ medium │ high │ HIGH     │
  │ hypothesis_generation            │ low  │ low    │ med  │ HIGH     │
  │ roadmap_building                 │ low  │ low    │ low  │ LOW      │
  │ report_generation                │ low  │ low    │ none │ LOW      │
  │ system_monitor_tick              │ low  │ low    │ none │ CRITICAL │
  │ alert_drain                      │ low  │ low    │ none │ CRITICAL │
  │ stage_cache_write                │ low  │ low    │ none │ LOW      │
  │ experiment_checkpoint            │ low  │ low    │ none │ LOW      │
  └──────────────────────────────────┴──────┴────────┴──────┴──────────┘

Usage:
    registry = TaskRegistry()

    # Get a task definition
    defn = registry.get("abstract_embedding")
    print(defn.priority, defn.resource.cpu_cost)

    # Build an execution plan in dependency order
    dag  = registry.dag
    plan = dag.topological_sort()   # → ["query_reformulation", "arxiv_retrieval", ...]

    # Find tasks ready to run (all dependencies complete)
    completed = {"query_reformulation", "arxiv_retrieval"}
    ready     = dag.get_ready_tasks(completed)

    # Critical path analysis
    path = dag.critical_path()

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RE-EXPORT: Priority (keep this module self-contained)
# ─────────────────────────────────────────────────────────────────────────────

class Priority(IntEnum):
    CRITICAL = 0
    HIGH     = 1
    MEDIUM   = 2
    LOW      = 3
    IDLE     = 4


PRIORITY_NAMES = {0: "CRITICAL", 1: "HIGH", 2: "MEDIUM", 3: "LOW", 4: "IDLE"}


# ─────────────────────────────────────────────────────────────────────────────
# TASK LIFECYCLE
# ─────────────────────────────────────────────────────────────────────────────

class TaskState(str):
    PENDING  = "pending"    # registered, not yet submitted to scheduler
    READY    = "ready"      # all dependencies satisfied, ready to queue
    QUEUED   = "queued"     # in PriorityQueue, awaiting dispatch
    RUNNING  = "running"    # dispatched to a worker thread
    DONE     = "done"       # completed successfully
    FAILED   = "failed"     # failed after retries exhausted
    SKIPPED  = "skipped"    # skipped (optional task, dependency failed)


# ─────────────────────────────────────────────────────────────────────────────
# RESOURCE PROFILE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ResourceProfile:
    """
    Declares the expected resource demand of a task.

    Fed directly to ResourcePolicy.evaluate() so the scheduler can
    gate dispatch based on live CPU/RAM/GPU telemetry.

    Fields:
      cpu_cost        : "none" | "low" | "medium" | "high"
      ram_cost        : "none" | "low" | "medium" | "high"
      gpu_cost        : "none" | "low" | "medium" | "high"
      estimated_s     : expected wall-clock seconds (0 = unknown)
      parallelisable  : True if this task can run concurrently with others
      burst_phase     : which pipeline phase this dominates
                        ("io_bound" | "cpu_bound" | "gpu_bound" | "llm_bound" | "mixed")

    parallelisable=False tasks are dispatched one at a time even when
    multiple worker slots are available (e.g. UMAP uses all CPU cores).
    """
    cpu_cost:       str   = "medium"
    ram_cost:       str   = "medium"
    gpu_cost:       str   = "none"
    estimated_s:    float = 0.0
    parallelisable: bool  = True
    burst_phase:    str   = "mixed"   # "io_bound"|"cpu_bound"|"gpu_bound"|"llm_bound"|"mixed"

    @property
    def is_gpu_intensive(self) -> bool:
        return self.gpu_cost in ("medium", "high")

    @property
    def is_cpu_intensive(self) -> bool:
        return self.cpu_cost == "high"

    @property
    def is_ram_intensive(self) -> bool:
        return self.ram_cost in ("medium", "high")

    @property
    def cost_signature(self) -> str:
        """Compact string for logging: e.g. 'CPU:high RAM:medium GPU:none'"""
        return f"CPU:{self.cpu_cost} RAM:{self.ram_cost} GPU:{self.gpu_cost}"

    def to_dict(self) -> dict:
        return {
            "cpu_cost":       self.cpu_cost,
            "ram_cost":       self.ram_cost,
            "gpu_cost":       self.gpu_cost,
            "estimated_s":    self.estimated_s,
            "parallelisable": self.parallelisable,
            "burst_phase":    self.burst_phase,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RETRY POLICY (task-level, not pipeline-level)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskRetryPolicy:
    """
    Per-task retry configuration.

    max_retries=2 means up to 3 total attempts (1 initial + 2 retries).
    backoff_s uses exponential backoff: attempt k waits backoff_s * 2^(k-1).
    """
    max_retries:    int   = 2
    backoff_s:      float = 2.0
    timeout_s:      float = 300.0   # max wall-clock seconds per attempt

    @property
    def max_total_time_s(self) -> float:
        """Worst-case total time including retries."""
        backoff_total = sum(self.backoff_s * (2 ** i) for i in range(self.max_retries))
        return self.timeout_s * (self.max_retries + 1) + backoff_total


# ─────────────────────────────────────────────────────────────────────────────
# TASK DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaskDefinition:
    """
    Static definition of a schedulable task.

    TaskDefinition is the equivalent of an OS process descriptor —
    it holds everything the scheduler needs to know about a task
    *before* it is submitted, without containing any executable code.

    The actual callable is injected at submission time by
    TaskRegistry.build_task_instance().

    Fields:
      name            : unique identifier (used as DAG node key)
      description     : human-readable purpose
      priority        : scheduling priority level
      resource        : ResourceProfile (CPU/RAM/GPU cost)
      dependencies    : task names that must complete before this runs
      optional        : if True, failure doesn't block dependents
      tags            : arbitrary labels for filtering / grouping
      retry_policy    : per-task retry configuration
      stage_group     : pipeline phase grouping for Gantt chart colouring
    """
    name:           str
    description:    str
    priority:       Priority          = Priority.MEDIUM
    resource:       ResourceProfile   = field(default_factory=ResourceProfile)
    dependencies:   list[str]         = field(default_factory=list)
    optional:       bool              = False
    tags:           list[str]         = field(default_factory=list)
    retry_policy:   TaskRetryPolicy   = field(default_factory=TaskRetryPolicy)
    stage_group:    str               = "general"

    # Runtime state (mutated by the scheduler, not set at definition time)
    state:          str  = TaskState.PENDING
    run_count:      int  = 0
    last_duration_s: float = 0.0

    @property
    def priority_name(self) -> str:
        return PRIORITY_NAMES.get(int(self.priority), "UNKNOWN")

    @property
    def is_ready(self) -> bool:
        return self.state == TaskState.READY

    @property
    def is_done(self) -> bool:
        return self.state in (TaskState.DONE, TaskState.SKIPPED)

    @property
    def is_failed(self) -> bool:
        return self.state == TaskState.FAILED

    def to_dict(self) -> dict:
        return {
            "name":           self.name,
            "description":    self.description,
            "priority":       self.priority_name,
            "resource":       self.resource.to_dict(),
            "dependencies":   self.dependencies,
            "optional":       self.optional,
            "tags":           self.tags,
            "stage_group":    self.stage_group,
            "state":          self.state,
            "run_count":      self.run_count,
            "last_duration_s": self.last_duration_s,
            "max_total_time_s": self.retry_policy.max_total_time_s,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TASK DAG
# ─────────────────────────────────────────────────────────────────────────────

class TaskDAG:
    """
    Directed Acyclic Graph of task dependencies.

    Nodes: task names (str)
    Edges: (dependency → dependent) — "A must complete before B"

    Provides:
      - Cycle detection (raises ValueError on register if cycle detected)
      - Topological sort (Kahn's algorithm — standard OS job scheduling)
      - get_ready_tasks() — tasks whose dependencies are all satisfied
      - critical_path() — longest path through the DAG (pipeline bottleneck)
      - parallelism_map() — tasks that can run concurrently per level

    The topological sort uses Kahn's algorithm because:
      1. It naturally detects cycles (remaining nodes = cycle members)
      2. It produces a BFS-ordered level-by-level sort (useful for
         identifying which stages can be parallelised)
      3. It runs in O(V + E) — efficient for small pipeline DAGs
    """

    def __init__(self) -> None:
        # adjacency list: node → set of nodes it must precede
        self._graph:     dict[str, set[str]] = defaultdict(set)
        # reverse adjacency: node → set of its dependencies
        self._rev_graph: dict[str, set[str]] = defaultdict(set)
        self._nodes:     set[str] = set()

    def add_node(self, name: str) -> None:
        """Register a task node (idempotent)."""
        self._nodes.add(name)
        if name not in self._graph:
            self._graph[name] = set()
        if name not in self._rev_graph:
            self._rev_graph[name] = set()

    def add_edge(self, from_task: str, to_task: str) -> None:
        """
        Add a dependency edge: from_task must complete before to_task.
        Raises ValueError if this edge would create a cycle.
        """
        self.add_node(from_task)
        self.add_node(to_task)
        self._graph[from_task].add(to_task)
        self._rev_graph[to_task].add(from_task)

        if self._has_cycle():
            # Roll back
            self._graph[from_task].discard(to_task)
            self._rev_graph[to_task].discard(from_task)
            raise ValueError(
                f"Adding edge {from_task} → {to_task} would create a cycle in the TaskDAG"
            )

    def _has_cycle(self) -> bool:
        """DFS-based cycle detection. O(V + E)."""
        WHITE, GRAY, BLACK = 0, 1, 2
        colour: dict[str, int] = {n: WHITE for n in self._nodes}

        def dfs(node: str) -> bool:
            colour[node] = GRAY
            for neighbour in self._graph.get(node, set()):
                if colour[neighbour] == GRAY:
                    return True
                if colour[neighbour] == WHITE and dfs(neighbour):
                    return True
            colour[node] = BLACK
            return False

        return any(colour[n] == WHITE and dfs(n) for n in self._nodes)

    def topological_sort(self) -> list[str]:
        """
        Kahn's algorithm — returns tasks in valid execution order.

        Level-by-level: all tasks at level 0 have no dependencies
        (can start immediately), level 1 tasks depend only on level 0, etc.

        Returns a flat list (not levelled) for simple sequential use.
        Raises ValueError if a cycle is detected (should not happen if
        add_edge() was used consistently, but guards against manual edits).

        This is the standard algorithm used by GNU Make, Bazel, and
        virtually every build / workflow scheduler.
        """
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for dep in self._rev_graph[node]:
                in_degree[node] += 1

        queue: deque[str] = deque(n for n in self._nodes if in_degree[n] == 0)
        sorted_nodes: list[str] = []

        while queue:
            node = queue.popleft()
            sorted_nodes.append(node)
            for successor in sorted(self._graph[node]):   # sorted for determinism
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        if len(sorted_nodes) != len(self._nodes):
            cycle_nodes = self._nodes - set(sorted_nodes)
            raise ValueError(
                f"Cycle detected in TaskDAG involving nodes: {cycle_nodes}"
            )

        return sorted_nodes

    def topological_levels(self) -> list[list[str]]:
        """
        Returns tasks grouped by level (parallelism-aware sort).

        Level 0: tasks with no dependencies (run first)
        Level 1: tasks depending only on level-0 tasks
        Level k: tasks whose earliest dependency is at level k-1

        Tasks within the same level have no mutual dependencies and
        CAN be dispatched concurrently by the SchedulerAgent.
        """
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        for node in self._nodes:
            for dep in self._rev_graph[node]:
                in_degree[node] += 1

        current_level = [n for n in sorted(self._nodes) if in_degree[n] == 0]
        levels: list[list[str]] = []

        while current_level:
            levels.append(sorted(current_level))
            next_level: list[str] = []
            for node in current_level:
                for successor in sorted(self._graph[node]):
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        next_level.append(successor)
            current_level = next_level

        return levels

    def get_ready_tasks(self, completed: set[str]) -> list[str]:
        """
        Returns task names whose all dependencies are in `completed`
        and which are not themselves already completed.

        This is the core scheduling query — called every tick by the
        SchedulerAgent to find what can be dispatched next.
        """
        ready: list[str] = []
        for node in self._nodes:
            if node in completed:
                continue
            deps = self._rev_graph[node]
            if deps.issubset(completed):
                ready.append(node)
        return sorted(ready)   # deterministic order

    def get_dependents(self, task_name: str) -> set[str]:
        """Return all tasks that directly depend on task_name."""
        return set(self._graph.get(task_name, set()))

    def get_dependencies(self, task_name: str) -> set[str]:
        """Return all direct dependencies of task_name."""
        return set(self._rev_graph.get(task_name, set()))

    def get_transitive_dependents(self, task_name: str) -> set[str]:
        """Return all tasks (directly or transitively) depending on task_name."""
        visited: set[str] = set()
        stack = list(self._graph.get(task_name, set()))
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(self._graph.get(node, set()))
        return visited

    def critical_path(
        self,
        durations: dict[str, float] | None = None,
    ) -> tuple[list[str], float]:
        """
        Compute the critical path — the longest path through the DAG.

        The critical path determines the minimum possible pipeline
        completion time. Any task on the critical path is a bottleneck:
        delaying it delays the entire pipeline.

        durations: {task_name: estimated_seconds}
                   Uses 1.0 for all tasks if not provided.

        Returns (path_list, total_seconds).

        Algorithm: dynamic programming on topological order.
        Standard technique in project management (CPM) and OS job
        scheduling for heterogeneous task graphs.
        """
        dur = durations or {n: 1.0 for n in self._nodes}
        topo = self.topological_sort()

        # dist[node] = (longest_distance_to_node, predecessor)
        dist: dict[str, float] = {n: 0.0 for n in self._nodes}
        pred: dict[str, str | None] = {n: None for n in self._nodes}

        for node in topo:
            node_dur = dur.get(node, 1.0)
            for successor in self._graph.get(node, set()):
                candidate = dist[node] + node_dur
                if candidate > dist[successor]:
                    dist[successor] = candidate
                    pred[successor] = node

        # Find the end node with the highest dist
        end_node = max(self._nodes, key=lambda n: dist[n] + dur.get(n, 1.0))
        total = dist[end_node] + dur.get(end_node, 1.0)

        # Reconstruct path
        path: list[str] = []
        node: str | None = end_node
        while node is not None:
            path.append(node)
            node = pred[node]
        path.reverse()

        return path, round(total, 2)

    def parallelism_map(self) -> dict[int, list[str]]:
        """
        Returns {level: [task_names]} where tasks in the same level
        have no mutual dependencies and can run in parallel.
        Identical to topological_levels() but keyed by integer.
        """
        return {i: level for i, level in enumerate(self.topological_levels())}

    def max_parallelism(self) -> int:
        """Maximum number of tasks that could run concurrently."""
        levels = self.topological_levels()
        return max((len(lvl) for lvl in levels), default=1)

    def validate(self) -> list[str]:
        """
        Validate the DAG for common issues.
        Returns a list of warning strings (empty = valid).
        """
        warnings: list[str] = []

        # Check for isolated nodes (no deps, no dependents) — usually a mistake
        for node in self._nodes:
            if not self._rev_graph[node] and not self._graph[node]:
                warnings.append(f"Isolated node (no edges): '{node}'")

        # Check for very long chains (potential bottleneck)
        try:
            topo = self.topological_sort()
            if len(topo) > 0:
                levels = self.topological_levels()
                if len(levels) > 10:
                    warnings.append(
                        f"DAG has {len(levels)} levels — deep dependency chain may limit parallelism"
                    )
        except ValueError as exc:
            warnings.append(f"DAG validation error: {exc}")

        return warnings

    def summary(self) -> dict:
        return {
            "nodes":          len(self._nodes),
            "edges":          sum(len(v) for v in self._graph.values()),
            "levels":         len(self.topological_levels()),
            "max_parallelism": self.max_parallelism(),
            "warnings":       self.validate(),
        }

    def iter_edges(self) -> Iterator[tuple[str, str]]:
        """Yield all (from, to) edges in the DAG."""
        for node, successors in self._graph.items():
            for succ in sorted(successors):
                yield (node, succ)


# ─────────────────────────────────────────────────────────────────────────────
# TASK REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class TaskRegistry:
    """
    Central registry of all TaskDefinitions and the TaskDAG.

    At startup, the registry is pre-populated with all ResearchFlow AI
    pipeline tasks via _register_defaults(). At runtime, the SchedulerAgent
    calls build_execution_plan() to get an ordered list of ready tasks.

    Key methods:
      register()              — add a custom TaskDefinition
      get()                   — retrieve a definition by name
      build_execution_plan()  — topological sort → ordered submission list
      get_ready_tasks()       — tasks ready given completed set
      mark_done() / fail()    — update task state
      execution_status()      — live snapshot for dashboard
    """

    def __init__(self) -> None:
        self._definitions: dict[str, TaskDefinition] = {}
        self._dag = TaskDAG()
        self._register_defaults()
        logger.info(
            "TaskRegistry initialised | %d tasks | DAG: %s",
            len(self._definitions), self._dag.summary(),
        )

    # ── REGISTRATION ─────────────────────────────────────────────────────────

    def register(self, defn: TaskDefinition) -> None:
        """
        Register a TaskDefinition. Idempotent — re-registering with the
        same name overwrites the previous definition.
        Adds the task node and all dependency edges to the DAG.
        """
        self._definitions[defn.name] = defn
        self._dag.add_node(defn.name)
        for dep in defn.dependencies:
            if dep not in self._definitions:
                logger.warning(
                    "Task '%s' depends on '%s' which is not yet registered",
                    defn.name, dep,
                )
            try:
                self._dag.add_edge(dep, defn.name)
            except ValueError as exc:
                logger.error("DAG edge error: %s", exc)

        logger.debug(
            "Registered task: '%s' [%s] deps=%s",
            defn.name, defn.priority_name, defn.dependencies,
        )

    def register_many(self, definitions: list[TaskDefinition]) -> None:
        """Batch register multiple TaskDefinitions."""
        for defn in definitions:
            self.register(defn)

    # ── LOOKUP ───────────────────────────────────────────────────────────────

    def get(self, name: str) -> TaskDefinition:
        """Retrieve a TaskDefinition by name. Raises KeyError if not found."""
        if name not in self._definitions:
            raise KeyError(f"Task '{name}' not found in TaskRegistry")
        return self._definitions[name]

    def get_or_none(self, name: str) -> TaskDefinition | None:
        return self._definitions.get(name)

    def all_names(self) -> list[str]:
        return sorted(self._definitions.keys())

    def by_priority(self, priority: Priority) -> list[TaskDefinition]:
        return [d for d in self._definitions.values() if d.priority == priority]

    def by_tag(self, tag: str) -> list[TaskDefinition]:
        return [d for d in self._definitions.values() if tag in d.tags]

    def by_stage_group(self, group: str) -> list[TaskDefinition]:
        return [d for d in self._definitions.values() if d.stage_group == group]

    # ── DAG ACCESS ───────────────────────────────────────────────────────────

    @property
    def dag(self) -> TaskDAG:
        return self._dag

    # ── EXECUTION PLAN ───────────────────────────────────────────────────────

    def build_execution_plan(self) -> list[TaskDefinition]:
        """
        Returns TaskDefinitions in valid topological execution order.
        This is the list the SchedulerAgent submits to the PriorityQueue
        at the start of a pipeline run.

        Note: The PriorityQueue reorders this list dynamically based on
        Priority + ResourcePolicy. The execution plan is the *initial*
        submission order, not the final dispatch order.
        """
        topo = self._dag.topological_sort()
        plan = [self._definitions[name] for name in topo if name in self._definitions]
        logger.info(
            "Execution plan: %d tasks | order=%s",
            len(plan), [t.name for t in plan],
        )
        return plan

    def build_execution_levels(self) -> list[list[TaskDefinition]]:
        """
        Returns TaskDefinitions grouped by parallelism level.
        Tasks in the same level have no mutual dependencies and can
        be submitted to the scheduler concurrently.
        """
        levels = self._dag.topological_levels()
        return [
            [self._definitions[name] for name in level if name in self._definitions]
            for level in levels
        ]

    def get_ready_tasks(self, completed: set[str]) -> list[TaskDefinition]:
        """
        Returns definitions for tasks whose all dependencies are satisfied.
        Called every scheduler tick to find what can be submitted next.
        """
        ready_names = self._dag.get_ready_tasks(completed)
        return [
            self._definitions[name] for name in ready_names
            if name in self._definitions
            and self._definitions[name].state not in (
                TaskState.DONE, TaskState.FAILED, TaskState.SKIPPED,
                TaskState.QUEUED, TaskState.RUNNING,
            )
        ]

    # ── STATE MANAGEMENT ────────────────────────────────────────────────────

    def mark_queued(self, name: str) -> None:
        if name in self._definitions:
            self._definitions[name].state = TaskState.QUEUED

    def mark_running(self, name: str) -> None:
        if name in self._definitions:
            self._definitions[name].state = TaskState.RUNNING
            self._definitions[name].run_count += 1

    def mark_done(self, name: str, duration_s: float = 0.0) -> None:
        if name in self._definitions:
            self._definitions[name].state = TaskState.DONE
            self._definitions[name].last_duration_s = duration_s

    def mark_failed(self, name: str) -> None:
        if name in self._definitions:
            self._definitions[name].state = TaskState.FAILED
            # Mark optional dependents as SKIPPED
            for dep_name in self._dag.get_transitive_dependents(name):
                dep = self._definitions.get(dep_name)
                if dep and dep.optional:
                    dep.state = TaskState.SKIPPED
                    logger.warning(
                        "Optional task '%s' skipped due to failure of '%s'",
                        dep_name, name,
                    )

    def reset(self) -> None:
        """Reset all task states to PENDING (for a new pipeline run)."""
        for defn in self._definitions.values():
            defn.state    = TaskState.PENDING
            defn.run_count = 0
            defn.last_duration_s = 0.0
        logger.info("TaskRegistry reset: all tasks → PENDING")

    # ── CRITICAL PATH & ANALYSIS ─────────────────────────────────────────────

    def critical_path(self) -> tuple[list[str], float]:
        """
        Returns (critical_path_task_names, estimated_total_seconds).
        Uses estimated_s from each task's ResourceProfile as the duration.
        """
        durations = {
            name: defn.resource.estimated_s
            for name, defn in self._definitions.items()
        }
        return self._dag.critical_path(durations)

    def estimated_total_s(self) -> float:
        """Sum of all task estimated_s values (upper bound, no parallelism)."""
        return sum(d.resource.estimated_s for d in self._definitions.values())

    def parallelism_map(self) -> dict[int, list[str]]:
        return self._dag.parallelism_map()

    # ── EXECUTION STATUS ─────────────────────────────────────────────────────

    def execution_status(self) -> dict:
        """
        Live snapshot of task states — for the Streamlit scheduler panel.
        """
        by_state: dict[str, list[str]] = defaultdict(list)
        for name, defn in self._definitions.items():
            by_state[defn.state].append(name)

        critical_path_names, cp_seconds = self.critical_path()

        return {
            "total_tasks":      len(self._definitions),
            "by_state":         dict(by_state),
            "critical_path":    critical_path_names,
            "critical_path_s":  cp_seconds,
            "dag_summary":      self._dag.summary(),
            "execution_plan":   [d.to_dict() for d in self.build_execution_plan()],
        }

    def task_table(self) -> list[dict]:
        """Serialisable list of all task definitions — for dashboard table."""
        plan = self.build_execution_plan()
        return [defn.to_dict() for defn in plan]

    # ── DEFAULT TASK CATALOGUE ────────────────────────────────────────────────

    def _register_defaults(self) -> None:
        """
        Pre-populate the registry with all ResearchFlow AI pipeline tasks.

        Task definitions are ordered here by natural pipeline flow.
        The DAG enforces correct execution ordering regardless of
        registration order — but listing them in flow order makes
        this file self-documenting.
        """

        # ── OS / System tasks (always CRITICAL) ──────────────────────────────

        self.register(TaskDefinition(
            name="system_monitor_tick",
            description="Poll CPU/RAM/GPU/disk telemetry via psutil (background daemon)",
            priority=Priority.CRITICAL,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="none",
                estimated_s=0.1, parallelisable=True, burst_phase="io_bound",
            ),
            dependencies=[],
            optional=False,
            tags=["system", "monitoring", "daemon"],
            retry_policy=TaskRetryPolicy(max_retries=5, backoff_s=0.5, timeout_s=5.0),
            stage_group="system",
        ))

        self.register(TaskDefinition(
            name="alert_drain",
            description="Drain LabAgent alert queue → route to Streamlit + log",
            priority=Priority.CRITICAL,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="none",
                estimated_s=0.05, parallelisable=True, burst_phase="io_bound",
            ),
            dependencies=[],
            optional=False,
            tags=["system", "alerts", "daemon"],
            retry_policy=TaskRetryPolicy(max_retries=3, backoff_s=0.2, timeout_s=2.0),
            stage_group="system",
        ))

        # ── Stage 1: Query Reformulation ─────────────────────────────────────

        self.register(TaskDefinition(
            name="query_reformulation",
            description="GPT-4o expands raw query into structured arXiv API syntax",
            priority=Priority.HIGH,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="low",
                estimated_s=5.0, parallelisable=True, burst_phase="llm_bound",
            ),
            dependencies=[],
            optional=False,
            tags=["nlp", "llm", "retrieval", "stage1"],
            retry_policy=TaskRetryPolicy(max_retries=3, backoff_s=2.0, timeout_s=30.0),
            stage_group="retrieval",
        ))

        # ── Stage 2: arXiv Retrieval ──────────────────────────────────────────

        self.register(TaskDefinition(
            name="cache_check",
            description="Check local JSON cache for previously retrieved paper sets",
            priority=Priority.HIGH,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="none",
                estimated_s=0.5, parallelisable=True, burst_phase="io_bound",
            ),
            dependencies=["query_reformulation"],
            optional=False,
            tags=["cache", "io", "retrieval", "stage2"],
            retry_policy=TaskRetryPolicy(max_retries=1, backoff_s=0.5, timeout_s=5.0),
            stage_group="retrieval",
        ))

        self.register(TaskDefinition(
            name="arxiv_retrieval",
            description="Fetch top-N papers from arXiv API with pagination and dedup",
            priority=Priority.HIGH,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="none",
                estimated_s=20.0, parallelisable=True, burst_phase="io_bound",
            ),
            dependencies=["cache_check"],
            optional=False,
            tags=["arxiv", "api", "io", "retrieval", "stage2"],
            retry_policy=TaskRetryPolicy(max_retries=3, backoff_s=5.0, timeout_s=120.0),
            stage_group="retrieval",
        ))

        # ── Stage 3: Embedding Generation ─────────────────────────────────────

        self.register(TaskDefinition(
            name="abstract_embedding",
            description="SPECTER/MiniLM dense embeddings for all paper abstracts",
            priority=Priority.MEDIUM,
            resource=ResourceProfile(
                cpu_cost="high", ram_cost="medium", gpu_cost="medium",
                estimated_s=60.0, parallelisable=False, burst_phase="gpu_bound",
            ),
            dependencies=["arxiv_retrieval"],
            optional=False,
            tags=["nlp", "embeddings", "gpu", "stage3"],
            retry_policy=TaskRetryPolicy(max_retries=2, backoff_s=5.0, timeout_s=300.0),
            stage_group="embedding",
        ))

        self.register(TaskDefinition(
            name="faiss_index_build",
            description="Build FAISS index from embeddings for semantic search",
            priority=Priority.MEDIUM,
            resource=ResourceProfile(
                cpu_cost="medium", ram_cost="medium", gpu_cost="medium",
                estimated_s=10.0, parallelisable=False, burst_phase="cpu_bound",
            ),
            dependencies=["abstract_embedding"],
            optional=False,
            tags=["faiss", "vector_store", "gpu", "stage3"],
            retry_policy=TaskRetryPolicy(max_retries=2, backoff_s=3.0, timeout_s=60.0),
            stage_group="embedding",
        ))

        # ── Stage 4: Clustering ───────────────────────────────────────────────

        self.register(TaskDefinition(
            name="hdbscan_clustering",
            description="Density-based clustering of embedding vectors (no fixed k)",
            priority=Priority.MEDIUM,
            resource=ResourceProfile(
                cpu_cost="high", ram_cost="medium", gpu_cost="none",
                estimated_s=10.0, parallelisable=False, burst_phase="cpu_bound",
            ),
            dependencies=["abstract_embedding"],
            optional=False,
            tags=["clustering", "hdbscan", "cpu", "stage4"],
            retry_policy=TaskRetryPolicy(max_retries=2, backoff_s=3.0, timeout_s=120.0),
            stage_group="clustering",
        ))

        self.register(TaskDefinition(
            name="umap_reduction",
            description="UMAP 2D projection of embeddings for interactive scatter plot",
            priority=Priority.LOW,
            resource=ResourceProfile(
                cpu_cost="high", ram_cost="high", gpu_cost="none",
                estimated_s=20.0, parallelisable=False, burst_phase="cpu_bound",
            ),
            dependencies=["abstract_embedding"],
            optional=True,   # visualisation — non-critical for gap analysis
            tags=["umap", "visualisation", "cpu", "ram", "stage4"],
            retry_policy=TaskRetryPolicy(max_retries=1, backoff_s=5.0, timeout_s=180.0),
            stage_group="clustering",
        ))

        # ── Stage 5: Cluster Labeling ─────────────────────────────────────────

        self.register(TaskDefinition(
            name="cluster_labeling",
            description="GPT-4o generates thematic labels from sample abstracts per cluster",
            priority=Priority.MEDIUM,
            resource=ResourceProfile(
                cpu_cost="medium", ram_cost="low", gpu_cost="high",
                estimated_s=30.0, parallelisable=True, burst_phase="llm_bound",
            ),
            dependencies=["hdbscan_clustering"],
            optional=False,
            tags=["nlp", "llm", "clustering", "gpu", "stage5"],
            retry_policy=TaskRetryPolicy(max_retries=3, backoff_s=3.0, timeout_s=120.0),
            stage_group="labeling",
        ))

        # ── Stage 6: Gap Detection ────────────────────────────────────────────

        self.register(TaskDefinition(
            name="gap_analysis_react",
            description="LangChain ReAct agent — identifies research gaps via tool loop",
            priority=Priority.HIGH,
            resource=ResourceProfile(
                cpu_cost="medium", ram_cost="medium", gpu_cost="high",
                estimated_s=45.0, parallelisable=True, burst_phase="llm_bound",
            ),
            dependencies=["cluster_labeling", "faiss_index_build"],
            optional=False,
            tags=["nlp", "llm", "agents", "react", "gpu", "stage6"],
            retry_policy=TaskRetryPolicy(max_retries=2, backoff_s=5.0, timeout_s=300.0),
            stage_group="gap_detection",
        ))

        # ── Stage 7: Hypothesis Generation ───────────────────────────────────

        self.register(TaskDefinition(
            name="hypothesis_generation",
            description="HypothesisAgent: novelty scoring + methodology planning",
            priority=Priority.HIGH,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="medium",
                estimated_s=20.0, parallelisable=True, burst_phase="llm_bound",
            ),
            dependencies=["gap_analysis_react"],
            optional=False,
            tags=["nlp", "llm", "hypotheses", "gpu", "stage7"],
            retry_policy=TaskRetryPolicy(max_retries=2, backoff_s=3.0, timeout_s=120.0),
            stage_group="hypothesis",
        ))

        # ── Stage 8: Roadmap Building ──────────────────────────────────────────

        self.register(TaskDefinition(
            name="roadmap_building",
            description="RoadmapAgent: tier classification, paper selection, annotations",
            priority=Priority.LOW,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="low",
                estimated_s=30.0, parallelisable=True, burst_phase="llm_bound",
            ),
            dependencies=["hypothesis_generation", "umap_reduction"],
            optional=False,
            tags=["nlp", "llm", "roadmap", "stage8"],
            retry_policy=TaskRetryPolicy(max_retries=2, backoff_s=3.0, timeout_s=180.0),
            stage_group="roadmap",
        ))

        # ── Stage 9: Report Generation ─────────────────────────────────────────

        self.register(TaskDefinition(
            name="report_generation",
            description="LangChain doc chain compiles all outputs → Markdown + PDF",
            priority=Priority.LOW,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="none",
                estimated_s=10.0, parallelisable=True, burst_phase="llm_bound",
            ),
            dependencies=["roadmap_building"],
            optional=False,
            tags=["report", "llm", "export", "stage9"],
            retry_policy=TaskRetryPolicy(max_retries=2, backoff_s=2.0, timeout_s=60.0),
            stage_group="report",
        ))

        # ── Utility / Background ───────────────────────────────────────────────

        self.register(TaskDefinition(
            name="stage_cache_write",
            description="Persist completed stage output to disk cache",
            priority=Priority.LOW,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="none",
                estimated_s=1.0, parallelisable=True, burst_phase="io_bound",
            ),
            dependencies=["arxiv_retrieval"],
            optional=True,
            tags=["cache", "io", "utility"],
            retry_policy=TaskRetryPolicy(max_retries=1, backoff_s=1.0, timeout_s=10.0),
            stage_group="utility",
        ))

        self.register(TaskDefinition(
            name="experiment_checkpoint",
            description="Record resource snapshot to LabAgent experiment tracker",
            priority=Priority.LOW,
            resource=ResourceProfile(
                cpu_cost="low", ram_cost="low", gpu_cost="none",
                estimated_s=0.5, parallelisable=True, burst_phase="io_bound",
            ),
            dependencies=[],
            optional=True,
            tags=["monitoring", "checkpoint", "utility"],
            retry_policy=TaskRetryPolicy(max_retries=1, backoff_s=0.5, timeout_s=5.0),
            stage_group="utility",
        ))

        logger.debug(
            "Default tasks registered: %d | DAG levels: %d | max parallelism: %d",
            len(self._definitions),
            len(self._dag.topological_levels()),
            self._dag.max_parallelism(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON (importable by SchedulerAgent)
# ─────────────────────────────────────────────────────────────────────────────

_default_registry: TaskRegistry | None = None


def get_registry() -> TaskRegistry:
    """Return the module-level singleton TaskRegistry (created on first call)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = TaskRegistry()
    return _default_registry


def reset_registry() -> None:
    """Reset the singleton — used between test runs."""
    global _default_registry
    if _default_registry:
        _default_registry.reset()


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """
    Visual walkthrough of the TaskRegistry and DAG analysis.
    Run with: python -m scheduler.task_registry
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    registry = TaskRegistry()

    # ── Task table ───────────────────────────────────────────────────────────
    print("\n═══ TASK CATALOGUE ═══")
    print(f"{'Name':<32} {'Priority':<10} {'CPU':<8} {'RAM':<8} {'GPU':<8} {'Est(s)':<8} {'Para'}")
    print("─" * 84)
    for defn in registry.build_execution_plan():
        r = defn.resource
        print(
            f"{defn.name:<32} {defn.priority_name:<10} "
            f"{r.cpu_cost:<8} {r.ram_cost:<8} {r.gpu_cost:<8} "
            f"{r.estimated_s:<8.1f} {str(r.parallelisable)}"
        )

    # ── DAG summary ───────────────────────────────────────────────────────────
    print("\n═══ DAG SUMMARY ═══")
    for k, v in registry.dag.summary().items():
        print(f"  {k:<20} {v}")

    # ── Execution levels (parallelism map) ────────────────────────────────────
    print("\n═══ PARALLELISM MAP (tasks per level) ═══")
    for level, tasks in registry.parallelism_map().items():
        print(f"  Level {level}: {tasks}")

    # ── Critical path ─────────────────────────────────────────────────────────
    path, total_s = registry.critical_path()
    print(f"\n═══ CRITICAL PATH ({total_s:.1f}s estimated) ═══")
    for i, name in enumerate(path):
        defn = registry.get(name)
        arrow = " →" if i < len(path) - 1 else ""
        print(f"  {name} ({defn.resource.estimated_s:.1f}s){arrow}")

    # ── Simulate a run ────────────────────────────────────────────────────────
    print("\n═══ SIMULATED EXECUTION ═══")
    completed: set[str] = set()
    tick = 0
    while True:
        ready = registry.get_ready_tasks(completed)
        if not ready:
            break
        tick += 1
        print(f"\n  Tick {tick} — ready to dispatch: {[t.name for t in ready]}")
        for t in ready:
            registry.mark_running(t.name)
            registry.mark_done(t.name, t.resource.estimated_s)
            completed.add(t.name)
            print(f"    ✓ {t.name} ({t.resource.estimated_s:.1f}s)")

    print(f"\n  Pipeline complete in {tick} ticks | {len(completed)} tasks done")

    # ── Execution status snapshot ─────────────────────────────────────────────
    print("\n═══ EXECUTION STATUS ═══")
    status = registry.execution_status()
    for k, v in status.items():
        if k != "execution_plan":
            print(f"  {k:<20} {v}")


if __name__ == "__main__":
    _demo()
