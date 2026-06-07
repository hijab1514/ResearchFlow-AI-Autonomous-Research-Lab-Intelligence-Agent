"""
app/backend.py
==============
Shared backend layer for the Streamlit app. ONE place that:
  - imports the real pipeline / agents (with graceful demo fallback)
  - owns the lab + scheduler lifecycle (session_state cached)
  - runs the pipeline and ALWAYS stores the full PipelineResult so every
    page (home, research, hypotheses, history) sees the same state
  - reads live telemetry, scheduler stats, and the real Gantt event log

main.py and pages 01/02 import from here. This removes the bootstrap that was
duplicated across files, and the asyncio _event_queue workaround now lives in
exactly one function (_run_pipeline_blocking) — fix stream() once, change here once.
"""

from __future__ import annotations

import sys
import time
import asyncio
from pathlib import Path
from datetime import datetime, timedelta

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

# Optional deps
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except Exception:
    HAS_TORCH = False

try:
    import yaml
    HAS_YAML = True
except Exception:
    HAS_YAML = False

# Real backend
BACKEND_AVAILABLE = False
BACKEND_ERROR = ""
try:
    from pipelines.research_pipeline import ResearchPipeline, PipelineResult  # noqa: F401
    from pipelines.pipeline_base import PipelineConfig
    from agents.lab_agent import LabAgent
    from agents.scheduler_agent import SchedulerAgent
    BACKEND_AVAILABLE = True
except Exception as e:
    BACKEND_ERROR = f"{type(e).__name__}: {e}"

REPORTS_DIR = REPO_ROOT / "reports"

DEFAULT_CONFIG = {
    "scheduler": {"cpu_threshold": 85, "ram_threshold": 20, "gpu_threshold": 90},
    "arxiv": {"max_results": 100},
}


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
def get_config() -> dict:
    if "config" in st.session_state:
        return st.session_state["config"]
    cfg = DEFAULT_CONFIG
    cfg_path = REPO_ROOT / "configs" / "scheduler_config.yaml"
    if HAS_YAML and cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = {**DEFAULT_CONFIG, **(yaml.safe_load(f) or {})}
        except Exception:
            pass
    st.session_state["config"] = cfg
    return cfg


# --------------------------------------------------------------------------- #
# LAB + SCHEDULER LIFECYCLE (session_state cached — no duplicate threads)
# --------------------------------------------------------------------------- #
def get_lab_and_scheduler():
    if not BACKEND_AVAILABLE:
        return None, None
    ss = st.session_state
    if ss.get("_lab") is None:
        lab = LabAgent()
        lab.start()
        ss["_lab"] = lab
    if ss.get("_scheduler") is None:
        sched = SchedulerAgent(lab_agent=ss["_lab"], max_workers=2)
        sched.start()
        ss["_scheduler"] = sched
    return ss["_lab"], ss["_scheduler"]


def _build_pipeline(max_papers: int):
    """Fresh pipeline per run — its asyncio.Queue can't cross event loops."""
    lab, sched = get_lab_and_scheduler()
    config = PipelineConfig(
        llm_model="gpt-4o", max_papers=max_papers,
        hypotheses_per_gap=1, max_hypotheses=3,
        roadmap_target_papers=10, roadmap_weeks=4,
        use_stage_cache=True, verbose=False,
    )
    return ResearchPipeline(lab_agent=lab, scheduler=sched, config=config)


def _run_pipeline_blocking(pipeline, query, on_event=None):
    """
    Drive the async pipeline from sync Streamlit. We run pipeline.run() and drain
    its event queue for progress, because stream() yields events but never returns
    the PipelineResult (see research_pipeline.stream). The _event_queue access is
    the single workaround; fix stream() to surface the result and update here only.
    """
    async def _drive():
        task = asyncio.create_task(pipeline.run(query))
        q = pipeline._event_queue
        while not task.done():
            try:
                ev = await asyncio.wait_for(q.get(), timeout=0.3)
                if on_event:
                    on_event(ev)
            except asyncio.TimeoutError:
                pass
        while not q.empty():
            ev = q.get_nowait()
            if on_event:
                on_event(ev)
        return task.result()
    return asyncio.run(_drive())


# --------------------------------------------------------------------------- #
# RUN — single entry point; always stores full state
# --------------------------------------------------------------------------- #
def run_research(topic: str, max_papers: int, force_demo: bool = False,
                 on_event=None) -> dict:
    """
    Run the pipeline (or demo) and store EVERYTHING in session_state so every
    page agrees:
        raw_result   -> the PipelineResult (None in demo)
        raw_is_demo  -> bool
        last_result  -> normalized summary dict (home view)
        history      -> appended normalized dict
    Returns the normalized dict.
    """
    ss = st.session_state
    t0 = time.time()

    if BACKEND_AVAILABLE and not force_demo:
        pipeline = _build_pipeline(max_papers)
        result = _run_pipeline_blocking(pipeline, topic, on_event=on_event)
        ss["raw_result"] = result
        ss["raw_is_demo"] = False
        norm = normalize(result)
        if not norm.get("elapsed_s"):
            norm["elapsed_s"] = round(time.time() - t0, 2)
    else:
        time.sleep(0.4)
        ss["raw_result"] = None
        ss["raw_is_demo"] = True
        norm = _demo_normalized(topic, max_papers, t0)

    ss["last_result"] = norm
    ss.setdefault("history", []).append(norm)
    return norm


def normalize(result) -> dict:
    clusters = [{
        "label": getattr(c, "label", f"Cluster {getattr(c, 'cluster_id', '?')}"),
        "size": getattr(c, "paper_count", 0),
        "reproducibility": getattr(c, "reproducibility_score", 0.0),
    } for c in (getattr(result, "clusters", None) or [])]
    gaps = [{
        "title": getattr(g, "title", ""),
        "confidence": getattr(g, "confidence", 0.0),
        "evidence": getattr(g, "description", ""),
    } for g in (getattr(result, "gaps", None) or [])]
    return {
        "topic": getattr(result, "query", ""),
        "n_papers": len(getattr(result, "papers", None) or []),
        "clusters": clusters, "gaps": gaps,
        "elapsed_s": getattr(result, "total_duration_s", 0.0),
        "status": getattr(result, "status", "completed"),
        "error": getattr(result, "error_message", ""),
        "report_md": getattr(result, "report_md", ""),
        "run_id": getattr(result, "run_id", ""),
        "started_at": getattr(result, "started_at", ""),
        "demo": False,
    }


def _demo_normalized(topic: str, max_papers: int, t0: float) -> dict:
    clusters = [
        {"label": "Privacy-preserving aggregation", "size": 18, "reproducibility": 0.62},
        {"label": "Communication-efficient updates", "size": 14, "reproducibility": 0.71},
        {"label": "Heterogeneous / non-IID data", "size": 11, "reproducibility": 0.40},
        {"label": "On-device personalization", "size": 7, "reproducibility": 0.55},
    ]
    gaps = [
        {"title": f"Robustness of {topic or 'the topic'} under adversarial clients",
         "confidence": 0.78,
         "evidence": "Only 2 of 50 papers address Byzantine clients; recency skewed pre-2022."},
        {"title": "Reproducibility in on-device personalization", "confidence": 0.66,
         "evidence": "Lowest open-code score (0.40) of all clusters."},
    ]
    return {"topic": topic, "n_papers": min(max_papers, 50), "clusters": clusters,
            "gaps": gaps, "elapsed_s": round(time.time() - t0, 2),
            "status": "completed", "error": "", "report_md": "",
            "run_id": "", "started_at": "", "demo": True}


# --------------------------------------------------------------------------- #
# TELEMETRY (for sidebar + lab dashboard)
# --------------------------------------------------------------------------- #
def get_system_stats() -> dict:
    """LabAgent.snapshot() first (real), psutil/torch fallback."""
    lab = st.session_state.get("_lab")
    if lab is not None:
        try:
            snap = lab.snapshot()
            if snap is not None:
                gpu = getattr(snap, "gpu", None)
                return {
                    "cpu": getattr(getattr(snap, "cpu", None), "overall_pct", None),
                    "ram": getattr(getattr(snap, "memory", None), "used_pct", None),
                    "gpu": (getattr(gpu, "vram_used_pct", None)
                            if gpu is not None and getattr(gpu, "available", False) else None),
                    "bottleneck": getattr(getattr(snap, "bottleneck", None), "value", None),
                    "source": "LabAgent",
                }
        except Exception:
            pass
    out = {"cpu": None, "ram": None, "gpu": None, "bottleneck": None, "source": "psutil"}
    if HAS_PSUTIL:
        out["cpu"] = psutil.cpu_percent(interval=None)
        out["ram"] = psutil.virtual_memory().percent
    if HAS_TORCH:
        try:
            free, total = torch.cuda.mem_get_info()
            out["gpu"] = round((1 - free / total) * 100, 1)
        except Exception:
            pass
    return out


def read_scheduler_stats() -> dict:
    sched = st.session_state.get("_scheduler")
    if sched is None:
        return {}
    try:
        status = sched.status()
        return status.get("stats", {}) if isinstance(status, dict) else {}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# GANTT — now wired to the REAL scheduler.gantt_events()
# --------------------------------------------------------------------------- #
def _to_dt(iso):
    if not iso:
        return None
    try:
        return datetime.fromisoformat(str(iso))
    except Exception:
        return None


def _status_rank(s: str) -> int:
    return {"completed": 3, "failed": 3, "running": 2, "deferred": 1}.get(s, 0)


def read_gantt():
    """
    Returns (bars, is_real). Each bar: {task, start(datetime), end(datetime),
    status, burst}. Uses scheduler.gantt_events(); keeps the terminal record per
    task_id (completed/failed > running). Falls back to a synthetic timeline.
    """
    sched = st.session_state.get("_scheduler")
    if sched is None:
        return _demo_gantt(), False
    try:
        events = sched.gantt_events()
    except Exception:
        return _demo_gantt(), False

    by_id: dict = {}
    for ev in events or []:
        if not ev.get("start_time"):
            continue
        tid = ev.get("task_id")
        cur = by_id.get(tid)
        if cur is None or _status_rank(ev.get("status", "")) >= _status_rank(cur.get("status", "")):
            by_id[tid] = ev

    bars = []
    for ev in by_id.values():
        start = _to_dt(ev.get("start_time"))
        if start is None:
            continue
        end = _to_dt(ev.get("end_time")) or datetime.utcnow()
        bars.append({"task": ev.get("task_name", "task"), "start": start, "end": end,
                     "status": ev.get("status", "completed"), "burst": ev.get("burst_time_s")})
    if not bars:
        return _demo_gantt(), False
    return bars, True


def _demo_gantt():
    base = datetime(2024, 1, 1)
    stages = [
        ("query_reformulation", 5, "completed"),
        ("arxiv_retrieval", 20, "completed"),
        ("embedding_generation", 60, "completed"),
        ("clustering", 3, "deferred"),
        ("clustering", 10, "completed"),
        ("cluster_labeling", 30, "completed"),
        ("gap_analysis_react", 45, "completed"),
        ("hypothesis_generation", 20, "completed"),
        ("report_generation", 10, "completed"),
    ]
    bars, t = [], 0.0
    for name, dur, status in stages:
        bars.append({"task": name, "start": base + timedelta(seconds=t),
                     "end": base + timedelta(seconds=t + dur),
                     "status": status, "burst": float(dur)})
        t += dur
    return bars
