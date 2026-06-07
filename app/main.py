"""
ResearchFlow AI — Streamlit entry point (app/main.py)

Run from the repo root:
    streamlit run app/main.py

Wired to the REAL pipelines/research_pipeline.py:
  - ResearchPipeline(lab_agent, scheduler, config)   # required args
  - async run(query) -> PipelineResult               # query only; max_papers via config
  - Streamlit (sync) drives the coroutine via asyncio.run
  - live progress is drained from the pipeline's event queue

Runs in DEMO MODE with no API key / no backend so it always launches.
Backend wiring lives in ONE place: the BACKEND ADAPTER block.
"""

from __future__ import annotations

import sys
import time
import asyncio
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. PATH BOOTSTRAP — app/main.py -> app/ -> repo root
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

# --------------------------------------------------------------------------- #
# Optional deps — never crash the UI if one is missing
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# 2. BACKEND ADAPTER  ←←←  the only block tied to your real code
# --------------------------------------------------------------------------- #
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


def _get_lab_and_scheduler():
    """
    LabAgent + SchedulerAgent are background-thread services. Create them ONCE
    and cache in session_state so we don't spawn new threads on every rerun.
    (Thread-based services are safe to persist; the pipeline's asyncio.Queue is
    NOT — see _build_pipeline.)
    """
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


def _build_pipeline(max_papers: int) -> "ResearchPipeline":
    """
    Build a FRESH ResearchPipeline per run.

    Why fresh: ResearchPipeline creates an asyncio.Queue in __init__. Reusing one
    pipeline across multiple asyncio.run() calls (each a new event loop) causes
    'Future attached to a different loop' errors. Lab/scheduler are reused; the
    pipeline (cheap to build — agents are lazy) is rebuilt each run.
    """
    lab, sched = _get_lab_and_scheduler()
    config = PipelineConfig(
        llm_model="gpt-4o",
        max_papers=max_papers,        # <-- slider feeds config, not run()
        hypotheses_per_gap=1,
        max_hypotheses=3,
        roadmap_target_papers=10,
        roadmap_weeks=4,
        use_stage_cache=True,
        verbose=False,
    )
    return ResearchPipeline(lab_agent=lab, scheduler=sched, config=config)


def _run_pipeline_blocking(pipeline, query: str, on_event=None) -> "PipelineResult":
    """
    Drive the async pipeline from sync Streamlit.

    Note: pipeline.stream() yields ProgressEvents but DISCARDS the PipelineResult
    (it never returns task.result()). So we run pipeline.run() ourselves and drain
    the event queue for live progress, which gives us BOTH progress and the result.
    """
    async def _drive():
        task = asyncio.create_task(pipeline.run(query))
        q = pipeline._event_queue  # private, but the only place the result-run emits
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


def _normalize(result: "PipelineResult") -> dict:
    """Map PipelineResult onto the keys the UI renderer expects."""
    clusters = [
        {
            "label": getattr(c, "label", f"Cluster {getattr(c, 'cluster_id', '?')}"),
            "size": getattr(c, "paper_count", 0),
            "reproducibility": getattr(c, "reproducibility_score", 0.0),
        }
        for c in (result.clusters or [])
    ]
    gaps = [
        {
            "title": getattr(g, "title", ""),
            "confidence": getattr(g, "confidence", 0.0),
            "evidence": getattr(g, "description", ""),
        }
        for g in (result.gaps or [])
    ]
    return {
        "topic": result.query,
        "n_papers": len(result.papers or []),
        "clusters": clusters,
        "gaps": gaps,
        "elapsed_s": getattr(result, "total_duration_s", 0.0),
        "status": getattr(result, "status", "completed"),
        "error": getattr(result, "error_message", ""),
        "report_md": getattr(result, "report_md", ""),
        "demo": False,
    }


def run_research(topic: str, max_papers: int, force_demo: bool = False,
                 on_event=None) -> dict:
    """Single integration point between UI and backend."""
    t0 = time.time()
    if BACKEND_AVAILABLE and not force_demo:
        pipeline = _build_pipeline(max_papers)
        result = _run_pipeline_blocking(pipeline, topic, on_event=on_event)
        out = _normalize(result)
        if not out.get("elapsed_s"):
            out["elapsed_s"] = round(time.time() - t0, 2)
        return out
    time.sleep(0.6)
    return _demo_result(topic, max_papers, t0)


def _demo_result(topic: str, max_papers: int, t0: float) -> dict:
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
        {"title": "Reproducibility in on-device personalization",
         "confidence": 0.66,
         "evidence": "Lowest open-code score (0.40) of all clusters; few released datasets."},
        {"title": "Cross-silo vs cross-device evaluation parity",
         "confidence": 0.59,
         "evidence": "Benchmarks rarely report both settings; comparison is inconsistent."},
    ]
    return {
        "topic": topic, "n_papers": min(max_papers, 50),
        "clusters": clusters, "gaps": gaps,
        "elapsed_s": round(time.time() - t0, 2),
        "status": "completed", "error": "", "report_md": "", "demo": True,
    }


# --------------------------------------------------------------------------- #
# Config + system stats
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = {
    "scheduler": {"cpu_threshold": 85, "ram_threshold": 20, "gpu_threshold": 90},
    "arxiv": {"max_results": 100},
}


def load_config() -> dict:
    cfg_path = REPO_ROOT / "configs" / "scheduler_config.yaml"
    if HAS_YAML and cfg_path.exists():
        try:
            with open(cfg_path) as f:
                return {**DEFAULT_CONFIG, **(yaml.safe_load(f) or {})}
        except Exception:
            pass
    return DEFAULT_CONFIG


def get_system_stats() -> dict:
    stats = {"cpu": None, "ram": None, "gpu": None}
    if HAS_PSUTIL:
        stats["cpu"] = psutil.cpu_percent(interval=None)  # non-blocking
        stats["ram"] = psutil.virtual_memory().percent
    if HAS_TORCH:
        try:
            free, total = torch.cuda.mem_get_info()
            stats["gpu"] = round((1 - free / total) * 100, 1)
        except Exception:
            stats["gpu"] = None
    return stats


# --------------------------------------------------------------------------- #
# 3. SESSION STATE
# --------------------------------------------------------------------------- #
def init_session_state() -> None:
    ss = st.session_state
    ss.setdefault("config", load_config())
    ss.setdefault("history", [])
    ss.setdefault("last_result", None)
    ss.setdefault("demo_toggle", not BACKEND_AVAILABLE)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.title("⚙️ ResearchFlow AI")
        if BACKEND_AVAILABLE:
            st.success("Backend connected", icon="✅")
        else:
            st.warning("Backend not connected — demo mode", icon="🧪")
            with st.expander("Why?"):
                st.caption(BACKEND_ERROR or "Pipeline import failed.")

        st.divider()
        st.subheader("Live system")
        stats = get_system_stats()
        if not HAS_PSUTIL:
            st.caption("psutil not installed — `pip install psutil`")
        else:
            cpu, ram = stats["cpu"] or 0, stats["ram"] or 0
            st.metric("CPU", f"{cpu:.0f}%")
            st.progress(min(int(cpu), 100))
            st.metric("RAM", f"{ram:.0f}%")
            st.progress(min(int(ram), 100))
            if stats["gpu"] is not None:
                st.metric("GPU VRAM", f"{stats['gpu']:.0f}%")
                st.progress(min(int(stats["gpu"]), 100))
            else:
                st.caption("GPU: not detected")
        st.button("🔄 Refresh stats", use_container_width=True)

        st.divider()
        st.caption("Pages ▸ research agent · lab assistant · "
                   "hypothesis studio · session history")


# --------------------------------------------------------------------------- #
# Home
# --------------------------------------------------------------------------- #
def render_home() -> None:
    st.title("ResearchFlow AI")
    st.caption("Autonomous research intelligence + OS-integrated experiment "
               "orchestration.")

    with st.form("research_form"):
        topic = st.text_input(
            "Research topic",
            placeholder="e.g. Multimodal Large Language Models for Medical Diagnosis",
        )
        c1, c2 = st.columns([3, 2])
        with c1:
            max_papers = st.slider("Max papers to retrieve", 10, 200, 50, step=10)
        with c2:
            demo = st.toggle(
                "Demo mode",
                value=st.session_state.demo_toggle,
                help="Sample data, no API key / model download / cost.",
            )
        submitted = st.form_submit_button("🚀 Run analysis", use_container_width=True)

    if submitted:
        if not topic and not demo:
            st.error("Enter a topic, or turn on demo mode.")
        else:
            prog = st.progress(0)
            status = st.empty()

            def _on_event(ev) -> None:
                try:
                    prog.progress(min(int(ev.progress_pct), 100))
                    status.text(str(ev.message))
                except Exception:
                    pass

            with st.spinner("Running pipeline…"):
                result = run_research(
                    topic, max_papers, force_demo=demo,
                    on_event=None if demo else _on_event,
                )
            prog.empty()
            status.empty()

            if result.get("status") == "failed":
                st.error(f"Pipeline failed: {result.get('error', 'unknown error')}")
            st.session_state.last_result = result
            st.session_state.history.append(result)

    result = st.session_state.last_result
    if result:
        st.divider()
        if result.get("demo"):
            st.info("Showing demo output (sample data).", icon="🧪")
        m1, m2, m3 = st.columns(3)
        m1.metric("Papers analyzed", result["n_papers"])
        m2.metric("Clusters", len(result["clusters"]))
        m3.metric("Elapsed", f"{result['elapsed_s']}s")

        st.subheader("Thematic clusters")
        st.dataframe(
            [{"Cluster": c["label"], "Papers": c["size"],
              "Reproducibility": c["reproducibility"]} for c in result["clusters"]],
            use_container_width=True, hide_index=True,
        )

        st.subheader("Research gaps")
        for g in result["gaps"]:
            with st.container(border=True):
                top = st.columns([5, 1])
                top[0].markdown(f"**{g['title']}**")
                top[1].markdown(f"`{g['confidence']:.2f}`")
                st.caption(g["evidence"])

        if result.get("report_md"):
            with st.expander("Full report (Markdown)"):
                st.markdown(result["report_md"])

        st.caption("Open the **Research Agent** page for the cluster map, and "
                   "**Lab Assistant** for the Gantt timeline.")
    else:
        st.divider()
        st.markdown("Enter a topic above to start.")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(
        page_title="ResearchFlow AI", page_icon="🔬",
        layout="wide", initial_sidebar_state="expanded",
    )
    init_session_state()
    render_sidebar()
    render_home()


if __name__ == "__main__":
    main()
