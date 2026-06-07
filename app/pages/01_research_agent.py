"""
app/pages/01_research_agent.py
==============================
Research Agent page — query → cluster map (UMAP scatter) → research gaps.

Streamlit auto-discovers this from app/pages/. Reachable via the sidebar once
you run `streamlit run app/main.py` from the repo root.

This page works with the RAW PipelineResult (not the normalized dict that
main.py's home view uses), because the cluster scatter needs per-paper data
(coordinates, titles, cluster ids) that the normalized dict throws away.

NOTE ON DUPLICATION: the lab/scheduler/pipeline bootstrap helpers below are
copied from main.py. That's the normal Streamlit "pages run standalone" pattern,
but it WILL drift. The clean fix is to extract them into app/backend.py and have
both main.py and this page import from it. Say the word and I'll do that next.
"""

from __future__ import annotations

import sys
import math
import random
import asyncio
from pathlib import Path

# --------------------------------------------------------------------------- #
# PATH BOOTSTRAP — this file is app/pages/01_research_agent.py
# parents[0]=pages, parents[1]=app, parents[2]=repo root
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

# Optional plotting deps
try:
    import pandas as pd
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False

# --------------------------------------------------------------------------- #
# BACKEND ADAPTER (duplicated from main.py — see header note)
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
    """Reuse the thread services main.py already cached; create if absent."""
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
    """Fresh pipeline per run (its asyncio.Queue can't cross event loops)."""
    lab, sched = _get_lab_and_scheduler()
    config = PipelineConfig(
        llm_model="gpt-4o", max_papers=max_papers,
        hypotheses_per_gap=1, max_hypotheses=3,
        roadmap_target_papers=10, roadmap_weeks=4,
        use_stage_cache=True, verbose=False,
    )
    return ResearchPipeline(lab_agent=lab, scheduler=sched, config=config)


def _run_pipeline_blocking(pipeline, query, on_event=None):
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
# COORDINATE EXTRACTION  (the UMAP-coords gap is handled here)
# --------------------------------------------------------------------------- #
def _extract_coords(result, papers):
    """
    Try to find real 2D coordinates. Returns list[(x, y)] aligned to `papers`,
    or None if nothing usable is found.

    Looked for, in order:
      1. result.umap_coords / result.umap_2d   (NOT currently stored — see below)
      2. per-Paper attrs: umap_x/umap_y or x/y
    """
    coords = getattr(result, "umap_coords", None)
    if coords is None:
        coords = getattr(result, "umap_2d", None)
    if coords is not None:
        try:
            arr = [(float(c[0]), float(c[1])) for c in coords]
            if len(arr) == len(papers):
                return arr
        except Exception:
            pass

    pts, ok = [], True
    for p in papers:
        x = getattr(p, "umap_x", getattr(p, "x", None))
        y = getattr(p, "umap_y", getattr(p, "y", None))
        if x is None or y is None:
            ok = False
            break
        pts.append((float(x), float(y)))
    return pts if (ok and pts) else None


def _synthesize_coords(papers):
    """Deterministic per-cluster blobs so the scatter renders when no real coords exist."""
    random.seed(42)
    centers, pts = {}, []
    for p in papers:
        cid = getattr(p, "cluster_id", -1)
        if cid not in centers:
            ang = len(centers) * 2.399963229728653  # golden angle
            r = 3.0 + 0.4 * len(centers)
            centers[cid] = (r * math.cos(ang), r * math.sin(ang))
        cx, cy = centers[cid]
        pts.append((cx + random.gauss(0, 0.6), cy + random.gauss(0, 0.6)))
    return pts


# --------------------------------------------------------------------------- #
# VIEW BUILDING — uniform structure for both real and demo results
# --------------------------------------------------------------------------- #
def _build_view(result, demo: bool) -> dict:
    if demo or result is None:
        return _demo_view()

    papers = list(result.papers or [])
    cluster_objs = list(result.clusters or [])
    label_by_id = {getattr(c, "cluster_id", i): getattr(c, "label", f"Cluster {i}")
                   for i, c in enumerate(cluster_objs)}

    real_coords = _extract_coords(result, papers)
    coords = real_coords if real_coords is not None else _synthesize_coords(papers)

    points = []
    for p, (x, y) in zip(papers, coords):
        cid = getattr(p, "cluster_id", -1)
        label = "Unclustered" if cid is not None and cid < 0 else label_by_id.get(cid, f"Cluster {cid}")
        title = getattr(p, "title", "") or ""
        year = getattr(p, "year", getattr(p, "published", "")) or ""
        points.append({
            "x": x, "y": y,
            "cluster": str(label),
            "title": (title[:80] + "…") if len(title) > 80 else title,
            "year": str(year)[:4],
        })

    clusters = [{
        "id": getattr(c, "cluster_id", i),
        "label": getattr(c, "label", f"Cluster {i}"),
        "size": getattr(c, "paper_count", 0),
        "reproducibility": getattr(c, "reproducibility_score", 0.0),
        "trend": getattr(c, "temporal_trend", ""),
    } for i, c in enumerate(cluster_objs)]

    gaps = [{
        "title": getattr(g, "title", ""),
        "confidence": getattr(g, "confidence", 0.0),
        "evidence": getattr(g, "description", ""),
    } for g in (result.gaps or [])]

    return {
        "topic": getattr(result, "query", ""),
        "n_papers": len(papers),
        "elapsed_s": getattr(result, "total_duration_s", 0.0),
        "coords_real": real_coords is not None,
        "points": points,
        "clusters": clusters,
        "gaps": gaps,
        "demo": False,
    }


def _demo_view() -> dict:
    random.seed(1)
    labels = ["Privacy-preserving aggregation", "Communication-efficient updates",
              "Heterogeneous / non-IID data", "On-device personalization"]
    sizes = [18, 14, 11, 7]
    repro = [0.62, 0.71, 0.40, 0.55]
    points = []
    for ci, (lab_, n) in enumerate(zip(labels, sizes)):
        ang = ci * 2.399963229728653
        cx, cy = 3.5 * math.cos(ang), 3.5 * math.sin(ang)
        for _ in range(n):
            points.append({
                "x": cx + random.gauss(0, 0.6), "y": cy + random.gauss(0, 0.6),
                "cluster": lab_, "title": "(demo paper)", "year": "2023",
            })
    clusters = [{"id": i, "label": l, "size": s, "reproducibility": r, "trend": "rising"}
                for i, (l, s, r) in enumerate(zip(labels, sizes, repro))]
    gaps = [
        {"title": "Robustness under adversarial clients", "confidence": 0.78,
         "evidence": "Only 2 of 50 papers address Byzantine clients."},
        {"title": "Reproducibility in on-device personalization", "confidence": 0.66,
         "evidence": "Lowest open-code score (0.40) of all clusters."},
    ]
    return {"topic": "Federated Learning for Edge Devices", "n_papers": sum(sizes),
            "elapsed_s": 0.0, "coords_real": True, "points": points,
            "clusters": clusters, "gaps": gaps, "demo": True}


# --------------------------------------------------------------------------- #
# RENDER
# --------------------------------------------------------------------------- #
def _render_scatter(view: dict) -> None:
    st.subheader("Research landscape")
    if not view["points"]:
        st.info("No papers to plot.")
        return
    if not HAS_PLOTLY:
        st.warning("plotly / pandas not installed — `pip install plotly pandas`. "
                   "Showing a table instead.")
        st.dataframe(view["points"], use_container_width=True, hide_index=True)
        return

    df = pd.DataFrame(view["points"])
    fig = px.scatter(
        df, x="x", y="y", color="cluster",
        hover_data={"title": True, "year": True, "x": False, "y": False},
        height=520,
    )
    fig.update_traces(marker=dict(size=9, opacity=0.8))
    fig.update_layout(
        legend_title_text="Cluster",
        xaxis_title=None, yaxis_title=None,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

    if not view["coords_real"]:
        st.caption(
            "⚠️ Synthetic layout — no real UMAP coordinates found in the result. "
            "Positions are grouped by cluster but are not the true embedding "
            "projection. Fix: store the coords on PipelineResult (see note below)."
        )


def _render_clusters(view: dict) -> None:
    st.subheader("Thematic clusters")
    if not view["clusters"]:
        st.info("No clusters.")
        return
    st.dataframe(
        [{"Cluster": c["label"], "Papers": c["size"],
          "Reproducibility": round(c["reproducibility"], 2), "Trend": c["trend"]}
         for c in view["clusters"]],
        use_container_width=True, hide_index=True,
    )


def _render_gaps(view: dict) -> None:
    st.subheader("Research gaps")
    if not view["gaps"]:
        st.info("No gaps detected.")
        return
    for g in view["gaps"]:
        with st.container(border=True):
            top = st.columns([5, 1])
            top[0].markdown(f"**{g['title']}**")
            top[1].markdown(f"`{g['confidence']:.2f}`")
            st.caption(g["evidence"])


# --------------------------------------------------------------------------- #
# PAGE
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Research Agent · ResearchFlow AI",
                       page_icon="🔬", layout="wide")
    st.title("🔬 Research Agent")
    st.caption("Retrieve → embed → cluster → detect gaps.")

    if not BACKEND_AVAILABLE:
        st.warning("Backend not connected — demo mode only.", icon="🧪")
        with st.expander("Why?"):
            st.caption(BACKEND_ERROR)

    ss = st.session_state
    with st.form("research_agent_form"):
        topic = st.text_input("Research topic",
                              value=ss.get("topic", ""),
                              placeholder="e.g. Federated Learning for Edge Devices")
        c1, c2 = st.columns([3, 2])
        with c1:
            max_papers = st.slider("Max papers", 10, 200, 50, step=10)
        with c2:
            demo = st.toggle("Demo mode", value=not BACKEND_AVAILABLE,
                            help="Sample data, no API key / cost.")
        submitted = st.form_submit_button("🚀 Run", use_container_width=True)

    if submitted:
        if not topic and not demo:
            st.error("Enter a topic, or turn on demo mode.")
        else:
            ss["topic"] = topic
            prog = st.progress(0)
            status = st.empty()

            def _on_event(ev):
                try:
                    prog.progress(min(int(ev.progress_pct), 100))
                    status.text(str(ev.message))
                except Exception:
                    pass

            with st.spinner("Running pipeline…"):
                if demo or not BACKEND_AVAILABLE:
                    ss["raw_result"] = None  # signals demo to _build_view
                    ss["raw_is_demo"] = True
                else:
                    pipeline = _build_pipeline(max_papers)
                    result = _run_pipeline_blocking(pipeline, topic, on_event=_on_event)
                    ss["raw_result"] = result
                    ss["raw_is_demo"] = False
                    if getattr(result, "status", "") == "failed":
                        st.error(f"Pipeline failed: {getattr(result, 'error_message', '')}")
            prog.empty()
            status.empty()

    # Render whatever result we have (from this page, or carried from home)
    if "raw_result" in ss or ss.get("raw_is_demo"):
        view = _build_view(ss.get("raw_result"), demo=ss.get("raw_is_demo", False))
        st.divider()
        if view["demo"]:
            st.info("Showing demo output (sample data).", icon="🧪")
        m1, m2, m3 = st.columns(3)
        m1.metric("Papers", view["n_papers"])
        m2.metric("Clusters", len(view["clusters"]))
        m3.metric("Elapsed", f"{view['elapsed_s']}s")
        _render_scatter(view)
        _render_clusters(view)
        _render_gaps(view)
    else:
        st.divider()
        st.markdown("Enter a topic above and run to see the cluster map and gaps.")


main()
