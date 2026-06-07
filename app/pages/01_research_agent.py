"""
app/pages/01_research_agent.py
==============================
Research Agent page — query → cluster map (UMAP scatter) → research gaps.

Running goes through app.backend.run_research (shared with home + other pages),
so this page no longer duplicates the pipeline bootstrap. The scatter-building
(coords extraction + fallback) stays here because it's specific to this page.
"""

from __future__ import annotations

import sys
import math
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402
from app.backend import BACKEND_AVAILABLE, BACKEND_ERROR, run_research  # noqa: E402

try:
    import pandas as pd
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# ---- coordinate extraction (UMAP-coords gap handled here) ------------------ #
def _extract_coords(result, papers):
    coords = getattr(result, "umap_coords", None) or getattr(result, "umap_2d", None)
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
    random.seed(42)
    centers, pts = {}, []
    for p in papers:
        cid = getattr(p, "cluster_id", -1)
        if cid not in centers:
            ang = len(centers) * 2.399963229728653
            r = 3.0 + 0.4 * len(centers)
            centers[cid] = (r * math.cos(ang), r * math.sin(ang))
        cx, cy = centers[cid]
        pts.append((cx + random.gauss(0, 0.6), cy + random.gauss(0, 0.6)))
    return pts


def _build_view(result, demo: bool) -> dict:
    if demo or result is None:
        return _demo_view()
    papers = list(getattr(result, "papers", None) or [])
    cluster_objs = list(getattr(result, "clusters", None) or [])
    label_by_id = {getattr(c, "cluster_id", i): getattr(c, "label", f"Cluster {i}")
                   for i, c in enumerate(cluster_objs)}
    real = _extract_coords(result, papers)
    coords = real if real is not None else _synthesize_coords(papers)
    points = []
    for p, (x, y) in zip(papers, coords):
        cid = getattr(p, "cluster_id", -1)
        label = "Unclustered" if (cid is not None and cid < 0) else label_by_id.get(cid, f"Cluster {cid}")
        title = (getattr(p, "title", "") or "")
        year = getattr(p, "year", getattr(p, "published", "")) or ""
        points.append({"x": x, "y": y, "cluster": str(label),
                       "title": (title[:80] + "…") if len(title) > 80 else title,
                       "year": str(year)[:4]})
    clusters = [{"label": getattr(c, "label", f"Cluster {i}"),
                 "size": getattr(c, "paper_count", 0),
                 "reproducibility": getattr(c, "reproducibility_score", 0.0),
                 "trend": getattr(c, "temporal_trend", "")} for i, c in enumerate(cluster_objs)]
    gaps = [{"title": getattr(g, "title", ""), "confidence": getattr(g, "confidence", 0.0),
             "evidence": getattr(g, "description", "")} for g in (getattr(result, "gaps", None) or [])]
    return {"topic": getattr(result, "query", ""), "n_papers": len(papers),
            "elapsed_s": getattr(result, "total_duration_s", 0.0),
            "coords_real": real is not None, "points": points,
            "clusters": clusters, "gaps": gaps, "demo": False}


def _demo_view() -> dict:
    random.seed(1)
    labels = ["Privacy-preserving aggregation", "Communication-efficient updates",
              "Heterogeneous / non-IID data", "On-device personalization"]
    sizes, repro = [18, 14, 11, 7], [0.62, 0.71, 0.40, 0.55]
    points = []
    for ci, (lab_, n) in enumerate(zip(labels, sizes)):
        ang = ci * 2.399963229728653
        cx, cy = 3.5 * math.cos(ang), 3.5 * math.sin(ang)
        for _ in range(n):
            points.append({"x": cx + random.gauss(0, 0.6), "y": cy + random.gauss(0, 0.6),
                           "cluster": lab_, "title": "(demo paper)", "year": "2023"})
    clusters = [{"label": l, "size": s, "reproducibility": r, "trend": "rising"}
                for l, s, r in zip(labels, sizes, repro)]
    gaps = [{"title": "Robustness under adversarial clients", "confidence": 0.78,
             "evidence": "Only 2 of 50 papers address Byzantine clients."},
            {"title": "Reproducibility in on-device personalization", "confidence": 0.66,
             "evidence": "Lowest open-code score (0.40) of all clusters."}]
    return {"topic": "Federated Learning for Edge Devices", "n_papers": sum(sizes),
            "elapsed_s": 0.0, "coords_real": True, "points": points,
            "clusters": clusters, "gaps": gaps, "demo": True}


def _render_scatter(view) -> None:
    st.subheader("Research landscape")
    if not view["points"]:
        st.info("No papers to plot.")
        return
    if not HAS_PLOTLY:
        st.warning("plotly / pandas not installed — `pip install plotly pandas`.")
        st.dataframe(view["points"], use_container_width=True, hide_index=True)
        return
    df = pd.DataFrame(view["points"])
    fig = px.scatter(df, x="x", y="y", color="cluster",
                     hover_data={"title": True, "year": True, "x": False, "y": False},
                     height=520)
    fig.update_traces(marker=dict(size=9, opacity=0.8))
    fig.update_layout(legend_title_text="Cluster", margin=dict(l=10, r=10, t=10, b=10))
    fig.update_xaxes(showticklabels=False, title=None)
    fig.update_yaxes(showticklabels=False, title=None)
    st.plotly_chart(fig, use_container_width=True)
    if not view["coords_real"]:
        st.caption("⚠️ Synthetic layout — no real UMAP coordinates in the result. "
                   "Store them on PipelineResult (result.umap_coords) for the true map.")


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
        topic = st.text_input("Research topic", value=ss.get("topic", ""),
                              placeholder="e.g. Federated Learning for Edge Devices")
        c1, c2 = st.columns([3, 2])
        with c1:
            max_papers = st.slider("Max papers", 10, 200, 50, step=10)
        with c2:
            demo = st.toggle("Demo mode", value=not BACKEND_AVAILABLE)
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
                norm = run_research(topic, max_papers, force_demo=demo,
                                    on_event=None if demo else _on_event)
            prog.empty()
            status.empty()
            if norm.get("status") == "failed":
                st.error(f"Pipeline failed: {norm.get('error', '')}")

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
        st.subheader("Thematic clusters")
        st.dataframe([{"Cluster": c["label"], "Papers": c["size"],
                       "Reproducibility": round(c["reproducibility"], 2), "Trend": c["trend"]}
                      for c in view["clusters"]], use_container_width=True, hide_index=True)
        st.subheader("Research gaps")
        for g in view["gaps"]:
            with st.container(border=True):
                top = st.columns([5, 1])
                top[0].markdown(f"**{g['title']}**")
                top[1].markdown(f"`{g['confidence']:.2f}`")
                st.caption(g["evidence"])
    else:
        st.divider()
        st.markdown("Enter a topic above and run to see the cluster map and gaps.")


main()
