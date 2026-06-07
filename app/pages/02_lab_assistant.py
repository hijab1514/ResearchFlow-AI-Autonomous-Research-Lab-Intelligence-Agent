"""
app/pages/02_lab_assistant.py
=============================
Lab Assistant — live system dashboard + scheduler stats + REAL Gantt timeline.

Telemetry, scheduler stats, and Gantt events all come from app.backend, which
reads the live LabAgent/SchedulerAgent. The Gantt is now wired to the real
scheduler.gantt_events() (no more probing).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402
from app.backend import (  # noqa: E402
    get_config, get_system_stats, read_scheduler_stats, read_gantt,
)

try:
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


def _gauge(value, title, threshold):
    v = value if value is not None else 0
    color = "#d62728" if v >= threshold else "#2ca02c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v, number={"suffix": "%"},
        title={"text": title},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "threshold": {"line": {"color": "black", "width": 3},
                             "thickness": 0.75, "value": threshold}}))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _render_dashboard():
    cfg = get_config().get("scheduler", {})
    cpu_th = cfg.get("cpu_threshold", 85)
    ram_th = 100 - cfg.get("ram_threshold", 20)
    gpu_th = cfg.get("gpu_threshold", 90)
    snap = get_system_stats()

    st.subheader("Live system")
    st.caption(f"Source: {snap['source']}"
               + (f" · bottleneck: **{snap['bottleneck']}**" if snap.get("bottleneck") else ""))
    cols = st.columns(3)
    for col, val, title, th in [(cols[0], snap["cpu"], "CPU", cpu_th),
                                (cols[1], snap["ram"], "RAM", ram_th),
                                (cols[2], snap["gpu"], "GPU VRAM", gpu_th)]:
        with col:
            if val is None:
                st.metric(title, "n/a")
                st.caption("not available")
            elif HAS_PLOTLY:
                st.plotly_chart(_gauge(val, title, th), use_container_width=True)
            else:
                st.metric(title, f"{val:.0f}%")
                st.progress(min(int(val), 100))
    st.button("🔄 Refresh", use_container_width=True)
    st.caption("Manual refresh. Add `streamlit-autorefresh` for continuous polling.")


def _render_scheduler_stats():
    stats = read_scheduler_stats()
    st.subheader("Scheduler")
    if not stats:
        st.info("No scheduler running yet (start a research run), or no stats exposed.")
        return
    items = [(k, v) for k, v in stats.items() if isinstance(v, (int, float, str))]
    cols = st.columns(min(4, max(1, len(items))))
    for i, (k, v) in enumerate(items):
        cols[i % len(cols)].metric(k.replace("_", " ").title(), v)


def _render_gantt():
    st.subheader("Task execution timeline")
    bars, is_real = read_gantt()
    if not HAS_PLOTLY:
        st.warning("plotly not installed — `pip install plotly pandas`.")
        st.dataframe([{"Task": b["task"], "Status": b["status"]} for b in bars],
                     use_container_width=True, hide_index=True)
        return
    df = pd.DataFrame([{
        "Task": b["task"], "Start": b["start"], "Finish": b["end"],
        "Status": b["status"],
        "Burst": f"{b['burst']:.1f}s" if isinstance(b.get("burst"), (int, float)) else "—",
    } for b in bars])
    color_map = {"completed": "#2ca02c", "deferred": "#ff7f0e",
                 "failed": "#d62728", "running": "#1f77b4"}
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task",
                      color="Status", color_discrete_map=color_map,
                      hover_data=["Burst"], height=440)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), xaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)
    if not is_real:
        st.caption("⚠️ Synthetic timeline — no scheduler runs recorded yet. "
                   "Run a research analysis and real task bursts appear here.")


def main():
    st.set_page_config(page_title="Lab Assistant · ResearchFlow AI",
                       page_icon="⚙️", layout="wide")
    st.title("⚙️ Lab Assistant")
    st.caption("Live resource telemetry · scheduler state · execution timeline.")
    _render_dashboard()
    st.divider()
    _render_scheduler_stats()
    st.divider()
    _render_gantt()


main()
