"""
app/pages/02_lab_assistant.py
=============================
Lab Assistant page — live system dashboard (CPU/RAM/GPU) + scheduler stats +
task execution Gantt timeline.

This page READS telemetry from the LabAgent / SchedulerAgent that main.py or the
Research Agent page already created and cached in session_state. It does NOT
create them — opening the dashboard has no side effects. If no services exist
yet (you opened this page first), it falls back to live psutil readings + a
demo timeline.

UNKNOWN INTERFACE: I haven't seen scheduler_agent.py / gantt_logger.py, so the
Gantt-event access is probed defensively across several likely names. If none
match, a synthetic timeline renders with an on-screen warning. Paste those two
files and I'll wire the real recorder.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime, timedelta

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

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
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# --------------------------------------------------------------------------- #
# READ TELEMETRY (no side effects — only reads what already exists)
# --------------------------------------------------------------------------- #
def _read_snapshot() -> dict:
    """
    Prefer LabAgent.snapshot() (matches research_pipeline._resource_dict);
    fall back to raw psutil/torch if no lab agent is running.
    """
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

    # Fallback: direct readings
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


def _read_scheduler_stats() -> dict:
    sched = st.session_state.get("_scheduler")
    if sched is None:
        return {}
    try:
        status = sched.status()
        if isinstance(status, dict):
            return status.get("stats", status)
    except Exception:
        pass
    return {}


def _read_gantt() -> tuple[list[dict], bool]:
    """
    Returns (events, is_real). Each event normalized to:
        {"task": str, "start_s": float, "end_s": float, "status": str}
    Probes several likely access points on the scheduler / its gantt logger.
    """
    sched = st.session_state.get("_scheduler")
    raw = None
    if sched is not None:
        # Try logger objects, then direct methods/attrs
        for owner in (getattr(sched, "gantt_logger", None),
                      getattr(sched, "gantt", None),
                      getattr(sched, "_gantt", None),
                      sched):
            if owner is None:
                continue
            for attr in ("events", "records", "entries", "to_list", "get_events", "timeline"):
                cand = getattr(owner, attr, None)
                if cand is None:
                    continue
                try:
                    raw = cand() if callable(cand) else cand
                except Exception:
                    raw = None
                if raw:
                    break
            if raw:
                break

    if not raw:
        return _demo_gantt(), False

    # Normalize whatever shape we got
    events, t0 = [], None
    for r in raw:
        get = (lambda k, d=None: r.get(k, d)) if isinstance(r, dict) else (lambda k, d=None: getattr(r, k, d))
        name = get("name", get("task", get("task_name", "task")))
        start = get("start", get("start_s", get("started_at")))
        end = get("end", get("end_s", get("ended_at")))
        status = str(get("status", "completed"))
        s = _to_seconds(start)
        e = _to_seconds(end)
        if s is None:
            continue
        if e is None:
            e = s
        t0 = s if t0 is None else min(t0, s)
        events.append({"task": str(name), "_s": s, "_e": e, "status": status})

    if not events:
        return _demo_gantt(), False
    for ev in events:
        ev["start_s"] = round(ev.pop("_s") - t0, 2)
        ev["end_s"] = round(ev.pop("_e") - t0, 2)
    return events, True


def _to_seconds(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:  # ISO timestamp
        return datetime.fromisoformat(str(v)).timestamp()
    except Exception:
        return None


def _demo_gantt() -> list[dict]:
    stages = [
        ("query_reformulation", 5, "completed"),
        ("arxiv_retrieval", 20, "completed"),
        ("embedding_generation", 60, "completed"),
        ("clustering", 10, "deferred"),     # show a deferral event
        ("clustering", 10, "completed"),
        ("cluster_labeling", 30, "completed"),
        ("gap_detection", 45, "completed"),
        ("hypothesis_generation", 20, "completed"),
        ("roadmap_building", 30, "completed"),
        ("report_generation", 10, "completed"),
    ]
    out, t = [], 0.0
    for name, dur, status in stages:
        gap = 3.0 if status == "deferred" else dur  # deferral = waiting, then real run
        out.append({"task": name, "start_s": round(t, 2),
                    "end_s": round(t + gap, 2), "status": status})
        t += gap
    return out


# --------------------------------------------------------------------------- #
# RENDER
# --------------------------------------------------------------------------- #
def _gauge(value, title, threshold):
    if not HAS_PLOTLY:
        return None
    v = value if value is not None else 0
    color = "#d62728" if v >= threshold else "#2ca02c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=v,
        number={"suffix": "%"},
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "threshold": {"line": {"color": "black", "width": 3},
                          "thickness": 0.75, "value": threshold},
        },
    ))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def _render_dashboard() -> None:
    cfg = st.session_state.get("config", {}).get("scheduler", {})
    cpu_th = cfg.get("cpu_threshold", 85)
    ram_th = 100 - cfg.get("ram_threshold", 20)   # "available<20%" => "used>80%"
    gpu_th = cfg.get("gpu_threshold", 90)

    snap = _read_snapshot()
    st.subheader("Live system")
    st.caption(f"Source: {snap['source']}"
               + (f" · bottleneck: **{snap['bottleneck']}**" if snap.get("bottleneck") else ""))

    c1, c2, c3 = st.columns(3)
    pairs = [(c1, snap["cpu"], "CPU", cpu_th),
             (c2, snap["ram"], "RAM", ram_th),
             (c3, snap["gpu"], "GPU VRAM", gpu_th)]
    for col, val, title, th in pairs:
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
    st.caption("Manual refresh. For continuous updates, the background "
               "system_monitor thread should drive this — add `st_autorefresh` "
               "if you want auto-polling.")


def _render_scheduler_stats() -> None:
    stats = _read_scheduler_stats()
    st.subheader("Scheduler")
    if not stats:
        st.info("No scheduler running (open a research run first), or no stats exposed.")
        return
    items = [(k, v) for k, v in stats.items() if isinstance(v, (int, float, str))]
    if not items:
        st.json(stats)
        return
    cols = st.columns(min(4, len(items)))
    for i, (k, v) in enumerate(items):
        cols[i % len(cols)].metric(k.replace("_", " ").title(), v)


def _render_gantt() -> None:
    st.subheader("Task execution timeline")
    events, is_real = _read_gantt()
    if not HAS_PLOTLY:
        st.warning("plotly not installed — `pip install plotly pandas`.")
        st.dataframe(events, use_container_width=True, hide_index=True)
        return

    base = datetime(2024, 1, 1)
    df = pd.DataFrame([{
        "Task": e["task"],
        "Start": base + timedelta(seconds=e["start_s"]),
        "Finish": base + timedelta(seconds=max(e["end_s"], e["start_s"] + 0.5)),
        "Status": e["status"],
        "Duration": f"{e['end_s'] - e['start_s']:.1f}s",
    } for e in events])

    color_map = {"completed": "#2ca02c", "deferred": "#ff7f0e",
                 "retrying": "#d62728", "running": "#1f77b4"}
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task",
                      color="Status", color_discrete_map=color_map,
                      hover_data=["Duration"], height=420)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=None, yaxis_title=None)
    st.plotly_chart(fig, use_container_width=True)

    if not is_real:
        st.caption("⚠️ Synthetic timeline — no Gantt records found on the "
                   "scheduler. Wire scheduler/gantt_logger.py to expose events "
                   "(e.g. `scheduler.gantt_logger.events`).")


def main() -> None:
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
