"""
app/main.py
===========
Streamlit entry point (home hub). All backend logic lives in app/backend.py;
this file is just UI + the shared run launcher.

Run from the repo root:
    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402
from app.backend import (  # noqa: E402
    BACKEND_AVAILABLE, BACKEND_ERROR, HAS_PSUTIL,
    get_config, get_system_stats, run_research,
)


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
        cfg = get_config().get("scheduler", {})
        stats = get_system_stats()
        if not HAS_PSUTIL and stats["source"] == "psutil":
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
            if stats.get("bottleneck"):
                st.caption(f"Bottleneck: **{stats['bottleneck']}**")
        st.button("🔄 Refresh stats", use_container_width=True)


def render_home() -> None:
    st.title("ResearchFlow AI")
    st.caption("Autonomous research intelligence + OS-integrated experiment "
               "orchestration.")

    with st.form("research_form"):
        topic = st.text_input(
            "Research topic",
            value=st.session_state.get("topic", ""),
            placeholder="e.g. Multimodal Large Language Models for Medical Diagnosis",
        )
        c1, c2 = st.columns([3, 2])
        with c1:
            max_papers = st.slider("Max papers to retrieve", 10, 200, 50, step=10)
        with c2:
            demo = st.toggle("Demo mode", value=not BACKEND_AVAILABLE,
                            help="Sample data, no API key / model download / cost.")
        submitted = st.form_submit_button("🚀 Run analysis", use_container_width=True)

    if submitted:
        if not topic and not demo:
            st.error("Enter a topic, or turn on demo mode.")
        else:
            st.session_state["topic"] = topic
            prog = st.progress(0)
            status = st.empty()

            def _on_event(ev):
                try:
                    prog.progress(min(int(ev.progress_pct), 100))
                    status.text(str(ev.message))
                except Exception:
                    pass

            with st.spinner("Running pipeline…"):
                result = run_research(topic, max_papers, force_demo=demo,
                                      on_event=None if demo else _on_event)
            prog.empty()
            status.empty()
            if result.get("status") == "failed":
                st.error(f"Pipeline failed: {result.get('error', 'unknown error')}")

    result = st.session_state.get("last_result")
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
        st.caption("Open **Research Agent** for the cluster map · **Hypothesis "
                   "Studio** for thesis ideas · **Lab Assistant** for the timeline.")
    else:
        st.divider()
        st.markdown("Enter a topic above to start.")


def main() -> None:
    st.set_page_config(page_title="ResearchFlow AI", page_icon="🔬",
                       layout="wide", initial_sidebar_state="expanded")
    render_sidebar()
    render_home()


if __name__ == "__main__":
    main()
