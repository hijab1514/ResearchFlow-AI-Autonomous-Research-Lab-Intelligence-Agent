"""
ResearchFlow AI — Streamlit entry point (app/main.py)

Run from the repo root:
    streamlit run app/main.py

Jobs of this file:
  1. Fix sys.path so top-level packages (agents/, nlp/, scheduler/, ...) import
     correctly when Streamlit runs a script inside app/.
  2. Boot shared state (config, session_state) used by app/pages/.
  3. Provide a working home screen + research launcher.

Runs in DEMO MODE with no API key and no backend, so it launches today.
To use your real pipeline, edit ONLY the run_research() adapter block.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# 1. PATH BOOTSTRAP (before importing your packages)
# app/main.py -> app/ -> repo root
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
# 2. BACKEND ADAPTER  ←←←  EDIT THIS BLOCK TO MATCH YOUR CODE
# --------------------------------------------------------------------------- #
BACKEND_AVAILABLE = False
BACKEND_ERROR = ""
try:
    # >>> EDIT to match your real module + class <
    from pipelines.research_pipeline import ResearchPipeline  # type: ignore
    BACKEND_AVAILABLE = True
except Exception as e:
    BACKEND_ERROR = f"{type(e).__name__}: {e}"


def run_research(topic: str, max_papers: int, force_demo: bool = False) -> dict:
    """
    Single integration point between UI and backend.

    Expected return shape:
        {
            "topic": str, "n_papers": int,
            "clusters": [{"label": str, "size": int, "reproducibility": float}],
            "gaps": [{"title": str, "confidence": float, "evidence": str}],
            "elapsed_s": float, "demo": bool,
        }
    """
    t0 = time.time()
    if BACKEND_AVAILABLE and not force_demo:
        # >>> EDIT to match ResearchPipeline's real interface <
        pipe = ResearchPipeline()
        raw = pipe.run(topic=topic, max_papers=max_papers)
        result = _normalize(raw)
        result["elapsed_s"] = round(time.time() - t0, 2)
        result["demo"] = False
        return result
    time.sleep(0.6)
    return _demo_result(topic, max_papers, t0)


def _normalize(raw) -> dict:
    """Map your pipeline's return onto the keys the UI expects."""
    if raw is None:
        return {"topic": "", "n_papers": 0, "clusters": [], "gaps": []}
    get = (lambda k, d: getattr(raw, k, raw.get(k, d))) if hasattr(raw, "get") \
        else (lambda k, d: getattr(raw, k, d))
    return {
        "topic": get("topic", ""),
        "n_papers": get("n_papers", 0),
        "clusters": get("clusters", []),
        "gaps": get("gaps", []),
    }


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
        "elapsed_s": round(time.time() - t0, 2), "demo": True,
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
                st.caption("Edit the adapter block in app/main.py to wire it in.")

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
            placeholder="e.g. Federated Learning for Edge Devices",
        )
        c1, c2 = st.columns([3, 2])
        with c1:
            max_papers = st.slider("Max papers to retrieve", 10, 200, 50, step=10)
        with c2:
            demo = st.toggle(
                "Demo mode",
                value=st.session_state.demo_toggle,
                help="Run with built-in sample data (no API key / backend needed).",
            )
        submitted = st.form_submit_button("🚀 Run analysis", use_container_width=True)

    if submitted:
        if not topic and not demo:
            st.error("Enter a topic, or turn on demo mode.")
        else:
            with st.spinner("Retrieving → embedding → clustering → gap detection…"):
                result = run_research(topic, max_papers, force_demo=demo)
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

        st.caption("Open the **Research Agent** page for the cluster map, and "
                   "**Lab Assistant** for the Gantt timeline.")
    else:
        st.divider()
        st.markdown("Enter a topic above to start. The full cluster map and OS "
                    "scheduling timeline live in the sidebar pages.")


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
