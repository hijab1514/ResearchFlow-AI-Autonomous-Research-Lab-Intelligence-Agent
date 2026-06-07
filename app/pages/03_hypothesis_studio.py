"""
app/pages/03_hypothesis_studio.py
=================================
Hypothesis Studio — thesis ideation workspace.

Displays the ScoredHypotheses produced by a research run: ranked cards,
a novelty-vs-feasibility quadrant, and a composite-score bar chart.

This is a VIEWER. Hypotheses are generated as stage 7 of the full pipeline
(they depend on gaps + clusters + papers), so this page reads the result that
the Research Agent page (01) stored in session_state["raw_result"]. It does not
re-run the pipeline.

NOTE: main.py's home view stores a *normalized* dict that drops hypotheses, so
to populate this page you must run from the Research Agent page (01). Unifying
that is exactly what the app/backend.py extraction would fix.

Field names taken from research_pipeline._generate_report:
  h.rank · h.one_liner · h.composite_score · h.novelty_score
  h.feasibility_score · h.research_question · h.hypothesis_statement
  h.methodology.methodology_type.value · h.methodology.estimated_months
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

try:
    import pandas as pd
    import plotly.express as px
    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False


# --------------------------------------------------------------------------- #
# DATA
# --------------------------------------------------------------------------- #
def _norm_hypothesis(h, idx: int) -> dict:
    meth = getattr(h, "methodology", None)
    meth_type = getattr(getattr(meth, "methodology_type", None), "value", None)
    if meth_type is None:
        meth_type = str(getattr(meth, "methodology_type", "") or "—")
    return {
        "rank": getattr(h, "rank", idx + 1),
        "one_liner": getattr(h, "one_liner", getattr(h, "title", f"Hypothesis {idx + 1}")),
        "composite": float(getattr(h, "composite_score", 0.0) or 0.0),
        "novelty": float(getattr(h, "novelty_score", 0.0) or 0.0),
        "feasibility": float(getattr(h, "feasibility_score", 0.0) or 0.0),
        "question": getattr(h, "research_question", ""),
        "statement": getattr(h, "hypothesis_statement", ""),
        "method": meth_type,
        "months": getattr(meth, "estimated_months", None),
    }


def _build_view(result, demo: bool) -> dict:
    if demo or result is None or not getattr(result, "hypotheses", None):
        return _demo_view(empty=(result is not None and not demo))
    items = [_norm_hypothesis(h, i) for i, h in enumerate(result.hypotheses)]
    items.sort(key=lambda d: d["composite"], reverse=True)
    return {"topic": getattr(result, "query", ""), "items": items, "demo": False}


def _demo_view(empty: bool = False) -> dict:
    items = [
        {"rank": 1, "one_liner": "Byzantine-robust aggregation for edge FL",
         "composite": 0.81, "novelty": 0.86, "feasibility": 0.74,
         "question": "Can a trust-weighted aggregation rule maintain accuracy under "
                     "up to 30% adversarial edge clients without a trusted root?",
         "statement": "Trust-weighted aggregation degrades < 5% accuracy at 30% "
                      "Byzantine clients vs. FedAvg's > 40% collapse.",
         "method": "empirical_benchmark", "months": 8},
        {"rank": 2, "one_liner": "Reproducibility harness for on-device personalization",
         "composite": 0.68, "novelty": 0.62, "feasibility": 0.83,
         "question": "Does a standardized on-device eval harness change the relative "
                     "ranking of published personalization methods?",
         "statement": "Re-evaluation under a unified harness reorders the top-5 "
                      "methods in published leaderboards.",
         "method": "replication_study", "months": 5},
        {"rank": 3, "one_liner": "Cross-silo / cross-device parity benchmark",
         "composite": 0.59, "novelty": 0.71, "feasibility": 0.48,
         "question": "How much does dual-setting evaluation alter reported "
                     "communication-efficiency gains?",
         "statement": "Methods tuned for cross-device lose > 15% of their reported "
                      "efficiency gain in cross-silo settings.",
         "method": "mixed_methods", "months": 10},
    ]
    return {"topic": "Federated Learning for Edge Devices",
            "items": items, "demo": True, "empty_source": empty}


# --------------------------------------------------------------------------- #
# RENDER
# --------------------------------------------------------------------------- #
def _render_quadrant(items: list[dict]) -> None:
    st.subheader("Novelty vs. feasibility")
    if not HAS_PLOTLY:
        st.dataframe(items, use_container_width=True, hide_index=True)
        return
    df = pd.DataFrame([{
        "Novelty": d["novelty"], "Feasibility": d["feasibility"],
        "Composite": d["composite"], "Idea": d["one_liner"], "Rank": d["rank"],
    } for d in items])
    fig = px.scatter(
        df, x="Novelty", y="Feasibility", size="Composite", color="Composite",
        text="Rank", hover_name="Idea",
        color_continuous_scale="Viridis", size_max=34, height=460,
        range_x=[0, 1], range_y=[0, 1],
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_vline(x=0.5, line_dash="dot", line_color="gray", opacity=0.5)
    fig.update_traces(textposition="middle center",
                      textfont=dict(color="white", size=11))
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Upper-right = high novelty + high feasibility (best thesis bets). "
               "Bubble size = composite score.")


def _render_cards(items: list[dict]) -> None:
    st.subheader("Ranked hypotheses")
    for d in items:
        with st.container(border=True):
            head = st.columns([6, 2])
            head[0].markdown(f"**#{d['rank']} · {d['one_liner']}**")
            head[1].markdown(
                f"`composite {d['composite']:.2f}`"
            )
            sc = st.columns(3)
            sc[0].metric("Novelty", f"{d['novelty']:.2f}")
            sc[1].metric("Feasibility", f"{d['feasibility']:.2f}")
            months = f"{d['months']} mo" if d["months"] is not None else "—"
            sc[2].metric("Effort", months)

            if d["question"]:
                st.markdown(f"**Research question** — {d['question']}")
            if d["statement"]:
                st.markdown(f"**H₁** — {d['statement']}")
            st.caption(f"Suggested methodology: `{d['method']}`")


def _render_scores(items: list[dict]) -> None:
    if not HAS_PLOTLY or len(items) < 2:
        return
    st.subheader("Composite scores")
    df = pd.DataFrame([{"Idea": d["one_liner"], "Composite": d["composite"]}
                       for d in items]).sort_values("Composite")
    fig = px.bar(df, x="Composite", y="Idea", orientation="h",
                 range_x=[0, 1], height=max(220, 60 * len(items)),
                 color="Composite", color_continuous_scale="Viridis")
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10),
                      yaxis_title=None, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Hypothesis Studio · ResearchFlow AI",
                       page_icon="💡", layout="wide")
    st.title("💡 Hypothesis Studio")
    st.caption("Scoped thesis directions ranked by novelty, feasibility, and "
               "composite score.")

    result = st.session_state.get("raw_result")
    is_demo = st.session_state.get("raw_is_demo", result is None)
    view = _build_view(result, demo=is_demo)

    if view["demo"]:
        if view.get("empty_source"):
            st.warning("This run produced no hypotheses — showing demo ideas.", icon="🧪")
        else:
            st.info("No research run found. Showing demo ideas — run from the "
                    "**Research Agent** page to populate real hypotheses.", icon="🧪")

    if view["topic"]:
        st.markdown(f"**Topic:** {view['topic']}")

    items = view["items"]
    st.metric("Hypotheses", len(items))
    st.divider()
    _render_quadrant(items)
    st.divider()
    _render_cards(items)
    _render_scores(items)


main()
