"""
app/pages/05_literature_finder.py — Research Copilot dashboard
==============================================================
Professional literature-intelligence UI: relevance-gated search, analytics
cards, trend/venue charts, and an offline "field briefing" that reads the top
papers and extracts themes. No API key, no GPU — runs anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402
from agents.literature_agent import LiteratureAgent, Paper  # noqa: E402

try:
    import pandas as pd
    HAS_PD = True
except Exception:
    HAS_PD = False


@st.cache_data(show_spinner=False, ttl=3600)
def _search(topic, limit, source, sort, year_from, oa_only, min_cit, strictness):
    papers = LiteratureAgent().find_top_papers(
        topic=topic, limit=limit, source=source, sort=sort, year_from=year_from,
        open_access_only=oa_only, min_citations=min_cit, strictness=strictness)
    return [p.__dict__ for p in papers]


def _badge(rel: float) -> str:
    pct = round(rel * 100)
    color = "#16a34a" if rel >= 0.66 else "#ca8a04" if rel >= 0.45 else "#dc2626"
    return (f"<span style='background:{color}1a;color:{color};padding:2px 8px;"
            f"border-radius:999px;font-size:0.75rem;font-weight:600'>{pct}% match</span>")


def main() -> None:
    st.set_page_config(page_title="Research Copilot · ResearchFlow AI",
                       page_icon="📚", layout="wide")
    st.markdown("""<style>
      .stMetric {background:#ffffff08;border:1px solid #ffffff14;border-radius:12px;
                 padding:12px 16px}
      div[data-testid="stExpander"]{border-radius:10px}
    </style>""", unsafe_allow_html=True)

    st.title("📚 Research Copilot")
    st.caption("Relevance-gated literature intelligence. Real papers from OpenAlex + arXiv, "
               "ranked for on-topic precision. No API key required.")

    with st.form("lit_form"):
        topic = st.text_input("Research topic",
                              placeholder="e.g. aircraft defect detection")
        a, b, c, d = st.columns(4)
        source = a.selectbox("Source", ["journals", "both", "preprints"],
                            format_func={"journals": "Journals", "both": "Journals + arXiv",
                                         "preprints": "arXiv"}.get)
        sort = b.selectbox("Rank by", ["best", "citations", "recent"],
                          format_func={"best": "Best match", "citations": "Most cited",
                                       "recent": "Newest"}.get)
        limit = c.slider("Results", 5, 40, 15, step=5)
        strictness = d.slider("On-topic strictness", 0.25, 0.85, 0.45, step=0.05,
                             help="Higher = only tightly-matching papers. Raise this if "
                                  "you still see loosely-related results.")
        e, f, g = st.columns(3)
        year_from = e.number_input("From year (0 = any)", 0, 2026, 0, step=1)
        min_cit = f.number_input("Min citations", 0, 100000, 0, step=10)
        oa_only = g.toggle("Open-access only")
        go = st.form_submit_button("🔎 Search", use_container_width=True)

    if not go:
        st.info("Enter a topic and search. The **strictness** slider controls how "
                "tightly results must match — the fix for off-topic results.")
        return
    if not topic.strip():
        st.error("Enter a topic first.")
        return

    with st.spinner(f"Searching for '{topic}'…"):
        try:
            raw = _search(topic.strip(), limit, source, sort,
                         year_from or None, oa_only, int(min_cit), strictness)
        except Exception as exc:
            st.error(f"Search failed (likely network): {type(exc).__name__}. Retry.")
            return
    if not raw:
        st.warning("No on-topic papers found. Try lowering strictness, widening the "
                   "year range, or a broader topic.")
        return

    papers = [Paper(**d) for d in raw]
    agent = LiteratureAgent()
    brief = agent.field_briefing(papers, topic)

    # ---- analytics cards ---- #
    yr = brief.get("year_range")
    m = st.columns(5)
    m[0].metric("Papers", brief["count"])
    m[1].metric("Year span", f"{yr[0]}–{yr[1]}" if yr else "—")
    m[2].metric("Median citations", brief["median_citations"])
    m[3].metric("Open access", f"{brief['open_access_pct']}%")
    m[4].metric("Top venue", (brief["top_venues"][0][0][:18] + "…")
                if brief["top_venues"] else "—")

    tabs = st.tabs(["📄 Papers", "📈 Trends", "🏛 Venues", "🧭 Field briefing"])

    # ---- Papers ---- #
    with tabs[0]:
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Reading list (.md)", agent.to_markdown(papers, topic),
                           file_name=f"reading_list_{topic[:24].replace(' ','_')}.md",
                           use_container_width=True)
        d2.download_button("⬇️ Citations (.bib)", agent.to_bibtex(papers),
                           file_name=f"citations_{topic[:24].replace(' ','_')}.bib",
                           use_container_width=True)
        st.divider()
        for i, p in enumerate(papers, 1):
            with st.container(border=True):
                h = st.columns([7, 2, 2])
                h[0].markdown(f"**{i}. {p.title}**")
                h[1].markdown(_badge(p.relevance), unsafe_allow_html=True)
                h[2].markdown(f"`📈 {p.citations:,}`" if p.citations is not None else "`preprint`")
                st.caption(f"{p.author_str} — *{p.venue or p.source}*, {p.year or 'n.d.'}"
                           + ("  ·  🔓 open access" if p.is_open_access else ""))
                if p.abstract:
                    with st.expander("Abstract"):
                        st.write(p.abstract)
                links = [f"[🔗 Source]({p.url})"] if p.url else []
                if p.pdf_url:
                    links.append(f"[📄 PDF]({p.pdf_url})")
                if links:
                    st.markdown(" · ".join(links))

    # ---- Trends ---- #
    with tabs[1]:
        if HAS_PD:
            yrs = [p.year for p in papers if p.year]
            if yrs:
                df = pd.DataFrame({"year": yrs})
                counts = df.groupby("year").size().rename("papers")
                st.markdown("**Publication activity over time**")
                st.bar_chart(counts)
                cite_by_year = pd.DataFrame(
                    [{"year": p.year, "citations": p.citations or 0}
                     for p in papers if p.year]).groupby("year")["citations"].sum()
                st.markdown("**Citations by year (impact over time)**")
                st.line_chart(cite_by_year)
            else:
                st.info("No year data to chart.")
        else:
            st.info("Install pandas for charts: `pip install pandas`")

    # ---- Venues ---- #
    with tabs[2]:
        if brief["top_venues"] and HAS_PD:
            vdf = pd.DataFrame(brief["top_venues"], columns=["venue", "papers"]).set_index("venue")
            st.markdown("**Where this work is published**")
            st.bar_chart(vdf)
        else:
            st.info("No venue data.")

    # ---- Field briefing (offline 'reads the papers') ---- #
    with tabs[3]:
        st.markdown("**Key themes across the top papers**")
        st.write(" · ".join(f"`{t}`" for t in brief["themes"]) or "—")
        col = st.columns(2)
        if brief["seminal"]:
            s = brief["seminal"]
            col[0].markdown(f"**🏆 Most-cited (seminal)**\n\n{s.title}\n\n"
                            f"*{s.venue}, {s.year} · {s.citations:,} citations*")
        if brief["newest"]:
            n = brief["newest"]
            col[1].markdown(f"**🆕 Most recent**\n\n{n.title}\n\n*{n.venue}, {n.year}*")
        st.caption("This briefing is computed locally from titles + abstracts (no LLM). "
                   "With an OpenAI key on a deployed instance, this becomes a written "
                   "synthesis of the field.")


main()
