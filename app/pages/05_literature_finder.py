"""
app/pages/05_literature_finder.py
=================================
Literature Finder — type a topic, get the TOP real papers, ranked and exportable.

This is the page that makes the project useful for actual research, and it works
on ANY machine: it uses the LiteratureAgent (free OpenAlex + arXiv APIs), so it
needs no OpenAI key, no GPU, and none of the Numba/UMAP stack that Smart App
Control blocks. It runs even when the sidebar says "Backend not connected".
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402
from agents.literature_agent import LiteratureAgent  # noqa: E402


@st.cache_data(show_spinner=False, ttl=3600)
def _search(topic, limit, source, sort, year_from, oa_only, min_cit):
    agent = LiteratureAgent()
    papers = agent.find_top_papers(
        topic=topic, limit=limit, source=source, sort=sort,
        year_from=year_from, open_access_only=oa_only, min_citations=min_cit,
    )
    # return plain dicts so Streamlit can cache them
    return [p.__dict__ for p in papers]


def main() -> None:
    st.set_page_config(page_title="Literature Finder · ResearchFlow AI",
                       page_icon="📚", layout="wide")
    st.title("📚 Literature Finder")
    st.caption("Type a topic → the top real papers, ranked by citations. "
               "Live data from OpenAlex + arXiv. No API key needed.")

    with st.form("lit_form"):
        topic = st.text_input(
            "Research topic",
            placeholder="e.g. multimodal large language models for MRI",
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            source = st.selectbox("Source", ["journals", "both", "preprints"],
                                  format_func={"journals": "Journals (OpenAlex)",
                                               "both": "Journals + arXiv",
                                               "preprints": "arXiv preprints"}.get)
        with c2:
            sort = st.selectbox("Rank by", ["citations", "recent", "relevance"],
                                format_func={"citations": "Most cited (top papers)",
                                             "recent": "Newest first",
                                             "relevance": "Relevance"}.get)
        with c3:
            limit = st.slider("How many", 5, 40, 15, step=5)

        c4, c5, c6 = st.columns(3)
        with c4:
            year_from = st.number_input("From year (0 = any)", 0, 2026, 0, step=1)
        with c5:
            min_cit = st.number_input("Min citations", 0, 100000, 0, step=10)
        with c6:
            oa_only = st.toggle("Open-access only", value=False)

        go = st.form_submit_button("🔎 Find papers", use_container_width=True)

    if go and not topic.strip():
        st.error("Enter a topic first.")
        return
    if not go:
        st.info("Enter a topic above and hit **Find papers**. "
                "Tip: rank by *Most cited* to see the foundational papers in a field.")
        return

    with st.spinner(f"Searching OpenAlex/arXiv for '{topic}'…"):
        try:
            papers = _search(topic.strip(), limit, source, sort,
                             year_from or None, oa_only, int(min_cit))
        except Exception as exc:
            st.error(f"Search failed — likely a network issue. ({type(exc).__name__})")
            st.caption("OpenAlex and arXiv are public APIs; check your connection "
                       "and retry. No API key is required.")
            return

    if not papers:
        st.warning("No papers matched. Try a broader topic, lower the min-citations "
                   "filter, or widen the year range.")
        return

    # ---- summary + export ---- #
    cited = [p["citations"] for p in papers if p["citations"] is not None]
    m1, m2, m3 = st.columns(3)
    m1.metric("Papers found", len(papers))
    m2.metric("Top citation count", max(cited) if cited else "—")
    m3.metric("Open access", sum(1 for p in papers if p["is_open_access"]))

    agent = LiteratureAgent()
    from agents.literature_agent import Paper
    objs = [Paper(**p) for p in papers]
    d1, d2 = st.columns(2)
    d1.download_button("⬇️ Reading list (Markdown)",
                       agent.to_markdown(objs, topic),
                       file_name=f"reading_list_{topic[:30].replace(' ', '_')}.md",
                       mime="text/markdown", use_container_width=True)
    d2.download_button("⬇️ Citations (BibTeX)",
                       agent.to_bibtex(objs),
                       file_name=f"citations_{topic[:30].replace(' ', '_')}.bib",
                       mime="text/plain", use_container_width=True)

    st.divider()

    # ---- ranked results ---- #
    for i, p in enumerate(papers, 1):
        with st.container(border=True):
            head = st.columns([8, 2])
            head[0].markdown(f"**{i}. {p['title']}**")
            if p["citations"] is not None:
                head[1].markdown(f"`📈 {p['citations']:,} cites`")
            else:
                head[1].markdown("`preprint`")

            authors = p["authors"]
            meta = (", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")) \
                if authors else "Unknown authors"
            venue = p["venue"] or p["source"]
            st.caption(f"{meta} — *{venue}*, {p['year'] or 'n.d.'}"
                       + ("  ·  🔓 open access" if p["is_open_access"] else ""))

            if p["abstract"]:
                with st.expander("Abstract"):
                    st.write(p["abstract"])

            links = []
            if p["url"]:
                links.append(f"[🔗 Source]({p['url']})")
            if p["pdf_url"]:
                links.append(f"[📄 PDF]({p['pdf_url']})")
            if links:
                st.markdown(" · ".join(links))


main()
