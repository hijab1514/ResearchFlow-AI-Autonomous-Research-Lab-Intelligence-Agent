"""
agents/literature_agent.py
==========================
LiteratureAgent — finds the TOP real papers for a topic and ranks them.

Why this exists: the original ResearchAgent depends on UMAP/HDBSCAN (blocked by
Windows Smart App Control on some machines) and OpenAI (costs money). This agent
needs NEITHER. It queries free, keyless scholarly APIs and returns real, ranked
results — so "type a topic → get the top journal papers" works on any machine
with an internet connection, no API key, no GPU, no Numba.

Sources:
  - OpenAlex  (https://openalex.org) — 250M+ works, journal articles with real
    citation counts and venues. This is what makes "TOP papers" meaningful:
    we sort by cited_by_count. Free, no key.
  - arXiv     (export.arxiv.org)     — preprints, for the newest unpublished work.

Pure standard library (urllib, json, xml). No third-party deps, nothing to block.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

OPENALEX_WORKS = "https://api.openalex.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "ResearchFlow-AI/1.0 (mailto:you@example.com)"  # OpenAlex polite pool


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    citations: int | None = None          # None = unknown (e.g. arXiv preprint)
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    doi: str = ""
    is_open_access: bool = False
    source: str = ""                       # "OpenAlex" | "arXiv"

    @property
    def author_str(self) -> str:
        if not self.authors:
            return "Unknown authors"
        if len(self.authors) <= 3:
            return ", ".join(self.authors)
        return f"{', '.join(self.authors[:3])} et al."


# --------------------------------------------------------------------------- #
# HTTP (stdlib, injectable for testing)
# --------------------------------------------------------------------------- #
def _http_get(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


class LiteratureAgent:
    """
    Finds and ranks real papers. The only public method you need is
    find_top_papers(); the rest are source-specific helpers and exporters.
    """

    def __init__(self, http_get: Callable[[str], str] = _http_get,
                 mailto: str | None = None) -> None:
        self._get = http_get
        if mailto:
            global USER_AGENT
            USER_AGENT = f"ResearchFlow-AI/1.0 (mailto:{mailto})"

    # ---- main entry point -------------------------------------------------- #
    def find_top_papers(
        self,
        topic: str,
        limit: int = 15,
        source: str = "journals",       # "journals" (OpenAlex) | "preprints" (arXiv) | "both"
        sort: str = "citations",        # "citations" | "recent" | "relevance"
        year_from: int | None = None,
        open_access_only: bool = False,
        min_citations: int = 0,
    ) -> list[Paper]:
        topic = (topic or "").strip()
        if not topic:
            return []

        papers: list[Paper] = []
        try:
            if source in ("journals", "both"):
                papers += self._openalex(topic, limit, sort, year_from,
                                         open_access_only, min_citations)
            if source in ("preprints", "both"):
                papers += self._arxiv(topic, limit, sort)
        except Exception as exc:                       # network/parse failure
            logger.warning("Literature search failed: %s", exc)
            raise

        # de-dupe by lowercased title
        seen, unique = set(), []
        for p in papers:
            key = re.sub(r"\W+", "", p.title.lower())[:80]
            if key and key not in seen:
                seen.add(key)
                unique.append(p)

        # final ranking across merged sources
        if sort == "citations":
            unique.sort(key=lambda p: (p.citations or -1), reverse=True)
        elif sort == "recent":
            unique.sort(key=lambda p: (p.year or 0), reverse=True)
        return unique[:limit]

    # ---- OpenAlex (journals, real citation counts) ------------------------- #
    def _openalex(self, topic, limit, sort, year_from, oa_only, min_citations) -> list[Paper]:
        sort_map = {
            "citations": "cited_by_count:desc",
            "recent": "publication_date:desc",
            "relevance": "relevance_score:desc",
        }
        filters = ["type:article"]
        if year_from:
            filters.append(f"from_publication_date:{int(year_from)}-01-01")
        if oa_only:
            filters.append("is_oa:true")
        if min_citations > 0:
            filters.append(f"cited_by_count:>{int(min_citations) - 1}")

        params = {
            "search": topic,
            "per_page": str(min(max(limit, 1), 50)),
            "sort": sort_map.get(sort, "relevance_score:desc"),
            "filter": ",".join(filters),
            "mailto": "you@example.com",
        }
        url = f"{OPENALEX_WORKS}?{urllib.parse.urlencode(params)}"
        data = json.loads(self._get(url))
        return [self._parse_openalex(w) for w in data.get("results", [])]

    @staticmethod
    def _parse_openalex(w: dict) -> Paper:
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        oa = w.get("open_access") or {}
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        landing = loc.get("landing_page_url") or w.get("id") or ""
        authors = [(a.get("author") or {}).get("display_name", "")
                   for a in (w.get("authorships") or [])]
        authors = [a for a in authors if a][:10]
        return Paper(
            title=w.get("title") or w.get("display_name") or "(untitled)",
            authors=authors,
            year=w.get("publication_year"),
            venue=src.get("display_name") or "",
            citations=w.get("cited_by_count", 0),
            abstract=LiteratureAgent._reconstruct_abstract(w.get("abstract_inverted_index")),
            url=(f"https://doi.org/{doi}" if doi else landing),
            pdf_url=oa.get("oa_url") or "",
            doi=doi,
            is_open_access=bool(oa.get("is_oa")),
            source="OpenAlex",
        )

    @staticmethod
    def _reconstruct_abstract(inv: dict | None) -> str:
        if not inv:
            return ""
        positions = [(i, word) for word, idxs in inv.items() for i in idxs]
        positions.sort()
        text = " ".join(w for _, w in positions)
        return text[:600] + ("…" if len(text) > 600 else "")

    # ---- arXiv (preprints) ------------------------------------------------- #
    def _arxiv(self, topic, limit, sort) -> list[Paper]:
        sort_by = "submittedDate" if sort == "recent" else "relevance"
        params = {
            "search_query": f"all:{topic}",
            "start": "0",
            "max_results": str(min(max(limit, 1), 50)),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        return self._parse_arxiv(self._get(url))

    @staticmethod
    def _parse_arxiv(xml_text: str) -> list[Paper]:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        out = []
        for e in root.findall("a:entry", ns):
            title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
            summary = (e.findtext("a:summary", default="", namespaces=ns) or "").strip()
            published = e.findtext("a:published", default="", namespaces=ns) or ""
            link = e.findtext("a:id", default="", namespaces=ns) or ""
            authors = [a.findtext("a:name", default="", namespaces=ns)
                       for a in e.findall("a:author", ns)]
            pdf = ""
            for ln in e.findall("a:link", ns):
                if ln.get("title") == "pdf":
                    pdf = ln.get("href", "")
            year = int(published[:4]) if published[:4].isdigit() else None
            out.append(Paper(
                title=title.replace("\n", " "),
                authors=[a for a in authors if a][:10],
                year=year, venue="arXiv (preprint)",
                citations=None,
                abstract=(summary[:600] + ("…" if len(summary) > 600 else "")),
                url=link, pdf_url=pdf, doi="", is_open_access=True, source="arXiv",
            ))
        return out

    # ---- exporters (the "real-life problem" payoff: a usable reading list) - #
    @staticmethod
    def to_markdown(papers: list[Paper], topic: str = "") -> str:
        lines = [f"# Reading list: {topic}".rstrip(), ""]
        for i, p in enumerate(papers, 1):
            cite = f" · {p.citations} citations" if p.citations is not None else ""
            lines += [
                f"## {i}. {p.title}",
                f"*{p.author_str} — {p.venue or p.source}, {p.year or 'n.d.'}{cite}*",
                "",
                (p.abstract or "_No abstract available._"),
                f"\n[Link]({p.url})" + (f" · [PDF]({p.pdf_url})" if p.pdf_url else ""),
                "",
            ]
        return "\n".join(lines)

    @staticmethod
    def to_bibtex(papers: list[Paper]) -> str:
        out = []
        for i, p in enumerate(papers, 1):
            first = (p.authors[0].split()[-1] if p.authors else "anon").lower()
            key = re.sub(r"\W+", "", f"{first}{p.year or ''}{i}")
            out.append(
                f"@article{{{key},\n"
                f"  title   = {{{p.title}}},\n"
                f"  author  = {{{' and '.join(p.authors) or 'Unknown'}}},\n"
                f"  journal = {{{p.venue}}},\n"
                f"  year    = {{{p.year or ''}}},\n"
                + (f"  doi     = {{{p.doi}}},\n" if p.doi else "")
                + "}"
            )
        return "\n\n".join(out)


if __name__ == "__main__":
    # quick live check (needs internet): python -m agents.literature_agent
    agent = LiteratureAgent()
    for p in agent.find_top_papers("graph neural networks for drug discovery", limit=5):
        print(f"[{p.citations}] {p.title[:70]} — {p.venue}, {p.year}")
