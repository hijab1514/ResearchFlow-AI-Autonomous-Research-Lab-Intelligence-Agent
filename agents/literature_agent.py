"""
agents/literature_agent.py  (v2 — relevance-gated Research Copilot engine)
==========================================================================
Finds the top REAL papers for a topic and — critically — keeps them ON-TOPIC.

The v1 bug: sorting by citations alone returned high-citation papers that only
weakly matched the query (e.g. "aircraft" -> a 1986 computer-reliability paper).
v2 fixes this with precision-first retrieval:

  1. Search title+abstract specifically (not OpenAlex's broad fulltext `search`).
  2. Fetch a CANDIDATE POOL larger than the result count.
  3. Score each candidate's ACTUAL relevance to the query locally
     (token coverage, title weighting, phrase bonus, fuzzy typo tolerance).
  4. GATE: drop anything below a relevance floor — this removes off-topic noise.
  5. Rank the survivors by a blended score: relevance (primary) + citations + recency.

Plus an offline "field briefing" (themes / seminal / newest) so the copilot
"reads the top papers and extracts findings" with no LLM and no API key.

Pure standard library. Free APIs (OpenAlex + arXiv). No key, no GPU, no Numba.
"""

from __future__ import annotations

import difflib
import json
import logging
import math
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

logger = logging.getLogger(__name__)

OPENALEX_WORKS = "https://api.openalex.org/works"
ARXIV_API = "http://export.arxiv.org/api/query"
USER_AGENT = "ResearchFlow-AI/2.0 (mailto:you@example.com)"

_STOP = {
    "the", "a", "an", "of", "for", "and", "or", "to", "in", "on", "with", "using",
    "based", "via", "by", "from", "at", "is", "are", "study", "approach", "method",
    "methods", "analysis", "system", "systems", "model", "models", "new", "novel",
    "paper", "research", "towards", "toward", "use", "used", "data",
}


@dataclass
class Paper:
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    citations: int | None = None
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    doi: str = ""
    is_open_access: bool = False
    source: str = ""
    relevance: float = 0.0          # 0..1, computed locally (the on-topic gate)

    @property
    def author_str(self) -> str:
        if not self.authors:
            return "Unknown authors"
        return ", ".join(self.authors[:3]) + (" et al." if len(self.authors) > 3 else "")


def _http_get(url: str, timeout: float = 20.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(t) >= 2 and t not in _STOP]


class LiteratureAgent:
    def __init__(self, http_get: Callable[[str], str] = _http_get,
                 mailto: str | None = None) -> None:
        self._get = http_get
        self._mailto = mailto or "you@example.com"

    # ---- main entry -------------------------------------------------------- #
    def find_top_papers(
        self,
        topic: str,
        limit: int = 15,
        source: str = "journals",
        sort: str = "best",            # "best" (relevance+citations) | "citations" | "recent"
        year_from: int | None = None,
        open_access_only: bool = False,
        min_citations: int = 0,
        strictness: float = 0.45,      # relevance floor 0..1 — higher = stricter on-topic
    ) -> list[Paper]:
        topic = (topic or "").strip()
        if not topic:
            return []
        q_terms = _tokens(topic)
        pool = max(limit * 3, 30)      # fetch wide, then filter to precision

        papers: list[Paper] = []
        if source in ("journals", "both"):
            papers += self._openalex(topic, pool, year_from, open_access_only, min_citations)
        if source in ("preprints", "both"):
            papers += self._arxiv(topic, pool)

        # de-dupe by title
        seen, unique = set(), []
        for p in papers:
            key = re.sub(r"\W+", "", p.title.lower())[:80]
            if key and key not in seen:
                seen.add(key)
                unique.append(p)

        # score relevance locally (works for every source, incl. arXiv)
        for p in unique:
            p.relevance = self._relevance(q_terms, p)

        # GATE: drop off-topic. If the gate is too aggressive (few survivors),
        # relax once so a valid niche query still returns something.
        gated = [p for p in unique if p.relevance >= strictness]
        if len(gated) < min(5, limit) and unique:
            relaxed = max(0.25, strictness - 0.2)
            gated = [p for p in unique if p.relevance >= relaxed]

        gated = self._rank(gated, sort)
        return gated[:limit]

    # ---- relevance (the fix) ---------------------------------------------- #
    @staticmethod
    def _relevance(q_terms: list[str], p: Paper) -> float:
        if not q_terms:
            return 1.0
        title_toks = set(_tokens(p.title))
        abs_toks = set(_tokens(p.abstract))
        hay = title_toks | abs_toks

        def present(term: str, bucket: set[str]) -> bool:
            if term in bucket:
                return True
            # fuzzy: tolerate typos / morphology (e.g. "aircarft"~"aircraft")
            return bool(difflib.get_close_matches(term, bucket, n=1, cutoff=0.86))

        title_hits = sum(1 for t in q_terms if present(t, title_toks))
        any_hits = sum(1 for t in q_terms if present(t, hay))
        coverage = any_hits / len(q_terms)                 # how many query terms appear
        title_w = title_hits / len(q_terms)                # appearing in TITLE matters more

        # phrase bonus: consecutive query terms appearing in the title
        phrase_bonus = 0.0
        tl = (p.title or "").lower()
        for i in range(len(q_terms) - 1):
            if f"{q_terms[i]} {q_terms[i+1]}" in tl:
                phrase_bonus = 0.15
                break

        score = 0.6 * coverage + 0.4 * title_w + phrase_bonus
        return max(0.0, min(1.0, score))

    @staticmethod
    def _rank(papers: list[Paper], sort: str) -> list[Paper]:
        if not papers:
            return papers
        if sort == "citations":
            papers.sort(key=lambda p: (p.relevance >= 0.45, p.citations or -1), reverse=True)
            return papers
        if sort == "recent":
            papers.sort(key=lambda p: (p.year or 0, p.relevance), reverse=True)
            return papers
        # "best": blend relevance (primary) + citation strength + recency
        cites = [p.citations for p in papers if p.citations]
        max_c = max(cites) if cites else 1
        years = [p.year for p in papers if p.year]
        y_lo, y_hi = (min(years), max(years)) if years else (0, 1)
        span = max(1, y_hi - y_lo)

        def blended(p: Paper) -> float:
            c = math.log1p(p.citations or 0) / math.log1p(max_c) if max_c else 0
            r = (p.year - y_lo) / span if p.year else 0
            return 0.65 * p.relevance + 0.25 * c + 0.10 * r

        papers.sort(key=blended, reverse=True)
        return papers

    # ---- OpenAlex (precision: title+abstract search, relevance sort) ------- #
    def _openalex(self, topic, pool, year_from, oa_only, min_citations) -> list[Paper]:
        filters = ["type:article", f"title_and_abstract.search:{topic}"]
        if year_from:
            filters.append(f"from_publication_date:{int(year_from)}-01-01")
        if oa_only:
            filters.append("is_oa:true")
        if min_citations > 0:
            filters.append(f"cited_by_count:>{int(min_citations) - 1}")
        params = {
            "per_page": str(min(pool, 50)),
            "sort": "relevance_score:desc",
            "filter": ",".join(filters),
            "mailto": self._mailto,
        }
        url = f"{OPENALEX_WORKS}?{urllib.parse.urlencode(params)}"
        try:
            data = json.loads(self._get(url))
            results = data.get("results", [])
        except Exception as exc:
            logger.warning("OpenAlex tight search failed (%s); trying broad search", exc)
            results = []
        # fallback: if tight title/abstract search is too sparse, broaden once
        if len(results) < 5:
            params.pop("filter", None)
            params["search"] = topic
            params["filter"] = "type:article"
            url = f"{OPENALEX_WORKS}?{urllib.parse.urlencode(params)}"
            try:
                results = json.loads(self._get(url)).get("results", [])
            except Exception:
                pass
        return [self._parse_openalex(w) for w in results]

    @staticmethod
    def _parse_openalex(w: dict) -> Paper:
        loc = w.get("primary_location") or {}
        src = loc.get("source") or {}
        oa = w.get("open_access") or {}
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        landing = loc.get("landing_page_url") or w.get("id") or ""
        authors = [(a.get("author") or {}).get("display_name", "")
                   for a in (w.get("authorships") or [])]
        return Paper(
            title=w.get("title") or w.get("display_name") or "(untitled)",
            authors=[a for a in authors if a][:10],
            year=w.get("publication_year"),
            venue=src.get("display_name") or "",
            citations=w.get("cited_by_count", 0),
            abstract=LiteratureAgent._abs(w.get("abstract_inverted_index")),
            url=(f"https://doi.org/{doi}" if doi else landing),
            pdf_url=oa.get("oa_url") or "",
            doi=doi, is_open_access=bool(oa.get("is_oa")), source="OpenAlex",
        )

    @staticmethod
    def _abs(inv: dict | None) -> str:
        if not inv:
            return ""
        pos = [(i, w) for w, idxs in inv.items() for i in idxs]
        pos.sort()
        t = " ".join(w for _, w in pos)
        return t[:700] + ("…" if len(t) > 700 else "")

    # ---- arXiv ------------------------------------------------------------- #
    def _arxiv(self, topic, pool) -> list[Paper]:
        params = {"search_query": f"all:{topic}", "start": "0",
                  "max_results": str(min(pool, 50)),
                  "sortBy": "relevance", "sortOrder": "descending"}
        try:
            return self._parse_arxiv(self._get(f"{ARXIV_API}?{urllib.parse.urlencode(params)}"))
        except Exception as exc:
            logger.warning("arXiv search failed: %s", exc)
            return []

    @staticmethod
    def _parse_arxiv(xml_text: str) -> list[Paper]:
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(xml_text)
        out = []
        for e in root.findall("a:entry", ns):
            title = (e.findtext("a:title", "", ns) or "").strip().replace("\n", " ")
            summary = (e.findtext("a:summary", "", ns) or "").strip()
            published = e.findtext("a:published", "", ns) or ""
            link = e.findtext("a:id", "", ns) or ""
            authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
            pdf = next((ln.get("href", "") for ln in e.findall("a:link", ns)
                        if ln.get("title") == "pdf"), "")
            out.append(Paper(
                title=title, authors=[a for a in authors if a][:10],
                year=int(published[:4]) if published[:4].isdigit() else None,
                venue="arXiv (preprint)", citations=None,
                abstract=summary[:700] + ("…" if len(summary) > 700 else ""),
                url=link, pdf_url=pdf, is_open_access=True, source="arXiv"))
        return out

    # ---- offline "field briefing" (extract findings, no LLM) --------------- #
    @staticmethod
    def field_briefing(papers: list[Paper], topic: str) -> dict:
        if not papers:
            return {}
        q = set(_tokens(topic))
        terms = Counter()
        for p in papers:
            for t in _tokens(p.title) + _tokens(p.abstract):
                if t not in q:
                    terms[t] += 1
        themes = [t for t, _ in terms.most_common(12)]
        cited = [p for p in papers if p.citations]
        years = [p.year for p in papers if p.year]
        venues = Counter(p.venue for p in papers if p.venue)
        seminal = max(cited, key=lambda p: p.citations) if cited else None
        newest = max((p for p in papers if p.year), key=lambda p: p.year, default=None)
        med = sorted(p.citations for p in cited)[len(cited) // 2] if cited else 0
        return {
            "themes": themes,
            "year_range": (min(years), max(years)) if years else None,
            "median_citations": med,
            "top_venues": venues.most_common(5),
            "open_access_pct": round(100 * sum(p.is_open_access for p in papers) / len(papers)),
            "seminal": seminal,
            "newest": newest,
            "count": len(papers),
        }

    # ---- exporters --------------------------------------------------------- #
    @staticmethod
    def to_markdown(papers: list[Paper], topic: str = "") -> str:
        lines = [f"# Reading list: {topic}".rstrip(), ""]
        for i, p in enumerate(papers, 1):
            cite = f" · {p.citations} citations" if p.citations is not None else ""
            lines += [f"## {i}. {p.title}",
                      f"*{p.author_str} — {p.venue or p.source}, {p.year or 'n.d.'}"
                      f"{cite} · match {round(p.relevance*100)}%*", "",
                      (p.abstract or "_No abstract available._"),
                      f"\n[Link]({p.url})" + (f" · [PDF]({p.pdf_url})" if p.pdf_url else ""), ""]
        return "\n".join(lines)

    @staticmethod
    def to_bibtex(papers: list[Paper]) -> str:
        out = []
        for i, p in enumerate(papers, 1):
            first = (p.authors[0].split()[-1] if p.authors else "anon").lower()
            key = re.sub(r"\W+", "", f"{first}{p.year or ''}{i}")
            out.append(f"@article{{{key},\n  title   = {{{p.title}}},\n"
                        f"  author  = {{{' and '.join(p.authors) or 'Unknown'}}},\n"
                        f"  journal = {{{p.venue}}},\n  year    = {{{p.year or ''}}},\n"
                        + (f"  doi     = {{{p.doi}}},\n" if p.doi else "") + "}")
        return "\n\n".join(out)
