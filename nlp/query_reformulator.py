"""
nlp/query_reformulator.py
==========================
ResearchFlow AI — GPT-4o arXiv Query Expansion

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL       = os.getenv("REFORMULATOR_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("REFORMULATOR_TEMP", "0.2"))
MAX_QUERIES         = int(os.getenv("REFORMULATOR_MAX_QUERIES", "4"))


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ReformulationResult:
    """Output of a single query expansion call."""
    original_query:  str
    expanded_queries: list[str]   # includes original as first entry
    domain_keywords: list[str]    # extracted domain terms for arXiv category hints
    arxiv_categories: list[str]   # suggested arXiv categories (e.g. cs.CV, cs.CL)

    @property
    def all_queries(self) -> list[str]:
        """Deduplicated list: original + expanded."""
        seen: set[str] = set()
        result: list[str] = []
        for q in [self.original_query, *self.expanded_queries]:
            q_lower = q.lower().strip()
            if q_lower and q_lower not in seen:
                seen.add(q_lower)
                result.append(q)
        return result

    def to_dict(self) -> dict:
        return {
            "original_query":   self.original_query,
            "expanded_queries": self.expanded_queries,
            "all_queries":      self.all_queries,
            "domain_keywords":  self.domain_keywords,
            "arxiv_categories": self.arxiv_categories,
        }

    def __repr__(self) -> str:
        return (
            f"ReformulationResult("
            f"original='{self.original_query[:40]}', "
            f"expanded={len(self.expanded_queries)} queries)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_REFORMULATION_PROMPT = """You are an expert academic search assistant specializing in computer science and engineering research.

Given a natural language research topic, generate {max_queries} distinct arXiv search queries that maximise paper retrieval coverage.

Rules:
- Expand all abbreviations (LLM → large language model, VLM → vision language model)
- Generate queries covering different angles: methodology, application, dataset, evaluation
- Each query should be 3–7 words, suitable for arXiv full-text search
- Include domain synonyms and closely related sub-topics
- Suggest the most relevant arXiv categories (e.g. cs.CV, cs.CL, cs.AI, cs.LG, cs.IR)
- Extract 5–8 key domain terms from the topic

Return ONLY a JSON object with these exact keys:
  "expanded_queries" : array of {max_queries} query strings
  "domain_keywords"  : array of 5–8 key domain terms
  "arxiv_categories" : array of 2–4 arXiv category codes

No markdown fences. No preamble. Valid JSON only.

Research topic: {query}

JSON:"""


# ─────────────────────────────────────────────────────────────────────────────
# QUERY REFORMULATOR
# ─────────────────────────────────────────────────────────────────────────────

class QueryReformulator:
    """
    Expands a natural language research topic into multiple structured
    arXiv API search queries using GPT-4o.

    Design decisions:
      - Single LLM call returns all expanded queries in one JSON response
        (cheaper and faster than N separate calls)
      - Falls back to rule-based expansion if LLM call fails
      - original_query is always included as the first entry so retrieval
        never degrades below a single-query baseline
    """

    def __init__(
        self,
        model_name:  str   = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_queries: int   = MAX_QUERIES,
    ) -> None:
        self._llm         = ChatOpenAI(model=model_name, temperature=temperature)
        self._max_queries = max_queries
        logger.info(
            "QueryReformulator init | model=%s | max_queries=%d",
            model_name, max_queries,
        )

    # ── MAIN API ─────────────────────────────────────────────────────────────

    async def reformulate(self, query: str) -> ReformulationResult:
        """
        Expand a raw research topic into structured arXiv search queries.

        Returns ReformulationResult with expanded_queries, domain_keywords,
        and suggested arXiv categories.
        Falls back to rule-based expansion on LLM failure.
        """
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        logger.info("Reformulating query: '%s'", query)

        prompt = _REFORMULATION_PROMPT.format(
            query=query,
            max_queries=self._max_queries,
        )

        try:
            response = await self._llm.ainvoke(prompt)
            raw = response.content.strip()
            result = self._parse_response(query, raw)
            logger.info(
                "Query expanded: %d queries | categories=%s",
                len(result.expanded_queries), result.arxiv_categories,
            )
            return result

        except Exception as exc:
            logger.warning(
                "LLM reformulation failed: %s — using rule-based fallback", exc
            )
            return self._fallback(query)

    async def reformulate_batch(
        self, queries: list[str]
    ) -> list[ReformulationResult]:
        """Expand multiple queries in parallel."""
        return list(await asyncio.gather(*[self.reformulate(q) for q in queries]))

    # ── SYNC WRAPPER ─────────────────────────────────────────────────────────

    def reformulate_sync(self, query: str) -> ReformulationResult:
        """Synchronous wrapper for non-async callers."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.reformulate(query))
        finally:
            loop.close()

    # ── PARSING ──────────────────────────────────────────────────────────────

    def _parse_response(self, original: str, raw: str) -> ReformulationResult:
        """Parse the LLM's JSON response into a ReformulationResult."""
        # Strip markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed — attempting extraction from text")
            data = self._extract_from_text(raw)

        expanded = data.get("expanded_queries", [])
        keywords = data.get("domain_keywords", [])
        cats     = data.get("arxiv_categories", [])

        # Validate and clean
        expanded = [
            str(q).strip() for q in expanded
            if q and str(q).strip().lower() != original.lower()
        ][:self._max_queries]

        return ReformulationResult(
            original_query=original,
            expanded_queries=expanded,
            domain_keywords=[str(k).strip() for k in keywords[:8]],
            arxiv_categories=[str(c).strip() for c in cats[:4]],
        )

    @staticmethod
    def _extract_from_text(raw: str) -> dict:
        """
        Best-effort extraction when JSON parsing fails.
        Looks for list-like patterns in the raw text.
        """
        queries: list[str] = []
        for line in raw.split("\n"):
            line = line.strip().lstrip("•-*123456789. ")
            if 3 <= len(line.split()) <= 10:
                queries.append(line)
        return {
            "expanded_queries": queries[:4],
            "domain_keywords":  [],
            "arxiv_categories": [],
        }

    def _fallback(self, query: str) -> ReformulationResult:
        """
        Rule-based fallback expansion when LLM is unavailable.
        Applies common abbreviation expansions and adds sub-topic variants.
        """
        # Common abbreviation expansions
        expansions: dict[str, str] = {
            r"\bLLM\b":    "large language model",
            r"\bVLM\b":    "vision language model",
            r"\bNLP\b":    "natural language processing",
            r"\bCV\b":     "computer vision",
            r"\bRL\b":     "reinforcement learning",
            r"\bGAN\b":    "generative adversarial network",
            r"\bSLM\b":    "small language model",
            r"\bRAG\b":    "retrieval augmented generation",
            r"\bFL\b":     "federated learning",
            r"\bSFT\b":    "supervised fine-tuning",
        }

        expanded_query = query
        for pattern, replacement in expansions.items():
            expanded_query = re.sub(pattern, replacement, expanded_query, flags=re.IGNORECASE)

        words   = query.lower().split()
        queries = [
            expanded_query,
            f"{query} survey",
            f"{query} benchmark",
            f"{query} deep learning",
        ]

        # Guess arXiv categories from keywords
        cats: list[str] = []
        kw_lower = query.lower()
        if any(w in kw_lower for w in ["vision", "image", "visual", "detection"]):
            cats.append("cs.CV")
        if any(w in kw_lower for w in ["language", "text", "nlp", "bert", "llm"]):
            cats.append("cs.CL")
        if any(w in kw_lower for w in ["learning", "neural", "deep", "train"]):
            cats.append("cs.LG")
        if any(w in kw_lower for w in ["agent", "reasoning", "planning"]):
            cats.append("cs.AI")
        if not cats:
            cats = ["cs.LG", "cs.AI"]

        return ReformulationResult(
            original_query=query,
            expanded_queries=queries[:self._max_queries],
            domain_keywords=words[:6],
            arxiv_categories=cats[:4],
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_reformulator: QueryReformulator | None = None


def get_reformulator() -> QueryReformulator:
    global _default_reformulator
    if _default_reformulator is None:
        _default_reformulator = QueryReformulator()
    return _default_reformulator


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

async def _demo() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    reformulator = QueryReformulator()

    queries = [
        "Multimodal LLMs for medical diagnosis",
        "Federated learning edge devices",
        "Zero-shot generalization in robotics",
    ]

    print("\n═══ QUERY REFORMULATOR DEMO ═══\n")

    for q in queries:
        result = await reformulator.reformulate(q)
        print(f"  Original:   {result.original_query}")
        print(f"  Expanded:")
        for eq in result.all_queries:
            print(f"    → {eq}")
        print(f"  Keywords:   {', '.join(result.domain_keywords)}")
        print(f"  Categories: {', '.join(result.arxiv_categories)}")
        print()


if __name__ == "__main__":
    asyncio.run(_demo())
