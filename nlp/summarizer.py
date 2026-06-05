"""
nlp/summarizer.py
=================
ResearchFlow AI — LangChain Cluster Summarization Chain

Author : ResearchFlow AI
License: MIT
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

DEFAULT_MODEL       = os.getenv("SUMMARIZER_MODEL", "gpt-4o")
DEFAULT_TEMPERATURE = float(os.getenv("SUMMARIZER_TEMP", "0.0"))
MAX_ABSTRACTS       = int(os.getenv("SUMMARIZER_MAX_ABS", "5"))


# ─────────────────────────────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ClusterSummary:
    """Output of summarizing one cluster."""
    cluster_id:    int
    label:         str        # ≤8-word thematic label
    summary:       str        # 2–3 sentence description
    keywords:      list[str]  # 5 representative keywords
    n_papers_used: int        # how many abstracts were sampled

    def to_dict(self) -> dict:
        return {
            "cluster_id":    self.cluster_id,
            "label":         self.label,
            "summary":       self.summary,
            "keywords":      self.keywords,
            "n_papers_used": self.n_papers_used,
        }

    def __repr__(self) -> str:
        return f"ClusterSummary(id={self.cluster_id}, label='{self.label}')"


# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_LABEL_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an expert academic research classifier.

Read the following paper abstracts from a single research cluster and generate:
1. A concise thematic label (5–8 words) — the research sub-field these papers share
2. A 2–3 sentence description of what this cluster covers
3. Five representative keywords (comma-separated)

Return ONLY a JSON object with keys: "label", "summary", "keywords" (array).
No markdown. No preamble. Valid JSON only.

Abstracts:
{text}

JSON:""",
)

_REFINE_PROMPT = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template="""You are refining an existing cluster summary with additional abstracts.

Existing summary:
{existing_answer}

Additional abstracts:
{text}

Update the summary if needed. Return the same JSON format:
{{"label": "...", "summary": "...", "keywords": [...]}}

Return ONLY valid JSON. No markdown. No preamble.""",
)


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARIZER
# ─────────────────────────────────────────────────────────────────────────────

class ClusterSummarizer:
    """
    LangChain summarization chain that generates thematic labels
    and descriptions for HDBSCAN clusters.

    Uses the 'refine' chain strategy so large clusters can be
    processed in chunks without truncating abstracts.
    """

    def __init__(
        self,
        model_name:  str   = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_abstracts: int = MAX_ABSTRACTS,
    ) -> None:
        self._llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._max_abstracts = max_abstracts
        self._chain = load_summarize_chain(
            llm=self._llm,
            chain_type="refine",
            question_prompt=_LABEL_PROMPT,
            refine_prompt=_REFINE_PROMPT,
            return_intermediate_steps=False,
        )
        logger.info(
            "ClusterSummarizer init | model=%s | max_abstracts=%d",
            model_name, max_abstracts,
        )

    # ── SINGLE CLUSTER ───────────────────────────────────────────────────────

    async def summarize_cluster(
        self,
        cluster_id: int,
        papers:     list[Any],
    ) -> ClusterSummary:
        """
        Generate a thematic label + summary for one cluster.

        Args:
            cluster_id : integer cluster ID
            papers     : Paper objects belonging to this cluster

        Returns ClusterSummary with label, summary, and keywords.
        """
        if not papers:
            return ClusterSummary(
                cluster_id=cluster_id,
                label=f"Cluster {cluster_id}",
                summary="No papers in this cluster.",
                keywords=[],
                n_papers_used=0,
            )

        # Sample up to max_abstracts representative papers
        sample = papers[:self._max_abstracts]
        docs = [
            Document(
                page_content=(
                    f"Title: {getattr(p, 'title', 'Unknown')}\n"
                    f"Abstract: {getattr(p, 'abstract', '')[:500]}"
                )
            )
            for p in sample
        ]

        try:
            response = await self._chain.ainvoke({"input_documents": docs})
            raw = response.get("output_text", "").strip()
            return self._parse_response(cluster_id, raw, len(sample))
        except Exception as exc:
            logger.error("Summarizer failed for cluster %d: %s", cluster_id, exc)
            return self._fallback(cluster_id, sample)

    # ── ALL CLUSTERS ─────────────────────────────────────────────────────────

    async def summarize_all(
        self,
        papers:          list[Any],
        cluster_labels:  list[int],
    ) -> list[ClusterSummary]:
        """
        Summarize all clusters in parallel.

        Args:
            papers        : all Paper objects
            cluster_labels: integer label per paper (parallel to papers)

        Returns list of ClusterSummary, one per cluster (excl. noise).
        """
        # Group papers by cluster
        cluster_map: dict[int, list[Any]] = {}
        for paper, label in zip(papers, cluster_labels):
            if label < 0:
                continue   # skip noise
            cluster_map.setdefault(label, []).append(paper)

        logger.info(
            "Summarizing %d clusters (parallel)...", len(cluster_map)
        )

        tasks = [
            self.summarize_cluster(cid, cluster_papers)
            for cid, cluster_papers in sorted(cluster_map.items())
        ]
        summaries = await asyncio.gather(*tasks)
        return list(summaries)

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _parse_response(
        self,
        cluster_id: int,
        raw:        str,
        n_used:     int,
    ) -> ClusterSummary:
        """Parse JSON response from the LLM."""
        import json

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()

        try:
            data = json.loads(raw)
            return ClusterSummary(
                cluster_id=cluster_id,
                label=str(data.get("label", f"Cluster {cluster_id}")),
                summary=str(data.get("summary", "")),
                keywords=list(data.get("keywords", [])),
                n_papers_used=n_used,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "JSON parse failed for cluster %d: %s — using raw text",
                cluster_id, exc,
            )
            # Graceful fallback: use raw as label
            return ClusterSummary(
                cluster_id=cluster_id,
                label=raw[:60] if raw else f"Cluster {cluster_id}",
                summary=raw,
                keywords=[],
                n_papers_used=n_used,
            )

    def _fallback(
        self,
        cluster_id: int,
        papers:     list[Any],
    ) -> ClusterSummary:
        """Minimal fallback summary when LLM call fails."""
        titles = [getattr(p, "title", "")[:40] for p in papers[:3]]
        return ClusterSummary(
            cluster_id=cluster_id,
            label=f"Cluster {cluster_id}",
            summary=f"Papers include: {'; '.join(titles)}",
            keywords=[],
            n_papers_used=len(papers),
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_default_summarizer: ClusterSummarizer | None = None


def get_summarizer() -> ClusterSummarizer:
    global _default_summarizer
    if _default_summarizer is None:
        _default_summarizer = ClusterSummarizer()
    return _default_summarizer


# ─────────────────────────────────────────────────────────────────────────────
# CLI / DEMO
# ─────────────────────────────────────────────────────────────────────────────

async def _demo() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    class FakePaper:
        def __init__(self, title: str, abstract: str) -> None:
            self.title    = title
            self.abstract = abstract

    cluster_0 = [
        FakePaper(
            "Zero-Shot Radiology Report Generation",
            "We propose a vision-language model for automatic radiology report "
            "generation using zero-shot learning. Our model achieves state-of-the-art "
            "results on CheXpert and MIMIC-CXR benchmarks without labelled training data.",
        ),
        FakePaper(
            "Cross-Modal Alignment for Clinical NLP",
            "We present a cross-modal alignment framework that bridges visual and "
            "textual representations in clinical settings. Code available at GitHub.",
        ),
        FakePaper(
            "Multimodal Transformers for Medical Imaging",
            "A transformer-based architecture that jointly encodes radiology images "
            "and clinical notes for downstream diagnostic tasks.",
        ),
    ]

    cluster_1 = [
        FakePaper(
            "Federated Learning with Heterogeneous Data",
            "Communication-efficient federated optimization for edge devices with "
            "non-IID data distributions using gradient compression.",
        ),
        FakePaper(
            "Privacy-Preserving ML on IoT Sensors",
            "We apply federated averaging to IoT sensor data while preserving "
            "differential privacy guarantees. Dataset released publicly.",
        ),
    ]

    all_papers = cluster_0 + cluster_1
    all_labels = [0, 0, 0, 1, 1]

    summarizer = ClusterSummarizer()

    print("\n═══ SUMMARIZER DEMO ═══\n")
    print(f"  Clusters: 2 | Papers: {len(all_papers)}\n")

    summaries = await summarizer.summarize_all(all_papers, all_labels)

    for s in summaries:
        print(f"  Cluster {s.cluster_id}: '{s.label}'")
        print(f"  Summary:  {s.summary}")
        print(f"  Keywords: {', '.join(s.keywords)}")
        print(f"  Papers used: {s.n_papers_used}")
        print()


if __name__ == "__main__":
    asyncio.run(_demo())
