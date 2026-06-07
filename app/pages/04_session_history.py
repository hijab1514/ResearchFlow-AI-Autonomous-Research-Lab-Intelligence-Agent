"""
app/pages/04_session_history.py
===============================
Session History — browse past research runs.

Reads from three sources, merged and deduped (newest first):
  1. experiment_tracker.run_tracker  — if it exposes a list method (PROBED;
     wire precisely once you share run_tracker.py)
  2. reports/*.md                    — the PERSISTENT source. _generate_report
     writes {run_id}_research_report.md on every run, so these survive restarts.
  3. session_state["history"]        — current-session runs from main.py home.

The reports-dir source is what makes history outlive a server restart; the
session_state source only covers the current process.
"""

from __future__ import annotations

import sys
import re
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st  # noqa: E402

REPORTS_DIR = REPO_ROOT / "reports"


# --------------------------------------------------------------------------- #
# SOURCE 1 — experiment_tracker (probed; replace once run_tracker.py is shared)
# --------------------------------------------------------------------------- #
def _from_tracker() -> list[dict]:
    try:
        from experiment_tracker import run_tracker as rt  # type: ignore
    except Exception:
        return []

    runs = None
    # Try a module-level function, then a RunTracker class
    for fname in ("list_runs", "all_runs", "get_runs", "load_runs"):
        fn = getattr(rt, fname, None)
        if callable(fn):
            try:
                runs = fn()
                break
            except Exception:
                pass
    if runs is None:
        Cls = getattr(rt, "RunTracker", None)
        if Cls is not None:
            try:
                inst = Cls()
                for m in ("list_runs", "all_runs", "get_runs"):
                    fn = getattr(inst, m, None)
                    if callable(fn):
                        runs = fn()
                        break
            except Exception:
                pass
    if not runs:
        return []

    out = []
    for r in runs:
        get = (lambda k, d=None: r.get(k, d)) if isinstance(r, dict) \
            else (lambda k, d=None: getattr(r, k, d))
        out.append(_record(
            run_id=get("run_id", get("id", "")),
            topic=get("query", get("topic", "")),
            ts=str(get("started_at", get("timestamp", ""))),
            status=str(get("status", "completed")),
            n_papers=get("n_papers", _len(get("papers"))),
            n_clusters=get("n_clusters", _len(get("clusters"))),
            n_gaps=get("n_gaps", _len(get("gaps"))),
            duration=get("total_duration_s", get("duration_s")),
            source="tracker",
        ))
    return out


# --------------------------------------------------------------------------- #
# SOURCE 2 — reports/*.md (persistent)
# --------------------------------------------------------------------------- #
_PATTERNS = {
    "topic":      re.compile(r"\*\*Query:\*\*\s*(.+)"),
    "ts":         re.compile(r"\*\*Generated:\*\*\s*([0-9T:\-\. ]+)"),
    "n_papers":   re.compile(r"Papers retrieved:\*\*\s*(\d+)"),
    "n_clusters": re.compile(r"Clusters identified:\*\*\s*(\d+)"),
    "n_gaps":     re.compile(r"Research gaps:\*\*\s*(\d+)"),
    "duration":   re.compile(r"Total pipeline time:\*\*\s*([\d.]+)"),
}


def _from_reports() -> list[dict]:
    if not REPORTS_DIR.exists():
        return []
    out = []
    for path in sorted(REPORTS_DIR.glob("*_research_report.md")):
        try:
            text = path.read_text()
        except Exception:
            continue
        fields = {}
        for key, pat in _PATTERNS.items():
            m = pat.search(text)
            fields[key] = m.group(1).strip() if m else None
        run_id = path.name.replace("_research_report.md", "")
        ts = fields["ts"] or datetime.fromtimestamp(path.stat().st_mtime).isoformat()[:19]
        out.append(_record(
            run_id=run_id, topic=fields["topic"] or "(unknown)", ts=ts,
            status="completed",
            n_papers=_int(fields["n_papers"]), n_clusters=_int(fields["n_clusters"]),
            n_gaps=_int(fields["n_gaps"]), duration=_float(fields["duration"]),
            source="report", report_md=text, report_path=str(path),
        ))
    return out


# --------------------------------------------------------------------------- #
# SOURCE 3 — session_state (current session only)
# --------------------------------------------------------------------------- #
def _from_session() -> list[dict]:
    hist = st.session_state.get("history", [])
    out = []
    for h in hist:
        if not isinstance(h, dict):
            continue
        out.append(_record(
            run_id=h.get("run_id", ""), topic=h.get("topic", ""),
            ts=h.get("started_at", ""), status=h.get("status", "completed"),
            n_papers=h.get("n_papers", 0), n_clusters=_len(h.get("clusters")),
            n_gaps=_len(h.get("gaps")), duration=h.get("elapsed_s"),
            source="session", report_md=h.get("report_md", ""),
        ))
    return out


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #
def _record(run_id, topic, ts, status, n_papers, n_clusters, n_gaps,
            duration, source, report_md="", report_path="") -> dict:
    return {"run_id": run_id or "", "topic": topic or "", "ts": ts or "",
            "status": status, "n_papers": n_papers or 0,
            "n_clusters": n_clusters or 0, "n_gaps": n_gaps or 0,
            "duration": duration, "source": source,
            "report_md": report_md, "report_path": report_path}


def _len(x):
    try:
        return len(x)
    except Exception:
        return 0


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _collect() -> list[dict]:
    runs = _from_tracker() + _from_reports() + _from_session()
    # Dedupe: prefer richer source (tracker > report > session) per run_id/topic+ts
    rank = {"tracker": 0, "report": 1, "session": 2}
    best: dict[str, dict] = {}
    for r in runs:
        key = r["run_id"] or f"{r['topic']}::{r['ts']}"
        if key not in best or rank[r["source"]] < rank[best[key]["source"]]:
            best[key] = r
    merged = list(best.values())
    merged.sort(key=lambda d: d["ts"], reverse=True)
    return merged


def _demo() -> list[dict]:
    return [
        _record("rp_demo_a", "Federated Learning for Edge Devices",
                "2024-05-02T14:21:08", "completed", 50, 4, 3, 168.4, "demo"),
        _record("rp_demo_b", "Multimodal LLMs for Medical Diagnosis",
                "2024-05-01T09:03:55", "completed", 80, 6, 5, 241.0, "demo"),
    ]


# --------------------------------------------------------------------------- #
# RENDER
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Session History · ResearchFlow AI",
                       page_icon="🗂️", layout="wide")
    st.title("🗂️ Session History")
    st.caption("Past research runs — persisted from reports/ plus this session.")

    runs = _collect()
    is_demo = not runs
    if is_demo:
        runs = _demo()
        st.info("No past runs found yet. Showing demo entries — run a research "
                "analysis and they'll appear here (persisted via reports/).",
                icon="🧪")

    sources = sorted({r["source"] for r in runs})
    st.caption(f"Sources: {', '.join(sources)} · {len(runs)} run(s)")

    st.dataframe(
        [{
            "When": r["ts"][:19], "Topic": r["topic"], "Status": r["status"],
            "Papers": r["n_papers"], "Clusters": r["n_clusters"],
            "Gaps": r["n_gaps"],
            "Duration": (f"{r['duration']:.1f}s" if isinstance(r["duration"], (int, float)) else "—"),
            "Source": r["source"],
        } for r in runs],
        use_container_width=True, hide_index=True,
    )

    st.divider()
    st.subheader("Run detail")
    labels = [f"{r['ts'][:19]} · {r['topic'][:50] or r['run_id']}" for r in runs]
    idx = st.selectbox("Select a run", range(len(runs)),
                       format_func=lambda i: labels[i]) if runs else None

    if idx is not None:
        r = runs[idx]
        m = st.columns(4)
        m[0].metric("Papers", r["n_papers"])
        m[1].metric("Clusters", r["n_clusters"])
        m[2].metric("Gaps", r["n_gaps"])
        m[3].metric("Duration",
                    f"{r['duration']:.1f}s" if isinstance(r["duration"], (int, float)) else "—")
        st.caption(f"Run ID: `{r['run_id'] or '—'}` · source: {r['source']}")

        if r.get("report_md"):
            with st.expander("Report", expanded=False):
                st.markdown(r["report_md"])
            st.download_button("⬇️ Download report (.md)", r["report_md"],
                               file_name=f"{r['run_id'] or 'report'}.md",
                               mime="text/markdown")
        elif r["source"] in ("report",) and r.get("report_path"):
            st.caption(f"Report file: {r['report_path']}")
        else:
            st.caption("No stored report for this run "
                       "(session-only entries don't persist the report body).")


main()
