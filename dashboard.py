"""
LLM-as-a-Judge Reliability Dashboard (STUB)

"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow importing from src when dashboard runs from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from compute_metrics import (
    _group_by_item,
    metric1_per_item_variance,
    metric2_exact_agreement,
    metric3_score_histogram,
)
from utils import ENCODING, REPO_ROOT, load_jsonl
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
CONTENT_DIR = REPO_ROOT / "dashboard_content"


def _load_content(name, ext="md"):
    """Load text from dashboard_content/{name}.{ext}. Returns empty string if missing."""
    path = CONTENT_DIR / f"{name}.{ext}"
    if not path.exists():
        return ""
    return path.read_text(encoding=ENCODING).strip()


def _load_captions():
    """Load captions from dashboard_content/captions.json. Returns dict, empty on error."""
    path = CONTENT_DIR / "captions.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding=ENCODING) as f:
            return json.load(f)
    except Exception:
        return {}


_captions = _load_captions()


# --- Page config and title ---
st.set_page_config(page_title="LLM-as-a-Judge Reliability", layout="wide")

# Load and inject custom CSS
def _load_css():
    path = CONTENT_DIR / "dashboard.css"
    if path.exists():
        return path.read_text(encoding=ENCODING)
    return ""

_css = _load_css()
if _css:
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

st.title(_captions.get("title", "LLM-as-a-Judge Reliability & Execution Integrity"))

# --- Tab structure ---
tab_names = ["Overview", "Run Experiment", "View Results"]

tab_names.append("Manage")
tabs = st.tabs(tab_names)

tab_overview, tab_run, tab_view = tabs[0], tabs[1], tabs[2]
tab_otel = None  # TODO: tabs[3] when Telemetry exists
tab_manage = tabs[-1]


# ==================== TAB 1:  Overview ====================
with tab_overview:
    # TODO: Load and render dashboard_content/overview.md
    overview_md = _load_content("overview")
    if overview_md:

        st.markdown(overview_md)

    else:
        st.info("Overview content not found. Create dashboard_content/overview.md.")
    st.caption(_captions.get("overview_footer", ""))


# ==================== TAB 2: View Results ====================
with tab_view:
    jsonl_files = list(RESULTS_DIR.glob("*.jsonl")) if RESULTS_DIR.exists() else []
    if not jsonl_files:
        st.info("No result files in results")
    else:
        selected = st.selectbox("Result file", [f.name for f in jsonl_files], key="view_results_file")
        path = RESULTS_DIR / selected
        rows = load_jsonl(path)
        if rows:
            df = pd.DataFrame(rows)
            by_item = _group_by_item(rows)
            m1 = metric1_per_item_variance(by_item)
            m2 = metric2_exact_agreement(by_item)
            counts = metric3_score_histogram(rows)

            # Metrics section
            st.subheader("Reliability metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean variance", f"{m1['mean_variance']:.4f}")
                st.caption("Per-item score variance across K runs")
            with col2:
                st.metric("% zero variance", f"{m1['pct_items_zero_variance']:.1f}%")
                st.caption(f"{m1['zero_var_count']} / {m1['n_items']} items")
            with col3:
                st.metric("Exact agreement", f"{m2['mean_agreement_rate']:.1%}")
                st.caption("Mean pairwise score match rate")

            # Score distribution
            st.subheader("Score distribution")
            if counts:
                hist_df = pd.DataFrame(
                    {"score": list(counts.keys()), "count": list(counts.values())}
                ).sort_values("score")
                st.bar_chart(hist_df.set_index("score"))
            else:
                st.info("No scores to display.")

            with st.expander("Raw judgments", expanded=True):
                st.dataframe(df, use_container_width=True)

        else:
            st.info("File is empty.")


# ==================== TAB 3: Run Experiment ==============
with tab_run:
    st.header("Run Experiment")
    st.info("TODO Implement Run Experiment form.")

    # Show raw results (most recent) so you can see output format
    jsonl_files = list(RESULTS_DIR.glob("*.jsonl")) if RESULTS_DIR.exists() else []
    if jsonl_files:
        latest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
        st.subheader("Latest results (raw)")
        st.caption(f"{latest.name}")
        rows = load_jsonl(latest)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("File is empty.")



# ==================== TAB 4: Telemetry (OTEL) ====================

if tab_otel:
    with tab_otel:
        # TODO: Sync with View Results 
        st.header("Telemetry - OTEL")
        st.info("TODO: Implement Telemetry tab")


# ==================== TAB 5: Manage ====================
with tab_manage:
    # TODO: Delete experiments  
    
    st.header("Manage")
    st.info("TODO: Implement Manage tab..")
