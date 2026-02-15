"""
LLM-as-a-Judge Reliability Dashboard (STUB)

"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# Paths relative to repo root (where dashboard.py lives).
REPO_ROOT = Path(__file__).resolve().parent
ENCODING = "utf-8"
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
        rows = []
        with path.open("r", encoding=ENCODING) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        if rows:
            df = pd.DataFrame(rows)

            # Raw data
            with st.expander("Raw judgments", expanded=True):
                st.dataframe(df, use_container_width=True)

            # TODO: Per-item score variance (Var_i = sample variance across K runs)
            with st.expander("Per-item score variance", expanded=False):
                st.info("TODO: Var_i = sample variance across K runs per item.")

            # TODO: Pairwise agreement rate (matching pairs / K(K-1)/2)
            with st.expander("Pairwise agreement rate", expanded=False):
                st.info("TODO: Agreement_i = matching pairs / total pairs (total pairs = K(K-1)/2).")

            # TODO: Distribution of disagreements
            with st.expander("Distribution of disagreements", expanded=False):
                st.info("TODO: Show distribution of score disagreements across items.")

            # TODO: Latency vs score correlation
            with st.expander("Latency vs score", expanded=False):
                st.info("TODO: Scatter plot of latency_ms vs score, correlation.")

            # TODO: Summary statistics
            with st.expander("Summary statistics", expanded=False):
                st.info("TODO: Overall summary stats across all items and runs.")
        else:
            st.info("File is empty.")


# ==================== TAB 3: Run Experiment ==============
# ====
with tab_run:
    # TODO: Subset dropdown
    
    st.header("Run Experiment")
    st.info("TODO Implement Run Experiment formm ")



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
