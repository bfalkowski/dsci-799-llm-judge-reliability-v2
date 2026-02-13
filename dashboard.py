"""
LLM-as-a-Judge Reliability Dashboard (STUB)

"""

import json
from pathlib import Path

import streamlit as st

# Paths relative to repo root (where dashboard.py lives).
REPO_ROOT = Path(__file__).resolve().parent

RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"

CONTENT_DIR = REPO_ROOT / "dashboard_content"


def _load_content(name, ext="md"):
    """Load text from dashboard_content/{name}.{ext}. Returns empty string if missing."""
    path = CONTENT_DIR / f"{name}.{ext}"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _load_captions():
    """Load captions from dashboard_content/captions.json. Returns dict, empty on error."""
    path = CONTENT_DIR / "captions.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


_captions = _load_captions()


# --- Page config and title ---
st.set_page_config(page_title="LLM-as-a-Judge Reliability", layout="wide")

# TODO: Add custom CSS

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

        # TODO: Panel 1 — Experiment Overview 
    # TODO: Panel 2 — Score 
    # TODO: Panel 3 —TBD
    # TODO: Panel 4 — TBD
    # TODO: Panel 5 — TBD
    st.info("TODO: Implement View Results")


# ==================== TAB 3: Run Experiment ==============
# ====
with tab_run:
    # TODO: Subset dropdown
    
    st.header("Run Experiment")
    st.info("TODO Implement Run Experiment formm ")


# ==================== TAB 4: Manage ====================
with tab_manage:
    # TODO: Delete experiments  
    
    st.header("Manage")
    st.info("TODO: Implement Manage tab..")


# ==================== TAB 5: Telemetry (OTEL) ====================

if tab_otel:
    with tab_otel:
        # TODO: Sync with View Results 
        st.header("Telemetry - OTEL")
        st.info("TODO: Implement Telemetry tab")
