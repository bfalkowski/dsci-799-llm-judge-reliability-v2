"""
LLM-as-a-Judge Reliability Dashboard (STUB)

"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing from src when dashboard runs from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from compute_metrics import (
    _group_by_item,
    metric1_per_item_variance,
    metric2_exact_agreement,
    metric3_score_histogram,
    otel_metrics,
    variance,
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
tab_names = ["Overview", "Run Experiment", "View Results", "Telemetry"]
tab_names.append("Manage")
tabs = st.tabs(tab_names)

tab_overview, tab_run, tab_view, tab_otel = tabs[0], tabs[1], tabs[2], tabs[3]
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

            # Overall reliability: stable vs unstable
            st.subheader("Overall reliability")
            st.caption(
                "Non-zero variance = judge instability (same response, different scores across repeats)."
            )
            var_data = [{"item_id": i, "variance": variance(s)} for i, s in by_item.items()]
            if var_data:
                stable = sum(1 for v in var_data if v["variance"] == 0)
                unstable = len(var_data) - stable
                summary_df = pd.DataFrame([
                    {"status": "Stable (0 variance)", "items": stable},
                    {"status": "Unstable (>0 variance)", "items": unstable},
                ])
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(
                        summary_df,
                        x="items",
                        y="status",
                        orientation="h",
                        color="status",
                        color_discrete_map={
                            "Stable (0 variance)": "#808080",
                            "Unstable (>0 variance)": "#808080",
                        },
                        pattern_shape="status",
                        pattern_shape_map={
                            "Stable (0 variance)": "",
                            "Unstable (>0 variance)": "/",
                        },
                    )
                    fig.update_layout(
                        xaxis_title="Number of items",
                        yaxis_title="",
                        height=280,
                        showlegend=False,
                        margin=dict(t=20, b=40),
                    )
                    fig.update_traces(marker_pattern_fillmode="overlay")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    # Variance distribution: how many items fall in each variance bucket
                    var_df = pd.DataFrame(var_data)
                    var_df["bucket"] = pd.cut(
                        var_df["variance"],
                        bins=[-0.1, 0, 0.25, 0.5, 1.0, 10],
                        labels=["0", "0–0.25", "0.25–0.5", "0.5–1", "1+"],
                    )
                    bucket_counts = var_df.groupby("bucket", observed=True).size().reset_index(name="count")
                    dist_chart = (
                        alt.Chart(bucket_counts)
                        .mark_bar(color="#808080")
                        .encode(
                            x=alt.X("bucket:N", title="Variance range"),
                            y=alt.Y("count:Q", title="Items"),
                        )
                        .properties(height=280)
                    )
                    st.altair_chart(dist_chart, use_container_width=True)

            # Score distribution
            st.subheader("Score distribution")
            if counts:
                hist_df = pd.DataFrame(
                    {"score": list(counts.keys()), "count": list(counts.values())}
                ).sort_values("score")
                hist_df["score"] = hist_df["score"].astype(str)
                score_chart = (
                    alt.Chart(hist_df)
                    .mark_bar(color="#808080")
                    .encode(
                        x=alt.X(
                            "score:N",
                            title="Score",
                            axis=alt.Axis(labelAngle=-90),
                        ),
                        y=alt.Y("count:Q", title="Count"),
                    )
                    .properties(height=280)
                )
                st.altair_chart(score_chart, use_container_width=True)
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
with tab_otel:
    st.header("Telemetry (OTEL)")
    st.caption(
        "OpenTelemetry data from judge runs: trace/span IDs, token usage, and span status. "
        "Used to validate reliability via token-based metrics."
    )
    jsonl_files = list(RESULTS_DIR.glob("*.jsonl")) if RESULTS_DIR.exists() else []
    if not jsonl_files:
        st.info("No result files. Run the judge pipeline to collect OTEL data.")
    else:
        selected = st.selectbox(
            "Result file",
            [f.name for f in jsonl_files],
            key="otel_file",
        )
        path = RESULTS_DIR / selected
        rows = load_jsonl(path)
        if not rows:
            st.info("File is empty.")
        else:
            otel = otel_metrics(rows)
            if otel.get("has_otel"):
                # ---- Section 1: Run overview ----
                st.subheader("1. Run overview")
                st.caption("Basic stats for this execution. Each API call = one span.")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total spans", otel["total_spans"])
                with col2:
                    st.metric("Span status OK", otel["span_status_ok"])
                with col3:
                    st.metric("Span status Error", otel["span_status_error"])
                with col4:
                    st.metric("Trace IDs", len(otel["trace_ids"]))
                with st.expander("Trace IDs", expanded=False):
                    for tid in otel.get("trace_ids", [])[:20]:
                        st.code(tid, language=None)
                    if len(otel.get("trace_ids", [])) > 20:
                        st.caption(f"... and {len(otel['trace_ids']) - 20} more")

                if otel.get("total_input_tokens") is not None:
                    # ---- Section 2: Across repeats of the same item (RELIABILITY) ----
                    st.subheader("2. Across repeats of the same item")
                    st.info(
                        "**Reliability focus:** Same item, same prompt, K repeats—does the judge behave consistently? "
                        "Output length can vary when the model words justifications differently. Low variance = more stable."
                    )
                    c1, c2, c3 = st.columns(3)
                    tok_var = otel.get("mean_token_variance_per_item")
                    with c1:
                        st.metric("Token length variance (mean)", f"{tok_var:.2f}" if tok_var is not None else "—")
                        st.caption("Lower = more consistent wording across repeats")
                    with c2:
                        st.metric("Total input tokens", f"{otel['total_input_tokens']:,}")
                    with c3:
                        st.metric("Total output tokens", f"{otel['total_output_tokens']:,}")
                    # Per-item within-item variance (reliability detail)
                    per_item = otel.get("per_item_token_details", [])
                    if per_item:
                        rel_df = pd.DataFrame([
                            {"item_id": p["item_id"], "within_item_variance": p["within_item_variance"], "repeats": p["repeats"]}
                            for p in per_item
                        ])
                        st.caption("Per-item: variance in token count across K repeats (0 = perfectly consistent).")
                        st.dataframe(rel_df, use_container_width=True)

                    # ---- Section 3: Across different items ----
                    st.subheader("3. Across different items")
                    st.caption(
                        "Each item = different question & response. Different prompts → different token counts. "
                        "This is expected, not a reliability signal."
                    )
                    if otel.get("between_item_range") is not None:
                        bc1, bc2, bc3 = st.columns(3)
                        with bc1:
                            st.metric("Min tokens (per item)", f"{otel['between_item_min_tokens']:.0f}")
                        with bc2:
                            st.metric("Max tokens (per item)", f"{otel['between_item_max_tokens']:.0f}")
                        with bc3:
                            st.metric("Range (difference)", f"{otel['between_item_range']:.0f}")
                    if per_item:
                        item_df = pd.DataFrame(per_item)
                        st.dataframe(item_df, use_container_width=True)
                        chart_data = []
                        for p in per_item:
                            chart_data.append({"item_id": str(p["item_id"]), "tokens": p["mean_input_tokens"], "type": "Input (prompt)"})
                            chart_data.append({"item_id": str(p["item_id"]), "tokens": p["mean_output_tokens"], "type": "Output (justification)"})
                        tok_fig = px.bar(
                            pd.DataFrame(chart_data),
                            x="item_id",
                            y="tokens",
                            color="type",
                            barmode="group",
                            color_discrete_map={"Input (prompt)": "#606060", "Output (justification)": "#a0a0a0"},
                        )
                        tok_fig.update_layout(
                            xaxis_title="Item",
                            yaxis_title="Mean tokens",
                            height=280,
                            legend_title="Token type",
                        )
                        st.plotly_chart(tok_fig, use_container_width=True)
            else:
                st.info(
                    "This file has no OTEL metadata (trace_id, input_tokens, etc.). "
                    "Run the instrumented judge pipeline to collect OTEL data."
                )


# ==================== TAB 5: Manage ====================
with tab_manage:
    # TODO: Delete experiments  
    
    st.header("Manage")
    st.info("TODO: Implement Manage tab..")
