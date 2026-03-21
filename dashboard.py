"""
LLM-as-a-Judge Reliability Dashboard (STUB)

"""

import json
import sys
from datetime import datetime
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
tab_names = ["Overview", "Run Experiment", "View Results", "Compare Judges", "Telemetry"]
tab_names.append("Manage")
tabs = st.tabs(tab_names)

tab_overview, tab_run, tab_view, tab_compare, tab_otel = tabs[0], tabs[1], tabs[2], tabs[3], tabs[4]
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


# ==================== TAB 4: Compare Judges ==============
with tab_compare:
    st.header("Compare Judges")
    st.caption(
        "Select multiple result files to compare reliability metrics and score distributions across judges."
    )
    jsonl_files = list(RESULTS_DIR.glob("*.jsonl")) if RESULTS_DIR.exists() else []
    if not jsonl_files:
        st.info("No result files. Run experiments with different judges first.")
    else:
        selected = st.multiselect(
            "Result files to compare (select 2 or more)",
            [f.name for f in sorted(jsonl_files, key=lambda p: p.name)],
            default=[],
            key="compare_files",
        )
        if len(selected) < 2:
            st.info("Select at least 2 files to compare.")
        else:
            # Load each file and compute metrics
            compare_data = []
            skipped = []
            for fname in selected:
                path = RESULTS_DIR / fname
                rows = load_jsonl(path)
                if not rows:
                    # Extract judge name from filename (mtbench_judge-XYZ_K...)
                    judge = fname.replace("mtbench_judge-", "").split("_K")[0] if "mtbench_judge-" in fname else fname
                    compare_data.append({
                        "Judge": judge,
                        "File": fname,
                        "Items": 0,
                        "Mean score": "—",
                        "Mean variance": "—",
                        "% zero variance": "—",
                        "Exact agreement": "—",
                    })
                    skipped.append(fname)
                    continue
                judge = rows[0].get("judge_model", fname)
                by_item = _group_by_item(rows)
                m1 = metric1_per_item_variance(by_item)
                m2 = metric2_exact_agreement(by_item)
                mean_score = sum(s for scores in by_item.values() for s in scores) / sum(len(s) for s in by_item.values()) if by_item else 0
                compare_data.append({
                    "Judge": judge,
                    "File": fname,
                    "Items": m1["n_items"],
                    "Mean score": round(mean_score, 2),
                    "Mean variance": round(m1["mean_variance"], 4),
                    "% zero variance": round(m1["pct_items_zero_variance"], 1),
                    "Exact agreement": f"{m2['mean_agreement_rate']:.1%}",
                })

            if skipped:
                st.warning(f"Skipped (empty): {', '.join(skipped)}")
            if compare_data:
                st.subheader("Per-judge metrics")
                st.dataframe(pd.DataFrame(compare_data), use_container_width=True)

                # Reliability chart: % zero variance (higher = more reliable)
                rel_data = [
                    {"judge": r["Judge"], "pct_zero_variance": r["% zero variance"]}
                    for r in compare_data
                    if isinstance(r["% zero variance"], (int, float))
                ]
                if rel_data:
                    st.subheader("Reliability: % items with zero variance")
                    st.caption("Higher = more stable. Same item, same response → identical scores across repeats.")
                    rel_df = pd.DataFrame(rel_data)
                    fig = px.bar(
                        rel_df,
                        x="pct_zero_variance",
                        y="judge",
                        orientation="h",
                    )
                    fig.update_traces(marker_color="#808080")
                    fig.update_layout(
                        xaxis_title="% zero variance",
                        yaxis_title="",
                        xaxis=dict(range=[0, 105]),
                        height=max(200, 60 * len(rel_data)),
                        showlegend=False,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No reliability data to chart (select files with valid judgments).")

                # Score spread per item across judges
                st.subheader("Score spread per item across judges")
                st.caption(
                    "Spread = max(score across judges) − min(score across judges). Higher = more disagreement. "
                    "All compared files must share the same dataset (same item set)."
                )
                # Build item sets and mean scores per judge; validate same dataset across files
                file_item_sets = {}
                spread_by_item: dict = {}
                for fname in selected:
                    if fname in skipped:
                        continue
                    path = RESULTS_DIR / fname
                    rows = load_jsonl(path)
                    judge = rows[0].get("judge_model", fname.replace("mtbench_judge-", "").split("_K")[0]) if rows else fname
                    by_item = _group_by_item(rows)
                    item_ids = frozenset(by_item.keys())
                    file_item_sets[fname] = (len(item_ids), item_ids)
                    for item_id, scores in by_item.items():
                        spread_by_item.setdefault(item_id, {})[judge] = sum(scores) / len(scores)
                # Validate: same number of items and same item_ids across all files
                spread_valid = len(file_item_sets) >= 2
                if spread_valid:
                    ref_count, ref_items = next(iter(file_item_sets.values()))
                    for fname, (n, items) in file_item_sets.items():
                        if n != ref_count or items != ref_items:
                            spread_valid = False
                            break
                if not spread_valid or len(file_item_sets) < 2:
                    st.warning(
                        "All compared files must have the same number of items and represent the same dataset (frozen subset). "
                        "Cannot compute spread across judges."
                    )
                else:
                    # Compute spread per item: max - min across judges
                    spread_rows = []
                    for item_id, judge_scores in spread_by_item.items():
                        if len(judge_scores) >= 2:
                            all_means = list(judge_scores.values())
                            mn, mx = min(all_means), max(all_means)
                            spread = mx - mn
                            spread_rows.append({
                                "item_id": item_id,
                                "spread": spread,
                                "min_score": mn,
                                "max_score": mx,
                                "judge_scores": ", ".join(f"{j}: {s:.1f}" for j, s in sorted(judge_scores.items())),
                            })
                    if spread_rows:
                        spread_df = pd.DataFrame(spread_rows).sort_values("spread", ascending=False)
                        # Summary metrics
                        mean_spread = spread_df["spread"].mean()
                        pct_spread_ge2 = 100 * (spread_df["spread"] >= 2).sum() / len(spread_df)
                        max_spread = spread_df["spread"].max()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean spread", f"{mean_spread:.2f}")
                        with col2:
                            st.metric("% items with spread ≥ 2", f"{pct_spread_ge2:.1f}%")
                        with col3:
                            st.metric("Max spread", f"{max_spread:.0f}")
                        # Bar chart: one bar per item, sorted by spread desc
                        spread_fig = px.bar(
                            spread_df,
                            x="item_id",
                            y="spread",
                        )
                        spread_fig.update_traces(
                            marker_color="#808080",
                            hovertemplate="<b>Item %{x}</b><br>Spread: %{y}<br>Min score: %{customdata[0]:.1f}<br>Max score: %{customdata[1]:.1f}<br>%{customdata[2]}<extra></extra>",
                            customdata=spread_df[["min_score", "max_score", "judge_scores"]].values,
                        )
                        spread_fig.add_hline(y=2, line_color="#bbb", line_width=1)
                        spread_fig.update_layout(
                            xaxis_title="Item",
                            yaxis_title="Spread",
                            yaxis=dict(range=[0, 9]),
                            height=400,
                            showlegend=False,
                            xaxis={"categoryorder": "array", "categoryarray": spread_df["item_id"].tolist()},
                        )
                        st.plotly_chart(spread_fig, use_container_width=True)
                    else:
                        st.info(
                            "Need items scored by at least 2 judges. "
                            "Ensure selected files share common item_ids (e.g. same dataset)."
                        )


# ==================== TAB 2: Run Experiment ==============
with tab_run:
    st.header("Run Experiment")

    judge_options = [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
        "Custom...",
    ]
    judge_choice = st.selectbox(
        "Judge model",
        judge_options,
        index=0,
        key="run_judge",
        help="gpt-* → OpenAI; claude-* → Anthropic. Set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env.",
    )
    if judge_choice == "Custom...":
        judge_choice = st.text_input(
            "Custom judge model",
            value="gpt-4o-mini",
            key="run_judge_custom",
            help="e.g. gpt-4o-mini (OpenAI) or claude-sonnet-4-20250514 (Anthropic)",
        )
    k_choice = st.selectbox("Repeats per item (K)", [2, 3, 5, 10], index=2, key="run_k")
    dataset_choice = st.selectbox(
        "Dataset",
        ["Subset (5 items)", "Full (30 items)"],
        key="run_dataset",
        help="Subset = mt_bench_subset.json. Full = mt_bench_full.json (run build_mt_bench_full.py first if missing).",
    )
    input_path = REPO_ROOT / "data" / ("mt_bench_full.json" if "Full" in dataset_choice else "mt_bench_subset.json")

    if st.button("Run experiment", type="primary", key="run_btn"):
        if "Full" in dataset_choice and not input_path.exists():
            st.error("mt_bench_full.json not found. Run: python src/build_mt_bench_full.py")
        else:
            with st.spinner("Running experiment (calling judge API)..."):
                try:
                    from run_repeated_judging import run_experiment

                    out_path = run_experiment(judge_model=judge_choice, repeats=k_choice, input_path=str(input_path))
                    st.success(f"Done. Output: {out_path}")
                except Exception as e:
                    st.error(str(e))

    st.divider()
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



# ==================== TAB 5: Telemetry (OTEL) ====================
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


# ==================== TAB 6: Manage ====================
with tab_manage:
    st.header("Manage")
    st.caption("View and delete result files from the results directory.")
    jsonl_files = sorted(RESULTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True) if RESULTS_DIR.exists() else []
    if not jsonl_files:
        st.info("No result files.")
    else:
        # Build file table with record counts
        file_info = []
        for path in jsonl_files:
            stat = path.stat()
            try:
                rows = load_jsonl(path)
                n_records = len(rows)
            except Exception:
                n_records = 0
            file_info.append({
                "name": path.name,
                "path": path,
                "records": n_records,
                "size_kb": round(stat.st_size / 1024, 1),
                "mtime": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
        # Table with sticky header (st.data_editor provides it)
        df = pd.DataFrame([
            {"Delete": False, "File": f["name"], "Records": f["records"], "Size (KB)": f["size_kb"], "Modified": f["mtime"]}
            for f in file_info
        ])
        edited = st.data_editor(
            df,
            column_config={
                "Delete": st.column_config.CheckboxColumn("Delete", help="Select to delete", width="small"),
                "File": st.column_config.TextColumn("File", width="large"),
                "Records": st.column_config.NumberColumn("Records", width="small"),
                "Size (KB)": st.column_config.NumberColumn("Size (KB)", width="small"),
                "Modified": st.column_config.TextColumn("Modified", width="medium"),
            },
            disabled=["File", "Records", "Size (KB)", "Modified"],
            hide_index=True,
            use_container_width=True,
            key="manage_file_editor",
        )
        to_delete = [f["path"] for f, row in zip(file_info, edited.itertuples(index=False)) if row.Delete]

        st.divider()
        confirm = False
        if to_delete:
            total_records = sum(f["records"] for f in file_info if f["path"] in to_delete)
            st.warning(f"Selected {len(to_delete)} file(s) ({total_records} records total).")
            confirm = st.checkbox("I confirm I want to delete these files", key="manage_confirm")
        if st.button("Delete", type="primary", key="manage_delete_btn", disabled=len(to_delete) == 0):
            if not confirm:
                st.warning("Check the confirmation box to proceed.")
            else:
                errors = []
                for path in to_delete:
                    try:
                        path.unlink()
                    except Exception as e:
                        errors.append(f"{path.name}: {e}")
                for err in errors:
                    st.error(err)
                if not errors:
                    st.success(f"Deleted {len(to_delete)} file(s).")
                    st.rerun()
