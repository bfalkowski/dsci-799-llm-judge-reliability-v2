"""Streamlit dashboard for LLM-as-a-judge reliability experiments."""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

import altair as alt
import pandas as pd
import plotly.express as px
import streamlit as st

# Allow importing from src when dashboard runs from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from constants import JUDGE_MODEL
from judge import build_rubric_generator_user_prompt, call_text_model, is_claude_model
from metric_rubric import METRIC_GLOSS_DEFAULTS, gloss_for_metric
from run_repeated_judging import (
    CONDITION_FILENAME_SLUG,
    load_judge_metric_prompt,
    load_judge_prompt,
    run_experiment,
)
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

# Filename slug → condition_name (matches run_repeated_judging.CONDITION_FILENAME_SLUG)
_COND_SLUG_TO_NAME = {
    "gen": "generic_overall",
    "metric": "metric_rubric",
    "custom": "per_item_custom",
}


def _first_jsonl_row(path: Path) -> dict:
    try:
        with path.open(encoding=ENCODING) as f:
            for line in f:
                line = line.strip()
                if line:
                    return json.loads(line)
    except Exception:
        pass
    return {}


def _condition_label_for_file(path: Path, first_row: Optional[dict] = None) -> str:
    row = first_row if first_row is not None else _first_jsonl_row(path)
    c = row.get("condition_name")
    if c and isinstance(c, str) and c.strip():
        return c.strip()
    m = re.search(r"_cond-([^_]+)_", path.name)
    if m:
        slug = m.group(1)
        return _COND_SLUG_TO_NAME.get(slug, slug)
    return "(legacy — not in filename or rows)"


def _summarize_result_file(path: Path) -> dict:
    row = _first_jsonl_row(path)
    return {
        "path": path,
        "name": path.name,
        "condition": _condition_label_for_file(path, row),
        "dataset_id": (str(row.get("dataset_id") or "").strip() or "—"),
        "judge_model": str(row.get("judge_model") or "—"),
        "metric_name": row.get("metric_name"),
    }


def _all_result_summaries() -> list:
    if not RESULTS_DIR.exists():
        return []
    out = [_summarize_result_file(p) for p in sorted(RESULTS_DIR.glob("*.jsonl"), key=lambda p: p.name)]
    return out


def _rows_for_single_metric(rows: list, metric_name: Optional[str]):
    """Condition B: restrict rows before grouping; avoids relying on compute_metrics keyword compat."""
    if metric_name is None:
        return rows
    return [r for r in rows if str(r.get("metric_name")) == str(metric_name)]


def _normalize_judge_dataset(data):
    """Ensure each row has item_id, question, response, judge_instructions."""
    if not isinstance(data, list):
        raise ValueError("Dataset file must contain a JSON array of objects.")
    out = []
    for row in data:
        if not isinstance(row, dict):
            continue
        if "item_id" not in row or "question" not in row or "response" not in row:
            continue
        out.append({
            "item_id": str(row["item_id"]),
            "question": str(row.get("question", "")),
            "response": str(row.get("response", "")),
            "judge_instructions": str(row.get("judge_instructions", "") or ""),
        })
    return out


def _discover_dataset_paths():
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("mt_bench*.json"), key=lambda p: p.name)


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


def _load_help_text():
    """Load heading/tooltip copy from dashboard_content/help_text.json. Returns dict, empty on error."""
    path = CONTENT_DIR / "help_text.json"
    if not path.exists():
        return {}
    try:
        with open(path, encoding=ENCODING) as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _help_text(key: str) -> Optional[str]:
    """Non-empty string for st.title/header/subheader `help=`, or None if missing or blank."""
    raw = _help.get(key)
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    return s if s else None


_captions = _load_captions()
_help = _load_help_text()
load_dotenv(REPO_ROOT / ".env", override=False)

# Preset judge / rubric-generator models (gpt-* → OpenAI, claude-* → Anthropic)
JUDGE_MODEL_PRESETS = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4",
    "claude-sonnet-4-20250514",
    "claude-opus-4-20250514",
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
    "Custom...",
]


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
tab_names = [
    "Overview",
    "Dataset & prompts",
    "Run Experiment",
    "View Results",
    "Compare Judges",
    "Telemetry",
    "Manage",
]
tabs = st.tabs(tab_names)

tab_overview, tab_dataset, tab_run, tab_view, tab_compare, tab_otel, tab_manage = (
    tabs[0],
    tabs[1],
    tabs[2],
    tabs[3],
    tabs[4],
    tabs[5],
    tabs[6],
)


# ==================== TAB 1:  Overview ====================
with tab_overview:
    overview_md = _load_content("overview")
    if overview_md:

        st.markdown(overview_md)

    else:
        st.info("Overview content not found. Create dashboard_content/overview.md.")
    st.caption(_captions.get("overview_footer", ""))


# ==================== TAB: Dataset & prompts ====================
with tab_dataset:
    st.header(
        "Dataset & prompts",
        anchor=False,
        help=_help_text("dataset_tab_header"),
    )
    st.caption(
        "View and edit the judge dataset JSON. **judge_instructions** are used for **Per-item custom** runs only; "
        "they are ignored for **Generic overall**. **Metric rubric** uses separate prompts (see previews below)."
    )
    paths = _discover_dataset_paths()
    if not paths:
        st.warning(f"No `mt_bench*.json` files in `{DATA_DIR}`. Add `mt_bench_subset.json` or run `build_mt_bench_full.py`.")
    else:
        labels = [p.name for p in paths]
        choice = st.selectbox("Dataset file", labels, key="dataset_file_select")
        data_path = DATA_DIR / choice

        try:
            raw = data_path.read_text(encoding=ENCODING)
        except Exception as e:
            st.error(str(e))
        else:
            try:
                records = _normalize_judge_dataset(json.loads(raw))
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Invalid dataset: {e}")
            else:
                if not records:
                    st.info("No valid items in file.")
                else:
                    st.caption(f"`{data_path.relative_to(REPO_ROOT)}` — {len(records)} items")
                    _fb = st.session_state.pop("dataset_rubric_feedback", None)
                    if _fb:
                        if _fb.get("success"):
                            st.success(_fb["success"])
                        if _fb.get("warning"):
                            st.warning(_fb["warning"])

                    df = pd.DataFrame(records)
                    edited = st.data_editor(
                        df,
                        column_config={
                            "item_id": st.column_config.TextColumn("item_id", disabled=True, width="small"),
                            "question": st.column_config.TextColumn("question", disabled=True, width="large"),
                            "response": st.column_config.TextColumn("response", disabled=True, width="large"),
                            "judge_instructions": st.column_config.TextColumn(
                                "judge_instructions",
                                help="Per-item instructions prepended to the judge prompt when non-empty.",
                                width="large",
                            ),
                        },
                        hide_index=True,
                        use_container_width=True,
                        key="dataset_prompts_editor",
                        num_rows="fixed",
                    )

                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Save changes to file", type="primary", key="dataset_save_btn"):
                            to_save = []
                            for _, row in edited.iterrows():
                                to_save.append({
                                    "item_id": str(row["item_id"]),
                                    "question": str(row["question"]),
                                    "response": str(row["response"]),
                                    "judge_instructions": str(row.get("judge_instructions", "") or ""),
                                })
                            try:
                                data_path.write_text(
                                    json.dumps(to_save, indent=2, ensure_ascii=False) + "\n",
                                    encoding=ENCODING,
                                )
                                st.success(f"Saved {len(to_save)} items to {data_path.name}")
                            except Exception as e:
                                st.error(str(e))

                    with c2:
                        st.caption("Commit saved files in git so runs stay reproducible.")

                    st.divider()
                    st.subheader(
                        "Generate custom judge instructions",
                        anchor=False,
                        help=_help_text("dataset_generate_rubrics"),
                    )
                    st.caption(
                        "One API call per row using the question and reference response. Fills **judge_instructions** "
                        "with add/deduct scoring guidance for a 0–100 scale. **Re-run overwrites** every row; the file is saved when finished."
                    )
                    rg1, rg2 = st.columns([3, 1])
                    with rg1:
                        _preset_ids = [p for p in JUDGE_MODEL_PRESETS if p != "Custom..."]
                        _env_m = (os.environ.get("JUDGE_MODEL") or JUDGE_MODEL).strip()
                        _rubric_idx = (
                            JUDGE_MODEL_PRESETS.index(_env_m)
                            if _env_m in _preset_ids
                            else JUDGE_MODEL_PRESETS.index("Custom...")
                        )
                        rubric_model_pick = st.selectbox(
                            "Model for rubric generation",
                            JUDGE_MODEL_PRESETS,
                            index=_rubric_idx,
                            key="rubric_gen_model_select",
                            help="Pick a preset or **Custom...** to type any model id (independent of the Run Experiment judge).",
                        )
                        if rubric_model_pick == "Custom...":
                            gen_model = st.text_input(
                                "Custom rubric model id",
                                value=_env_m if _env_m not in _preset_ids else "gpt-4o-mini",
                                key="rubric_gen_model_custom",
                                help="Any OpenAI or Anthropic model string your account supports.",
                            ).strip()
                        else:
                            gen_model = rubric_model_pick
                    with rg2:
                        gen_temp = st.slider("Temp", 0.0, 1.0, 0.2, 0.05, key="rubric_gen_temp")
                    if st.button(
                        "Generate / overwrite custom judges for all items",
                        type="secondary",
                        key="rubric_gen_btn",
                    ):
                        load_dotenv(REPO_ROOT / ".env", override=False)
                        model_id = (gen_model or "").strip() or JUDGE_MODEL
                        key_name = "ANTHROPIC_API_KEY" if is_claude_model(model_id) else "OPENAI_API_KEY"
                        if not (os.environ.get(key_name) or "").strip():
                            st.error(f"{key_name} is not set. Add it to .env for {model_id}.")
                        else:
                            to_save = []
                            errors = []
                            n_items = len(edited)
                            prog = st.progress(0.0, text="Starting…")
                            for j, (_, row) in enumerate(edited.iterrows()):
                                q = str(row["question"])
                                resp = str(row["response"])
                                item_id = str(row["item_id"])
                                try:
                                    up = build_rubric_generator_user_prompt(q, resp)
                                    text, _, _ = call_text_model(
                                        up, model_id, temperature=float(gen_temp)
                                    )
                                    instr = (text or "").strip()
                                except Exception as e:
                                    errors.append((item_id, str(e)))
                                    instr = str(row.get("judge_instructions", "") or "")
                                to_save.append({
                                    "item_id": item_id,
                                    "question": q,
                                    "response": resp,
                                    "judge_instructions": instr,
                                })
                                prog.progress(
                                    (j + 1) / max(n_items, 1),
                                    text=f"Generated {j + 1} / {n_items}",
                                )
                            try:
                                data_path.write_text(
                                    json.dumps(to_save, indent=2, ensure_ascii=False) + "\n",
                                    encoding=ENCODING,
                                )
                            except Exception as e:
                                st.error(str(e))
                            else:
                                fb = {
                                    "success": f"Wrote {len(to_save)} items to {data_path.name} (model {model_id}).",
                                }
                                if errors:
                                    preview = "; ".join(f"{iid}: {msg[:80]}" for iid, msg in errors[:5])
                                    more = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
                                    fb["warning"] = (
                                        f"{len(errors)} item(s) failed (previous instructions kept): {preview}{more}"
                                    )
                                st.session_state["dataset_rubric_feedback"] = fb
                                st.rerun()

                    st.divider()
                    st.subheader(
                        "Judge prompt previews",
                        anchor=False,
                        help=_help_text("dataset_judge_previews_intro"),
                    )
                    st.caption("**A**, **B**, **C** align with **Run Experiment**. Hover **ⓘ** on each blue title for a short explanation.")
                    tmpl = load_judge_prompt()
                    preview_ids = [str(r["item_id"]) for r in records]

                    st.subheader(
                        "A — Generic overall",
                        anchor=False,
                        help=_help_text("dataset_prompt_generic"),
                    )
                    with st.expander("Show template", expanded=False):
                        st.code(tmpl, language=None)

                    st.subheader(
                        "B — Metric rubric",
                        anchor=False,
                        help=_help_text("dataset_metric_prompts"),
                    )
                    with st.expander("Show criterion table & prompt preview", expanded=False):
                        _metric_keys = sorted(METRIC_GLOSS_DEFAULTS.keys())
                        st.caption("**Criterion definitions** (long text scrolls inside the table)")
                        st.dataframe(
                            pd.DataFrame(
                                [{"metric": k, "gloss": METRIC_GLOSS_DEFAULTS[k]} for k in _metric_keys]
                            ),
                            hide_index=True,
                            use_container_width=True,
                            height=240,
                        )
                        _metric_pick = st.selectbox(
                            "Criterion to preview",
                            _metric_keys,
                            key="dataset_metric_criterion",
                        )
                        _mtpl = load_judge_metric_prompt()
                        try:
                            _mgloss = gloss_for_metric(_metric_pick)
                            _m_shape = _mtpl.format(
                                metric_name=_metric_pick,
                                metric_gloss=_mgloss,
                                question="{question}",
                                response="{response}",
                            )
                        except (KeyError, ValueError) as e:
                            _m_shape = f"(Could not build preview: {e})"
                        st.code(_m_shape, language=None)

                    st.subheader(
                        "C — Per-item custom",
                        anchor=False,
                        help=_help_text("dataset_prompt_per_item_custom"),
                    )
                    with st.expander("Show filled prompt for one row", expanded=False):
                        _c_item = st.selectbox(
                            "Row to preview",
                            preview_ids,
                            key="dataset_preview_item_custom",
                            help="Uses this row’s question, response, and judge_instructions (including unsaved edits).",
                        )
                        _crow = edited[edited["item_id"].astype(str) == _c_item].iloc[0]
                        _cq = str(_crow["question"])
                        _cr = str(_crow["response"])
                        _ci = str(_crow.get("judge_instructions", "") or "").strip()
                        _crubric = (
                            f"Item-specific judge instructions:\n{_ci}\n\n"
                            if _ci
                            else ""
                        )
                        try:
                            _filled_c = tmpl.format(
                                question=_cq,
                                response=_cr,
                                item_specific_rubric=_crubric,
                            )
                        except KeyError as e:
                            _filled_c = f"(Template missing placeholder: {e})"
                        st.code(_filled_c, language=None)


# ==================== TAB 2: View Results ====================
with tab_view:
    summaries = _all_result_summaries()
    if not summaries:
        st.info("No result files in results")
    else:
        st.caption(
            "Filter by **condition** and **dataset** so you only see runs from one experiment design at a time "
            "(new runs log `condition_name` on every row and encode `_cond-gen|metric|custom_` in the filename)."
        )
        cond_values = sorted({s["condition"] for s in summaries}, key=str)
        cond_filter = st.selectbox(
            "Condition",
            ["(all)"] + cond_values,
            key="view_filter_condition",
            help="generic_overall = no per-item rubric in prompt; per_item_custom = uses judge_instructions; metric_rubric = metric slice.",
        )
        dset_candidates = sorted(
            {s["dataset_id"] for s in summaries if cond_filter == "(all)" or s["condition"] == cond_filter},
            key=str,
        )
        dset_filter = st.selectbox(
            "Dataset",
            ["(all)"] + dset_candidates,
            key="view_filter_dataset",
        )

        def _view_pick(s):
            if cond_filter != "(all)" and s["condition"] != cond_filter:
                return False
            if dset_filter != "(all)" and s["dataset_id"] != dset_filter:
                return False
            return True

        filtered = [s for s in summaries if _view_pick(s)]
        if not filtered:
            st.warning("No files match these filters.")
        else:
            labels = [
                f"{s['name']}  ·  {s['condition']}  ·  {s['dataset_id']}  ·  {s['judge_model']}"
                for s in filtered
            ]
            label_to_name = {labels[i]: filtered[i]["name"] for i in range(len(filtered))}
            pick_label = st.selectbox("Result file", labels, key="view_results_file_pick")
            selected = label_to_name[pick_label]
            path = RESULTS_DIR / selected
            rows = load_jsonl(path)
            if rows:
                df = pd.DataFrame(rows)
                metric_opts = sorted(
                    {str(r["metric_name"]) for r in rows if r.get("metric_name")},
                    key=str,
                )
                view_metric_filter = None
                if len(metric_opts) == 1:
                    view_metric_filter = metric_opts[0]
                    st.caption(f"Condition **B** — reliability for metric **{view_metric_filter}** only.")
                elif len(metric_opts) > 1:
                    view_metric_filter = st.selectbox(
                        "Metric (this file)",
                        metric_opts,
                        key="view_metric_pick",
                        help="One JSONL row per (item, repeat, metric). Charts use the selected metric.",
                    )

                rows_m = _rows_for_single_metric(rows, view_metric_filter)
                by_item = _group_by_item(rows_m)
                m1 = metric1_per_item_variance(by_item)
                m2 = metric2_exact_agreement(by_item)
                counts = metric3_score_histogram(rows_m)
    
                # Metrics section
                st.subheader("Reliability metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean variance", f"{m1['mean_variance']:.4f}")
                    st.caption("Per-item score variance across K runs")
                with col2:
                    st.metric("Mean within-item SD", f"{m1['mean_within_item_std']:.3f}")
                    st.caption("Average √(per-item variance); same units as score (0–100). 0 = perfectly stable repeats.")
                with col3:
                    st.metric("% zero variance", f"{m1['pct_items_zero_variance']:.1f}%")
                    st.caption(f"{m1['zero_var_count']} / {m1['n_items']} items")
                with col4:
                    st.metric("Repeat agreement (exact)", f"{m2['mean_agreement_rate']:.1%}")
                    st.caption("Per item: fraction of repeat pairs with identical integer score; then mean over items—not the same as mean score.")
    
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
        "Pick **condition**, then **dataset** — only result files that match both are available to compare. "
        "Run the same condition and dataset across judges for a valid comparison."
    )
    summaries_cmp = _all_result_summaries()
    if not summaries_cmp:
        st.info("No result files. Run experiments with different judges first.")
    else:
        cmp_cond_vals = sorted({s["condition"] for s in summaries_cmp}, key=str)
        cond_filter_cmp = st.selectbox(
            "1. Condition",
            cmp_cond_vals,
            index=0,
            key="compare_filter_condition",
            help="Required. Files from other conditions are hidden so you cannot mix A / B / C by accident.",
        )
        dset_cmp_candidates = sorted(
            {s["dataset_id"] for s in summaries_cmp if s["condition"] == cond_filter_cmp},
            key=str,
        )
        if not dset_cmp_candidates:
            st.warning("No runs found for this condition.")
            filtered_cmp = []
            dset_filter_cmp = None
        else:
            dset_filter_cmp = st.selectbox(
                "2. Dataset",
                dset_cmp_candidates,
                index=0,
                key="compare_filter_dataset",
                help="Required. Subset vs full bench must match across compared files.",
            )
            filtered_cmp = [
                s
                for s in summaries_cmp
                if s["condition"] == cond_filter_cmp and s["dataset_id"] == dset_filter_cmp
            ]
        cmp_labels = [
            f"{s['name']}  ·  {s['condition']}  ·  {s['dataset_id']}  ·  {s['judge_model']}"
            for s in filtered_cmp
        ]
        _cmp_label_to_name = {cmp_labels[i]: filtered_cmp[i]["name"] for i in range(len(filtered_cmp))}
        pick_cmp = st.multiselect(
            "3. Result files (select 2+ judges)",
            cmp_labels,
            default=[],
            key="compare_files_pick",
        )
        selected = [_cmp_label_to_name[L] for L in pick_cmp]
        if len(selected) < 2:
            st.info("Select at least 2 files to compare.")
        else:
            metric_sets = [
                {str(r["metric_name"]) for r in load_jsonl(RESULTS_DIR / fn) if r.get("metric_name")}
                for fn in selected
            ]
            compare_metric_choice = None
            skip_compare = False
            each_file_has_metrics = all(len(s) > 0 for s in metric_sets)
            some_have_metrics = any(len(s) > 0 for s in metric_sets)
            if each_file_has_metrics:
                _common = set.intersection(*metric_sets)
                if not _common:
                    st.error(
                        "Each selected file has condition B rows, but no metric_name is shared across all of them."
                    )
                    skip_compare = True
                else:
                    compare_metric_choice = st.selectbox(
                        "Metric (condition B)",
                        sorted(_common),
                        key="compare_b_metric",
                        help="Restrict analysis to this metric.",
                    )
            elif some_have_metrics:
                st.warning(
                    "Some files have metric_name rows and some do not; prefer comparing all-B or all non-B runs."
                )

            if skip_compare:
                st.info("Pick files that share the same metrics to compare condition B results.")
            else:
                # Load each file and compute metrics
                compare_data = []
                skipped = []
                for fname in selected:
                    path = RESULTS_DIR / fname
                    summ = _summarize_result_file(path)
                    rows = load_jsonl(path)
                    if compare_metric_choice is not None:
                        rows = [r for r in rows if str(r.get("metric_name")) == str(compare_metric_choice)]
                    if not rows:
                        # Extract judge name from filename (mtbench_judge-XYZ_K...)
                        judge = fname.replace("mtbench_judge-", "").split("_K")[0] if "mtbench_judge-" in fname else fname
                        compare_data.append({
                            "Condition": summ["condition"],
                            "Dataset": summ["dataset_id"],
                            "Judge": judge,
                            "File": fname,
                            "Items": 0,
                            "Mean score": "—",
                            "Mean variance": "—",
                            "Mean within-item SD": "—",
                            "% zero variance": "—",
                            "Repeat agreement (exact)": "—",
                        })
                        skipped.append(fname)
                        continue
                    judge = rows[0].get("judge_model", fname)
                    by_item = _group_by_item(rows)
                    m1 = metric1_per_item_variance(by_item)
                    m2 = metric2_exact_agreement(by_item)
                    mean_score = sum(s for scores in by_item.values() for s in scores) / sum(len(s) for s in by_item.values()) if by_item else 0
                    compare_data.append({
                        "Condition": summ["condition"],
                        "Dataset": summ["dataset_id"],
                        "Judge": judge,
                        "File": fname,
                        "Items": m1["n_items"],
                        "Mean score": round(mean_score, 2),
                        "Mean variance": round(m1["mean_variance"], 4),
                        "Mean within-item SD": round(m1["mean_within_item_std"], 4),
                        "% zero variance": round(m1["pct_items_zero_variance"], 1),
                        "Repeat agreement (exact)": f"{m2['mean_agreement_rate']:.1%}",
                    })

                if skipped:
                    st.warning(f"Skipped (empty): {', '.join(skipped)}")
                if compare_data:
                    st.subheader("Per-judge metrics")
                    st.dataframe(pd.DataFrame(compare_data), use_container_width=True)
                    st.caption(
                        "**Mean score** is the average of all valid **integer** judgments (items × repeats), so decimals are normal. "
                        "**Mean within-item SD** is the average of √(per-item variance) — typical spread of repeat scores in **points** on the 0–100 scale. "
                        "**Repeat agreement (exact)** is separate: fraction of repeat pairs that match exactly, averaged over items."
                    )

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
                    file_item_sets = {}
                    spread_by_item: dict = {}
                    for fname in selected:
                        if fname in skipped:
                            continue
                        path = RESULTS_DIR / fname
                        rows = load_jsonl(path)
                        if compare_metric_choice is not None:
                            rows = [r for r in rows if str(r.get("metric_name")) == str(compare_metric_choice)]
                        judge = rows[0].get("judge_model", fname.replace("mtbench_judge-", "").split("_K")[0]) if rows else fname
                        by_item = _group_by_item(rows)
                        item_ids = frozenset(by_item.keys())
                        file_item_sets[fname] = (len(item_ids), item_ids)
                        for item_id, scores in by_item.items():
                            spread_by_item.setdefault(item_id, {})[judge] = sum(scores) / len(scores)
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
                            _ymax = max(float(spread_df["spread"].max()), 5.0) * 1.08
                            spread_fig.update_layout(
                                xaxis_title="Item",
                                yaxis_title="Spread",
                                yaxis=dict(range=[0, _ymax]),
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
    st.caption(
        "Each run writes `condition_name`, `dataset_id`, and score range on every JSONL row, and the filename includes "
        "`_cond-gen_`, `_cond-custom_`, or `_cond-metric_`. Use **View Results** / **Compare Judges** filters to view "
        "generic vs per-item-custom vs metric runs separately."
    )

    judge_choice = st.selectbox(
        "Judge model",
        JUDGE_MODEL_PRESETS,
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
    k_choice = st.selectbox(
        "Repeats per item (K)",
        [2, 3, 5, 10],
        index=2,
        key="run_k",
        help="Default paper setting: K=5.",
    )
    temp_choice = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        key="run_temp",
        help="Default: 0 for reproducibility. Higher values increase randomness.",
    )
    condition_labels = {
        "Generic overall": "generic_overall",
        "Per-item custom": "per_item_custom",
        "Metric rubric": "metric_rubric",
    }
    condition_display = st.selectbox(
        "Condition",
        list(condition_labels.keys()),
        index=0,
        key="run_condition",
        help="Logged as condition_name on each row. Generic ignores judge_instructions; custom uses them; metric rubric runs one judge call per metric per item per repeat (B2).",
    )
    condition_name = condition_labels[condition_display]
    metrics_pick_run: list = []
    if condition_name == "metric_rubric":
        _metric_options = sorted(METRIC_GLOSS_DEFAULTS.keys())
        _default_metrics = [m for m in ("accuracy", "helpfulness", "relevance") if m in METRIC_GLOSS_DEFAULTS]
        metrics_pick_run = st.multiselect(
            "Metrics",
            options=_metric_options,
            default=_default_metrics,
            key="run_metrics_multiselect",
            help="Select one or more dimensions from metric_rubric.py. One judge call per item × repeat × metric.",
        )
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
            metric_names_arg = None
            block_run = False
            if condition_name == "metric_rubric":
                metric_names_arg = list(metrics_pick_run)
                if not metric_names_arg:
                    st.error("Condition B requires at least one metric — select one or more in the Metrics list.")
                    block_run = True
            if not block_run and (condition_name != "metric_rubric" or metric_names_arg):
                with st.spinner("Running experiment (calling judge API)..."):
                    try:
                        out_path = run_experiment(
                            judge_model=judge_choice,
                            repeats=k_choice,
                            input_path=str(input_path),
                            temperature=float(temp_choice),
                            condition_name=condition_name,
                            metric_names=metric_names_arg,
                            dataset_id=input_path.stem,
                        )
                        _slug = CONDITION_FILENAME_SLUG.get(
                            condition_name,
                            condition_name.replace("_", "")[:12],
                        )
                        st.success(f"Done. Output: {out_path}")
                        st.info(
                            f"Tagged **{condition_name}** · dataset **{input_path.stem}** · look for `_cond-{_slug}_` in the "
                            "filename. On **View Results** / **Compare Judges**, set Condition (and Dataset) filters to this run."
                        )
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
