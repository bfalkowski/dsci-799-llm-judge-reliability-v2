"""Streamlit dashboard for LLM-as-a-judge reliability experiments."""

import json
import os
import re
import sys
from collections import Counter
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
from constants import JUDGE_MODEL, JUDGE_MODEL_BATCH_PRESETS
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
    metric_repeat_variability_headlines,
    otel_metrics,
    variance,
)
from utils import ENCODING, REPO_ROOT, load_jsonl
RESULTS_DIR = REPO_ROOT / "results"
DATA_DIR = REPO_ROOT / "data"
CONTENT_DIR = REPO_ROOT / "dashboard_content"

# Run tab: show raw JSONL preview only after a completed run (cleared when a new run starts).
_RUN_RAW_PREVIEW_PATH_KEY = "run_experiment_raw_preview_path"

# Filename slug → condition_name (matches run_repeated_judging.CONDITION_FILENAME_SLUG)
_COND_SLUG_TO_NAME = {
    "gen": "generic_overall",
    "metric": "metric_rubric",
    "custom": "per_item_custom",
}
# Run Experiment tab: radio labels → condition_name (matches Dataset A / B / C wording)
RUN_CONDITION_LABEL_TO_NAME = {
    "A — Generic overall": "generic_overall",
    "B — Metric rubric": "metric_rubric",
    "C — Per-item custom": "per_item_custom",
}

# Run Experiment: output destination (radio value → must match key= storage)
_RUN_OUTPUT_NEW_JSONL = "new_jsonl"
_RUN_OUTPUT_RESUME_JSONL = "resume_jsonl"


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
    multi_flag = bool(row.get("multi_judge_run"))
    if not multi_flag and "multi" in path.name and "judges" in path.name:
        multi_flag = True
    return {
        "path": path,
        "name": path.name,
        "condition": _condition_label_for_file(path, row),
        "dataset_id": (str(row.get("dataset_id") or "").strip() or "—"),
        "judge_model": str(row.get("judge_model") or "—"),
        "metric_name": row.get("metric_name"),
        "multi_judge_run": multi_flag,
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


def _unique_judge_models_in_rows(rows: list) -> list:
    """Preserve first-seen order of distinct judge_model values (non-empty strings)."""
    seen: list = []
    for r in rows:
        j = r.get("judge_model")
        if j is None:
            continue
        sj = str(j).strip()
        if sj and sj not in seen:
            seen.append(sj)
    return seen


def _rows_for_judge_model(rows: list, judge_model: str) -> list:
    jm = str(judge_model).strip()
    return [r for r in rows if str(r.get("judge_model", "")).strip() == jm]


def _short_run_tag_from_results_filename(fname: str) -> str:
    """
    Compact run id from results filename: UTC stamp plus optional condition slug and K
    (e.g. gen · K2 · 20260404T184805) so tables and charts do not show the full basename.
    """
    name = Path(fname).name
    m_time = re.search(r"_(\d{8}T\d{6})\.jsonl$", name)
    stamp = m_time.group(1) if m_time else ""
    m_cond = re.search(r"_cond-([^_]+)_", name)
    cond = (m_cond.group(1) if m_cond else "").strip()
    m_k = re.search(r"_K(\d+)_", name)
    k_part = f"K{m_k.group(1)}" if m_k else ""
    parts = [p for p in (cond, k_part, stamp) if p]
    if parts:
        return " · ".join(parts)
    stem = Path(name).stem
    if stem.startswith("mtbench_judge-"):
        stem = stem[len("mtbench_judge-") :]
    return stem[:40] + ("…" if len(stem) > 40 else "")


def _compare_slice_label(fname: str, judge_key: str) -> str:
    """Disambiguate duplicate model ids: model plus short run tag from filename."""
    tag = _short_run_tag_from_results_filename(fname)
    return f"{judge_key} · {tag}"


def _compare_result_file_pick_label(s: dict) -> str:
    """Short label for Compare tab file multiselect (no long basename)."""
    tag = _short_run_tag_from_results_filename(s["name"])
    mj = " · multi-judge" if s.get("multi_judge_run") else ""
    return f"{tag}{mj}  ·  {s['condition']}  ·  {s['dataset_id']}"


def _disambiguate_compare_slice_labels(slices: list) -> None:
    """Prefer bare model id (`gpt-4o-mini`); add run tag only when the same model appears on multiple slices."""
    if not slices:
        return
    counts = Counter(str(s["judge_key"]).strip() for s in slices)
    for s in slices:
        jk = str(s["judge_key"]).strip()
        if counts[jk] > 1:
            s["label"] = _compare_slice_label(s["fname"], jk)
        else:
            s["label"] = jk


def _api_vendor_label(judge_key: str) -> str:
    """Bucket for cross-provider comparison (batch runs mix OpenAI and Anthropic)."""
    if is_claude_model(judge_key):
        return "Anthropic"
    return "OpenAI"


def _iter_judge_slices_for_compare(
    fname: str,
    rows: list,
):
    """
    Yield (judge_key, slice_rows) for one file after metric filter.
    judge_key is used for labeling; slice_rows are only that judge's rows.
    """
    judges = _unique_judge_models_in_rows(rows)
    if not judges:
        if not rows:
            return
        jb = (
            fname.replace("mtbench_judge-", "").split("_K")[0]
            if "mtbench_judge-" in fname
            else fname
        )
        yield jb, rows
        return
    for j in judges:
        part = _rows_for_judge_model(rows, j)
        if part:
            yield j, part


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

# Run Experiment: single judge or “all presets” into one JSONL (order matches constants).
RUN_ALL_JUDGES_LABEL = "Run all preset judges (one JSONL file)"
JUDGE_MODEL_PRESETS = list(JUDGE_MODEL_BATCH_PRESETS) + [RUN_ALL_JUDGES_LABEL]
# Dataset tab rubric generator: same model ids + custom id field.
RUBRIC_GEN_MODEL_PRESETS = list(JUDGE_MODEL_BATCH_PRESETS) + ["Custom..."]


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
    "Compare judges & vendors",
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
                        _preset_ids = [p for p in RUBRIC_GEN_MODEL_PRESETS if p != "Custom..."]
                        _env_m = (os.environ.get("JUDGE_MODEL") or JUDGE_MODEL).strip()
                        _rubric_idx = (
                            RUBRIC_GEN_MODEL_PRESETS.index(_env_m)
                            if _env_m in _preset_ids
                            else RUBRIC_GEN_MODEL_PRESETS.index("Custom...")
                        )
                        rubric_model_pick = st.selectbox(
                            "Model for rubric generation",
                            RUBRIC_GEN_MODEL_PRESETS,
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
                        _metric_keys = list(METRIC_GLOSS_DEFAULTS.keys())
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
            "One JSONL can include several **judge_model** values — pick a model below so metrics use only that slice. "
            "For OpenAI vs Anthropic rollups across judges, use **Compare judges & vendors**."
        )
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
                judges_in_file = _unique_judge_models_in_rows(rows)
                if len(judges_in_file) > 1:
                    _vj = st.selectbox(
                        "Judge model (required for multi-judge files)",
                        judges_in_file,
                        key=f"view_judge_model__{selected}",
                        help=(
                            "This file contains multiple **judge_model** values. All metrics and charts use **only** "
                            "the selected model’s rows."
                        ),
                    )
                    rows = _rows_for_judge_model(rows, _vj)
                elif len(judges_in_file) == 1:
                    rows = _rows_for_judge_model(rows, judges_in_file[0])

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
                df = pd.DataFrame(rows_m)
                by_item = _group_by_item(rows_m)
                hl = metric_repeat_variability_headlines(by_item)
                m1 = metric1_per_item_variance(by_item)
                m2 = metric2_exact_agreement(by_item)
                counts = metric3_score_histogram(rows_m)

                st.subheader("Repeat variability")
                if hl["n_judgments"]:
                    st.markdown(
                        f"**{hl['n_distinct_scores']} different scores** across **{hl['n_judgments']}** judgments on the same items. "
                        f"**Within-item** repeat pairs disagreed **{hl['pct_repeat_pairs_disagree']:.0f}%** of the time "
                        f"**({hl['n_repeat_pairs_disagree']}/{hl['n_repeat_pairs']} pairs)**. "
                        f"**{hl['pct_items_any_repeat_disagree']:.0f}%** of items **({hl['n_items_any_repeat_disagree']}/{hl['n_items']})** "
                        f"had at least one disagreeing repeat pair."
                    )
                    c0a, c0b, c0c = st.columns(3)
                    with c0a:
                        st.metric(
                            "Distinct scores / judgments",
                            f"{hl['n_distinct_scores']} / {hl['n_judgments']}",
                        )
                    with c0b:
                        st.metric("Repeat pairs disagreeing", f"{hl['pct_repeat_pairs_disagree']:.0f}%")
                        st.caption(
                            f"**{hl['n_repeat_pairs_disagree']}** / **{hl['n_repeat_pairs']}** within-item pairs differ."
                        )
                    with c0c:
                        st.metric("Items with any repeat disagreement", f"{hl['pct_items_any_repeat_disagree']:.0f}%")
                        st.caption(
                            f"**{hl['n_items_any_repeat_disagree']}** / **{hl['n_items']}** items (max ≠ min repeats)."
                        )
                else:
                    st.info("No scored judgments in this slice.")
    
                # Metrics section
                st.subheader("Reliability metrics (detail)")
                st.caption(
                    "**Mean variance** averages per-item **sample** variances (divide by K−1 within each item). "
                    "With **K=2**, one wild item can inflate the mean — use **Repeat variability** above and **Avg spread** below."
                )
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Avg score spread (points)", f"{m1['mean_within_item_range']:.2f}")
                    st.caption("Mean of max−min repeat scores per item (0–100 scale). With K=2 = mean |Δscore|.")
                with col2:
                    st.metric("Mean variance", f"{m1['mean_variance']:.4f}")
                    st.caption(f"Median: **{m1['median_variance']:.2f}** · worst item spread: **{m1['max_within_item_range']:.0f}** pts")
                with col3:
                    st.metric("Mean within-item SD", f"{m1['mean_within_item_std']:.3f}")
                    st.caption("Mean √(per-item sample variance); same units as score.")
                with col4:
                    st.metric("% zero variance", f"{m1['pct_items_zero_variance']:.1f}%")
                    st.caption(f"{m1['zero_var_count']} / {m1['n_items']} items · identical repeats")
                col5, col6 = st.columns(2)
                with col5:
                    st.metric("Repeat agreement (exact)", f"{m2['mean_agreement_rate']:.1%}")
                    st.caption("Per item: share of repeat pairs with same integer score; mean over items.")
                with col6:
                    st.metric("Items analyzed", f"{m1['n_items']}")
                    st.caption("Distinct item_ids in this slice")
    
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
    
                with st.expander("Judgments reflected above", expanded=False):
                    st.caption(
                        "Same rows as the **metrics and charts**: selected **judge model** and (under condition **B**) the "
                        "chosen **metric** only — not the full JSONL."
                    )
                    if df.empty:
                        st.info("No rows after filters.")
                    else:
                        st.dataframe(df, use_container_width=True)

            else:
                st.info("File is empty.")


# ==================== TAB 4: Compare judges & vendors ==============
with tab_compare:
    st.header("Compare judges & vendors")
    st.caption(
        "Pick **condition** and **dataset**, then one or more result files. Tables split by **judge_model**; "
        "the **API vendor** block aggregates OpenAI vs Anthropic judges when both appear in your selection."
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
        _cmp_label_to_name = {}
        cmp_labels = []
        for s in filtered_cmp:
            base = _compare_result_file_pick_label(s)
            label = base
            n = 2
            while label in _cmp_label_to_name:
                label = f"{base} ({n})"
                n += 1
            _cmp_label_to_name[label] = s["name"]
            cmp_labels.append(label)
        pick_cmp = st.multiselect(
            "3. Result files",
            cmp_labels,
            default=[],
            key="compare_files_pick",
            help=(
                "Pick **one** multi-judge JSONL to compare judges inside it, or **two or more** files (any mix). "
                "Each file is loaded **once**; rows are split by **judge_model** so variance and agreement stay within-judge."
            ),
        )
        selected = [_cmp_label_to_name[L] for L in pick_cmp]
        if len(selected) < 1:
            st.info("Select at least one result file.")
        else:
            loaded_cmp = {fn: load_jsonl(RESULTS_DIR / fn) for fn in selected}
            metric_sets = [
                {str(r["metric_name"]) for r in loaded_cmp[fn] if r.get("metric_name")}
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
                compare_slices = []
                for fname in selected:
                    path = RESULTS_DIR / fname
                    summ = _summarize_result_file(path)
                    rows_metric = _rows_for_single_metric(loaded_cmp[fname], compare_metric_choice)
                    for judge_key, slice_rows in _iter_judge_slices_for_compare(fname, rows_metric):
                        compare_slices.append({
                            "fname": fname,
                            "summ": summ,
                            "judge_key": judge_key,
                            "label": str(judge_key).strip(),
                            "rows": slice_rows,
                        })

                nonempty = [s for s in compare_slices if s["rows"]]
                _disambiguate_compare_slice_labels(nonempty)
                empty_files = [fn for fn in selected if not _rows_for_single_metric(loaded_cmp[fn], compare_metric_choice)]

                if len(nonempty) < 2:
                    st.info(
                        "Need **at least two judge groups** to compare: e.g. two files, or **one multi-judge** file with "
                        "two or more models (after the metric filter, if condition **B**)."
                    )
                    if empty_files:
                        st.warning(
                            "No rows after filters: "
                            + ", ".join(_short_run_tag_from_results_filename(f) for f in empty_files)
                        )
                else:
                    compare_data = []
                    for s in nonempty:
                        summ = s["summ"]
                        rows = s["rows"]
                        by_item = _group_by_item(rows)
                        hl_s = metric_repeat_variability_headlines(by_item)
                        m1 = metric1_per_item_variance(by_item)
                        m2 = metric2_exact_agreement(by_item)
                        mean_score = (
                            sum(sc for scores in by_item.values() for sc in scores)
                            / sum(len(scores) for scores in by_item.values())
                            if by_item
                            else 0
                        )
                        compare_data.append({
                            "Condition": summ["condition"],
                            "Dataset": summ["dataset_id"],
                            "Judge": s["label"],
                            "Items": m1["n_items"],
                            "Distinct scores / judgments": (
                                f"{hl_s['n_distinct_scores']} / {hl_s['n_judgments']}"
                                if hl_s["n_judgments"]
                                else "—"
                            ),
                            "% repeat pairs differ": (
                                round(hl_s["pct_repeat_pairs_disagree"], 1)
                                if hl_s["n_repeat_pairs"]
                                else "—"
                            ),
                            "% items any repeat disagree": (
                                round(hl_s["pct_items_any_repeat_disagree"], 1)
                                if hl_s["n_items"]
                                else "—"
                            ),
                            "Mean score": round(mean_score, 2),
                            "Avg spread (pts)": round(m1["mean_within_item_range"], 2),
                            "Mean variance": round(m1["mean_variance"], 4),
                            "Median var": round(m1["median_variance"], 4),
                            "Mean within-item SD": round(m1["mean_within_item_std"], 4),
                            "% zero variance": round(m1["pct_items_zero_variance"], 1),
                            "Repeat agreement (exact)": f"{m2['mean_agreement_rate']:.1%}",
                        })

                    st.subheader("Per-judge metrics")
                    st.dataframe(pd.DataFrame(compare_data), use_container_width=True)
                    st.caption(
                        "**Distinct scores / judgments**, **% repeat pairs differ**, and **% items any repeat disagree** are all **within-item** summaries (not a pooled min/max across all scores). "
                        "**Judge** is the model id (plus a run tag only if the same model appears twice)."
                    )

                    vendor_agg: dict = {}
                    for s in nonempty:
                        by_item_v = _group_by_item(s["rows"])
                        if not by_item_v:
                            continue
                        m1v = metric1_per_item_variance(by_item_v)
                        m2v = metric2_exact_agreement(by_item_v)
                        vlabel = _api_vendor_label(s["judge_key"])
                        vendor_agg.setdefault(vlabel, []).append({
                            "pct_zero": m1v["pct_items_zero_variance"],
                            "mean_w_std": m1v["mean_within_item_std"],
                            "mean_var": m1v["mean_variance"],
                            "agree": m2v["mean_agreement_rate"],
                            "n_items": m1v["n_items"],
                        })

                    show_vendors = {k for k in vendor_agg if vendor_agg[k]}
                    if len(show_vendors) >= 2:
                        st.subheader("API vendor summary")
                        st.caption(
                            "Item-**weighted** averages across judge models in each API bucket (OpenAI vs Anthropic). "
                            "Higher **% zero variance** and **repeat agreement** indicates more stable repeats here; "
                            "**lower within-item SD** means tighter repeat spread across K."
                        )
                        v_rows = []
                        for vname in sorted(show_vendors):
                            parts = vendor_agg[vname]
                            wtot = sum(p["n_items"] for p in parts)
                            if wtot <= 0:
                                continue
                            w_pct = sum(p["pct_zero"] * p["n_items"] for p in parts) / wtot
                            w_std = sum(p["mean_w_std"] * p["n_items"] for p in parts) / wtot
                            w_agr = sum(p["agree"] * p["n_items"] for p in parts) / wtot
                            w_var = sum(p["mean_var"] * p["n_items"] for p in parts) / wtot
                            v_rows.append({
                                "API": vname,
                                "Models": len(parts),
                                "Weighted avg % zero variance": round(w_pct, 2),
                                "Weighted avg within-item SD": round(w_std, 4),
                                "Weighted avg repeat agreement": round(w_agr, 4),
                                "Weighted avg mean variance": round(w_var, 4),
                            })
                        if v_rows:
                            st.dataframe(pd.DataFrame(v_rows), use_container_width=True, hide_index=True)
                            vdf = pd.DataFrame(v_rows)
                            c1, c2 = st.columns(2)
                            with c1:
                                fig_v = px.bar(
                                    vdf,
                                    x="API",
                                    y="Weighted avg % zero variance",
                                    color="API",
                                    color_discrete_map={"OpenAI": "#10a37f", "Anthropic": "#d4a574"},
                                )
                                fig_v.update_layout(
                                    yaxis_title="% zero variance (weighted)",
                                    yaxis=dict(range=[0, 105]),
                                    showlegend=False,
                                    height=320,
                                )
                                st.plotly_chart(fig_v, use_container_width=True)
                            with c2:
                                fig_w = px.bar(
                                    vdf,
                                    x="API",
                                    y="Weighted avg within-item SD",
                                    color="API",
                                    color_discrete_map={"OpenAI": "#10a37f", "Anthropic": "#d4a574"},
                                )
                                fig_w.update_layout(
                                    yaxis_title="Within-item SD (lower = tighter repeats)",
                                    showlegend=False,
                                    height=320,
                                )
                                st.plotly_chart(fig_w, use_container_width=True)
                    elif len(show_vendors) == 1:
                        only = next(iter(show_vendors))
                        st.caption(
                            f"**Vendor comparison:** only **{only}** judges appear in this selection — add results from "
                            "the other provider (e.g. use **Run all preset judges**) to compare OpenAI vs Anthropic."
                        )

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

                    st.subheader("Score spread per item across judges")
                    st.caption(
                        "Spread = max(mean score across judge groups) − min. Each group is one **`judge_model · file`** slice. "
                        "Requires the **same set of item_ids** (and same dataset) across all slices."
                    )
                    spread_by_item: dict = {}
                    slice_item_sets: list = []
                    for s in nonempty:
                        by_item = _group_by_item(s["rows"])
                        item_ids = frozenset(by_item.keys())
                        slice_item_sets.append((s["label"], len(item_ids), item_ids))
                        for item_id, scores in by_item.items():
                            spread_by_item.setdefault(item_id, {})[s["label"]] = sum(scores) / len(scores)

                    spread_valid = len(slice_item_sets) >= 2
                    if spread_valid:
                        _, ref_count, ref_items = slice_item_sets[0]
                        for _lbl, n, items in slice_item_sets:
                            if n != ref_count or items != ref_items:
                                spread_valid = False
                                break
                    if not spread_valid or len(slice_item_sets) < 2:
                        st.warning(
                            "All compared judge groups must share the **same item_ids** (same dataset / design). "
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
                                    "judge_scores": ", ".join(f"{j}: {v:.1f}" for j, v in sorted(judge_scores.items())),
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
                                "Need at least two judge groups with overlapping items. "
                                "Check **judge_model** splits and shared **item_id**s."
                            )


# ==================== TAB 2: Run Experiment ==============
with tab_run:
    st.header(
        "Run Experiment",
        anchor=False,
        help=_help_text("run_tab_header"),
    )
    st.caption(_help_text("run_tab_caption") or "")

    st.subheader(
        "1 — Dataset & model",
        anchor=False,
        help=_help_text("run_section_setup"),
    )
    dataset_choice = st.selectbox(
        "Dataset",
        ["Subset (5 items)", "Full (30 items)"],
        key="run_dataset",
        help="Subset → `data/mt_bench_subset.json`. Full → `data/mt_bench_full.json` (build it first if missing).",
    )
    input_path = REPO_ROOT / "data" / ("mt_bench_full.json" if "Full" in dataset_choice else "mt_bench_subset.json")

    judge_choice = st.selectbox(
        "Judge model",
        JUDGE_MODEL_PRESETS,
        index=0,
        key="run_judge",
        help="One model, or **Run all preset judges** to loop every preset and append every row to **one** JSONL (needs **both** API keys if you mix OpenAI and Claude).",
    )
    if judge_choice == RUN_ALL_JUDGES_LABEL:
        st.warning(
            f"**Batch run:** {len(JUDGE_MODEL_BATCH_PRESETS)} models × your dataset × K (× metrics under **B**). "
            "Ensure **OPENAI_API_KEY** and **ANTHROPIC_API_KEY** are set if you use both families. "
            "Each output line includes **`judge_model`** and **`multi_judge_run`: true**."
        )
    k_choice = st.selectbox(
        "Repeats per item (K)",
        [2, 3, 5, 10],
        index=2,
        key="run_k",
        help=_help_text("run_k_repeats")
        or "Each item is judged K times; round-robin order across items. Total calls scale with K (and metrics under **B**).",
    )
    temp_choice = st.number_input(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.0,
        step=0.1,
        key="run_temp",
        help="0 = deterministic as the API allows; higher = more randomness.",
    )

    st.divider()
    st.subheader(
        "2 — Condition (A / B / C)",
        anchor=False,
        help=_help_text("run_section_condition"),
    )
    _cond_pick = st.radio(
        "Choose condition",
        list(RUN_CONDITION_LABEL_TO_NAME.keys()),
        key="run_condition_radio",
        horizontal=True,
        help=_help_text("run_condition_radio"),
    )
    condition_name = RUN_CONDITION_LABEL_TO_NAME[_cond_pick]

    st.divider()
    st.subheader(
        "3 — Options for this condition",
        anchor=False,
        help=_help_text("run_section_options"),
    )
    metrics_pick_run: list = []
    if condition_name == "metric_rubric":
        _metric_options = list(METRIC_GLOSS_DEFAULTS.keys())
        metrics_pick_run = st.multiselect(
            "Metrics to score",
            options=_metric_options,
            default=_metric_options,
            key="run_metrics_multiselect",
            help="Each selected metric adds a separate judge call for every item × repeat. Total calls ≈ items × K × (number of metrics).",
        )
        st.info(
            "**B — Metric rubric:** tune the list above. Preview prompts under **Dataset & prompts → B**."
        )
    elif condition_name == "generic_overall":
        st.info(
            "**A — Generic overall:** no extra options. **judge_instructions** in the JSON are **not** used. "
            "Edit the file on **Dataset & prompts**."
        )
    else:
        st.info(
            "**C — Per-item custom:** each row’s **judge_instructions** are injected into the judge prompt. "
            "Fill them on **Dataset & prompts** (or use **Generate custom judge instructions**)."
        )

    st.subheader(
        "Total records (this run)",
        anchor=False,
        help=_help_text("run_total_records_formula"),
    )

    run_output_mode = st.radio(
        "Results output",
        options=[_RUN_OUTPUT_NEW_JSONL, _RUN_OUTPUT_RESUME_JSONL],
        format_func=lambda mode: (
            "New JSONL file (timestamped under results/)"
            if mode == _RUN_OUTPUT_NEW_JSONL
            else "Resume partial JSONL (append missing rows)"
        ),
        key="run_output_mode_radio",
        help=_help_text("run_output_mode"),
    )
    resume_partial = run_output_mode == _RUN_OUTPUT_RESUME_JSONL

    resume_path_arg: Optional[str] = None
    jsonl_for_resume: list = []
    if resume_partial:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        jsonl_for_resume = sorted(
            RESULTS_DIR.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if jsonl_for_resume:
            labels = [p.name for p in jsonl_for_resume]
            pick = st.selectbox(
                "File to resume",
                options=labels,
                key="run_resume_select",
                help=_help_text("run_resume_file_pick"),
            )
            st.caption(_help_text("run_resume_partial"))
            custom_resume = st.text_input(
                "Or explicit path (optional; overrides dropdown)",
                value="",
                key="run_resume_custom_path",
                help="Absolute path, or a filename under **results/**.",
            ).strip()
            if custom_resume:
                cand = Path(custom_resume)
                resume_path_arg = str(cand if cand.is_absolute() else (RESULTS_DIR / cand.name))
            else:
                resume_path_arg = str((RESULTS_DIR / pick).resolve())
        else:
            st.warning("No `.jsonl` files in **results/** yet. Choose **New JSONL file** or add a partial file first.")

    st.divider()
    if st.button("Run experiment", type="primary", key="run_btn"):
        if "Full" in dataset_choice and not input_path.exists():
            st.error("mt_bench_full.json not found. Run: python src/build_mt_bench_full.py")
        elif resume_partial and not jsonl_for_resume:
            st.error(
                "**Resume partial JSONL** is selected but **results/** has no `.jsonl` files. "
                "Switch to **New JSONL file** or add a partial run under **results/**."
            )
        else:
            metric_names_arg = None
            block_run = False
            if condition_name == "metric_rubric":
                metric_names_arg = list(metrics_pick_run)
                if not metric_names_arg:
                    st.error(
                        "**Metric rubric** needs at least one metric — select one or more under **Metrics to score**."
                    )
                    block_run = True
            if not block_run and (condition_name != "metric_rubric" or metric_names_arg):
                st.session_state[_RUN_RAW_PREVIEW_PATH_KEY] = None
                progress_ph = st.progress(0)
                cap_ph = st.caption("Preparing run…")

                def _experiment_progress(done: int, total: int) -> None:
                    if total <= 0:
                        return
                    p = min(float(done) / float(total), 1.0)
                    progress_ph.progress(p)
                    cap_ph.caption(f"Judgments written: **{done}** / **{total}**")

                try:
                    _run_kw = dict(
                        repeats=k_choice,
                        input_path=str(input_path),
                        temperature=float(temp_choice),
                        condition_name=condition_name,
                        metric_names=metric_names_arg,
                        dataset_id=input_path.stem,
                        progress_callback=_experiment_progress,
                    )
                    if resume_path_arg:
                        _run_kw["resume_path"] = resume_path_arg
                    if judge_choice == RUN_ALL_JUDGES_LABEL:
                        result = run_experiment(
                            judge_models=list(JUDGE_MODEL_BATCH_PRESETS),
                            **_run_kw,
                        )
                    else:
                        result = run_experiment(
                            judge_model=judge_choice,
                            **_run_kw,
                        )
                    out_path = result["output_path"]
                    exp_n = result["expected_rows"]
                    got_n = result["written_rows"]
                    progress_ph.progress(1.0)
                    cap_ph.caption(f"Complete: **{got_n}** / **{exp_n}** records (validated).")
                    _slug = CONDITION_FILENAME_SLUG.get(
                        condition_name,
                        condition_name.replace("_", "")[:12],
                    )
                    st.success(
                        f"Done. Output: `{out_path}` — **{got_n}** records written "
                        f"(expected **{exp_n}**; counts match)."
                    )
                    if result.get("resumed"):
                        st.info(
                            f"**Resumed** this file: **{result['session_new_rows']}** new rows appended; "
                            f"**{result['skipped_existing']}** judgment slots were already on disk."
                        )
                    st.info(
                        f"Tagged **{condition_name}** · dataset **{input_path.stem}** · look for `_cond-{_slug}_` in the "
                        "filename. On **View Results** / **Compare judges & vendors**, set **Condition** (and **Dataset**) to this run."
                    )
                    st.session_state[_RUN_RAW_PREVIEW_PATH_KEY] = str(out_path)
                except Exception as e:
                    st.error(str(e))

    st.divider()
    _preview_raw = st.session_state.get(_RUN_RAW_PREVIEW_PATH_KEY)
    if not _preview_raw:
        st.caption(
            "Raw JSONL preview appears here **after** a completed run on this tab (cleared when you start another run)."
        )
    else:
        _preview_path = Path(_preview_raw)
        if _preview_path.is_file():
            st.subheader(
                "This run — raw JSONL preview",
                anchor=False,
                help=_help_text("run_latest_results_raw"),
            )
            st.caption(_preview_path.name)
            rows = load_jsonl(_preview_path)
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.info("File is empty.")
        else:
            st.session_state[_RUN_RAW_PREVIEW_PATH_KEY] = None
            st.warning("Preview file is no longer on disk; cleared.")



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
