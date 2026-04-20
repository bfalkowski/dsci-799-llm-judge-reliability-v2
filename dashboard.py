"""Streamlit dashboard for LLM-as-a-judge repeat-stability experiments."""

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv

import altair as alt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
from vendor_billing_csv import parse_uploaded_files
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


# Bar colors for OpenAI vs Anthropic (match across Run summary, Compare, etc.)
VENDOR_BAR_COLOR_MAP = {"OpenAI": "#10a37f", "Anthropic": "#c4713f"}


# Mean-score line chart: teal/emerald family (OpenAI), orange/terracotta (Anthropic); cycle within family.
_OPENAI_LINE_FAMILY = (
    "#064e3b",
    "#047857",
    "#059669",
    "#10a37f",
    "#14b8a6",
    "#2dd4bf",
    "#5eead4",
)
_ANTHROPIC_LINE_FAMILY = (
    "#7c2d12",
    "#9a3412",
    "#c2410c",
    "#c4713f",
    "#ea580c",
    "#f97316",
    "#fb923c",
)


def _line_color_vendor_family(vendor: str, within_vendor_index: int) -> str:
    """Hue family by API; distinct shades per model (cycles if many judges per vendor)."""
    if vendor == "Anthropic":
        pal = _ANTHROPIC_LINE_FAMILY
    else:
        pal = _OPENAI_LINE_FAMILY
    return pal[within_vendor_index % len(pal)]


def _first_execution_id(rows: list) -> str:
    if not rows:
        return "—"
    eid = str(rows[0].get("execution_id") or "").strip()
    return eid if eid else "—"


def _mean_panel_score(rows: list) -> Optional[float]:
    """Mean of each judge's mean score (non-null rows only); judges weighted equally."""
    judges = _unique_judge_models_in_rows(rows)
    if not judges:
        return None
    jmeans: list = []
    for j in judges:
        part = _rows_for_judge_model(rows, j)
        scores = [float(r["score"]) for r in part if r.get("score") is not None]
        if scores:
            jmeans.append(sum(scores) / len(scores))
    if not jmeans:
        return None
    return sum(jmeans) / len(jmeans)


def _token_totals_by_judge(rows: list) -> dict:
    """Aggregate input/output tokens per judge_model (missing tokens → 0)."""
    out: dict = {}
    for r in rows:
        j = str(r.get("judge_model") or "").strip()
        if not j:
            continue
        inn = int(r.get("input_tokens") or 0)
        ott = int(r.get("output_tokens") or 0)
        if j not in out:
            out[j] = {"in": 0, "out": 0}
        out[j]["in"] += inn
        out[j]["out"] += ott
    return out


def _estimate_vendor_cost_us1m(tin: float, tout: float, rate_in: float, rate_out: float) -> float:
    return (tin / 1_000_000.0) * rate_in + (tout / 1_000_000.0) * rate_out


def _effective_rates_for_judge(
    judge: str,
    vendor: str,
    rate_oai_in: float,
    rate_oai_out: float,
    rate_ant_in: float,
    rate_ant_out: float,
    rate_overrides: Optional[dict],
) -> Tuple[float, float, bool]:
    """Resolve input/output USD-per-1M rates: per-judge overrides else vendor defaults."""
    if vendor == "Anthropic":
        vin, vout = float(rate_ant_in), float(rate_ant_out)
    else:
        vin, vout = float(rate_oai_in), float(rate_oai_out)
    o = (rate_overrides or {}).get(judge)
    if not isinstance(o, dict):
        o = {}
    rin = o.get("in")
    rout = o.get("out")
    eff_in = float(rin) if rin is not None else vin
    eff_out = float(rout) if rout is not None else vout
    return eff_in, eff_out, bool(eff_in or eff_out)


def _openai_export_key_for_judge(judge: str, openai_by_model: dict) -> Optional[str]:
    """Map JSONL judge id to OpenAI usage CSV model id (often with date suffix)."""
    if not judge or not openai_by_model:
        return None
    j = judge.strip().lower()
    for k in openai_by_model:
        kl = k.lower()
        if kl == j or kl.startswith(j + "-") or kl.startswith(j + "_"):
            return k
    for k in openai_by_model:
        if j in k.lower():
            return k
    return None


def _anthropic_export_cost_usd(judge: str, cost_by_disp: dict) -> Optional[float]:
    """Match JSONL claude-* id to Anthropic cost CSV display names."""
    if not judge or not cost_by_disp:
        return None
    j = judge.lower()
    total = 0.0
    matched = False
    for disp, val in cost_by_disp.items():
        dl = disp.lower()
        ok = False
        if "haiku" in j and "haiku" in dl:
            ok = True
        elif "opus" in j and "opus" in dl:
            ok = True
        elif "sonnet" in j and "4.6" in j and "sonnet" in dl and "4.6" in dl:
            ok = True
        elif "sonnet" in j and "4.6" not in j and "sonnet" in dl and "4.6" not in dl:
            ok = True
        if ok:
            total += float(val)
            matched = True
    return total if matched else None


def _run_summary_rel_row(
    file_tag: str,
    summ: dict,
    judge: str,
    metric_label: str,
    by_item: dict,
) -> dict:
    if not by_item:
        return {
            "File tag": file_tag,
            "Condition": summ["condition"],
            "Judge": judge,
            "Metric": metric_label,
            "Items": 0,
            "Mean score": None,
            "% repeat pairs differ": None,
            "% items any repeat disagree": None,
            "% zero variance": None,
            "Mean within-item SD": None,
            "Repeat agreement (exact)": None,
        }
    hl = metric_repeat_variability_headlines(by_item)
    m1 = metric1_per_item_variance(by_item)
    m2 = metric2_exact_agreement(by_item)
    mean_score = (
        sum(sc for scores in by_item.values() for sc in scores)
        / sum(len(scores) for scores in by_item.values())
    )
    return {
        "File tag": file_tag,
        "Condition": summ["condition"],
        "Judge": judge,
        "Metric": metric_label,
        "Items": m1["n_items"],
        "Mean score": round(mean_score, 2),
        "% repeat pairs differ": (
            round(hl["pct_repeat_pairs_disagree"], 1) if hl["n_repeat_pairs"] else None
        ),
        "% items any repeat disagree": (
            round(hl["pct_items_any_repeat_disagree"], 1) if hl["n_items"] else None
        ),
        "% zero variance": round(m1["pct_items_zero_variance"], 2),
        "Mean within-item SD": round(m1["mean_within_item_std"], 4),
        "Repeat agreement (exact)": round(m2["mean_agreement_rate"] * 100.0, 2),
    }


def _pct_zero_variance_for_pool(by_item: dict) -> Optional[float]:
    if not by_item:
        return None
    m1 = metric1_per_item_variance(by_item)
    if not m1.get("n_items"):
        return None
    return float(m1["pct_items_zero_variance"])


def _composite_pct_zero_equal_abc(
    pool: dict,
    fname_to_condition: dict,
) -> Tuple[Optional[float], dict]:
    """
    Repeat stability with equal weight per condition: B = mean of per-metric % zero variance;
    composite = mean of A, aggregated B, and C among conditions present in the selection.
    """
    pool_a: dict = {}
    pool_c: dict = {}
    pools_b_by_m = defaultdict(dict)
    for k, scores in pool.items():
        parts = k.split("\t")
        if not parts:
            continue
        fname = parts[0]
        cond = fname_to_condition.get(fname)
        if not cond:
            continue
        if cond == "generic_overall" and len(parts) == 2:
            pool_a[k] = scores
        elif cond == "per_item_custom" and len(parts) == 2:
            pool_c[k] = scores
        elif cond == "metric_rubric" and len(parts) == 3:
            mname = parts[1]
            pools_b_by_m[mname][k] = scores
    pct_a = _pct_zero_variance_for_pool(pool_a)
    pct_c = _pct_zero_variance_for_pool(pool_c)
    b_by_metric: dict = {}
    b_vals = []
    for mname in sorted(pools_b_by_m.keys()):
        pv = _pct_zero_variance_for_pool(pools_b_by_m[mname])
        b_by_metric[mname] = pv
        if pv is not None:
            b_vals.append(pv)
    pct_b = sum(b_vals) / len(b_vals) if b_vals else None
    present = [x for x in (pct_a, pct_b, pct_c) if x is not None]
    comp = sum(present) / len(present) if present else None
    detail = {
        "A": pct_a,
        "B_mean": pct_b,
        "C": pct_c,
        "B_by_metric": b_by_metric,
    }
    return comp, detail


def _rel_econ_economics_for_judge(
    judge: str,
    pr: dict,
    rate_oai_in: float,
    rate_oai_out: float,
    rate_ant_in: float,
    rate_ant_out: float,
    billing_parsed,
    rate_overrides: Optional[dict] = None,
) -> dict:
    """Token totals + estimated / export USD columns for Repeat stability × economics."""
    vendor = _api_vendor_label(judge)
    eff_in, eff_out, has_rate = _effective_rates_for_judge(
        judge,
        vendor,
        rate_oai_in,
        rate_oai_out,
        rate_ant_in,
        rate_ant_out,
        rate_overrides,
    )
    est = _estimate_vendor_cost_us1m(float(pr["in"]), float(pr["out"]), eff_in, eff_out)
    ex_in = ex_out = None
    ex_cost_usd = None
    ex_oai_line_usd = None
    if billing_parsed:
        if vendor == "OpenAI" and billing_parsed.openai_by_model:
            ok = _openai_export_key_for_judge(judge, billing_parsed.openai_by_model)
            if ok:
                v = billing_parsed.openai_by_model[ok]
                ex_in = v["input"]
                ex_out = v["output"]
        if vendor == "OpenAI" and billing_parsed.openai_line_cost_by_model:
            okey = _openai_export_key_for_judge(
                judge, billing_parsed.openai_line_cost_by_model
            )
            if okey:
                ex_oai_line_usd = float(billing_parsed.openai_line_cost_by_model[okey])
        if vendor == "Anthropic":
            if (
                billing_parsed.anthropic_tokens_by_model
                and judge in billing_parsed.anthropic_tokens_by_model
            ):
                t = billing_parsed.anthropic_tokens_by_model[judge]
                ex_in = t["in"]
                ex_out = t["out"]
            if billing_parsed.anthropic_cost_by_model:
                ex_cost_usd = _anthropic_export_cost_usd(
                    judge, billing_parsed.anthropic_cost_by_model
                )
    tok_sum = int(pr["in"]) + int(pr["out"])
    return {
        "Vendor": vendor,
        "JSONL input": pr["in"],
        "JSONL output": pr["out"],
        "JSONL tokens (sum)": tok_sum,
        "Est. USD (rates)": round(est, 4) if has_rate else None,
        "Export input": ex_in,
        "Export output": ex_out,
        "Export $ OpenAI (line items)": round(ex_oai_line_usd, 4)
        if ex_oai_line_usd is not None
        else None,
        "Export cost USD (Anthropic)": round(ex_cost_usd, 4)
        if ex_cost_usd is not None
        else None,
    }


def _rel_econ_unified_spend_series(df: pd.DataFrame) -> pd.Series:
    """One USD column for charts: Anthropic export, else OpenAI line items, else estimated rates."""
    cols_try = (
        "Export cost USD (Anthropic)",
        "Export $ OpenAI (line items)",
        "Est. USD (rates)",
    )

    def pick(row):
        for c in cols_try:
            if c not in df.columns:
                continue
            v = row.get(c)
            if v is not None and pd.notna(v) and float(v) > 0:
                return float(v)
        return float("nan")

    return df.apply(pick, axis=1)


def _rel_econ_log_scale_ok(vals: list) -> bool:
    s = [float(x) for x in vals if x is not None and pd.notna(x) and float(x) > 0]
    if len(s) < 2:
        return False
    return (max(s) / min(s)) > 12.0


def _rel_econ_log_decade_ticks(vals: list) -> Optional[list]:
    """Powers of ten from floor(log10 min) through ceil(log10 max) — no minor 2,3,4… labels."""
    pos = [float(x) for x in vals if x is not None and pd.notna(x) and float(x) > 0]
    if not pos:
        return None
    lo, hi = min(pos), max(pos)
    e0 = int(math.floor(math.log10(lo)))
    e1 = int(math.ceil(math.log10(hi)))
    if e1 < e0:
        e0, e1 = e1, e0
    return [10**e for e in range(e0, e1 + 1)]


def _rel_econ_left_margin_for_labels(judges: list) -> int:
    _jmax = max((len(str(j)) for j in judges), default=12)
    # ~7.5px/char at 14px proportional font + tick/pad; prior 11*+96 was overly wide.
    return max(128, min(420, (15 * _jmax) // 2 + 54))


def _rel_econ_style_single_hbar(
    fig: go.Figure,
    *,
    margin_l: int,
    n_judges: int,
    x_title: str,
    use_log_x: bool,
    log_vals: Optional[list],
    x_range: Optional[tuple] = None,
) -> None:
    """One horizontal-bar panel: tight gap below ticks; optional log x (tokens only when needed)."""
    fig.update_yaxes(
        autorange="reversed",
        ticklabelposition="outside",
        ticks="outside",
        ticklen=3,
        tickfont=dict(size=14, color="#111"),
        ticklabelstandoff=4,
        side="left",
        automargin=False,
    )
    fig.update_layout(
        height=max(200, 28 * n_judges + 72),
        margin=dict(l=margin_l, r=28, t=8, b=48),
        bargap=0.14,
        showlegend=False,
    )
    x_common = dict(
        title=dict(text=x_title),
        title_standoff=12,
        tickfont=dict(size=12, color="#222"),
        ticklabelstandoff=6,
        showline=True,
        linewidth=1,
        linecolor="#333",
        showgrid=True,
        gridcolor="rgba(0,0,0,0.08)",
        zeroline=False,
    )
    if use_log_x and log_vals is not None:
        ticks = _rel_econ_log_decade_ticks(log_vals)
        kw = dict(type="log", minor=dict(showgrid=False), **x_common)
        if ticks:
            kw["tickmode"] = "array"
            kw["tickvals"] = ticks
            kw["ticktext"] = [f"{t:g}" for t in ticks]
        fig.update_xaxes(**kw)
    else:
        kw = dict(type="linear", **x_common)
        if x_range is not None:
            kw["range"] = list(x_range)
        fig.update_xaxes(**kw)


def _rel_econ_condition_weighted_stability_chart(
    rel_econ_combined_df: pd.DataFrame,
    pooled_by_judge: dict,
    fname_to_condition: dict,
) -> None:
    """Horizontal bars: % zero variance with A, B (mean of metrics), C each counting once."""

    def _fmt_pct(x):
        if x is None:
            return "—"
        return f"{float(x):.1f}%"

    cdf = rel_econ_combined_df.copy()
    cdf["_usd_only"] = _rel_econ_unified_spend_series(cdf)
    has_usd = (cdf["_usd_only"].notna() & (cdf["_usd_only"] > 0)).any()
    cdf["_sort_key"] = cdf["_usd_only"] if has_usd else pd.to_numeric(cdf["JSONL tokens (sum)"], errors="coerce")
    cdf_plot = cdf.sort_values("_sort_key", ascending=True, na_position="first")
    judges = cdf_plot["Judge"].astype(str).tolist()
    vlist = cdf_plot["Vendor"].astype(str).tolist()
    bar_colors = [VENDOR_BAR_COLOR_MAP.get(v, "#6366f1") for v in vlist]
    n = len(judges)
    if n == 0:
        return

    pairs = []
    for j in judges:
        pool = pooled_by_judge.get(j, {})
        comp, det = _composite_pct_zero_equal_abc(pool, fname_to_condition)
        pairs.append((comp, det))

    if all(p[0] is None for p in pairs):
        st.info(
            "No **condition-weighted** repeat stability to show yet (need scored repeat data for judges in the selection)."
        )
        return

    st.subheader("Repeat stability — equal weight per condition (A, B, C)")
    st.caption(
        "**Condition B** is **one** score: the **mean** of **% zero variance** over **each metric’s** pool. **A** and **C** "
        "each use one pooled score. The **composite** averages those **condition-level** scores (only conditions present "
        "in your selected JSONLs)."
    )

    x_vals = []
    labels = []
    customdata = []
    for comp, det in pairs:
        if comp is not None:
            x_vals.append(float(comp))
            labels.append(f"{float(comp):.1f}%")
        else:
            x_vals.append(0.0)
            labels.append("—")
        bb = det.get("B_by_metric") or {}
        bb_s = ", ".join(f"{mk}: {_fmt_pct(bb[mk])}" for mk in sorted(bb.keys()))
        customdata.append(
            (
                _fmt_pct(det.get("A")),
                _fmt_pct(det.get("B_mean")),
                _fmt_pct(det.get("C")),
                bb_s,
            )
        )

    fig = go.Figure(
        data=[
            go.Bar(
                y=judges,
                x=x_vals,
                orientation="h",
                marker_color=bar_colors,
                text=labels,
                textposition="outside",
                cliponaxis=False,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{y}</b><br>Composite: %{x:.2f}%<br>A: %{customdata[0]}<br>B (mean of metrics): %{customdata[1]}<br>"
                    "C: %{customdata[2]}<br>B by metric: %{customdata[3]}<extra></extra>"
                ),
                showlegend=False,
            )
        ]
    )
    lm = _rel_econ_left_margin_for_labels(judges)
    _rel_econ_style_single_hbar(
        fig,
        margin_l=lm,
        n_judges=n,
        x_title="% zero variance (composite, condition-weighted)",
        use_log_x=False,
        log_vals=None,
        x_range=(0, 100),
    )
    _tok_sum = int(pd.to_numeric(cdf["JSONL tokens (sum)"], errors="coerce").fillna(0).sum())
    _oai_ex = pd.to_numeric(
        cdf.get("Export $ OpenAI (line items)", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    _ant_ex = pd.to_numeric(
        cdf.get("Export cost USD (Anthropic)", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    _key_suf = f"{n}_{_tok_sum}_{int(_oai_ex.sum()*100)}_{int(_ant_ex.sum()*100)}"
    with st.container(border=True):
        st.markdown("##### % zero variance — A, B (mean of metrics), C each count once")
        st.plotly_chart(fig, use_container_width=True, key=f"rel_econ_condabc_{_key_suf}")


def _short_metric_label(m: str, max_len: int = 36) -> str:
    s = str(m).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _rel_econ_mean_score_line_by_condition(
    reliability_rows: list,
    rel_econ_combined_df: pd.DataFrame,
) -> None:
    """Line chart: x = A, each B metric, C; y = mean score; one trace per judge (same order as combined table)."""
    if not reliability_rows:
        return
    parsed = []
    for rr in reliability_rows:
        try:
            ms = rr.get("Mean score")
            if ms is None:
                continue
            judge = str(rr.get("Judge") or "").strip()
            if not judge:
                continue
            parsed.append({
                "judge": judge,
                "condition": str(rr.get("Condition") or ""),
                "metric": rr.get("Metric"),
                "mean_score": float(ms),
            })
        except (TypeError, ValueError):
            continue
    if not parsed:
        return

    metrics_b = sorted(
        {
            str(r["metric"])
            for r in parsed
            if r["condition"] == "metric_rubric"
            and r["metric"] is not None
            and str(r["metric"]).strip() not in ("", "—")
        }
    )
    x_labels = (
        ["A — generic overall"]
        + [f"B — {_short_metric_label(m)}" for m in metrics_b]
        + ["C — per-item custom"]
    )

    cdf = rel_econ_combined_df.copy()
    cdf["_usd_only"] = _rel_econ_unified_spend_series(cdf)
    has_usd = (cdf["_usd_only"].notna() & (cdf["_usd_only"] > 0)).any()
    cdf["_sort_key"] = cdf["_usd_only"] if has_usd else pd.to_numeric(cdf["JSONL tokens (sum)"], errors="coerce")
    judges_order = cdf.sort_values("_sort_key", ascending=True, na_position="first")["Judge"].astype(str).tolist()

    def _avg(vals: list) -> Optional[float]:
        return sum(vals) / len(vals) if vals else None

    fig = go.Figure()
    n_traces = 0
    oai_line_i = 0
    ant_line_i = 0
    for judge in judges_order:
        y_pts: list = []
        va = [
            r["mean_score"]
            for r in parsed
            if r["judge"] == judge and r["condition"] == "generic_overall"
        ]
        y_pts.append(_avg(va))
        for m in metrics_b:
            vb = [
                r["mean_score"]
                for r in parsed
                if r["judge"] == judge
                and r["condition"] == "metric_rubric"
                and str(r["metric"]) == m
            ]
            y_pts.append(_avg(vb))
        vc = [
            r["mean_score"]
            for r in parsed
            if r["judge"] == judge and r["condition"] == "per_item_custom"
        ]
        y_pts.append(_avg(vc))
        if all(v is None for v in y_pts):
            continue
        _vendor = _api_vendor_label(judge)
        if _vendor == "Anthropic":
            color = _line_color_vendor_family(_vendor, ant_line_i)
            ant_line_i += 1
        else:
            color = _line_color_vendor_family(_vendor, oai_line_i)
            oai_line_i += 1
        y_plot = [float(v) if v is not None else None for v in y_pts]
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=y_plot,
                mode="lines+markers",
                name=judge,
                line=dict(color=color, width=2),
                marker=dict(size=9, color=color),
                connectgaps=False,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>%{x}<br>Mean score: %{y:.2f}<extra></extra>"
                ),
            )
        )
        n_traces += 1

    if n_traces == 0:
        st.info("No **mean score** rows to draw condition lines (check scored JSONLs).")
        return

    fig.update_layout(
        title=dict(text="", font=dict(size=1)),
        xaxis_title="Condition / B metric",
        yaxis_title="Mean score",
        yaxis=dict(range=[50, 100], showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
        xaxis=dict(tickangle=-28, showgrid=False),
        height=480,
        margin=dict(l=56, r=24, t=28, b=140),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.32,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        hovermode="x unified",
    )
    _tok_sum = int(pd.to_numeric(cdf["JSONL tokens (sum)"], errors="coerce").fillna(0).sum())
    _oai_ex = pd.to_numeric(
        cdf.get("Export $ OpenAI (line items)", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    _ant_ex = pd.to_numeric(
        cdf.get("Export cost USD (Anthropic)", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    _key_suf = f"{n_traces}_{len(x_labels)}_{_tok_sum}_{int(_oai_ex.sum()*100)}_{int(_ant_ex.sum()*100)}"

    st.subheader("Mean score across conditions (all judges)")
    st.caption(
        "One **line** per **judge_model**. **A** and **C** are the "
        "overall **mean score** for that condition (averaged across files if you picked several). **B** adds one point "
        "**per metric_name**. Missing points break the line (no score for that slice). **OpenAI** judges use **teal / "
        "emerald** shades; **Anthropic** judges use **orange / terracotta**. "
        "**Y-axis is 50\u2013100** to spread out typical judge means; values under 50 are clipped (hover still shows the real score)."
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, key=f"rel_econ_meanline_{_key_suf}")


def _compute_mcd_mcb(all_file_rows: dict, fname_to_condition: dict) -> list:
    """Compute leave-one-out MCD and MCB per judge from raw JSONL rows.

    all_file_rows: {filename: [row_dicts, …]}
    fname_to_condition: {filename: condition_name}

    Returns list of dicts: [{Judge, Vendor, MCD, MCB, MCD_A, MCB_A, …}, …]
    """
    judge_item_scores = defaultdict(lambda: defaultdict(list))
    for fname, rows in all_file_rows.items():
        cond = fname_to_condition.get(fname, "")
        for r in rows:
            s = r.get("score")
            if s is None:
                continue
            j = str(r.get("judge_model", "")).strip()
            item_id = str(r.get("item_id", "")).strip()
            metric = str(r.get("metric_name") or "").strip()
            if cond == "metric_rubric" and metric:
                key = (cond, metric, item_id)
            else:
                key = (cond, "", item_id)
            judge_item_scores[j][key].append(s)

    judges = sorted(judge_item_scores.keys())
    if len(judges) < 2:
        return []

    all_keys = set()
    for j in judges:
        all_keys.update(judge_item_scores[j].keys())

    judge_item_mean = {}
    for j in judges:
        for key in all_keys:
            scores = judge_item_scores[j].get(key, [])
            if scores:
                judge_item_mean[(j, key)] = sum(scores) / len(scores)

    conditions_in_keys = set()
    for (cond, metric, _item) in all_keys:
        conditions_in_keys.add(cond)

    cond_labels = {
        "generic_overall": "A",
        "metric_rubric": "B",
        "per_item_custom": "C",
    }

    def _loo_for_keys(keys_subset, judges_subset):
        mcd_j, mcb_j = {}, {}
        for j in judges_subset:
            abs_d, signed_d = [], []
            for key in keys_subset:
                if (j, key) not in judge_item_mean:
                    continue
                others = [
                    judge_item_mean[(jj, key)]
                    for jj in judges_subset
                    if jj != j and (jj, key) in judge_item_mean
                ]
                if not others:
                    continue
                diff = judge_item_mean[(j, key)] - sum(others) / len(others)
                abs_d.append(abs(diff))
                signed_d.append(diff)
            mcd_j[j] = sum(abs_d) / len(abs_d) if abs_d else None
            mcb_j[j] = sum(signed_d) / len(signed_d) if signed_d else None
        return mcd_j, mcb_j

    overall_mcd, overall_mcb = _loo_for_keys(all_keys, judges)

    per_cond = {}
    for cond in conditions_in_keys:
        ckeys = {k for k in all_keys if k[0] == cond}
        if ckeys:
            m, b = _loo_for_keys(ckeys, judges)
            lbl = cond_labels.get(cond, cond)
            per_cond[lbl] = (m, b)

    result = []
    for j in judges:
        row = {
            "Judge": j,
            "Vendor": _api_vendor_label(j),
            "MCD": round(overall_mcd.get(j) or 0, 1),
            "MCB": round(overall_mcb.get(j) or 0, 1),
        }
        for lbl in ("A", "B", "C"):
            if lbl in per_cond:
                m, b = per_cond[lbl]
                row[f"MCD ({lbl})"] = round(m.get(j) or 0, 1) if m.get(j) is not None else None
                row[f"MCB ({lbl})"] = round(b.get(j) or 0, 1) if b.get(j) is not None else None
        result.append(row)
    return result


def _rel_econ_mcd_mcb_chart(
    mcd_mcb_rows: list,
    rel_econ_combined_df: pd.DataFrame,
) -> None:
    """Horizontal grouped bars: MCD (magnitude) and MCB (signed bias) per judge."""
    if not mcd_mcb_rows:
        return

    cdf = rel_econ_combined_df.copy()
    cdf["_usd_only"] = _rel_econ_unified_spend_series(cdf)
    has_usd = (cdf["_usd_only"].notna() & (cdf["_usd_only"] > 0)).any()
    cdf["_sort_key"] = cdf["_usd_only"] if has_usd else pd.to_numeric(cdf["JSONL tokens (sum)"], errors="coerce")
    judge_order = cdf.sort_values("_sort_key", ascending=True, na_position="first")["Judge"].astype(str).tolist()

    mcd_map = {r["Judge"]: r for r in mcd_mcb_rows}
    judges = [j for j in judge_order if j in mcd_map]
    if len(judges) < 2:
        st.info("Need at least **2 judges** in the selected files to compute consensus deviation.")
        return

    mcd_vals = [mcd_map[j]["MCD"] for j in judges]
    mcb_vals = [mcd_map[j]["MCB"] for j in judges]
    vendors = [mcd_map[j]["Vendor"] for j in judges]

    per_cond_cols = [c for c in ("MCD (A)", "MCD (B)", "MCD (C)") if c in mcd_mcb_rows[0]]
    per_cond_b_cols = [c for c in ("MCB (A)", "MCB (B)", "MCB (C)") if c in mcd_mcb_rows[0]]

    hover_parts = ["<b>%{y}</b>"]
    hover_parts.append("MCD: %{customdata[0]:.1f} pts")
    hover_parts.append("MCB: %{customdata[1]:+.1f} pts")
    cd_indices = 2
    for col in per_cond_cols:
        lbl = col.replace("MCD ", "")
        hover_parts.append(f"MCD {lbl}: %{{customdata[{cd_indices}]:.1f}}")
        cd_indices += 1
    for col in per_cond_b_cols:
        lbl = col.replace("MCB ", "")
        hover_parts.append(f"MCB {lbl}: %{{customdata[{cd_indices}]:+.1f}}")
        cd_indices += 1
    hover_template = "<br>".join(hover_parts) + "<extra></extra>"

    customdata = []
    for j in judges:
        row_data = [mcd_map[j]["MCD"], mcd_map[j]["MCB"]]
        for col in per_cond_cols:
            row_data.append(mcd_map[j].get(col) or 0)
        for col in per_cond_b_cols:
            row_data.append(mcd_map[j].get(col) or 0)
        customdata.append(row_data)

    mcd_colors = [VENDOR_BAR_COLOR_MAP.get(v, "#6366f1") for v in vendors]
    mcb_colors = ["#e74c3c" if v < 0 else "#27ae60" for v in mcb_vals]

    mcd_labels = [f"{v:.1f}" for v in mcd_vals]
    mcb_labels = [f"{v:+.1f}" for v in mcb_vals]

    fig_mcd = go.Figure(
        data=[
            go.Bar(
                y=judges,
                x=mcd_vals,
                orientation="h",
                marker_color=mcd_colors,
                text=mcd_labels,
                textposition="outside",
                cliponaxis=False,
                customdata=customdata,
                hovertemplate=hover_template,
                showlegend=False,
            )
        ]
    )
    n = len(judges)
    lm = _rel_econ_left_margin_for_labels(judges)
    _rel_econ_style_single_hbar(
        fig_mcd, margin_l=lm, n_judges=n,
        x_title="Points",
        use_log_x=False, log_vals=None,
        x_range=(0, max(mcd_vals) * 1.25 if mcd_vals else 25),
    )

    fig_mcb = go.Figure(
        data=[
            go.Bar(
                y=judges,
                x=mcb_vals,
                orientation="h",
                marker_color=mcb_colors,
                text=mcb_labels,
                textposition="outside",
                cliponaxis=False,
                customdata=customdata,
                hovertemplate=hover_template,
                showlegend=False,
            )
        ]
    )
    x_abs_max = max(abs(v) for v in mcb_vals) * 1.35 if mcb_vals else 20
    _rel_econ_style_single_hbar(
        fig_mcb, margin_l=lm, n_judges=n,
        x_title="Points",
        use_log_x=False, log_vals=None,
        x_range=(-x_abs_max, x_abs_max),
    )
    fig_mcb.add_vline(x=0, line_color="#999", line_width=1, line_dash="dot")

    _tok_sum = int(pd.to_numeric(cdf["JSONL tokens (sum)"], errors="coerce").fillna(0).sum())
    _key_suf = f"mcdmcb_{n}_{_tok_sum}"

    st.subheader("Cross-judge consensus: MCD & MCB (leave-one-out)")
    st.caption(
        "**MCD** (Mean Consensus Deviation) = average absolute distance from the other judges' "
        "consensus score on each item. **MCB** (Mean Consensus Bias) = same but **signed**: "
        "**positive = lenient** (scores above consensus), **negative = harsh** (below). "
        "Consensus is **leave-one-out**: each judge is compared to the mean of the other judges, "
        "so its own score does not bias the reference."
    )

    tbl_rows = []
    for j in judges:
        r = mcd_map[j]
        tbl_row = {"Judge": j, "Vendor": r["Vendor"], "MCD (pts)": r["MCD"], "MCB (pts)": r["MCB"]}
        for col in per_cond_cols + per_cond_b_cols:
            tbl_row[col] = r.get(col)
        tbl_rows.append(tbl_row)
    st.dataframe(pd.DataFrame(tbl_rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("##### MCD — magnitude of disagreement")
            st.caption(
                "How far this judge typically sits from what the other judges scored on the same item. "
                "Higher = more of an outlier."
            )
            st.plotly_chart(fig_mcd, use_container_width=True, key=f"rel_econ_mcd_{_key_suf}")
    with c2:
        with st.container(border=True):
            st.markdown("##### MCB — scoring bias direction")
            st.caption(
                "Same distance but **signed**: **green / positive = lenient** (scores above the group), "
                "**red / negative = harsh** (below the group). A bar near zero means no systematic lean."
            )
            st.plotly_chart(fig_mcb, use_container_width=True, key=f"rel_econ_mcb_{_key_suf}")


def _rel_econ_combined_charts(rel_econ_combined_df: pd.DataFrame) -> None:
    """Aligned horizontal bars: when USD exists → cost, tokens, $/1M tokens, repeat stability; else tokens + stability."""
    cdf = rel_econ_combined_df.copy()
    cdf["_usd_only"] = _rel_econ_unified_spend_series(cdf)
    has_usd = (cdf["_usd_only"].notna() & (cdf["_usd_only"] > 0)).any()
    cdf["_sort_key"] = cdf["_usd_only"] if has_usd else pd.to_numeric(cdf["JSONL tokens (sum)"], errors="coerce")
    _zv = pd.to_numeric(cdf["% zero variance"], errors="coerce")
    has_var = bool(_zv.notna().any())
    n = len(cdf)
    if n == 0:
        return

    st.subheader("Economics & repeat stability (charts)")
    if has_usd:
        if has_var:
            st.caption(
                "Same **judge** order in every panel (highest **total spend** at the top). **Spend** and **USD / 1M tokens** "
                "use **linear** scales so cost differences read naturally. **Token** counts may use a **log** x-axis only "
                "when spreads are wide; exact values stay **on the bars**. **% zero variance** is share of items with "
                "identical repeat scores (higher ⇒ more stable repeats)."
            )
        else:
            st.caption(
                "Same **judge** order in every panel. **Spend** and **USD / 1M tokens** are **linear**; **tokens** may use "
                "**log** when spreads are wide. **(4)** No pooled **% zero variance** when nothing is scored for repeats."
            )
    elif has_var:
        st.caption(
            "Bars share the same **judge** order. **Top:** **JSONL tokens** (no USD column in this view). "
            "**Bottom:** **% zero variance** (repeat stability)."
        )
    else:
        st.caption(
            "**JSONL tokens** only — no USD and no repeat-stability column in this view."
        )

    cdf_plot = cdf.sort_values("_sort_key", ascending=True, na_position="first")
    judges = cdf_plot["Judge"].astype(str).tolist()
    vendors = cdf_plot["Vendor"].astype(str).tolist()
    bar_colors = [VENDOR_BAR_COLOR_MAP.get(v, "#6366f1") for v in vendors]

    usd_vals = [float(x) if x is not None and pd.notna(x) else float("nan") for x in cdf_plot["_usd_only"].tolist()]
    tok_vals = [float(x or 0) for x in cdf_plot["JSONL tokens (sum)"].tolist()]
    var_vals = cdf_plot["% zero variance"].tolist()
    agree_vals = cdf_plot["Repeat agreement (exact)"].tolist() if "Repeat agreement (exact)" in cdf_plot.columns else [None] * n

    var_txt = [
        (f"{float(v):.1f}%" if v is not None and pd.notna(v) else "") for v in var_vals
    ]

    per_m_vals: list = []
    for u, t in zip(usd_vals, tok_vals):
        if pd.notna(u) and float(u) > 0 and t > 0:
            per_m_vals.append(float(u) / (t / 1_000_000.0))
        else:
            per_m_vals.append(float("nan"))
    per_m_txt = []
    for x in per_m_vals:
        if x is not None and pd.notna(x) and float(x) > 0:
            per_m_txt.append(f"${float(x):.2f}")
        else:
            per_m_txt.append("—")

    tok_txt = []
    for t in tok_vals:
        if t > 0:
            tok_txt.append(f"{int(round(t)):,}")
        else:
            tok_txt.append("—")

    usd_txt = []
    for u in usd_vals:
        if u is not None and pd.notna(u) and float(u) > 0:
            usd_txt.append(f"${float(u):.2f}")
        else:
            usd_txt.append("—")

    _oai_ex = pd.to_numeric(
        cdf.get("Export $ OpenAI (line items)", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    _ant_ex = pd.to_numeric(
        cdf.get("Export cost USD (Anthropic)", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0)
    _tok_sum = int(pd.to_numeric(cdf["JSONL tokens (sum)"], errors="coerce").fillna(0).sum())
    _chart_key = (
        f"rel_econ_{has_usd}_{has_var}_{n}_{_tok_sum}_{int(_oai_ex.sum()*100)}_{int(_ant_ex.sum()*100)}"
    )
    lm = _rel_econ_left_margin_for_labels(judges)

    if has_usd:
        log_tok = _rel_econ_log_scale_ok(tok_vals)
        tok_x_title = "JSONL tokens, log scale" if log_tok else "JSONL tokens (sum)"

        fig_usd = go.Figure(
            data=[
                go.Bar(
                    y=judges,
                    x=usd_vals,
                    orientation="h",
                    marker_color=bar_colors,
                    text=usd_txt,
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "<b>%{y}</b><br>USD: %{customdata[0]:.4f}<br>Tokens: %{customdata[1]:,.0f}<extra></extra>"
                    ),
                    customdata=list(zip(usd_vals, tok_vals)),
                    showlegend=False,
                )
            ]
        )
        _rel_econ_style_single_hbar(
            fig_usd,
            margin_l=lm,
            n_judges=n,
            x_title="Total spend (US$)",
            use_log_x=False,
            log_vals=None,
        )

        fig_tok = go.Figure(
            data=[
                go.Bar(
                    y=judges,
                    x=tok_vals,
                    orientation="h",
                    marker_color=bar_colors,
                    text=tok_txt,
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "<b>%{y}</b><br>Tokens: %{x:,.0f}<br>USD: %{customdata[0]:.4f}<extra></extra>"
                    ),
                    customdata=[(u if pd.notna(u) else 0.0) for u in usd_vals],
                    showlegend=False,
                )
            ]
        )
        _rel_econ_style_single_hbar(
            fig_tok,
            margin_l=lm,
            n_judges=n,
            x_title=tok_x_title,
            use_log_x=log_tok,
            log_vals=tok_vals if log_tok else None,
        )

        fig_pm = go.Figure(
            data=[
                go.Bar(
                    y=judges,
                    x=per_m_vals,
                    orientation="h",
                    marker_color=bar_colors,
                    text=per_m_txt,
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "<b>%{y}</b><br>USD / 1M tokens: %{x:.4f}<br>Total tokens: %{customdata[0]:,.0f}<br>"
                        "% zero variance: %{customdata[1]}<br>Repeat agreement: %{customdata[2]}<extra></extra>"
                    ),
                    customdata=list(
                        zip(
                            tok_vals,
                            [
                                f"{float(v):.2f}%"
                                if v is not None and pd.notna(v)
                                else "—"
                                for v in var_vals
                            ],
                            [
                                f"{float(a):.1f}%"
                                if a is not None and pd.notna(a)
                                else "—"
                                for a in agree_vals
                            ],
                        )
                    ),
                    showlegend=False,
                )
            ]
        )
        _rel_econ_style_single_hbar(
            fig_pm,
            margin_l=lm,
            n_judges=n,
            x_title="US$ per 1M tokens",
            use_log_x=False,
            log_vals=None,
        )

        if has_var:
            fig_var = go.Figure(
                data=[
                    go.Bar(
                        y=judges,
                        x=var_vals,
                        orientation="h",
                        marker_color=bar_colors,
                        text=var_txt,
                        textposition="outside",
                        cliponaxis=False,
                        hovertemplate=(
                            "<b>%{y}</b><br>% zero variance: %{x:.2f}<br>Repeat agree.: %{customdata[0]}<extra></extra>"
                        ),
                        customdata=[
                            (
                                f"{float(a):.1f}%"
                                if a is not None and pd.notna(a)
                                else "—"
                            )
                            for a in agree_vals
                        ],
                        showlegend=False,
                    )
                ]
            )
            _rel_econ_style_single_hbar(
                fig_var,
                margin_l=lm,
                n_judges=n,
                x_title="% of items with zero variance (0–100)",
                use_log_x=False,
                log_vals=None,
                x_range=(0, 100),
            )
        else:
            fig_var = go.Figure(
                data=[
                    go.Bar(
                        y=judges,
                        x=[0.5] * len(judges),
                        orientation="h",
                        marker=dict(color="rgba(160,160,160,0.25)"),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                ]
            )
            _rel_econ_style_single_hbar(
                fig_var,
                margin_l=lm,
                n_judges=n,
                x_title="Pooled % zero variance (unavailable)",
                use_log_x=False,
                log_vals=None,
                x_range=(0, 100),
            )
            _ymid = judges[len(judges) // 2] if judges else ""
            fig_var.add_annotation(
                xref="x",
                yref="y",
                x=50,
                y=_ymid,
                text="No % zero variance (no pooled scores in this selection)",
                showarrow=False,
                font=dict(size=12, color="#555"),
            )

        with st.container(border=True):
            st.markdown("##### Total spend (invoice / estimated)")
            st.plotly_chart(fig_usd, use_container_width=True, key=f"{_chart_key}_usd")
        with st.container(border=True):
            st.markdown("##### JSONL tokens (summed over selected files)")
            st.plotly_chart(fig_tok, use_container_width=True, key=f"{_chart_key}_tok")
        with st.container(border=True):
            st.markdown("##### Effective cost: USD per 1M tokens")
            st.plotly_chart(fig_pm, use_container_width=True, key=f"{_chart_key}_pm")
        with st.container(border=True):
            st.markdown("##### Repeat stability: % zero variance")
            st.plotly_chart(fig_var, use_container_width=True, key=f"{_chart_key}_var")

    elif has_var:
        spend_vals = tok_vals
        log_top = _rel_econ_log_scale_ok(spend_vals)
        tok_title = "JSONL tokens, log scale" if log_top else "JSONL tokens (sum)"

        fig_t = go.Figure(
            data=[
                go.Bar(
                    y=judges,
                    x=spend_vals,
                    orientation="h",
                    marker_color=bar_colors,
                    text=tok_txt,
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Tokens: %{x:,.0f}<extra></extra>",
                    showlegend=False,
                )
            ]
        )
        _rel_econ_style_single_hbar(
            fig_t,
            margin_l=lm,
            n_judges=n,
            x_title=tok_title,
            use_log_x=log_top,
            log_vals=spend_vals if log_top else None,
        )

        fig_v = go.Figure(
            data=[
                go.Bar(
                    y=judges,
                    x=var_vals,
                    orientation="h",
                    marker_color=bar_colors,
                    text=var_txt,
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate=(
                        "<b>%{y}</b><br>% zero variance: %{x:.2f}<br>Repeat agree.: %{customdata[0]}<extra></extra>"
                    ),
                    customdata=[
                        (
                            f"{float(a):.1f}%"
                            if a is not None and pd.notna(a)
                            else "—"
                        )
                        for a in agree_vals
                    ],
                    showlegend=False,
                )
            ]
        )
        _rel_econ_style_single_hbar(
            fig_v,
            margin_l=lm,
            n_judges=n,
            x_title="% of items with zero variance (0–100)",
            use_log_x=False,
            log_vals=None,
            x_range=(0, 100),
        )

        with st.container(border=True):
            st.markdown("##### JSONL tokens (summed over selected files)")
            st.plotly_chart(fig_t, use_container_width=True, key=f"{_chart_key}_tok")
        with st.container(border=True):
            st.markdown("##### Repeat stability: % zero variance")
            st.plotly_chart(fig_v, use_container_width=True, key=f"{_chart_key}_var")

    else:
        log_top = _rel_econ_log_scale_ok(tok_vals)
        tok_title = "JSONL tokens, log scale" if log_top else "JSONL tokens"
        fig_only = go.Figure(
            data=[
                go.Bar(
                    y=judges,
                    x=tok_vals,
                    orientation="h",
                    marker_color=bar_colors,
                    text=tok_txt,
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>Tokens: %{x:,.0f}<extra></extra>",
                    showlegend=False,
                )
            ]
        )
        _rel_econ_style_single_hbar(
            fig_only,
            margin_l=lm,
            n_judges=n,
            x_title=tok_title,
            use_log_x=log_top,
            log_vals=tok_vals if log_top else None,
        )
        with st.container(border=True):
            st.markdown("##### JSONL tokens")
            st.plotly_chart(fig_only, use_container_width=True, key=f"{_chart_key}_single")

    st.caption(
        "Each bordered block is a separate chart; **titles** sit in the page, not inside the plot. "
        "**Spend** and **USD / 1M tokens** are **linear** so you can compare dollars. **Token** bars may use a **log** "
        "axis only when spreads are very wide (decade ticks; exact counts stay on the bars). Billing CSV changes "
        "refresh the same panels."
    )


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
st.set_page_config(page_title="LLM-as-a-Judge Repeat Stability", layout="wide")

# Load and inject custom CSS
def _load_css():
    path = CONTENT_DIR / "dashboard.css"
    if path.exists():
        return path.read_text(encoding=ENCODING)
    return ""

_css = _load_css()
if _css:
    st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)

st.title(_captions.get("title", "LLM-as-a-Judge Repeat Stability"))

# --- Tab structure ---
tab_names = [
    "Overview",
    "Dataset & prompts",
    "View Results",
    "Run summary",
]
tabs = st.tabs(tab_names)

(
    tab_overview,
    tab_dataset,
    tab_view,
    tab_run_summary,
) = (
    tabs[0],
    tabs[1],
    tabs[2],
    tabs[3],
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
                    st.caption(f"Condition **B** — repeat stability for metric **{view_metric_filter}** only.")
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
                st.subheader("Repeat stability metrics (detail)")
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
    
                st.subheader("Overall repeat stability")
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


# ==================== TAB: Compare judges & vendors (hidden for presentation) ==============
if False:  # tab removed for presentation
 with st.container():
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
                    _metric_line = (
                        f" · metric **{compare_metric_choice}**"
                        if compare_metric_choice
                        else ""
                    )
                    st.caption(
                        f"**Comparison:** {len(selected)} result file(s) · **{cond_filter_cmp}** · "
                        f"**{dset_filter_cmp}**{_metric_line}. Tables and charts below use only these files and filters."
                    )
                    with st.expander("Included result files (audit)", expanded=False):
                        for fn in selected:
                            raw_rows = loaded_cmp[fn]
                            filt_rows = _rows_for_single_metric(raw_rows, compare_metric_choice)
                            eid = _first_execution_id(filt_rows if filt_rows else raw_rows)
                            tag = _short_run_tag_from_results_filename(fn)
                            st.markdown(
                                f"- **`{fn}`** — run tag `{tag}` · **{len(filt_rows)}** rows after filters "
                                f"({len(raw_rows)} loaded) · `execution_id` **{eid}**"
                            )
                            st.caption(str((RESULTS_DIR / fn).resolve()))

                    compare_data = []
                    multi_file = len(selected) > 1
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
                        row_out = {
                            "Condition": summ["condition"],
                            "Dataset": summ["dataset_id"],
                            "Judge": s["label"],
                            "Vendor": _api_vendor_label(s["judge_key"]),
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
                        }
                        if multi_file:
                            row_out["Source file"] = _short_run_tag_from_results_filename(s["fname"])
                        compare_data.append(row_out)

                    st.subheader("Per-judge metrics")
                    st.dataframe(pd.DataFrame(compare_data), use_container_width=True)
                    _src_note = (
                        " **Source file** is the short run tag when multiple JSONLs are selected. "
                        if multi_file
                        else ""
                    )
                    st.caption(
                        "**Distinct scores / judgments**, **% repeat pairs differ**, and **% items any repeat disagree** are all **within-item** summaries (not a pooled min/max across all scores). "
                        "**Judge** is the model id (plus a run tag only if the same model appears twice)."
                        + _src_note
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
                                    color_discrete_map=VENDOR_BAR_COLOR_MAP,
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
                                    color_discrete_map=VENDOR_BAR_COLOR_MAP,
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
                        {
                            "judge": r["Judge"],
                            "pct_zero_variance": r["% zero variance"],
                            "Vendor": r["Vendor"],
                        }
                        for r in compare_data
                        if isinstance(r["% zero variance"], (int, float))
                    ]
                    if rel_data:
                        st.subheader("Repeat stability: % items with zero variance")
                        st.caption("Higher = more stable. Same item, same response → identical scores across repeats.")
                        rel_df = pd.DataFrame(rel_data)
                        fig = px.bar(
                            rel_df,
                            x="pct_zero_variance",
                            y="judge",
                            orientation="h",
                            color="Vendor",
                            color_discrete_map=VENDOR_BAR_COLOR_MAP,
                        )
                        fig.update_layout(
                            xaxis_title="% zero variance",
                            yaxis_title="",
                            xaxis=dict(range=[0, 105]),
                            height=max(200, 60 * len(rel_data)),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No repeat-stability data to chart (select files with valid judgments).")

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


# ==================== TAB: Run Experiment (hidden for presentation) ==============
if False:  # tab removed for presentation
 with st.container():
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



# ==================== TAB: Run summary ====================
with tab_run_summary:
    st.header("Run summary")
    st.caption(
        "Select every JSONL from a session (e.g. conditions **A**, **B**, **C** on the same benchmark). "
        "**Repeat stability (RS%)** is the share of items with identical integer scores across K repeats. "
        "**MCD / MCB** quantify cross-judge consensus alignment and directional scoring bias."
    )
    summaries_rs = _all_result_summaries()
    if not summaries_rs:
        st.info("No result files in results/.")
    else:
        rs_labels = [
            f"{s['name']}  ·  {s['condition']}  ·  {s['dataset_id']}" for s in summaries_rs
        ]
        rs_map = {rs_labels[i]: summaries_rs[i]["name"] for i in range(len(summaries_rs))}
        pick_rs = st.multiselect(
            "Result JSONL files (e.g. all runs from today)",
            rs_labels,
            default=[],
            key="run_summary_files_pick",
        )

        if not pick_rs:
            st.info("Select one or more result files to see repeat stability metrics.")
        else:
            selected_names = [rs_map[L] for L in pick_rs]
            file_stats: list = []
            by_judge_all: dict = {}
            reliability_rows: list = []
            pooled_by_judge: dict = {}
            conditions_seen: set = set()
            fname_to_condition: dict = {}
            all_file_rows: dict = {}

            for fname in selected_names:
                path = RESULTS_DIR / fname
                rows = load_jsonl(path)
                all_file_rows[fname] = rows
                summ = _summarize_result_file(path)
                fname_to_condition[fname] = summ["condition"]
                conditions_seen.add(str(summ["condition"]))
                pmean = _mean_panel_score(rows)
                tag = _short_run_tag_from_results_filename(fname)
                toks = _token_totals_by_judge(rows)
                for j, pr in toks.items():
                    if j not in by_judge_all:
                        by_judge_all[j] = {"in": 0, "out": 0}
                    by_judge_all[j]["in"] += pr["in"]
                    by_judge_all[j]["out"] += pr["out"]
                n_scored = sum(1 for r in rows if r.get("score") is not None)
                file_stats.append({
                    "File tag": tag,
                    "Filename": fname,
                    "Condition": summ["condition"],
                    "Dataset": summ["dataset_id"],
                    "Rows": len(rows),
                    "Rows with score": n_scored,
                    "Panel mean score": round(pmean, 2) if pmean is not None else None,
                })
                for j in _unique_judge_models_in_rows(rows):
                    slice_j = _rows_for_judge_model(rows, j)
                    bucket = pooled_by_judge.setdefault(j, {})
                    if summ["condition"] == "metric_rubric":
                        metrics = sorted(
                            {str(r.get("metric_name")) for r in slice_j if r.get("metric_name")}
                        )
                        for m in metrics:
                            slice_m = _rows_for_single_metric(slice_j, m)
                            by_item_m = _group_by_item(slice_m)
                            for item_id, scores in by_item_m.items():
                                bucket[f"{fname}\t{m}\t{item_id}"] = list(scores)
                            rr = _run_summary_rel_row(tag, summ, j, m, by_item_m)
                            rr["Filename"] = fname
                            reliability_rows.append(rr)
                    else:
                        by_item_j = _group_by_item(slice_j)
                        for item_id, scores in by_item_j.items():
                            bucket[f"{fname}\t{item_id}"] = list(scores)
                        rr = _run_summary_rel_row(tag, summ, j, "—", by_item_j)
                        rr["Filename"] = fname
                        reliability_rows.append(rr)

            n_sel = len(selected_names)
            conds_lbl = ", ".join(sorted(conditions_seen, key=str)) if conditions_seen else "—"
            combined_tag = f"All selected (n={n_sel})"
            synthetic_summ = {"condition": conds_lbl}

            rel_econ_combined_rows: list = []
            for j in sorted(by_judge_all.keys()):
                by_item_pool = pooled_by_judge.get(j, {})
                rr = _run_summary_rel_row(
                    combined_tag, synthetic_summ, j, "—", by_item_pool
                )
                rr["Filename"] = "—"
                rr["Vendor"] = _api_vendor_label(j)
                tok = by_judge_all[j]
                rr["JSONL tokens (sum)"] = int(tok["in"]) + int(tok["out"])
                rel_econ_combined_rows.append(rr)

            st.subheader("Per file")
            st.dataframe(pd.DataFrame(file_stats), use_container_width=True, hide_index=True)

            by_cond: dict = {}
            for fs in file_stats:
                c = fs["Condition"]
                by_cond.setdefault(c, {"panel_means": [], "files": []})
                if fs["Panel mean score"] is not None:
                    by_cond[c]["panel_means"].append(fs["Panel mean score"])
                by_cond[c]["files"].append(fs["File tag"])
            if by_cond:
                st.subheader("By condition (mean of file-level panel means)")
                cond_rows = []
                for c in sorted(by_cond.keys(), key=str):
                    pm = by_cond[c]["panel_means"]
                    cond_rows.append({
                        "Condition": c,
                        "Files (n)": len(by_cond[c]["files"]),
                        "Avg panel mean": round(sum(pm) / len(pm), 2) if pm else None,
                    })
                st.dataframe(pd.DataFrame(cond_rows), use_container_width=True, hide_index=True)

            if rel_econ_combined_rows:
                st.subheader("Repeat stability per judge (all selected files)")
                st.caption(
                    "One row per **judge_model**. Repeat-stability metrics pool **within-file** repeat variance only: "
                    "each judged item is keyed by **file** (and **metric** for condition B), so different rubrics "
                    "are not mixed under the same `item_id`."
                )
                rel_econ_combined_df = pd.DataFrame(rel_econ_combined_rows)
                _display_cols = [
                    c for c in rel_econ_combined_df.columns
                    if c not in {
                        "JSONL input", "JSONL output", "JSONL tokens (sum)",
                        "Est. USD (rates)", "Export input", "Export output",
                        "Export $ OpenAI (line items)", "Export cost USD (Anthropic)",
                    }
                ]
                st.dataframe(rel_econ_combined_df[_display_cols], use_container_width=True, hide_index=True)
                _rel_econ_condition_weighted_stability_chart(
                    rel_econ_combined_df,
                    pooled_by_judge,
                    fname_to_condition,
                )
                _rel_econ_mean_score_line_by_condition(reliability_rows, rel_econ_combined_df)

                mcd_mcb_rows = _compute_mcd_mcb(all_file_rows, fname_to_condition)
                if mcd_mcb_rows:
                    _rel_econ_mcd_mcb_chart(mcd_mcb_rows, rel_econ_combined_df)






# ==================== TAB: Telemetry (hidden for presentation) ====================
if False:  # tab removed for presentation
 with st.container():
    st.header("Telemetry (OTEL)")
    st.caption(
        "OpenTelemetry data from judge runs: trace/span IDs, token usage, and span status. "
        "Used to validate repeat stability via token-based metrics."
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
                        "**Repeat-stability focus:** Same item, same prompt, K repeats—does the judge behave consistently? "
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
                    # Per-item within-item variance (repeat-stability detail)
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
                        "This is expected, not a repeat-stability signal."
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


# ==================== TAB: Manage (hidden for presentation) ====================
if False:  # tab removed for presentation
 with st.container():
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
