"""Compute reliability metrics from judge JSONL output."""

import sys
from collections import Counter
from pathlib import Path

import itertools
from math import sqrt

from typing import Dict, List, Optional

from utils import REPO_ROOT, load_jsonl

RESULTS_DIR = REPO_ROOT / "results"


def _get_score(row: dict) -> Optional[int]:
    s = row.get("score")
    return int(s) if s is not None else None


def _group_by_item(rows: List[dict], metric_name: Optional[str] = None) -> Dict[str, List[int]]:
    """Group valid scores by item_id. If metric_name is set, only rows with that metric_name count (condition B)."""
    by_item: Dict[str, List[int]] = {}
    for r in rows:
        if metric_name is not None:
            rn = r.get("metric_name")
            if rn is None or str(rn) != str(metric_name):
                continue
        score = _get_score(r)
        if score is not None:
            item_id = str(r.get("item_id", ""))
            by_item.setdefault(item_id, []).append(score)
    return by_item


def variance(scores: List[float]) -> float:
    """Sample variance. Returns 0 for n<2."""
    n = len(scores)
    if n < 2:
        return 0.0
    mean = sum(scores) / n
    return sum((x - mean) ** 2 for x in scores) / (n - 1)


def per_item_variances(by_item: Dict[str, List[int]]) -> List[dict]:
    """Return per-item variance for charting: [{item_id, variance}, ...]."""
    return [
        {"item_id": item_id, "variance": variance(scores)}
        for item_id, scores in by_item.items()
    ]


def metric1_per_item_variance(by_item: Dict[str, List[int]]) -> dict:
    """Per-item variance, mean variance, % items with zero variance."""
    variances = []
    zero_var_count = 0
    for item_id, scores in by_item.items():
        var_i = variance(scores)
        variances.append(var_i)
        if var_i == 0:
            zero_var_count += 1

    n_items = len(by_item)
    mean_var = sum(variances) / n_items if n_items else 0
    pct_zero = 100 * zero_var_count / n_items if n_items else 0
    # Average of per-item sample SDs (sqrt of per-item variance); same units as score (0–100).
    mean_within_item_std = sum(sqrt(v) for v in variances) / n_items if n_items else 0.0

    return {
        "mean_variance": mean_var,
        "mean_within_item_std": mean_within_item_std,
        "pct_items_zero_variance": pct_zero,
        "n_items": n_items,
        "zero_var_count": zero_var_count,
    }


def metric2_exact_agreement(by_item: Dict[str, List[int]]) -> dict:
    """Per item: among all unordered pairs of repeat scores, fraction with exact integer equality.
    Return the mean of those fractions across items (not comparable to mean score, which averages raw scores)."""
    agreement_rates = []
    for scores in by_item.values():
        n = len(scores)
        total_pairs = n * (n - 1) // 2
        if total_pairs == 0:
            agreement_rates.append(1.0)
            continue
        matching = sum(1 for a, b in itertools.combinations(scores, 2) if a == b)
        agreement_rates.append(matching / total_pairs)

    n_items = len(by_item)
    mean_agreement = sum(agreement_rates) / n_items if n_items else 0
    return {
        "mean_agreement_rate": mean_agreement,
        "n_items": n_items,
    }


def per_item_table(rows: List[dict]) -> List[dict]:
    """Per-item aggregation for table: mean_score, variance, std_dev, min/max, agreement, latency stats."""
    from math import sqrt
    by_item: Dict[str, List[dict]] = {}
    for r in rows:
        item_id = str(r.get("item_id", ""))
        by_item.setdefault(item_id, []).append(r)

    result = []
    for item_id, item_rows in by_item.items():
        scores = [s for r in item_rows if (s := _get_score(r)) is not None]
        latencies = [r.get("latency_ms") for r in item_rows if r.get("latency_ms") is not None]
        latencies = [int(x) for x in latencies if isinstance(x, (int, float))]

        n = len(scores)
        mean_score = sum(scores) / n if n else None
        var = variance(scores)
        std_dev = sqrt(var) if var > 0 else 0.0

        # Agreement rate
        total_pairs = n * (n - 1) // 2 if n else 0
        if total_pairs == 0:
            agreement_rate = 1.0
        else:
            matching = sum(1 for a, b in itertools.combinations(scores, 2) if a == b)
            agreement_rate = matching / total_pairs

        mean_latency = sum(latencies) / len(latencies) if latencies else None
        if len(latencies) >= 2:
            lat_mean = sum(latencies) / len(latencies)
            latency_var = sum((x - lat_mean) ** 2 for x in latencies) / (len(latencies) - 1)
            latency_std_dev = sqrt(latency_var)
        else:
            latency_std_dev = 0.0 if latencies else None

        result.append({
            "item_id": item_id,
            "mean_score": round(mean_score, 2) if mean_score is not None else None,
            "variance": round(var, 4),
            "std_dev": round(std_dev, 4),
            "min_score": min(scores) if scores else None,
            "max_score": max(scores) if scores else None,
            "agreement_rate": round(agreement_rate, 4),
            "mean_latency": round(mean_latency, 0) if mean_latency is not None else None,
            "latency_std_dev": round(latency_std_dev, 1) if latency_std_dev is not None else None,
        })
    return result


def metric3_score_histogram(rows: List[dict], metric_name: Optional[str] = None) -> Dict[int, int]:
    """Score distribution (count per score). Optionally restrict to one metric_name (condition B)."""
    scores = []
    for r in rows:
        if metric_name is not None:
            rn = r.get("metric_name")
            if rn is None or str(rn) != str(metric_name):
                continue
        s = _get_score(r)
        if s is not None:
            scores.append(s)
    return dict(Counter(scores))


def otel_metrics(rows: List[dict]) -> dict:
    """OTEL-derived metrics: token usage, span status, trace coverage.

    Uses input_tokens, output_tokens, span_status when present (from instrumented runs).
    Returns empty/minimal dict for older JSONL without OTEL fields.
    """
    if not rows:
        return {}

    has_tokens = any(r.get("input_tokens") is not None or r.get("output_tokens") is not None for r in rows)
    has_trace = any(r.get("trace_id") for r in rows)

    result = {
        "total_spans": len(rows),
        "trace_ids": list({r["trace_id"] for r in rows if r.get("trace_id")}),
        "span_status_ok": sum(1 for r in rows if r.get("span_status") == "ok"),
        "span_status_error": sum(1 for r in rows if r.get("span_status") == "error"),
    }

    if has_tokens:
        input_tokens = [r["input_tokens"] for r in rows if r.get("input_tokens") is not None]
        output_tokens = [r["output_tokens"] for r in rows if r.get("output_tokens") is not None]
        result["total_input_tokens"] = sum(input_tokens)
        result["total_output_tokens"] = sum(output_tokens)
        result["mean_input_tokens"] = sum(input_tokens) / len(input_tokens) if input_tokens else 0
        result["mean_output_tokens"] = sum(output_tokens) / len(output_tokens) if output_tokens else 0
        # Per-item token variance (reliability: does token count vary for same item?)
        by_item_tokens: Dict[str, List[int]] = {}
        for r in rows:
            ti, to = r.get("input_tokens"), r.get("output_tokens")
            if ti is not None and to is not None:
                iid = str(r.get("item_id", ""))
                by_item_tokens.setdefault(iid, []).append(ti + to)
        token_variances = [variance(v) for v in by_item_tokens.values() if len(v) >= 2]
        result["mean_token_variance_per_item"] = sum(token_variances) / len(token_variances) if token_variances else 0
    else:
        result["total_input_tokens"] = None
        result["total_output_tokens"] = None
        result["mean_input_tokens"] = None
        result["mean_output_tokens"] = None
        result["mean_token_variance_per_item"] = None

    result["has_otel"] = has_trace or has_tokens

    # Per-item token details for dashboard (between-item & within-item)
    if has_tokens:
        by_item: Dict[str, List[dict]] = {}
        for r in rows:
            iid = str(r.get("item_id", ""))
            by_item.setdefault(iid, []).append(r)
        per_item_list = []
        for item_id, item_rows in by_item.items():
            inputs = [r["input_tokens"] for r in item_rows if r.get("input_tokens") is not None]
            outputs = [r["output_tokens"] for r in item_rows if r.get("output_tokens") is not None]
            totals = [
                r.get("input_tokens", 0) + r.get("output_tokens", 0)
                for r in item_rows
                if r.get("input_tokens") is not None and r.get("output_tokens") is not None
            ]
            scores = [_get_score(r) for r in item_rows if _get_score(r) is not None]
            mean_input = sum(inputs) / len(inputs) if inputs else 0
            mean_output = sum(outputs) / len(outputs) if outputs else 0
            within_var = variance(totals) if len(totals) >= 2 else 0
            per_item_list.append({
                "item_id": item_id,
                "mean_input_tokens": round(mean_input, 1),
                "mean_output_tokens": round(mean_output, 1),
                "mean_total_tokens": round(mean_input + mean_output, 1),
                "within_item_variance": round(within_var, 2),
                "repeats": len(item_rows),
                "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
            })
        result["per_item_token_details"] = sorted(per_item_list, key=lambda x: x["item_id"])
        # Between-item spread
        if per_item_list:
            mean_totals = [p["mean_total_tokens"] for p in per_item_list]
            result["between_item_min_tokens"] = min(mean_totals)
            result["between_item_max_tokens"] = max(mean_totals)
            result["between_item_range"] = max(mean_totals) - min(mean_totals)
    else:
        result["per_item_token_details"] = []
        result["between_item_min_tokens"] = None
        result["between_item_max_tokens"] = None
        result["between_item_range"] = None

    return result


def print_histogram(counts: Dict[int, int]) -> None:
    """Print ASCII histogram."""
    if not counts:
        print("  (no scores)")
        return
    max_count = max(counts.values()) if counts else 0
    width = 40
    for score in sorted(counts.keys()):
        n = counts[score]
        bar_len = int(width * n / max_count) if max_count else 0
        bar = "█" * bar_len
        print(f"  {score:2d} │ {bar} {n}")


def main():
    if len(sys.argv) >= 2:
        path = Path(sys.argv[1])
    else:
        jsonl_files = list(RESULTS_DIR.glob("*.jsonl")) if RESULTS_DIR.exists() else []
        if not jsonl_files:
            print("No JSONL files in results/. Run the judge first.")
            sys.exit(1)
        path = max(jsonl_files, key=lambda p: p.stat().st_mtime)

    print(f"File: {path}\n")
    rows = load_jsonl(path)
    if not rows:
        print("File is empty.")
        sys.exit(1)

    by_item = _group_by_item(rows)

    # 1. Per-item variance
    m1 = metric1_per_item_variance(by_item)
    print("1. PER-ITEM VARIANCE")
    print("   Mean variance across items:    {:.4f}".format(m1["mean_variance"]))
    print("   Mean within-item SD (score units): {:.4f}".format(m1["mean_within_item_std"]))
    print("   % of items with zero variance: {:.1f}%".format(m1["pct_items_zero_variance"]))
    print(f"   ({m1['zero_var_count']} / {m1['n_items']} items)")
    print()

    # 2. Repeat agreement (exact)
    m2 = metric2_exact_agreement(by_item)
    print("2. REPEAT AGREEMENT RATE (exact integer match within item)")
    print("   Mean agreement across items: {:.2%}".format(m2["mean_agreement_rate"]))
    print(f"   ({m2['n_items']} items)")
    print()

    # 3. Score distribution
    counts = metric3_score_histogram(rows)
    print("3. SCORE DISTRIBUTION (histogram)")
    print_histogram(counts)
    print()


if __name__ == "__main__":
    main()
