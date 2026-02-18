"""Compute reliability metrics from judge JSONL output."""

import sys
from collections import Counter
from pathlib import Path

import itertools

from typing import Dict, List, Optional

from utils import REPO_ROOT, load_jsonl

RESULTS_DIR = REPO_ROOT / "results"


def _get_score(row: dict) -> Optional[int]:
    s = row.get("score")
    return int(s) if s is not None else None


def _group_by_item(rows: List[dict]) -> Dict[str, List[int]]:
    """Group valid scores by item_id."""
    by_item: Dict[str, List[int]] = {}
    for r in rows:
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

    return {
        "mean_variance": mean_var,
        "pct_items_zero_variance": pct_zero,
        "n_items": n_items,
        "zero_var_count": zero_var_count,
    }


def metric2_exact_agreement(by_item: Dict[str, List[int]]) -> dict:
    """Exact agreement: fraction of score pairs that match per item, mean across items."""
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


def metric3_score_histogram(rows: List[dict]) -> Dict[int, int]:
    """Score distribution (count per score)."""
    scores = [s for r in rows if (s := _get_score(r)) is not None]
    return dict(Counter(scores))


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
    print("   % of items with zero variance: {:.1f}%".format(m1["pct_items_zero_variance"]))
    print(f"   ({m1['zero_var_count']} / {m1['n_items']} items)")
    print()

    # 2. Exact agreement rate
    m2 = metric2_exact_agreement(by_item)
    print("2. EXACT AGREEMENT RATE")
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
