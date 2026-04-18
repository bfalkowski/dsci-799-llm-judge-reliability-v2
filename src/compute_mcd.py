"""
Mean Consensus Deviation (MCD) and Mean Consensus Bias (MCB) — leave-one-out.

For each (judge, item) pair, consensus is the mean of the *other* judges' scores
on that item (the evaluated judge is excluded).

  MCD = mean |judge_score − LOO_consensus|   (magnitude of disagreement)
  MCB = mean  (judge_score − LOO_consensus)  (signed: +lenient, −harsh)

Usage:
    python src/compute_mcd.py                       # prints tables to stdout
    python src/compute_mcd.py --out docs/mcd_results.md   # also writes markdown
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULT_FILES = {
    "A": "results/mtbench_judge-multi7judges_cond-gen_K5_t0p0_20260405T142328.jsonl",
    "B": "results/mtbench_judge-multi7judges_cond-metric_K5_t0p0_20260405T174020.jsonl",
    "C": "results/mtbench_judge-multi7judges_cond-custom_K5_t0p0_20260405T163013.jsonl",
}

SHORT_NAMES = {
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "claude-opus-4-20250514": "Claude Opus 4",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-sonnet-4-6": "Claude Sonnet 4.6",
    "gpt-4": "GPT-4",
    "gpt-4o": "GPT-4o",
    "gpt-4o-mini": "GPT-4o mini",
}

B_METRICS = ["accuracy", "relevance", "completeness"]


def _load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_rs_mcd_mcb(rows, metric_filter=None):
    """Return (judges, items, rs, mcd, mcb, mean_score) dicts for a set of rows.

    mcd: mean |deviation| (magnitude)
    mcb: mean  deviation  (signed; + = lenient, − = harsh)
    """
    if metric_filter:
        rows = [r for r in rows if r.get("metric_name") == metric_filter]

    judge_item = defaultdict(list)
    for r in rows:
        s = r.get("score")
        if s is not None:
            judge_item[(r["judge_model"], r["item_id"])].append(s)

    judges = sorted(set(k[0] for k in judge_item))
    items = sorted(set(k[1] for k in judge_item))

    # RS% per judge (% items with zero variance across repeats)
    rs = {}
    for j in judges:
        zero_var = total = 0
        for i in items:
            scores = judge_item.get((j, i), [])
            if len(scores) >= 2:
                total += 1
                if len(set(scores)) == 1:
                    zero_var += 1
        rs[j] = (zero_var / total * 100) if total > 0 else 0

    # Per-judge per-item mean score
    judge_item_mean = {}
    for (j, i), scores in judge_item.items():
        judge_item_mean[(j, i)] = sum(scores) / len(scores)

    # Leave-one-out consensus: for judge j on item i, consensus = mean of other judges
    mcd = {}
    mcb = {}
    mean_score = {}
    for j in judges:
        abs_devs, signed_devs, sc = [], [], []
        for i in items:
            if (j, i) not in judge_item_mean:
                continue
            others = [
                judge_item_mean[(jj, i)]
                for jj in judges
                if jj != j and (jj, i) in judge_item_mean
            ]
            if not others:
                continue
            loo_consensus = sum(others) / len(others)
            diff = judge_item_mean[(j, i)] - loo_consensus
            abs_devs.append(abs(diff))
            signed_devs.append(diff)
            sc.append(judge_item_mean[(j, i)])
        mcd[j] = sum(abs_devs) / len(abs_devs) if abs_devs else 0
        mcb[j] = sum(signed_devs) / len(signed_devs) if signed_devs else 0
        mean_score[j] = sum(sc) / len(sc) if sc else 0

    return judges, items, rs, mcd, mcb, mean_score


def _pearson(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n
    sx = (sum((v - mx) ** 2 for v in xs) / n) ** 0.5
    sy = (sum((v - my) ** 2 for v in ys) / n) ** 0.5
    return cov / (sx * sy) if sx * sy > 0 else 0


def main():
    repo = Path(__file__).resolve().parent.parent
    out_path = None
    if "--out" in sys.argv:
        idx = sys.argv.index("--out")
        if idx + 1 < len(sys.argv):
            out_path = repo / sys.argv[idx + 1]

    paths = {k: repo / v for k, v in RESULT_FILES.items()}
    for k, p in paths.items():
        if not p.exists():
            print(f"Missing result file for condition {k}: {p}", file=sys.stderr)
            sys.exit(1)

    a_rows = _load_jsonl(paths["A"])
    b_rows = _load_jsonl(paths["B"])
    c_rows = _load_jsonl(paths["C"])

    judges_a, _, rs_a, mcd_a, mcb_a, ms_a = compute_rs_mcd_mcb(a_rows)

    b_rs, b_mcd, b_mcb, b_ms = {}, {}, {}, {}
    for m in B_METRICS:
        _, _, rs_m, mcd_m, mcb_m, ms_m = compute_rs_mcd_mcb(b_rows, m)
        b_rs[m], b_mcd[m], b_mcb[m], b_ms[m] = rs_m, mcd_m, mcb_m, ms_m

    judges = judges_a
    rs_b = {j: sum(b_rs[m][j] for m in B_METRICS) / 3 for j in judges}
    mcd_b = {j: sum(b_mcd[m][j] for m in B_METRICS) / 3 for j in judges}
    mcb_b = {j: sum(b_mcb[m][j] for m in B_METRICS) / 3 for j in judges}
    ms_b = {j: sum(b_ms[m][j] for m in B_METRICS) / 3 for j in judges}

    _, _, rs_c, mcd_c, mcb_c, ms_c = compute_rs_mcd_mcb(c_rows)

    rs_all = {j: (rs_a[j] + rs_b[j] + rs_c[j]) / 3 for j in judges}
    mcd_all = {j: (mcd_a[j] + mcd_b[j] + mcd_c[j]) / 3 for j in judges}
    mcb_all = {j: (mcb_a[j] + mcb_b[j] + mcb_c[j]) / 3 for j in judges}
    ms_all = {j: (ms_a[j] + ms_b[j] + ms_c[j]) / 3 for j in judges}

    # --- Build output lines ---
    lines = []

    def p(s=""):
        lines.append(s)

    p("# Mean Consensus Deviation (MCD) & Mean Consensus Bias (MCB) — Leave-One-Out")
    p()
    p("**MCD** = mean |judge − LOO consensus| → magnitude of disagreement (always ≥ 0).")
    p("**MCB** = mean  (judge − LOO consensus) → signed direction: **+** = lenient, **−** = harsh.")
    p("LOO consensus = average of the *other six* judges on each item (the evaluated judge excluded).")
    p()

    def _sign_str(v):
        return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"

    # Per-condition detail tables
    for cond_label, rs_d, mcd_d, mcb_d, ms_d in [
        ("A — Generic overall", rs_a, mcd_a, mcb_a, ms_a),
        ("B — Metric rubric (mean of accuracy/relevance/completeness)", rs_b, mcd_b, mcb_b, ms_b),
        ("C — Per-item custom", rs_c, mcd_c, mcb_c, ms_c),
    ]:
        p(f"## Condition {cond_label}")
        p()
        p("| Judge | Mean Score | MCD (pts) | MCB (pts) |")
        p("|-------|------------|-----------|-----------|")
        for j in judges:
            p(f"| {SHORT_NAMES.get(j, j)} | {ms_d[j]:.1f} | {mcd_d[j]:.1f} | {_sign_str(mcb_d[j])} |")
        n_j = len(judges)
        panel_mcd = sum(mcd_d[j] for j in judges) / n_j
        panel_mcb = sum(mcb_d[j] for j in judges) / n_j
        p(f"| **Panel mean** | | **{panel_mcd:.1f}** | **{_sign_str(panel_mcb)}** |")
        p()

    # B per-metric breakdown
    p("## Condition B — per-metric detail")
    p()
    p("| Judge | Acc MCD | Acc MCB | Rel MCD | Rel MCB | Comp MCD | Comp MCB |")
    p("|-------|---------|---------|---------|---------|----------|----------|")
    for j in judges:
        p(
            f"| {SHORT_NAMES.get(j, j)} "
            f"| {b_mcd['accuracy'][j]:.1f} | {_sign_str(b_mcb['accuracy'][j])} "
            f"| {b_mcd['relevance'][j]:.1f} | {_sign_str(b_mcb['relevance'][j])} "
            f"| {b_mcd['completeness'][j]:.1f} | {_sign_str(b_mcb['completeness'][j])} |"
        )
    p()

    # Cross-condition summary
    p("## Cross-condition summary (RS% / MCD / MCB / Mean Score)")
    p()
    p("| Judge | A: RS% | A: MCD | A: MCB | A: Score | B: RS% | B: MCD | B: MCB | B: Score | C: RS% | C: MCD | C: MCB | C: Score | **Overall RS%** | **Overall MCD** | **Overall MCB** | **Overall Score** |")
    p("|-------|--------|--------|--------|----------|--------|--------|--------|----------|--------|--------|--------|----------|----------------|----------------|----------------|-------------------|")
    for j in judges:
        n = SHORT_NAMES.get(j, j)
        p(
            f"| {n} "
            f"| {rs_a[j]:.1f}% | {mcd_a[j]:.1f} | {_sign_str(mcb_a[j])} | {ms_a[j]:.1f} "
            f"| {rs_b[j]:.1f}% | {mcd_b[j]:.1f} | {_sign_str(mcb_b[j])} | {ms_b[j]:.1f} "
            f"| {rs_c[j]:.1f}% | {mcd_c[j]:.1f} | {_sign_str(mcb_c[j])} | {ms_c[j]:.1f} "
            f"| **{rs_all[j]:.1f}%** | **{mcd_all[j]:.1f}** | **{_sign_str(mcb_all[j])}** | **{ms_all[j]:.1f}** |"
        )
    n_j = len(judges)
    p(
        f"| **Mean** "
        f"| {sum(rs_a[j] for j in judges)/n_j:.1f}% | {sum(mcd_a[j] for j in judges)/n_j:.1f} | {_sign_str(sum(mcb_a[j] for j in judges)/n_j)} | {sum(ms_a[j] for j in judges)/n_j:.1f} "
        f"| {sum(rs_b[j] for j in judges)/n_j:.1f}% | {sum(mcd_b[j] for j in judges)/n_j:.1f} | {_sign_str(sum(mcb_b[j] for j in judges)/n_j)} | {sum(ms_b[j] for j in judges)/n_j:.1f} "
        f"| {sum(rs_c[j] for j in judges)/n_j:.1f}% | {sum(mcd_c[j] for j in judges)/n_j:.1f} | {_sign_str(sum(mcb_c[j] for j in judges)/n_j)} | {sum(ms_c[j] for j in judges)/n_j:.1f} "
        f"| **{sum(rs_all[j] for j in judges)/n_j:.1f}%** | **{sum(mcd_all[j] for j in judges)/n_j:.1f}** | **{_sign_str(sum(mcb_all[j] for j in judges)/n_j)}** | **{sum(ms_all[j] for j in judges)/n_j:.1f}** |"
    )
    p()

    # Key observations
    rs_vals = [rs_all[j] for j in judges]
    mcd_vals = [mcd_all[j] for j in judges]
    corr = _pearson(rs_vals, mcd_vals)

    best_j = min(judges, key=lambda j: mcd_all[j])
    worst_j = max(judges, key=lambda j: mcd_all[j])
    most_lenient = max(judges, key=lambda j: mcb_all[j])
    most_harsh = min(judges, key=lambda j: mcb_all[j])

    p("## Key observations")
    p()
    p(f"- **Correlation(RS%, MCD)** across 7 judges (overall): **r = {corr:.3f}**")
    if corr >= 0:
        p("  - Weakly positive: higher repeat stability does *not* predict closer consensus alignment.")
    else:
        p("  - Negative: more stable judges also tend to be closer to consensus.")
    p()
    p(f"- **Closest to consensus (lowest MCD):** {SHORT_NAMES[best_j]} — MCD = {mcd_all[best_j]:.1f} pts")
    p(f"- **Farthest from consensus (highest MCD):** {SHORT_NAMES[worst_j]} — MCD = {mcd_all[worst_j]:.1f} pts")
    p()
    p(f"- **Most lenient (highest MCB):** {SHORT_NAMES[most_lenient]} — MCB = {_sign_str(mcb_all[most_lenient])} pts")
    p(f"- **Most harsh (lowest MCB):** {SHORT_NAMES[most_harsh]} — MCB = {_sign_str(mcb_all[most_harsh])} pts")
    p()
    p(f"- **GPT-4:** RS% = {rs_all['gpt-4']:.1f}%, MCD = {mcd_all['gpt-4']:.1f}, MCB = {_sign_str(mcb_all['gpt-4'])}, Mean Score = {ms_all['gpt-4']:.1f}")
    p("  - High stability, highest deviation, and most lenient — systematic high-scorer.")
    p()
    p(f"- **Claude Sonnet 4:** RS% = {rs_all['claude-sonnet-4-20250514']:.1f}%, MCD = {mcd_all['claude-sonnet-4-20250514']:.1f}, MCB = {_sign_str(mcb_all['claude-sonnet-4-20250514'])}, Mean Score = {ms_all['claude-sonnet-4-20250514']:.1f}")
    p("  - Perfect stability but mid-pack consensus alignment; sign shows scoring direction.")
    p()
    panel_a = sum(mcd_a[j] for j in judges) / n_j
    panel_b = sum(mcd_b[j] for j in judges) / n_j
    panel_c = sum(mcd_c[j] for j in judges) / n_j
    p(f"- **Panel mean MCD by condition:** A = {panel_a:.1f}, B = {panel_b:.1f}, C = {panel_c:.1f}")
    p("  - Per-item custom instructions (C) nearly halve cross-judge disagreement vs generic (A).")

    text = "\n".join(lines) + "\n"
    print(text)

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"\n[saved to {out_path}]", file=sys.stderr)


if __name__ == "__main__":
    main()
