## Purpose

This dashboard supports the **DSCI 799** capstone study on **repeat stability** in LLM-as-a-judge evaluation. We hold **questions and reference responses** fixed, ask one or more **judge models** to score the same answer **multiple times** under identical conditions, and measure whether scores **hold still**.

The primary metric is **Repeat Stability (RS%)** — the percentage of items where a judge produces the exact same integer score across all K repeats. A secondary analysis uses **Mean Consensus Deviation (MCD)** and **Mean Consensus Bias (MCB)** to quantify cross-judge agreement and directional scoring tendency.

---

## Experimental design

We use a **frozen** slice of **MT-Bench**-style items (`mt_bench_subset.json` or `mt_bench_full.json`). Every condition and every judge sees the **same items** for a fair comparison.

**Three judging conditions** (logged on every row as `condition_name`):

| Condition | Idea |
|-----------|------|
| **A — Generic overall** | One holistic 0–100 score from a single prompt. Per-item custom text is **not** injected. |
| **B — Metric rubric** | Separate scores per **dimension** (accuracy, relevance, completeness). Each dimension is its own judge call per item per repeat. Only metrics defined in `metric_rubric.py` are allowed. |
| **C — Per-item custom** | Per-item `judge_instructions` are prepended to the prompt. |

**Repeat schedule (K):** repeats use **round-robin** ordering — all items for repeat **0**, then all for repeat **1**, etc. — so the same prompt is not sent K times back-to-back.

---

## Key metrics

- **Repeat Stability (RS%)** — within a judge: share of items with **zero variance** across K repeats (identical integer scores every time).
- **Mean Consensus Deviation (MCD)** — across judges: average absolute distance from **leave-one-out** panel consensus per item. Lower = closer to the group.
- **Mean Consensus Bias (MCB)** — same as MCD but **signed**: **positive = lenient** (scores above consensus), **negative = harsh** (below). Shows systematic scoring direction.

---

## Workflow

1. **Dataset & prompts** — View the frozen JSON; preview judge prompts for conditions A, B, C.
2. **View results** — Single-file analysis: repeat stability metrics, score distribution.
3. **Run summary** — Session-level view: RS% by condition, mean score profiles, MCD & MCB charts.

---

## Bottom line

The dashboard runs and analyzes the **7 judge models × 3 conditions × K repeats** grid on a frozen benchmark, then reports **within-judge repeat stability** and **cross-judge consensus alignment** without mixing conditions or datasets.
