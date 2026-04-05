## Purpose

This app supports **LLM-as-a-judge reliability** work for **DSCI 799**: we hold **questions and reference responses** fixed, ask one or more **judge models** to score the same answer **many times**, and quantify whether scores **hold still** (within a judge) and how much judges **disagree** with each other (across models). Raw outputs are **JSONL judgment records** suitable for analysis and reproducibility.

---

## Experimental design

We use a **frozen** slice of **MT-Bench**-style items (`mt_bench_subset.json` or `mt_bench_full.json`). Every condition and every judge sees the **same items** for a fair comparison.

**Three judging conditions** (logged on every row as `condition_name`):

| Condition | Idea |
|-----------|------|
| **A — Generic overall** | One holistic 0–100 score from a single prompt. Per-item custom text is **not** injected. |
| **B — Metric rubric** | Separate scores per **dimension** (accuracy, relevance, completeness). Each dimension is its own judge call per item per repeat. Only metrics defined in `metric_rubric.py` are allowed (no silent fallbacks). |
| **C — Per-item custom** | For each item, optional `judge_instructions` are prepended to the prompt (manually or via **Generate custom judge instructions** on the dataset tab). |

**Repeat schedule (K):** for each judge model, the runner uses **round-robin** ordering: it scores **every item** for repeat **0**, then every item for repeat **1**, and so on (per metric under **B**). That way the **same prompt** is not sent **K** times back-to-back, which reduces reliance on identical consecutive requests (e.g. provider caching or correlated failures). Each JSONL row still stores the correct `idx` repeat index.

Runs are tagged in filenames with `_cond-gen_`, `_cond-metric_`, or `_cond-custom_`, and in rows with `dataset_id`, `score_min` / `score_max`, and (for B) `metric_name`.

---

## Workflow in this dashboard

1. **Dataset & prompts** — View or edit the frozen JSON; generate or overwrite **per-item instructions** for Condition C if needed.  
2. **Run experiment** — Choose condition, dataset (subset vs full), judge model, **K** repeats, temperature, and (for B) which **metrics** to score. Expect **items × K** judge calls for A/C, and **items × K × (#metrics)** for B. Repeats run in **round-robin** order (all items for `idx=0`, then all for `idx=1`, …).  
3. **View results** — Pick **condition** and **dataset** filters, then one result file. For Condition B, choose which **metric** to plot (variance and agreement are computed **within** that metric).  
4. **Compare judges** — Select **condition** and **dataset** first so only compatible files appear; then choose **two or more** result files. For B, pick a **metric** shared by all files. Tables report **mean score**, **mean variance**, **mean within-item SD** (typical repeat spread in points), **% items with zero variance**, and **repeat agreement (exact)**. Charts summarize stability and **cross-judge spread per item**.  
5. **Telemetry / Manage** — Optional tracing and housekeeping.

---

## Reliability metrics (short definitions)

- **Within a judge, same item:** we compare the **K** integer scores (0–100). **Mean variance** / **mean within-item SD** summarize how much repeats **jitter**. **Repeat agreement (exact)** is the average fraction of repeat pairs that match **exactly**—not the same thing as **mean score**, which can be a decimal because it averages many integers **across** items and repeats.  
- **Across judges:** after fixing condition, dataset, and (if B) metric, **spread** is roughly how far apart the judges’ **per-item means** fall on the scale.

---

## Data sources

MT-Bench question and reference material can be obtained from the **FastChat** MT-Bench assets (e.g. `question.jsonl`, reference answers); processed files live under `data/` with the `mt_bench*.json` naming used by the runner.

---

## Bottom line

The dashboard is the **control room** for running the **7 judge models × 3 conditions × K repeats** grid on a frozen benchmark slice, then inspecting **within-judge stability** and **between-judge disagreement** without mixing conditions or datasets by accident.
