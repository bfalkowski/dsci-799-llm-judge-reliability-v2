# DSCI 799 – Repeat Stability in LLM-as-a-Judge Evaluation

This repository contains the research artifacts for a DSCI 799 capstone project on **repeat
stability** in automated LLM-as-a-judge evaluation: how much judge scores move when the same
evaluation is run again under the same settings.

The project studies repeated executions of identical evaluations on fixed benchmark datasets
(e.g., MT-Bench), quantifies **repeat stability** (variance and agreement across repeats—not a
single abstract “reliability” label), and investigates when those scores correlate with
human-labeled ground truth and when they are too noisy to trust.

## Goals

- Measure **repeat stability**: score variance and agreement across repeated LLM-as-a-judge runs
- Analyze correlation between automated judgments and human preference data
- Identify conditions under which LLM-as-a-judge provides meaningful evaluation signals

## Non-Goals

- Training new foundation or judge models
- Optimizing prompt quality beyond what is needed for evaluation consistency

## Setup

From the repository root:

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your API key(s):

- **OpenAI** (gpt-*): `OPENAI_API_KEY` – required for GPT-4o-mini, GPT-4o, etc.
- **Anthropic** (claude-*): `ANTHROPIC_API_KEY` – required for Claude Sonnet, Haiku, Opus

Set the judge model with `JUDGE_MODEL` (default: `gpt-4o-mini`), or choose it in the dashboard.

## Running Experiments

**Via the dashboard:** Open the Run Experiment tab, select judge model and K (repeats), then click Run.

**Via CLI:**

```bash
cd src && python run_repeated_judging.py
```

Override via environment: `JUDGE_MODEL=claude-haiku-4-5-20251001 REPEATS=5 python run_repeated_judging.py`

Outputs are written to `results/` as JSONL (one judgment per line) with OTEL metadata (trace_id, span_id, token usage).

## Judge Support

The judge supports two providers:

| Provider | Models | API Key |
|----------|--------|---------|
| **OpenAI** | gpt-4o-mini, gpt-4o, gpt-4 | OPENAI_API_KEY |
| **Anthropic** | claude-sonnet-4, claude-haiku-4-5, claude-opus-4 | ANTHROPIC_API_KEY |

Structured output (JSON: score 1–10, justification) is used for both. OpenTelemetry records trace/span IDs and token usage per judgment for repeat-stability and cost analysis.

## Repository Structure

- `docs/` – proposal, literature, **`docs/final_plan.example.md`** (plan template; working copy **`docs/final_plan.md`** is gitignored)
- `experiments/` – experiment definitions and run configurations
- `data/` – MT-Bench subset and dataset metadata
- `results/` – judge output JSONL files (gitignored)
- `src/` – evaluation scripts (`judge.py`, `run_repeated_judging.py`, `compute_metrics.py`, `otel_setup.py`, `vendor_billing_csv.py` for dashboard billing CSV parsing)
- `dashboard.py` – Streamlit UI for running experiments and viewing results
- `dashboard_content/` – overview text, captions, and UI copy

## Starting the Dashboard

After [Setup](#setup):

```bash
streamlit run dashboard.py
```

Then open http://localhost:8501 in your browser.

For remote development (e.g. WSL, port forwarding):

```bash
streamlit run dashboard.py --server.address 0.0.0.0
```

Then use the URL Streamlit prints, or your host's forwarded port (e.g. `https://xxx.preview.app.github.dev:8501`).

## Dashboard Tabs

- **Overview** – project stages and goals
- **Dataset & prompts** – view/edit `mt_bench*.json`, per-item `judge_instructions`, prompt preview
- **Run Experiment** – select judge model, K repeats, run the pipeline
- **View Results** – repeat-stability metrics, charts, score distribution (single JSONL)
- **Compare judges & vendors** – multi-file comparison: per-judge metrics, vendor rollups, repeat-stability bar chart (e.g. % zero variance; OpenAI vs Anthropic colors), spread-by-item when item sets align
- **Run summary** – session-level rollups when you multi-select result JSONLs (e.g. conditions A, B, C):
  - Scores and token totals per file; **Repeat stability × economics** table (one row per judge, tokens summed across selected files)
  - **Cost inputs:** vendor default **USD per 1M** input/output rates; optional **per-model overrides** (table after files are selected); invoice totals from manual entry or **Apply** on uploaded billing CSVs (`src/vendor_billing_csv.py`)
  - **Charts:** separate bordered panels for spend, tokens, USD/1M tokens, and pooled **% zero variance**; linear scales for money unless noted
  - **Repeat stability (equal weight per condition):** condition **B** = mean of **% zero variance** over each B **metric** pool; composite = mean of present A / B / C (not every B item weighted as its own condition)
  - **Mean score line chart** across conditions (A, each B metric, C): one line per judge; **y-axis 50–100**; **teal family** (OpenAI) vs **orange family** (Anthropic) with distinct shades per model
- **Telemetry** – OTEL token usage, span status, per-item variance (when traces exist on the JSONL)
- **Manage** – list and delete result files

Python **3.8** is supported (e.g. `Tuple[...]` typing); use `streamlit run dashboard.py` with the same environment you used for `pip install`.
