# DSCI 799 – Reliability of LLM-as-a-Judge Evaluation

This repository contains the research artifacts for a DSCI 799 capstone project examining the
reliability of automated LLM-as-a-judge evaluation.

The project studies how repeated executions of identical evaluations on fixed benchmark datasets
(e.g., MT-Bench) can produce varying scores, and investigates when those scores correlate with
human-labeled ground truth and when they become unreliable.

## Goals

- Measure score variance and stability across repeated LLM-as-a-judge runs
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

Structured output (JSON: score 1–10, justification) is used for both. OpenTelemetry records trace/span IDs and token usage per judgment for reliability analysis.

## Repository Structure

- `docs/` – proposal drafts and literature review notes
- `experiments/` – experiment definitions and run configurations
- `data/` – MT-Bench subset and dataset metadata
- `results/` – judge output JSONL files (gitignored)
- `src/` – evaluation scripts (`judge.py`, `run_repeated_judging.py`, `compute_metrics.py`, `otel_setup.py`)
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
- **Run Experiment** – select judge model, K repeats, run the pipeline
- **View Results** – reliability metrics, charts, score distribution
- **Telemetry** – OTEL token usage, span status, per-item variance
- **Manage** – experiment management (placeholder)
