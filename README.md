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

Copy `.env.example` to `.env` and add your `OPENAI_API_KEY` for experiments that call the judge API.

## Repository Structure

- `docs/` – proposal drafts and literature review notes
- `experiments/` – experiment definitions and run configurations
- `data/` – dataset documentation (no raw datasets committed)
- `results/` – analysis outputs, tables, and plots
- `src/` – evaluation and analysis scripts
- `dashboard.py` – Streamlit UI for running experiments and viewing results (stub)
- `dashboard_content/` – text and copy for the dashboard (edit these to change UI text)

## Starting the Dashboard UI

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

**Note:** The dashboard is currently a stub with TODOs. Content in `dashboard_content/` is ready to display once the stub is implemented.
