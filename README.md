<<<<<<< HEAD
# dsci-799-llm-judge-reliability-v2
=======
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

## Repository Structure
- `docs/` – proposal drafts and literature review notes
- `experiments/` – experiment definitions and run configurations
- `data/` – dataset documentation (no raw datasets committed)
- `results/` – analysis outputs, tables, and plots
- `src/` – evaluation and analysis scripts

This repository is designed to support reproducibility and empirical analysis.
>>>>>>> 0515b1b (Add initial repository structure)
