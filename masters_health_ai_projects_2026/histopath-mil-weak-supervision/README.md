# histopath-mil-weak-supervision

## Goal
Classify whole-slide patch bags with weak labels

## What this project includes
- Reproducible synthetic-data pipeline
- Baseline machine-learning solution in Python
- Evaluation report with clinically relevant metrics
- Thesis-ready research framing

## Quick start
1. Create env and install dependencies:
   pip install -r requirements.txt
2. Run baseline experiment:
   python src/main.py

## Outputs
- metrics.json: evaluation metrics from the latest run
- model.joblib: fitted model artifact

## Thesis idea summary
Weakly supervised multiple-instance learning for histopath triage in low-label oncology settings

## Suggested publication angle
Frame the study as a pragmatic, reproducible baseline for low-resource clinical AI research and include ethics/fairness discussion.
