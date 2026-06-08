# cervical-screening-selfsupervised

## Goal
Leverage unlabeled cytology-like features via self-supervision

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
Self-supervised representation learning for low-label cancer screening pipelines

## Suggested publication angle
Frame the study as a pragmatic, reproducible baseline for low-resource clinical AI research and include ethics/fairness discussion.
