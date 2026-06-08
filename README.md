# Breast Cancer Wisconsin Diagnostic AI

This repository now includes a detailed and working breast cancer screening baseline with:

- A reusable Python pipeline for training, evaluation, and artifact export
- An interactive Streamlit interface for patient-level inference and threshold tuning
- A thesis-ready research proposal and reproducible outputs for writing/reporting

## Repository Layout

```text
breast cancer/
├── medical_ai_loader.py
├── streamlit_app.py
├── THESIS_PROPOSAL_BREAST_AI.md
├── requirements.txt
└── README.md
```

## What The Program Does

1. Loads the authentic Wisconsin Breast Cancer dataset from scikit-learn.
2. Trains an interpretable logistic regression baseline with standardized features.
3. Evaluates clinical metrics: precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and Brier score.
4. Generates precision-recall and calibration data for threshold and risk-quality analysis.
5. Exports thesis-ready artifacts:
   - `thesis_outputs/feature_summary.csv`
   - `thesis_outputs/effect_sizes.csv`
   - `thesis_outputs/evaluation_metrics.json`

## Quick Start

```bash
pip install -r requirements.txt
python medical_ai_loader.py --threshold 0.5 --feature-selection bwwpa --top-k 12 --use-small-dataset --dataset-path data/wisconsin_breast_cancer_live.csv --export-dir thesis_outputs_live
```

## Live Dataset File

This project now supports a real local CSV dataset file:

- `data/wisconsin_breast_cancer_live.csv`

The file is generated from the authentic scikit-learn Wisconsin dataset and can be reused directly for experiments and thesis writeups.

## Run The Interface

```bash
streamlit run streamlit_app.py
```

The interface provides:

- Real-time threshold control for clinical triage scenarios
- BWWPA feature-selection toggle with Top-K control
- Live dataset file path selector
- Confusion matrix and risk-quality plots
- Single-patient inference form for morphology inputs
- Thesis contribution ideas to guide your write-up

## Thesis Direction

Use `THESIS_PROPOSAL_BREAST_AI.md` as your thesis foundation. It includes:

- Problem framing and hypotheses
- Research questions aligned to clinical tradeoffs
- Methodology and experimental plan
- Deliverables you can directly include in a dissertation or publication draft

## Core Metrics Used

- Precision: $\frac{TP}{TP + FP}$
- Recall: $\frac{TP}{TP + FN}$
- F1-score: $2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$
- ROC-AUC and PR-AUC for ranking quality
- Brier score and calibration curve for probability reliability

## Important Note

This is a research and educational baseline, not a medical device and not for clinical decision-making without external validation and regulatory review.
