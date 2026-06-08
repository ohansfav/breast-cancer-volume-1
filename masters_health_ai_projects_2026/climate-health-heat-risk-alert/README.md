# Climate Health Heat Risk Alert

A thesis-ready climate-health project that predicts heat illness risk using a transparent machine learning baseline and an interactive Streamlit interface.

## Why This Matters
Extreme heat is one of the fastest-growing climate-health risks. Communities need early warning systems that are simple, interpretable, and adaptable to local vulnerability.

## What This Project Does
- Simulates climate and vulnerability data for a heat-risk screening task
- Trains a calibrated, interpretable classifier
- Reports precision, recall, F1, ROC-AUC, PR-AUC, Brier score, and calibration curves
- Exports thesis-ready artifacts for analysis and writing
- Provides a Streamlit interface for threshold tuning and single-case scoring

## Run It
```bash
pip install -r requirements.txt
python src/main.py --threshold 0.5 --export-dir thesis_outputs
streamlit run streamlit_app.py
```

## Thesis Angle
Possible thesis title:
**"Threshold-Calibrated Heat Risk Alerts for Climate Vulnerability Screening in Resource-Constrained Settings"**

Possible research questions:
- Which vulnerability variables most strongly predict heat illness risk?
- How should thresholds change to prioritize recall during extreme heat events?
- Does calibration improve the trustworthiness of public-health alerts?

## Artifacts
- `thesis_outputs/feature_summary.csv`
- `thesis_outputs/evaluation_metrics.json`

## Note
This is a research baseline, not a deployment-ready medical or emergency system.
