# Maternal Risk Rural Triage

A thesis-ready project for prioritizing maternal referrals in low-resource rural settings.

## Why This Matters
Many maternal outcomes worsen because risk is recognized too late and referral paths are slow. A simple triage tool can help prioritize limited transport and specialist attention.

## What This Project Does
- Simulates maternal risk data with access barriers and obstetric features
- Trains a classification baseline
- Reports precision, recall, F1, ROC-AUC, PR-AUC, and calibration
- Exports thesis-ready artifacts
- Provides an interactive Streamlit triage interface

## Run It
```bash
pip install -r requirements.txt
python src/main.py --export-dir thesis_outputs
streamlit run streamlit_app.py
```

## Thesis Angle
Possible title:
**"Risk-Calibrated Maternal Referral Triage for Rural Health Systems Using Interpretable Machine Learning"**

Possible research questions:
- Which predictors best identify high-risk pregnancies in low-resource contexts?
- How much does travel time change referral performance?
- Does threshold tuning improve sensitivity without overwhelming referral pathways?
