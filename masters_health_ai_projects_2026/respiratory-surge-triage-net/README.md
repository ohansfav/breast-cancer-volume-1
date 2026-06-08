# Respiratory Surge Triage Net

A thesis-ready project for respiratory outbreak triage and surge preparedness.

## Why This Matters
Respiratory outbreaks create spikes in emergency demand. Health systems need triage tools that can identify high-risk patients quickly and consistently.

## What This Project Does
- Simulates respiratory severity data
- Trains a classification baseline
- Reports precision, recall, F1, ROC-AUC, PR-AUC, and calibration
- Exports thesis-ready artifacts
- Provides a Streamlit triage interface

## Run It
```bash
pip install -r requirements.txt
python src/main.py --export-dir thesis_outputs
streamlit run streamlit_app.py
```

## Thesis Angle
Possible title:
**"Threshold-Calibrated Respiratory Surge Triage for Outbreak Preparedness Using Interpretable Machine Learning"**

Possible research questions:
- Which triage variables are most predictive during surge conditions?
- Does threshold tuning improve sensitivity without overwhelming care pathways?
- How can calibration improve confidence in outbreak referral policies?
