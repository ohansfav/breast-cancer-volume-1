# Hospital Bed Demand Forecast

A thesis-ready project for predicting hospital bed pressure under surge conditions.

## Why This Matters
Health systems are still struggling with demand spikes caused by outbreaks, seasonal waves, and staffing shortages. Bed planning is an unresolved operational challenge in many settings.

## What This Project Does
- Simulates a short-horizon bed demand forecasting dataset
- Trains a regression baseline
- Reports MAE, RMSE, and R2
- Exports thesis-ready CSV and JSON artifacts
- Provides a Streamlit forecasting interface

## Run It
```bash
pip install -r requirements.txt
python src/main.py --export-dir thesis_outputs
streamlit run streamlit_app.py
```

## Thesis Angle
Possible title:
**"Short-Horizon Hospital Bed Demand Forecasting for Surge Preparedness Using Interpretable Machine Learning"**

Possible research questions:
- Which operational and community factors most influence bed pressure?
- Can a simple regression baseline provide usable surge warnings?
- How does forecast error change under high-demand periods?

## Artifacts
- `thesis_outputs/feature_summary.csv`
- `thesis_outputs/evaluation_metrics.json`
