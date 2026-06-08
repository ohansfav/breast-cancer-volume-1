"""Streamlit interface for bed demand forecasting."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.main import BedDemandPipeline


@st.cache_resource
def train_pipeline() -> tuple[BedDemandPipeline, dict]:
    pipeline = BedDemandPipeline()
    pipeline.load_data()
    pipeline.split()
    pipeline.train()
    metrics = pipeline.evaluate()
    return pipeline, metrics


def build_inputs(pipeline: BedDemandPipeline) -> dict[str, float]:
    defaults = pipeline.data.median(numeric_only=True).to_dict() if pipeline.data is not None else {}
    st.sidebar.header("Forecast Inputs")
    values: dict[str, float] = {}
    for feature, default_value in defaults.items():
        values[feature] = st.sidebar.number_input(feature, value=float(default_value))
    return values


def main() -> None:
    st.set_page_config(page_title="Bed Demand Forecast", layout="wide")
    st.title("Hospital Bed Demand Forecast")
    st.caption("A thesis-ready tool for capacity planning under health system surge pressure.")

    pipeline, metrics = train_pipeline()
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{metrics['mae']:.3f}")
    c2.metric("RMSE", f"{metrics['rmse']:.3f}")
    c3.metric("R2", f"{metrics['r2']:.3f}")

    st.subheader("Observed vs Predicted")
    chart_df = pd.DataFrame({"observed": metrics["observed"], "predicted": metrics["predictions"]})
    st.line_chart(chart_df)

    st.subheader("Predict a New Day")
    inputs = build_inputs(pipeline)
    if st.button("Forecast Capacity", type="primary"):
        forecast = pipeline.predict_case(inputs)
        st.success(
            f"Predicted bed demand ratio: {forecast['predicted_bed_demand_ratio']:.3f} | {forecast['capacity_flag']}"
        )
        st.json(forecast)

    st.subheader("Thesis Ideas")
    st.markdown(
        "- Compare regression models for short-horizon demand prediction.\n"
        "- Study the effect of community infection pressure on occupancy.\n"
        "- Build an alert policy for surge versus watch states.\n"
        "- Extend to uncertainty-aware forecasting intervals."
    )


if __name__ == "__main__":
    main()
