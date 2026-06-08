"""Streamlit interface for heat-risk alert experimentation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.main import FEATURE_NOTES, HeatRiskPipeline


@st.cache_resource
def train_pipeline(threshold: float) -> tuple[HeatRiskPipeline, dict]:
    pipeline = HeatRiskPipeline()
    pipeline.load_data()
    pipeline.split()
    pipeline.train()
    metrics = pipeline.evaluate(threshold=threshold)
    return pipeline, metrics


def build_case_form(pipeline: HeatRiskPipeline) -> dict[str, float]:
    defaults = pipeline.data.median().to_dict() if pipeline.data is not None else {}
    st.sidebar.header("Exposure and Vulnerability")
    inputs: dict[str, float] = {}
    for feature, default_value in defaults.items():
        note = FEATURE_NOTES.get(feature, "")
        inputs[feature] = st.sidebar.number_input(feature, value=float(default_value), help=note)
    return inputs


def main() -> None:
    st.set_page_config(page_title="Heat Risk Alert", layout="wide")
    st.title("Climate-Health Heat Risk Alert")
    st.caption("A thesis-ready screening tool for climate-driven heat illness triage.")

    threshold = st.slider("High-risk decision threshold", 0.1, 0.9, 0.5, 0.05)
    pipeline, metrics = train_pipeline(threshold)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    c3.metric("Recall", f"{metrics['recall']:.3f}")
    c4.metric("Brier", f"{metrics['brier_score']:.3f}")

    st.subheader("Confusion Matrix")
    st.dataframe(pd.DataFrame(metrics["confusion_matrix"]), use_container_width=True)

    st.subheader("Calibration Curve")
    cal = pd.DataFrame(metrics["calibration_curve"])
    st.line_chart(cal.set_index("mean_predicted_value"))

    st.subheader("Single-Case Risk Estimate")
    patient = build_case_form(pipeline)
    if st.button("Score Heat Risk", type="primary"):
        result = pipeline.predict_case(patient, threshold=threshold)
        st.success(f"Predicted risk: {result['risk_label']} ({result['risk_probability']:.3f})")
        st.json(result)

    st.subheader("Thesis Ideas")
    st.markdown(
        "- Study alert thresholds for public-health referral policy.\n"
        "- Quantify the contribution of cooling access and hydration.\n"
        "- Test fairness across age and chronic-disease subgroups.\n"
        "- Add neighborhood heat-index data in future external validation."
    )


if __name__ == "__main__":
    main()
