"""Streamlit interface for maternal risk triage."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.main import FEATURE_NOTES, MaternalRiskPipeline


@st.cache_resource
def train_pipeline(threshold: float) -> tuple[MaternalRiskPipeline, dict]:
    pipeline = MaternalRiskPipeline()
    pipeline.load_data()
    pipeline.split()
    pipeline.train()
    metrics = pipeline.evaluate(threshold=threshold)
    return pipeline, metrics


def build_case_form(pipeline: MaternalRiskPipeline) -> dict[str, float]:
    defaults = pipeline.data.median(numeric_only=True).to_dict() if pipeline.data is not None else {}
    st.sidebar.header("Maternal Triage Inputs")
    values: dict[str, float] = {}
    for feature, default_value in defaults.items():
        values[feature] = st.sidebar.number_input(feature, value=float(default_value), help=FEATURE_NOTES.get(feature, ""))
    return values


def main() -> None:
    st.set_page_config(page_title="Maternal Risk Rural Triage", layout="wide")
    st.title("Maternal Risk Rural Triage")
    st.caption("A thesis-ready referral prioritization tool for low-resource maternal care settings.")

    threshold = st.slider("High-risk threshold", 0.1, 0.9, 0.5, 0.05)
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

    st.subheader("Single-Case Referral Score")
    case = build_case_form(pipeline)
    if st.button("Score Referral Risk", type="primary"):
        result = pipeline.predict_case(case, threshold=threshold)
        st.success(f"Predicted risk: {result['risk_label']} ({result['risk_probability']:.3f})")
        st.json(result)

    st.subheader("Thesis Ideas")
    st.markdown(
        "- Compare referral thresholds for recall-focused maternal triage.\n"
        "- Study the role of access barriers such as travel time.\n"
        "- Add fairness analysis by age and care-access subgroups.\n"
        "- Extend with external rural clinic validation later."
    )


if __name__ == "__main__":
    main()
