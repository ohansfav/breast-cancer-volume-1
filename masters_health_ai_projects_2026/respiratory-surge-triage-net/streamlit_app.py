"""Streamlit interface for respiratory surge triage."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.main import FEATURE_NOTES, RespiratorySurgePipeline


@st.cache_resource
def train_pipeline(threshold: float) -> tuple[RespiratorySurgePipeline, dict]:
    pipeline = RespiratorySurgePipeline()
    pipeline.load_data()
    pipeline.split()
    pipeline.train()
    metrics = pipeline.evaluate(threshold=threshold)
    return pipeline, metrics


def build_case_form(pipeline: RespiratorySurgePipeline) -> dict[str, float]:
    defaults = pipeline.data.median(numeric_only=True).to_dict() if pipeline.data is not None else {}
    st.sidebar.header("Respiratory Triage Inputs")
    values: dict[str, float] = {}
    for feature, default_value in defaults.items():
        values[feature] = st.sidebar.number_input(feature, value=float(default_value), help=FEATURE_NOTES.get(feature, ""))
    return values


def main() -> None:
    st.set_page_config(page_title="Respiratory Surge Triage", layout="wide")
    st.title("Respiratory Surge Triage Net")
    st.caption("A thesis-ready triage baseline for respiratory outbreaks and surge conditions.")

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

    st.subheader("Single-Case Triage Score")
    case = build_case_form(pipeline)
    if st.button("Score Respiratory Risk", type="primary"):
        result = pipeline.predict_case(case, threshold=threshold)
        st.success(f"Predicted risk: {result['risk_label']} ({result['risk_probability']:.3f})")
        st.json(result)

    st.subheader("Thesis Ideas")
    st.markdown(
        "- Test threshold policies during simulated outbreak surges.\n"
        "- Study oxygen saturation and exposure effects on severity.\n"
        "- Add abstention policy for borderline patients.\n"
        "- Extend to real ED triage and syndromic surveillance data."
    )


if __name__ == "__main__":
    main()
