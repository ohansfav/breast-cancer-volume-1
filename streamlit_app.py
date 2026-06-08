"""Interactive interface for breast cancer diagnostic baseline and thesis exploration."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from medical_ai_loader import BreastCancerDiagnosticPipeline, PipelineConfig


@st.cache_resource
def train_pipeline(threshold: float) -> tuple[BreastCancerDiagnosticPipeline, dict]:
    pipeline = BreastCancerDiagnosticPipeline(config=PipelineConfig(decision_threshold=threshold))
    metrics = pipeline.fit_and_evaluate(decision_threshold=threshold)
    return pipeline, metrics


def build_sidebar_inputs(pipeline: BreastCancerDiagnosticPipeline) -> dict[str, float]:
    default_values = pipeline.get_default_patient_profile()
    st.sidebar.header("Patient Morphology Inputs")
    st.sidebar.caption("Adjust fine needle aspirate morphology values and run model inference.")

    patient_features: dict[str, float] = {}
    for feature_name, default_value in default_values.items():
        patient_features[feature_name] = st.sidebar.number_input(
            label=feature_name,
            value=float(default_value),
            format="%.5f",
        )
    return patient_features


def main() -> None:
    st.set_page_config(page_title="Breast Cancer AI Screening", layout="wide")
    st.title("Breast Cancer Diagnostic AI Interface")
    st.caption(
        "Thesis-ready baseline using Wisconsin FNA morphology with interpretable metrics, "
        "threshold analysis, and single-case risk estimation."
    )

    threshold = st.slider("Decision threshold (benign probability)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    pipeline, metrics = train_pipeline(threshold=threshold)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    col2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("Precision", f"{metrics['precision']:.3f}")

    st.subheader("Confusion Matrix")
    confusion = pd.DataFrame(
        metrics["confusion_matrix"],
        index=["True malignant", "True benign"],
        columns=["Pred malignant", "Pred benign"],
    )
    st.dataframe(confusion, use_container_width=True)

    st.subheader("Precision-Recall Curve")
    pr_curve = pd.DataFrame(
        {
            "recall": metrics["precision_recall_curve"]["recall"],
            "precision": metrics["precision_recall_curve"]["precision"],
        }
    )
    st.line_chart(pr_curve.set_index("recall"))

    st.subheader("Calibration Curve")
    calibration = pd.DataFrame(
        {
            "mean_predicted_value": metrics["calibration_curve"]["mean_predicted_value"],
            "fraction_positives": metrics["calibration_curve"]["fraction_positives"],
        }
    )
    st.line_chart(calibration.set_index("mean_predicted_value"))

    st.subheader("Single-Patient Inference")
    patient_features = build_sidebar_inputs(pipeline)
    if st.button("Run Risk Prediction", type="primary"):
        prediction = pipeline.predict_single_case(patient_features, decision_threshold=threshold)
        st.success(
            "Prediction complete: "
            f"{prediction['predicted_label'].upper()} | "
            f"P(malignant)={prediction['probability_malignant']:.3f}"
        )
        st.json(prediction)

    st.subheader("Thesis Contribution Ideas")
    st.markdown(
        "- Calibrate thresholds for high-recall screening and compare clinical tradeoffs.\n"
        "- Perform subgroup stress tests by perturbing selected morphology features.\n"
        "- Benchmark logistic baseline versus tree/boosting models under same split policy.\n"
        "- Extend to uncertainty-aware referral policy using risk bands (low/medium/high)."
    )


if __name__ == "__main__":
    main()
