"""Thesis-ready breast cancer diagnostic baseline with reusable API and CLI support."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


CLINICAL_FEATURE_NOTES: Dict[str, str] = {
    "mean radius": "Proxy for lesion size; larger cell clusters can indicate aggressive growth.",
    "mean texture": "Captures grayscale heterogeneity linked to tissue disorganization.",
    "mean perimeter": "Represents boundary extent; irregular elongation may reflect invasion.",
    "mean area": "Two-dimensional lesion footprint; larger areas are often more suspicious.",
    "mean smoothness": "Boundary regularity; rough contours can correlate with malignancy.",
    "mean compactness": "Shape compactness; lower compactness can indicate structural distortion.",
    "mean concavity": "Degree of inward boundary curvature associated with irregular masses.",
    "mean concave points": "Frequency of concave segments, often elevated in malignant lesions.",
}


@dataclass(frozen=True)
class PipelineConfig:
    test_size: float = 0.2
    random_state: int = 42
    decision_threshold: float = 0.5


class BreastCancerDiagnosticPipeline:
    """Reusable pipeline for model training, evaluation, and thesis artifact export."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.logger = self._setup_logger()
        self.data: pd.DataFrame | None = None
        self.target: pd.Series | None = None
        self.target_names: np.ndarray | None = None
        self.feature_names: np.ndarray | None = None
        self.pipeline: Pipeline | None = None
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("breast_cancer_pipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        dataset = load_breast_cancer(as_frame=True)
        self.data = dataset.frame.copy()
        self.target = self.data["target"]
        self.target_names = dataset.target_names
        self.feature_names = dataset.feature_names
        self.logger.info("Loaded %d samples and %d features.", self.data.shape[0], len(self.feature_names))
        return self.data, self.target

    def inspect_features(self) -> None:
        if self.data is None:
            raise RuntimeError("Data must be loaded before feature inspection.")
        for feature, note in CLINICAL_FEATURE_NOTES.items():
            if feature in self.data.columns:
                self.logger.info("%s -> %s", feature, note)

    def summarize_data(self) -> pd.DataFrame:
        if self.data is None or self.target is None:
            raise RuntimeError("Data must be loaded before summarization.")
        summary = self.data.drop(columns=["target"]).describe().T
        summary["missing_values"] = self.data.drop(columns=["target"]).isna().sum()
        return summary

    def baseline_statistical_differentiation(self) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("Data must be loaded before statistical differentiation.")
        features = self.data.drop(columns=["target"])
        malignant = features[self.data["target"] == 0]
        benign = features[self.data["target"] == 1]

        malignant_mean = malignant.mean()
        benign_mean = benign.mean()
        pooled_std = np.sqrt((malignant.var(ddof=1) + benign.var(ddof=1)) / 2).replace(0, np.nan)
        cohen_d = (malignant_mean - benign_mean) / pooled_std

        return pd.DataFrame(
            {
                "malignant_mean": malignant_mean,
                "benign_mean": benign_mean,
                "mean_difference": malignant_mean - benign_mean,
                "cohen_d": cohen_d,
            }
        ).sort_values(by="cohen_d", key=lambda s: s.abs(), ascending=False)

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.data is None or self.target is None:
            raise RuntimeError("Data must be loaded before splitting.")
        X = self.data.drop(columns=["target"])
        y = self.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_model(self) -> Pipeline:
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=2000, solver="liblinear", random_state=self.config.random_state),
                ),
            ]
        )
        return self.pipeline

    def train_model(self) -> Pipeline:
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Training data is not ready. Call split_data first.")
        model = self.build_model()
        model.fit(self.X_train, self.y_train)
        self.logger.info("Model training completed.")
        return model

    def evaluate_model(self, decision_threshold: float | None = None) -> Dict[str, Any]:
        if self.pipeline is None:
            raise NotFittedError("The model has not been trained yet.")
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("Testing data is not available. Call split_data first.")

        threshold = decision_threshold if decision_threshold is not None else self.config.decision_threshold
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        precision, recall, pr_thresholds = precision_recall_curve(self.y_test, y_proba)
        frac_pos, mean_pred = calibration_curve(self.y_test, y_proba, n_bins=10, strategy="quantile")

        matrix = confusion_matrix(self.y_test, y_pred)
        metrics: Dict[str, Any] = {
            "decision_threshold": float(threshold),
            "accuracy": float(accuracy_score(self.y_test, y_pred)),
            "precision": float(precision_score(self.y_test, y_pred)),
            "recall": float(recall_score(self.y_test, y_pred)),
            "f1_score": float(f1_score(self.y_test, y_pred)),
            "roc_auc": float(roc_auc_score(self.y_test, y_proba)),
            "pr_auc": float(average_precision_score(self.y_test, y_proba)),
            "brier_score": float(brier_score_loss(self.y_test, y_proba)),
            "confusion_matrix": matrix.tolist(),
            "classification_report": classification_report(self.y_test, y_pred, target_names=self.target_names),
            "precision_recall_curve": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": pr_thresholds.tolist(),
            },
            "calibration_curve": {
                "fraction_positives": frac_pos.tolist(),
                "mean_predicted_value": mean_pred.tolist(),
            },
        }

        self.logger.info("Evaluation complete: recall=%.4f precision=%.4f roc_auc=%.4f", metrics["recall"], metrics["precision"], metrics["roc_auc"])
        return metrics

    def fit_and_evaluate(self, decision_threshold: float | None = None) -> Dict[str, Any]:
        self.load_data()
        self.inspect_features()
        self.split_data()
        self.train_model()
        return self.evaluate_model(decision_threshold=decision_threshold)

    def get_default_patient_profile(self) -> Dict[str, float]:
        if self.data is None:
            self.load_data()
        assert self.data is not None
        medians = self.data.drop(columns=["target"]).median()
        return {k: float(v) for k, v in medians.to_dict().items()}

    def predict_single_case(self, patient_features: Dict[str, float], decision_threshold: float | None = None) -> Dict[str, Any]:
        if self.pipeline is None:
            raise NotFittedError("Model has not been trained. Call fit_and_evaluate first.")
        if self.feature_names is None:
            raise RuntimeError("Feature names unavailable. Ensure dataset is loaded.")

        row = {feature: float(patient_features[feature]) for feature in self.feature_names}
        frame = pd.DataFrame([row])
        prob_benign = float(self.pipeline.predict_proba(frame)[0, 1])
        threshold = decision_threshold if decision_threshold is not None else self.config.decision_threshold
        pred = int(prob_benign >= threshold)
        label = "benign" if pred == 1 else "malignant"

        return {
            "predicted_label": label,
            "probability_benign": prob_benign,
            "probability_malignant": 1.0 - prob_benign,
            "decision_threshold": float(threshold),
        }

    def export_thesis_artifacts(self, output_dir: str | Path = "thesis_outputs", decision_threshold: float | None = None) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.data is None:
            self.load_data()
        summary = self.summarize_data()
        effect_table = self.baseline_statistical_differentiation()
        metrics = self.fit_and_evaluate(decision_threshold=decision_threshold)

        summary_file = output_path / "feature_summary.csv"
        effects_file = output_path / "effect_sizes.csv"
        metrics_file = output_path / "evaluation_metrics.json"

        summary.to_csv(summary_file)
        effect_table.to_csv(effects_file)
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        self.logger.info("Thesis artifacts exported to %s", output_path.resolve())
        return {
            "summary_csv": str(summary_file),
            "effect_sizes_csv": str(effects_file),
            "metrics_json": str(metrics_file),
            "metrics": metrics,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breast cancer diagnostic baseline and thesis artifact exporter")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for benign class probability")
    parser.add_argument("--export-dir", type=str, default="thesis_outputs", help="Directory for thesis-ready outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = BreastCancerDiagnosticPipeline(config=PipelineConfig(decision_threshold=args.threshold))
    artifacts = pipeline.export_thesis_artifacts(output_dir=args.export_dir, decision_threshold=args.threshold)
    print("\n=== Thesis Artifact Export Completed ===")
    print(f"summary_csv: {artifacts['summary_csv']}")
    print(f"effect_sizes_csv: {artifacts['effect_sizes_csv']}")
    print(f"metrics_json: {artifacts['metrics_json']}")
    print("\n=== Final Diagnostic Metrics ===")
    print(artifacts["metrics"]["classification_report"])


if __name__ == "__main__":
    main()
