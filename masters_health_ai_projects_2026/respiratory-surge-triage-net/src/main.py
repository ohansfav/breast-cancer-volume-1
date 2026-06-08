"""Respiratory surge triage baseline for outbreak preparedness."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.exceptions import NotFittedError
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


FEATURE_NOTES: Dict[str, str] = {
    "resp_rate": "Respiratory rate from triage assessment.",
    "spo2": "Oxygen saturation is a core severity marker.",
    "fever_c": "Fever indicates possible infection burden.",
    "cough_days": "Symptom duration proxy.",
    "age": "Age influences respiratory vulnerability.",
    "chronic_lung": "Underlying lung disease increases risk.",
    "crowding_exposure": "High-exposure environments raise transmission risk.",
    "vaccination_gap": "Lower vaccination can increase severe disease risk.",
}


@dataclass(frozen=True)
class Config:
    test_size: float = 0.25
    random_state: int = 42
    decision_threshold: float = 0.5


def make_synthetic_data(n_samples: int = 520, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    resp_rate = np.clip(rng.normal(20, 5, n_samples), 10, 40)
    spo2 = np.clip(rng.normal(95, 4, n_samples), 72, 100)
    fever_c = np.clip(rng.normal(37.6, 0.9, n_samples), 35, 41)
    cough_days = np.clip(rng.poisson(4, n_samples), 0, 20)
    age = np.clip(rng.normal(46, 19, n_samples), 0, 95)
    chronic_lung = np.clip(rng.poisson(0.7, n_samples), 0, 4)
    crowding_exposure = rng.uniform(0, 1, n_samples)
    vaccination_gap = rng.uniform(0, 1, n_samples)

    severity = (
        0.12 * (resp_rate - 18)
        - 0.16 * (spo2 - 92)
        + 0.8 * (fever_c - 37)
        + 0.08 * cough_days
        + 0.03 * (age - 40)
        + 0.7 * chronic_lung
        + 0.9 * crowding_exposure
        + 0.8 * vaccination_gap
        + rng.normal(0, 0.9, n_samples)
    )

    label = (severity > np.quantile(severity, 0.68)).astype(int)
    data = pd.DataFrame(
        {
            "resp_rate": resp_rate,
            "spo2": spo2,
            "fever_c": fever_c,
            "cough_days": cough_days,
            "age": age,
            "chronic_lung": chronic_lung,
            "crowding_exposure": crowding_exposure,
            "vaccination_gap": vaccination_gap,
        }
    )
    return data, pd.Series(label, name="respiratory_severity")


class RespiratorySurgePipeline:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.logger = self._setup_logger()
        self.data: pd.DataFrame | None = None
        self.target: pd.Series | None = None
        self.model: Pipeline | None = None
        self.feature_names: list[str] = []
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("respiratory_surge_triage")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        samples = 80 if os.getenv("FAST_TEST") == "1" else 240
        self.data, self.target = make_synthetic_data(samples, self.config.random_state)
        self.feature_names = list(self.data.columns)
        return self.data, self.target

    def summary(self) -> pd.DataFrame:
        if self.data is None:
            raise RuntimeError("Load data first.")
        summary = self.data.describe().T
        summary["missing_values"] = self.data.isna().sum()
        return summary

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.data is None or self.target is None:
            raise RuntimeError("Load data first.")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.data,
            self.target,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=self.target,
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_model(self) -> Pipeline:
        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        random_state=self.config.random_state,
                        n_estimators=20 if os.getenv("FAST_TEST") == "1" else 60,
                        learning_rate=0.07,
                        max_depth=3,
                    ),
                ),
            ]
        )
        return self.model

    def train(self) -> Pipeline:
        if self.X_train is None or self.y_train is None:
            self.split()
        model = self.build_model()
        model.fit(self.X_train, self.y_train)
        return model

    def evaluate(self, threshold: float | None = None) -> Dict[str, Any]:
        if self.model is None:
            raise NotFittedError("Model not trained.")
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("Split first.")
        used_threshold = self.config.decision_threshold if threshold is None else threshold
        proba = self.model.predict_proba(self.X_test)[:, 1]
        pred = (proba >= used_threshold).astype(int)
        precision, recall, thresholds = precision_recall_curve(self.y_test, proba)
        frac_pos, mean_pred = calibration_curve(self.y_test, proba, n_bins=10, strategy="quantile")
        return {
            "decision_threshold": float(used_threshold),
            "accuracy": float(accuracy_score(self.y_test, pred)),
            "precision": float(precision_score(self.y_test, pred)),
            "recall": float(recall_score(self.y_test, pred)),
            "f1_score": float(f1_score(self.y_test, pred)),
            "roc_auc": float(roc_auc_score(self.y_test, proba)),
            "pr_auc": float(average_precision_score(self.y_test, proba)),
            "brier_score": float(brier_score_loss(self.y_test, proba)),
            "confusion_matrix": confusion_matrix(self.y_test, pred).tolist(),
            "classification_report": classification_report(self.y_test, pred),
            "precision_recall_curve": {"precision": precision.tolist(), "recall": recall.tolist(), "thresholds": thresholds.tolist()},
            "calibration_curve": {"fraction_positives": frac_pos.tolist(), "mean_predicted_value": mean_pred.tolist()},
        }

    def predict_case(self, row: Dict[str, float], threshold: float | None = None) -> Dict[str, Any]:
        if self.model is None:
            raise NotFittedError("Train model first.")
        frame = pd.DataFrame([{feature: float(row[feature]) for feature in self.feature_names}])
        proba = float(self.model.predict_proba(frame)[0, 1])
        used_threshold = self.config.decision_threshold if threshold is None else threshold
        return {
            "risk_label": "high" if proba >= used_threshold else "lower",
            "risk_probability": proba,
            "decision_threshold": float(used_threshold),
        }

    def export_artifacts(self, output_dir: str | Path = "thesis_outputs", threshold: float | None = None) -> Dict[str, Any]:
        if self.data is None:
            self.load_data()
        if self.model is None:
            self.split()
            self.train()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        metrics = self.evaluate(threshold=threshold)
        summary_path = out / "feature_summary.csv"
        metrics_path = out / "evaluation_metrics.json"
        summary.to_csv(summary_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return {"summary_csv": str(summary_path), "metrics_json": str(metrics_path), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Respiratory surge triage baseline")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--export-dir", type=str, default="thesis_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = RespiratorySurgePipeline()
    artifacts = pipeline.export_artifacts(output_dir=args.export_dir, threshold=args.threshold)
    print("=== Respiratory Surge Triage ===")
    print(f"summary_csv: {artifacts['summary_csv']}")
    print(f"metrics_json: {artifacts['metrics_json']}")
    print(artifacts["metrics"]["classification_report"])


if __name__ == "__main__":
    main()
