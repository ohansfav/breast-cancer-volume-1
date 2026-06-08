"""Heat risk alert baseline for climate-health thesis work."""

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
    "temperature_c": "Ambient heat load; the principal exposure variable.",
    "humidity_pct": "Restricts evaporative cooling and worsens heat stress.",
    "wbgt": "Wet-bulb globe temperature proxy for physiological heat strain.",
    "age": "Older adults have reduced thermoregulatory reserve.",
    "chronic_conditions": "Comorbidity burden increases clinical vulnerability.",
    "outdoor_hours": "Exposure duration amplifies cumulative heat load.",
    "hydration_liters": "Lower hydration can increase heat illness risk.",
    "cooling_access": "Access to cooling devices protects against severe heat exposure.",
}


@dataclass(frozen=True)
class Config:
    test_size: float = 0.25
    random_state: int = 42
    decision_threshold: float = 0.5


def make_synthetic_data(n_samples: int = 600, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    temperature_c = rng.normal(35.0, 4.5, n_samples)
    humidity_pct = np.clip(rng.normal(58.0, 16.0, n_samples), 5, 100)
    wbgt = 0.7 * temperature_c + 0.15 * humidity_pct + rng.normal(0, 1.5, n_samples)
    age = np.clip(rng.normal(48, 18, n_samples), 0, 95)
    chronic_conditions = np.clip(rng.poisson(1.1, n_samples), 0, 5)
    outdoor_hours = np.clip(rng.normal(4.0, 2.5, n_samples), 0, 14)
    hydration_liters = np.clip(rng.normal(1.9, 0.7, n_samples), 0.2, 5.0)
    cooling_access = rng.uniform(0, 1, n_samples)

    vulnerability = (
        0.05 * (temperature_c - 30)
        + 0.03 * (humidity_pct - 50)
        + 0.04 * (wbgt - 30)
        + 0.04 * (age / 10)
        + 0.25 * chronic_conditions
        + 0.18 * outdoor_hours
        - 0.30 * hydration_liters
        - 1.6 * cooling_access
        + rng.normal(0, 0.8, n_samples)
    )

    label = (vulnerability > np.quantile(vulnerability, 0.65)).astype(int)
    data = pd.DataFrame(
        {
            "temperature_c": temperature_c,
            "humidity_pct": humidity_pct,
            "wbgt": wbgt,
            "age": age,
            "chronic_conditions": chronic_conditions,
            "outdoor_hours": outdoor_hours,
            "hydration_liters": hydration_liters,
            "cooling_access": cooling_access,
        }
    )
    return data, pd.Series(label, name="heat_risk")


class HeatRiskPipeline:
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
        logger = logging.getLogger("heat_risk_alert")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        samples = 80 if os.getenv("FAST_TEST") == "1" else 240
        self.data, self.target = make_synthetic_data(n_samples=samples, seed=self.config.random_state)
        self.feature_names = list(self.data.columns)
        self.logger.info("Loaded climate-health heat-risk dataset with %d samples.", len(self.data))
        return self.data, self.target

    def feature_summary(self) -> pd.DataFrame:
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
                        n_estimators=20 if os.getenv("FAST_TEST") == "1" else 40,
                        learning_rate=0.08,
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
        self.logger.info("Heat-risk model trained.")
        return model

    def evaluate(self, threshold: float | None = None) -> Dict[str, Any]:
        if self.model is None:
            raise NotFittedError("Model is not trained.")
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("Split the data first.")

        used_threshold = self.config.decision_threshold if threshold is None else threshold
        proba = self.model.predict_proba(self.X_test)[:, 1]
        pred = (proba >= used_threshold).astype(int)
        precision, recall, precision_thresholds = precision_recall_curve(self.y_test, proba)
        frac_pos, mean_pred = calibration_curve(self.y_test, proba, n_bins=10, strategy="quantile")

        metrics: Dict[str, Any] = {
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
            "precision_recall_curve": {
                "precision": precision.tolist(),
                "recall": recall.tolist(),
                "thresholds": precision_thresholds.tolist(),
            },
            "calibration_curve": {
                "fraction_positives": frac_pos.tolist(),
                "mean_predicted_value": mean_pred.tolist(),
            },
        }
        return metrics

    def predict_case(self, row: Dict[str, float], threshold: float | None = None) -> Dict[str, Any]:
        if self.model is None:
            raise NotFittedError("Train the model first.")
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

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary = self.feature_summary()
        metrics = self.evaluate(threshold=threshold)
        summary_path = output_path / "feature_summary.csv"
        metrics_path = output_path / "evaluation_metrics.json"
        summary.to_csv(summary_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return {"summary_csv": str(summary_path), "metrics_json": str(metrics_path), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Heat risk alert baseline")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--export-dir", type=str, default="thesis_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = HeatRiskPipeline()
    artifacts = pipeline.export_artifacts(output_dir=args.export_dir, threshold=args.threshold)
    print("=== Heat Risk Alert Results ===")
    print(f"summary_csv: {artifacts['summary_csv']}")
    print(f"metrics_json: {artifacts['metrics_json']}")
    print(artifacts["metrics"]["classification_report"])


if __name__ == "__main__":
    main()
