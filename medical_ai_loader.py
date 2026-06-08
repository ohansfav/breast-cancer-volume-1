"""Thesis-ready breast cancer diagnostic baseline with BWWPA feature selection support."""

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


DEFAULT_SMALL_DATASET_PATH = Path("data") / "wisconsin_breast_cancer_live.csv"


@dataclass(frozen=True)
class PipelineConfig:
    test_size: float = 0.2
    random_state: int = 42
    decision_threshold: float = 0.5
    feature_selection: str = "all"
    top_k: int = 0


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
        self.selected_features: list[str] = []
        self.feature_scores: pd.DataFrame | None = None

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

    def create_small_local_dataset(self, output_path: str | Path = DEFAULT_SMALL_DATASET_PATH, n_rows: int = 220) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        dataset = load_breast_cancer(as_frame=True)
        frame = dataset.frame.copy().sample(n=n_rows, random_state=self.config.random_state)
        frame.to_csv(output, index=False)
        self.logger.info("Saved small local dataset: %s", output)
        return output

    def load_data(self, use_small_dataset: bool = False, dataset_path: str | Path | None = None) -> Tuple[pd.DataFrame, pd.Series]:
        if use_small_dataset:
            path = Path(dataset_path) if dataset_path is not None else DEFAULT_SMALL_DATASET_PATH
            if not path.exists():
                self.create_small_local_dataset(output_path=path)
            frame = pd.read_csv(path)
            self.data = frame.copy()
            self.target = self.data["target"].astype(int)
            self.target_names = np.array(["malignant", "benign"])
            self.feature_names = np.array([c for c in self.data.columns if c != "target"])
            self.logger.info("Loaded local dataset %s with %d samples.", path, len(self.data))
            return self.data, self.target

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

        table = pd.DataFrame(
            {
                "malignant_mean": malignant_mean,
                "benign_mean": benign_mean,
                "mean_difference": malignant_mean - benign_mean,
                "abs_mean_difference": (malignant_mean - benign_mean).abs(),
                "cohen_d": cohen_d,
            }
        )
        return table.sort_values(by="cohen_d", key=lambda s: s.abs(), ascending=False)

    @staticmethod
    def _minmax(values: pd.Series) -> pd.Series:
        vmin = values.min()
        vmax = values.max()
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            return pd.Series(np.ones(len(values)), index=values.index)
        return (values - vmin) / (vmax - vmin)

    def bwwpa_feature_scores(self, diff_table: pd.DataFrame) -> pd.DataFrame:
        """Compute a BWWPA-inspired weighted feature score.

        This approximates a balanced weighted winner aggregation by combining
        standardized effect size and absolute cohort shift.
        """

        scores = diff_table.copy()
        scores["abs_cohen_d"] = scores["cohen_d"].abs().fillna(0.0)
        scores["norm_abs_cohen_d"] = self._minmax(scores["abs_cohen_d"])
        scores["norm_abs_mean_diff"] = self._minmax(scores["abs_mean_difference"].fillna(0.0))
        scores["bwwpa_score"] = 0.7 * scores["norm_abs_cohen_d"] + 0.3 * scores["norm_abs_mean_diff"]
        return scores.sort_values("bwwpa_score", ascending=False)

    def select_features(self, selection_method: str | None = None, top_k: int | None = None) -> list[str]:
        if self.data is None:
            raise RuntimeError("Data must be loaded before feature selection.")

        method = (selection_method or self.config.feature_selection).lower()
        k = self.config.top_k if top_k is None else top_k
        k = max(0, int(k))

        all_features = [c for c in self.data.columns if c != "target"]
        if method == "all" or k == 0:
            self.selected_features = all_features
            self.feature_scores = None
            return self.selected_features

        diff_table = self.baseline_statistical_differentiation()
        if method == "cohen_d":
            ranked = diff_table.sort_values(by="cohen_d", key=lambda s: s.abs(), ascending=False)
            self.feature_scores = ranked
        elif method == "bwwpa":
            ranked = self.bwwpa_feature_scores(diff_table)
            self.feature_scores = ranked
        else:
            self.selected_features = all_features
            self.feature_scores = None
            return self.selected_features

        used_k = min(k, len(ranked))
        self.selected_features = ranked.head(used_k).index.tolist()
        return self.selected_features

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        if self.data is None or self.target is None:
            raise RuntimeError("Data must be loaded before splitting.")
        if not self.selected_features:
            self.select_features()
        X = self.data[self.selected_features]
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
            "feature_selection": self.config.feature_selection,
            "top_k": int(self.config.top_k),
            "selected_features": self.selected_features,
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

    def fit_and_evaluate(
        self,
        decision_threshold: float | None = None,
        use_small_dataset: bool = False,
        dataset_path: str | Path | None = None,
        feature_selection: str | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        self.load_data(use_small_dataset=use_small_dataset, dataset_path=dataset_path)
        self.select_features(selection_method=feature_selection, top_k=top_k)
        self.inspect_features()
        self.split_data()
        self.train_model()
        return self.evaluate_model(decision_threshold=decision_threshold)

    def get_default_patient_profile(self) -> Dict[str, float]:
        if self.data is None:
            self.load_data()
        assert self.data is not None
        if not self.selected_features:
            self.select_features()
        medians = self.data[self.selected_features].median()
        return {k: float(v) for k, v in medians.to_dict().items()}

    def predict_single_case(self, patient_features: Dict[str, float], decision_threshold: float | None = None) -> Dict[str, Any]:
        if self.pipeline is None:
            raise NotFittedError("Model has not been trained. Call fit_and_evaluate first.")
        if self.feature_names is None or not self.selected_features:
            raise RuntimeError("Feature names unavailable. Ensure dataset is loaded.")

        row = {feature: float(patient_features[feature]) for feature in self.selected_features}
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

    def export_thesis_artifacts(
        self,
        output_dir: str | Path = "thesis_outputs",
        decision_threshold: float | None = None,
        use_small_dataset: bool = False,
        dataset_path: str | Path | None = None,
        feature_selection: str | None = None,
        top_k: int | None = None,
    ) -> Dict[str, Any]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if self.data is None:
            self.load_data(use_small_dataset=use_small_dataset, dataset_path=dataset_path)
        self.select_features(selection_method=feature_selection, top_k=top_k)
        summary = self.summarize_data()
        effect_table = self.baseline_statistical_differentiation()
        metrics = self.fit_and_evaluate(
            decision_threshold=decision_threshold,
            use_small_dataset=use_small_dataset,
            dataset_path=dataset_path,
            feature_selection=feature_selection,
            top_k=top_k,
        )

        summary_file = output_path / "feature_summary.csv"
        effects_file = output_path / "effect_sizes.csv"
        metrics_file = output_path / "evaluation_metrics.json"
        selected_file = output_path / "selected_features.json"

        summary.to_csv(summary_file)
        effect_table.to_csv(effects_file)
        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        selected_file.write_text(json.dumps(self.selected_features, indent=2), encoding="utf-8")

        self.logger.info("Thesis artifacts exported to %s", output_path.resolve())
        return {
            "summary_csv": str(summary_file),
            "effect_sizes_csv": str(effects_file),
            "metrics_json": str(metrics_file),
            "selected_features_json": str(selected_file),
            "metrics": metrics,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Breast cancer diagnostic baseline and thesis artifact exporter")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for benign class probability")
    parser.add_argument("--export-dir", type=str, default="thesis_outputs", help="Directory for thesis-ready outputs")
    parser.add_argument("--feature-selection", type=str, default="all", choices=["all", "cohen_d", "bwwpa"], help="Feature ranking strategy")
    parser.add_argument("--top-k", type=int, default=0, help="Number of top features to keep (0 means all)")
    parser.add_argument("--use-small-dataset", action="store_true", help="Use local small dataset CSV")
    parser.add_argument("--dataset-path", type=str, default=str(DEFAULT_SMALL_DATASET_PATH), help="Path for local dataset CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = BreastCancerDiagnosticPipeline(
        config=PipelineConfig(
            decision_threshold=args.threshold,
            feature_selection=args.feature_selection,
            top_k=args.top_k,
        )
    )
    artifacts = pipeline.export_thesis_artifacts(
        output_dir=args.export_dir,
        decision_threshold=args.threshold,
        use_small_dataset=args.use_small_dataset,
        dataset_path=args.dataset_path,
        feature_selection=args.feature_selection,
        top_k=args.top_k,
    )
    print("\n=== Thesis Artifact Export Completed ===")
    print(f"summary_csv: {artifacts['summary_csv']}")
    print(f"effect_sizes_csv: {artifacts['effect_sizes_csv']}")
    print(f"metrics_json: {artifacts['metrics_json']}")
    print(f"selected_features_json: {artifacts['selected_features_json']}")
    print("\n=== Final Diagnostic Metrics ===")
    print(artifacts["metrics"]["classification_report"])


if __name__ == "__main__":
    main()
