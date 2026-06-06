"""Breast Cancer Wisconsin exploratory clinical data analysis and baseline classification.

This module provides a reproducible, clinically framed pipeline for loading the
Breast Cancer Wisconsin dataset from scikit-learn, inspecting the feature
space, standardizing the predictors, training a baseline classifier, and
reporting evaluation metrics suitable for a medical AI portfolio project.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Clinical notes summarize why specific morphometric features matter in screening.
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
    """Configuration for the clinical analysis pipeline."""

    test_size: float = 0.2
    random_state: int = 42


class BreastCancerDiagnosticPipeline:
    """End-to-end breast cancer screening analysis pipeline.

    The Wisconsin Breast Cancer dataset contains digitized features derived from
    fine needle aspirate (FNA) images of breast masses. Features such as radius,
    texture, perimeter, and area are clinical morphology proxies: larger and
    more irregular values often correlate with malignant behavior.
    """

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
        self.y_pred: np.ndarray | None = None

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Create a console logger with compact medical-research style output."""

        logger = logging.getLogger("breast_cancer_pipeline")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        return logger

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load the authentic Breast Cancer Wisconsin dataset from scikit-learn."""

        try:
            dataset = load_breast_cancer(as_frame=True)
        except Exception as exc:
            self.logger.exception("Dataset loading failed.")
            raise RuntimeError("Unable to load the Breast Cancer Wisconsin dataset.") from exc

        self.data = dataset.frame.copy()
        self.target = self.data["target"]
        self.target_names = dataset.target_names
        self.feature_names = dataset.feature_names

        self.logger.info("Dataset loaded successfully.")
        self.logger.info("Samples: %d | Features: %d", self.data.shape[0], self.data.shape[1] - 1)
        self.logger.info(
            "Clinical labels: %s (0 = malignant, 1 = benign)",
            ", ".join(dataset.target_names),
        )

        return self.data, self.target

    def inspect_features(self) -> None:
        """Log the most clinically relevant morphology features."""

        if self.data is None:
            raise RuntimeError("Data must be loaded before feature inspection.")

        key_features = [
            "mean radius",
            "mean texture",
            "mean perimeter",
            "mean area",
            "mean smoothness",
            "mean compactness",
            "mean concavity",
            "mean concave points",
        ]
        available = [feature for feature in key_features if feature in self.data.columns]

        self.logger.info("Selected clinical morphology features: %s", ", ".join(available))
        for feature in available:
            self.logger.info("  %s -> %s", feature, CLINICAL_FEATURE_NOTES.get(feature, "Clinical note unavailable."))

    def baseline_statistical_differentiation(self) -> pd.DataFrame:
        """Compare malignant vs benign cohorts using mean/STD and standardized effect size.

        Cohen's d is used as a baseline signal-strength indicator:
        d = (mean_malignant - mean_benign) / pooled_std.
        """

        if self.data is None:
            raise RuntimeError("Data must be loaded before statistical differentiation.")

        features = self.data.drop(columns=["target"])
        malignant = features[self.data["target"] == 0]
        benign = features[self.data["target"] == 1]

        malignant_mean = malignant.mean()
        benign_mean = benign.mean()
        malignant_std = malignant.std(ddof=1)
        benign_std = benign.std(ddof=1)

        # Pooled standard deviation stabilizes cohort-to-cohort comparisons.
        pooled_std = np.sqrt((malignant_std.pow(2) + benign_std.pow(2)) / 2)
        pooled_std = pooled_std.replace(0, np.nan)
        cohen_d = (malignant_mean - benign_mean) / pooled_std

        diff_table = pd.DataFrame(
            {
                "malignant_mean": malignant_mean,
                "benign_mean": benign_mean,
                "mean_difference": malignant_mean - benign_mean,
                "cohen_d": cohen_d,
            }
        ).sort_values(by="cohen_d", key=lambda s: s.abs(), ascending=False)

        self.logger.info("Top 5 differentiating features (|Cohen's d|):")
        for feature_name, row in diff_table.head(5).iterrows():
            self.logger.info(
                "  %s | malignant=%.4f benign=%.4f delta=%.4f d=%.4f",
                feature_name,
                row["malignant_mean"],
                row["benign_mean"],
                row["mean_difference"],
                row["cohen_d"],
            )

        return diff_table

    def summarize_data(self) -> pd.DataFrame:
        """Produce a compact statistical summary for exploratory analysis."""

        if self.data is None or self.target is None:
            raise RuntimeError("Data must be loaded before summarization.")

        summary = self.data.drop(columns=["target"]).describe().T
        summary["missing_values"] = self.data.drop(columns=["target"]).isna().sum()
        target_counts = self.target.value_counts().sort_index()

        self.logger.info("Class distribution:")
        for label, count in target_counts.items():
            class_name = self.target_names[label] if self.target_names is not None else str(label)
            self.logger.info("  %s: %d", class_name.capitalize(), int(count))

        return summary

    def split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split the dataset into training and testing partitions."""

        if self.data is None or self.target is None:
            raise RuntimeError("Data must be loaded before splitting.")

        features = self.data.drop(columns=["target"])
        labels = self.target

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features,
            labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels,
        )

        self.logger.info(
            "Train/test split completed: %d training samples and %d testing samples.",
            len(self.X_train),
            len(self.X_test),
        )

        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_model(self) -> Pipeline:
        """Construct a scaled baseline classifier.

        Logistic regression is a strong and interpretable baseline for binary
        diagnostic screening, especially when features are standardized.
        """

        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        solver="liblinear",
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )
        return self.pipeline

    def train_model(self) -> Pipeline:
        """Fit the classifier on the training partition."""

        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Training data is not ready. Call split_data first.")

        model = self.build_model()
        model.fit(self.X_train, self.y_train)
        self.logger.info("Model training completed.")
        return model

    def evaluate_model(self) -> Dict[str, float | np.ndarray | str]:
        """Evaluate the fitted model and print clinically oriented metrics."""

        if self.pipeline is None:
            raise NotFittedError("The model has not been trained yet.")
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("Testing data is not available. Call split_data first.")

        self.y_pred = self.pipeline.predict(self.X_test)

        matrix = confusion_matrix(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred)
        recall = recall_score(self.y_test, self.y_pred)
        f1 = f1_score(self.y_test, self.y_pred)
        report = classification_report(self.y_test, self.y_pred, target_names=self.target_names)

        self.logger.info("Clinical confusion matrix:\n%s", matrix)
        self.logger.info("Precision: %.4f", precision)
        self.logger.info("Recall: %.4f", recall)
        self.logger.info("F1-score: %.4f", f1)
        self.logger.info("Classification report:\n%s", report)

        return {
            "confusion_matrix": matrix,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "classification_report": report,
        }

    def run(self) -> Dict[str, float | np.ndarray | str]:
        """Execute the full exploratory analysis and baseline classification workflow."""

        self.load_data()
        self.inspect_features()
        self.summarize_data()
        self.baseline_statistical_differentiation()
        self.split_data()
        self.train_model()
        return self.evaluate_model()


def main() -> None:
    """Run the pipeline as a command-line script."""

    pipeline = BreastCancerDiagnosticPipeline()
    results = pipeline.run()
    print("\n=== Final Diagnostic Metrics ===")
    print(results["classification_report"])


if __name__ == "__main__":
    main()
