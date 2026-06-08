"""Baseline pipeline for cervical-screening-selfsupervised.
Uses synthetic data so the project is runnable immediately.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_synthetic_data(n_samples: int = 180, n_features: int = 12, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    weights = rng.uniform(-1.2, 1.2, size=n_features)
    logits = x @ weights + 0.35 * rng.normal(size=n_samples)
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs > np.quantile(probs, 0.62)).astype(int)

    cols = [f"feature_{i:02d}" for i in range(n_features)]
    return pd.DataFrame(x, columns=cols), y


def fit_and_evaluate() -> dict[str, float]:
    if os.getenv("FAST_TEST") == "1":
        x, y = make_synthetic_data(n_samples=80, n_features=8)
    else:
        x, y = make_synthetic_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                CalibratedClassifierCV(
                    estimator=RandomForestClassifier(
                        n_estimators=5,
                        min_samples_leaf=5,
                        max_depth=4,
                        random_state=42,
                        n_jobs=1,
                    ),
                    method="sigmoid",
                    cv=2,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    proba = model.predict_proba(x_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_values = 2 * precision * recall / np.clip(precision + recall, 1e-10, None)
    best_idx = int(np.nanargmax(f1_values))
    threshold = float(thresholds[max(best_idx - 1, 0)]) if len(thresholds) > 0 else 0.5
    pred = (proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1": float(f1_score(y_test, pred)),
        "threshold": threshold,
        "positive_rate": float(pred.mean()),
    }

    out_dir = Path(__file__).resolve().parents[1]
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    joblib.dump(model, out_dir / "model.joblib", compress=3)
    return metrics


if __name__ == "__main__":
    result = fit_and_evaluate()
    print(json.dumps(result, indent=2))






