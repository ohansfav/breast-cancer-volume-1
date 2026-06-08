"""Hospital bed demand forecasting baseline for thesis work."""

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
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


FEATURE_NOTES: Dict[str, str] = {
    "day_of_week": "Weekly admission pattern proxy.",
    "season_index": "Seasonality and respiratory-wave proxy.",
    "community_infection_rate": "Community transmission pressure.",
    "admission_lag_1": "Recent demand persistence.",
    "discharge_lag_1": "Capacity released by recent discharges.",
    "temperature_c": "Weather stress linked to admissions.",
    "event_load": "Mass-gathering or shock-demand indicator.",
    "staff_absence": "Operational strain on usable beds.",
}


@dataclass(frozen=True)
class Config:
    test_size: float = 0.25
    random_state: int = 42


def make_synthetic_data(n_samples: int = 420, seed: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    day_of_week = rng.integers(0, 7, n_samples)
    season_index = rng.uniform(0, 1, n_samples)
    community_infection_rate = np.clip(rng.normal(0.18, 0.08, n_samples), 0, 1)
    admission_lag_1 = np.clip(rng.normal(0.72, 0.12, n_samples), 0, 1)
    discharge_lag_1 = np.clip(rng.normal(0.68, 0.15, n_samples), 0, 1)
    temperature_c = rng.normal(21.0, 8.0, n_samples)
    event_load = rng.uniform(0, 1, n_samples)
    staff_absence = np.clip(rng.normal(0.12, 0.07, n_samples), 0, 0.5)

    occupancy_ratio = (
        0.05 * day_of_week
        + 0.28 * season_index
        + 0.48 * community_infection_rate
        + 0.36 * admission_lag_1
        - 0.22 * discharge_lag_1
        + 0.01 * temperature_c
        + 0.25 * event_load
        + 0.16 * staff_absence
        + rng.normal(0, 0.05, n_samples)
    )
    occupancy_ratio = np.clip(occupancy_ratio, 0, 1.5)
    data = pd.DataFrame(
        {
            "day_of_week": day_of_week,
            "season_index": season_index,
            "community_infection_rate": community_infection_rate,
            "admission_lag_1": admission_lag_1,
            "discharge_lag_1": discharge_lag_1,
            "temperature_c": temperature_c,
            "event_load": event_load,
            "staff_absence": staff_absence,
        }
    )
    return data, pd.Series(occupancy_ratio, name="bed_demand_ratio")


class BedDemandPipeline:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self.logger = self._setup_logger()
        self.data: pd.DataFrame | None = None
        self.target: pd.Series | None = None
        self.model: LinearRegression | None = None
        self.feature_names: list[str] = []
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None

    @staticmethod
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("bed_demand_forecast")
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
        self.logger.info("Loaded hospital bed demand dataset with %d rows.", len(self.data))
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
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def build_model(self) -> LinearRegression:
        self.model = LinearRegression()
        return self.model

    def train(self) -> LinearRegression:
        if self.X_train is None or self.y_train is None:
            self.split()
        model = self.build_model()
        model.fit(self.X_train, self.y_train)
        self.logger.info("Bed-demand model trained.")
        return model

    def evaluate(self) -> Dict[str, Any]:
        if self.model is None:
            raise NotFittedError("Model not trained.")
        if self.X_test is None or self.y_test is None:
            raise RuntimeError("Split first.")
        pred = self.model.predict(self.X_test)
        rmse = float(np.sqrt(mean_squared_error(self.y_test, pred)))
        mae = mean_absolute_error(self.y_test, pred)
        r2 = r2_score(self.y_test, pred)
        return {
            "mae": float(mae),
            "rmse": rmse,
            "r2": float(r2),
            "predictions": pred.tolist(),
            "observed": self.y_test.tolist(),
        }

    def predict_case(self, row: Dict[str, float]) -> Dict[str, Any]:
        if self.model is None:
            raise NotFittedError("Train model first.")
        frame = pd.DataFrame([{feature: float(row[feature]) for feature in self.feature_names}])
        prediction = float(self.model.predict(frame)[0])
        return {
            "predicted_bed_demand_ratio": prediction,
            "capacity_flag": "surge" if prediction >= 0.85 else "watch",
        }

    def export_artifacts(self, output_dir: str | Path = "thesis_outputs") -> Dict[str, Any]:
        if self.data is None:
            self.load_data()
        if self.model is None:
            self.split()
            self.train()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary = self.summary()
        metrics = self.evaluate()
        summary_path = output_path / "feature_summary.csv"
        metrics_path = output_path / "evaluation_metrics.json"
        summary.to_csv(summary_path)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        return {"summary_csv": str(summary_path), "metrics_json": str(metrics_path), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hospital bed demand forecast baseline")
    parser.add_argument("--export-dir", type=str, default="thesis_outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = BedDemandPipeline()
    artifacts = pipeline.export_artifacts(output_dir=args.export_dir)
    print("=== Bed Demand Forecast Results ===")
    print(f"summary_csv: {artifacts['summary_csv']}")
    print(f"metrics_json: {artifacts['metrics_json']}")
    print(f"MAE: {artifacts['metrics']['mae']:.4f} | RMSE: {artifacts['metrics']['rmse']:.4f} | R2: {artifacts['metrics']['r2']:.4f}")


if __name__ == "__main__":
    main()
