"""Training utility for MIMIC-IV-ED triage parquet data using XGBoost.

This script avoids one-hot encoding and multi-label targets, relying on
ordinal encoding for categorical features. It expects a Parquet file such as
``MIMIC-IV-ED_triage_cleaned.parquet`` with a single target column.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier


@dataclass
class TrainingConfig:
    """Configuration for training an XGBoost model."""

    data_path: Path
    target: str
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.9
    colsample_bytree: float = 0.9


class MultiLabelTargetError(ValueError):
    """Raised when the provided target column contains multi-label data."""


def _split_columns(frame: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Identify numeric and categorical columns for preprocessing."""

    numeric_columns = frame.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = frame.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric_columns, categorical_columns


def _build_preprocessor(
    numeric_columns: Iterable[str], categorical_columns: Iterable[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
    )


def _validate_target(target_series: pd.Series) -> None:
    if target_series.ndim != 1:
        raise MultiLabelTargetError(
            "Multi-label targets are not supported; please provide a single target column."
        )

    if target_series.apply(lambda value: isinstance(value, (list, tuple, set))).any():
        raise MultiLabelTargetError(
            "Multi-label targets (collections within target rows) are not supported."
        )


def load_dataset(data_path: Path, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load dataset from Parquet and split into features and target."""

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    frame = pd.read_parquet(data_path)

    if target not in frame.columns:
        raise KeyError(f"Target column '{target}' not found in dataset.")

    y = frame[target]
    _validate_target(y)

    X = frame.drop(columns=[target])
    return X, y


def train_xgboost_model(config: TrainingConfig):
    """Train an XGBoost classifier and return the fitted pipeline and metrics."""

    X, y = load_dataset(config.data_path, config.target)
    numeric_columns, categorical_columns = _split_columns(X)
    preprocessor = _build_preprocessor(numeric_columns, categorical_columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    objective = "binary:logistic" if y.nunique() == 2 else "multi:softprob"

    model = XGBClassifier(
        objective=objective,
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        max_depth=config.max_depth,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        eval_metric="logloss",
        random_state=config.random_state,
        tree_method="hist",
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    f1 = f1_score(y_test, predictions, average="macro")
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=False)

    metrics = {
        "f1_macro": f1,
        "accuracy": accuracy,
        "classification_report": report,
    }

    return pipeline, metrics


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(
        description="Train an XGBoost model on MIMIC-IV-ED triage data without one-hot encoding.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the Parquet file (e.g., MIMIC-IV-ED_triage_cleaned.parquet)",
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Name of the target column to predict.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)

    args = parser.parse_args()
    return TrainingConfig(
        data_path=args.data_path,
        target=args.target,
        test_size=args.test_size,
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
    )


def main() -> None:
    config = parse_args()
    _, metrics = train_xgboost_model(config)
    print("Training complete. Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 (macro): {metrics['f1_macro']:.4f}")
    print("Classification report:\n" + metrics["classification_report"])


if __name__ == "__main__":
    main()
