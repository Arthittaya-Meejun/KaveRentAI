"""Evaluation helpers shared by binary-classification models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_classification_metrics(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Calculate probability and threshold metrics for binary outcomes."""
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")

    y_true = np.asarray(actual, dtype=int)
    y_probability = np.asarray(probabilities, dtype=float)
    if len(y_true) == 0 or len(y_true) != len(y_probability):
        raise ValueError("actual and probabilities must have equal non-zero length")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("actual must contain only binary values")
    if not np.isfinite(y_probability).all():
        raise ValueError("probabilities must contain only finite values")

    y_probability = np.clip(y_probability, 1e-9, 1 - 1e-9)
    y_predicted = (y_probability >= threshold).astype(int)
    true_positive = int(((y_true == 1) & (y_predicted == 1)).sum())
    true_negative = int(((y_true == 0) & (y_predicted == 0)).sum())
    false_positive = int(((y_true == 0) & (y_predicted == 1)).sum())
    false_negative = int(((y_true == 1) & (y_predicted == 0)).sum())

    return {
        "roc_auc": float(roc_auc_score(y_true, y_probability)),
        "average_precision": float(
            average_precision_score(y_true, y_probability)
        ),
        "log_loss": float(log_loss(y_true, y_probability, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_true, y_probability)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_predicted)
        ),
        "precision": float(
            precision_score(y_true, y_predicted, zero_division=0)
        ),
        "recall": float(recall_score(y_true, y_predicted, zero_division=0)),
        "f1": float(f1_score(y_true, y_predicted, zero_division=0)),
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(y_predicted.mean()),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "specificity": float(
            true_negative / (true_negative + false_positive)
        ),
    }


def build_threshold_analysis(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    *,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compare classification trade-offs across probability thresholds."""
    candidate_thresholds = (
        np.arange(0.20, 0.81, 0.05)
        if thresholds is None
        else np.asarray(thresholds, dtype=float)
    )
    rows = []
    for threshold in candidate_thresholds:
        metrics = calculate_classification_metrics(
            actual,
            probabilities,
            threshold=float(threshold),
        )
        rows.append(
            {
                "threshold": metrics["threshold"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "specificity": metrics["specificity"],
                "f1": metrics["f1"],
                "predicted_positive_rate": metrics[
                    "predicted_positive_rate"
                ],
            }
        )
    return pd.DataFrame(rows)


def build_prediction_table(
    data: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    target_column: str,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Attach probabilities and decisions to listing identifiers."""
    required = {"listing_id", "project_id", "date_listed", target_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Prediction data is missing columns: {missing}")
    if len(data) != len(probabilities):
        raise ValueError("Prediction count does not match data row count")

    result = data[
        ["listing_id", "project_id", "date_listed", target_column]
    ].copy()
    result["predicted_probability"] = np.asarray(probabilities, dtype=float)
    result["predicted_class"] = result["predicted_probability"].ge(threshold)
    return result
