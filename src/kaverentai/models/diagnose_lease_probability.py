"""Check seasonal dependence and time stability of lease probability."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kaverentai.features.lease_probability_features import (
    MODEL_FEATURES,
    TARGET_COLUMN,
)
from kaverentai.models.evaluate_classification import (
    calculate_classification_metrics,
)
from kaverentai.models.train_lease_probability import (
    CATBOOST_MODEL,
    build_model,
    load_lease_probability_data,
)
from kaverentai.paths import REPORT_DIR


TIME_FEATURES = {
    "season_listed",
    "listing_month_sin",
    "listing_month_cos",
    "week_listed",
}

FEATURE_SETS = {
    "full": MODEL_FEATURES,
    "without_season_category": tuple(
        feature for feature in MODEL_FEATURES if feature != "season_listed"
    ),
    "without_cyclic_month": tuple(
        feature
        for feature in MODEL_FEATURES
        if feature not in {"listing_month_sin", "listing_month_cos"}
    ),
    "without_all_time": tuple(
        feature for feature in MODEL_FEATURES if feature not in TIME_FEATURES
    ),
}

ROLLING_QUARTERS = (
    ("2025_q1", "2025-01-01", "2025-03-31"),
    ("2025_q2", "2025-04-01", "2025-06-30"),
    ("2025_q3", "2025-07-01", "2025-09-30"),
    ("2025_q4", "2025-10-01", "2025-12-31"),
)


def evaluate_feature_sets(data: pd.DataFrame) -> pd.DataFrame:
    """Measure how much validation performance depends on time features."""
    train = data.loc[data["data_split"].eq("train")].copy()
    validation = data.loc[data["data_split"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise ValueError("Both train and validation rows are required")

    rows = []
    for feature_set_name, features in FEATURE_SETS.items():
        model = build_model(CATBOOST_MODEL, features)
        model.fit(train.loc[:, features], train[TARGET_COLUMN])
        probabilities = model.predict_proba(
            validation.loc[:, features]
        )[:, 1]
        metrics = calculate_classification_metrics(
            validation[TARGET_COLUMN], probabilities
        )
        rows.append(
            {
                "feature_set": feature_set_name,
                "feature_count": len(features),
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        "log_loss", kind="stable"
    ).reset_index(drop=True)


def rolling_time_backtest(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate CatBoost on four expanding-window quarters inside Train."""
    train_period = data.loc[data["data_split"].eq("train")].copy()
    prediction_parts = []
    metric_rows = []

    for fold_name, check_start, check_end in ROLLING_QUARTERS:
        start = pd.Timestamp(check_start)
        end = pd.Timestamp(check_end)
        history = train_period.loc[train_period["date_listed"].lt(start)]
        check = train_period.loc[
            train_period["date_listed"].between(start, end)
        ]
        if history.empty or check.empty:
            raise ValueError(f"Rolling fold {fold_name} has no data")

        model = build_model(CATBOOST_MODEL)
        model.fit(history.loc[:, MODEL_FEATURES], history[TARGET_COLUMN])
        probabilities = model.predict_proba(
            check.loc[:, MODEL_FEATURES]
        )[:, 1]
        metrics = calculate_classification_metrics(
            check[TARGET_COLUMN], probabilities
        )
        metric_rows.append(
            {
                "fold": fold_name,
                "train_rows": len(history),
                "check_rows": len(check),
                "check_positive_rate": metrics["positive_rate"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
                "log_loss": metrics["log_loss"],
                "brier_score": metrics["brier_score"],
                "balanced_accuracy": metrics["balanced_accuracy"],
            }
        )

        fold_predictions = check[
            ["listing_id", "project_id", "date_listed", TARGET_COLUMN]
        ].copy()
        fold_predictions["fold"] = fold_name
        fold_predictions["predicted_probability"] = probabilities
        prediction_parts.append(fold_predictions)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_parts, ignore_index=True),
    )


def save_diagnostics(
    feature_ablation: pd.DataFrame,
    rolling_metrics: pd.DataFrame,
    rolling_predictions: pd.DataFrame,
) -> dict[str, Path]:
    """Save report-ready diagnostic tables."""
    table_dir = REPORT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "feature_ablation": table_dir
        / "lease_probability_feature_ablation.csv",
        "rolling_metrics": table_dir
        / "lease_probability_rolling_metrics.csv",
        "rolling_predictions": table_dir
        / "lease_probability_rolling_predictions.csv",
    }
    feature_ablation.to_csv(paths["feature_ablation"], index=False)
    rolling_metrics.to_csv(paths["rolling_metrics"], index=False)
    rolling_predictions.to_csv(paths["rolling_predictions"], index=False)
    return {name: path.resolve() for name, path in paths.items()}


def main() -> None:
    """Run diagnostics without using Validation to refit or inspect Test."""
    data = load_lease_probability_data()
    feature_ablation = evaluate_feature_sets(data)
    rolling_metrics, rolling_predictions = rolling_time_backtest(data)
    paths = save_diagnostics(
        feature_ablation,
        rolling_metrics,
        rolling_predictions,
    )

    print("Lease-probability feature ablation")
    print(
        feature_ablation[
            ["feature_set", "roc_auc", "log_loss", "balanced_accuracy"]
        ].to_string(index=False)
    )
    print("\nRolling quarterly backtest")
    print(
        rolling_metrics[
            ["fold", "check_rows", "roc_auc", "log_loss", "balanced_accuracy"]
        ].to_string(index=False)
    )
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
