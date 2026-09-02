"""Forecast monthly renewal rates at project level.

Two statistical models are compared:
1. Historical-rate baseline: predicts the weighted Train renewal rate.
2. Binomial Logistic Regression: learns from project-month features.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kaverentai.config import RENEWAL_LOGISTIC_C, RENEWAL_SELECTED_MODEL
from kaverentai.features.renewal_rate_features import (
    RATE_MODEL_FEATURES,
    TARGET_COLUMN,
)
from kaverentai.paths import MODELING_DATA_DIR, MODEL_DIR, REPORT_DIR


NUMERIC_FEATURES = tuple(
    feature
    for feature in RATE_MODEL_FEATURES
    if feature != "project_id_category"
)
CATEGORICAL_FEATURES = ("project_id_category",)
SELECTED_LOGISTIC_C = RENEWAL_LOGISTIC_C
SELECTED_MODEL = RENEWAL_SELECTED_MODEL

BACKTEST_QUARTERS = (
    ("2024-12-01", "2025-01-01", "2025-03-01"),
    ("2025-03-01", "2025-04-01", "2025-06-01"),
    ("2025-06-01", "2025-07-01", "2025-09-01"),
    ("2025-09-01", "2025-10-01", "2025-12-01"),
)


def load_renewal_rate_data(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the project-month modeling table."""
    data_path = Path(path) if path is not None else (
        MODELING_DATA_DIR / "renewal_rate_monthly.csv"
    )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Aggregate modeling data was not found: {data_path}. "
            "Run `python -m kaverentai.features.renewal_rate_features` first."
        )
    return pd.read_csv(data_path, parse_dates=["expiry_month"])


def build_rate_pipeline(c_value: float = SELECTED_LOGISTIC_C) -> Pipeline:
    """Create the preprocessing and Binomial Logistic Regression pipeline."""
    if c_value <= 0:
        raise ValueError("c_value must be greater than zero")
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="median")),
                        ("standardize", StandardScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "project",
                Pipeline(
                    steps=[
                        (
                            "fill_missing",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(
                                handle_unknown="ignore"
                            ),
                        ),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("prepare_data", preprocessing),
            ("model", LogisticRegression(C=c_value, max_iter=1_000)),
        ]
    )


def expand_binomial_rows(
    grouped_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Expand success/failure counts for scikit-learn's Logistic Regression."""
    repeated_rows = []
    outcomes = []
    for row_number, row in grouped_data.reset_index(drop=True).iterrows():
        total = int(row["expiring_contracts"])
        renewed = int(row["renewed_contracts"])
        repeated_rows.extend([row_number] * total)
        outcomes.extend([1] * renewed)
        outcomes.extend([0] * (total - renewed))

    features = grouped_data.reset_index(drop=True).loc[
        repeated_rows, RATE_MODEL_FEATURES
    ].reset_index(drop=True)
    target = pd.Series(outcomes, name="will_renew", dtype="int8")
    return features, target


def calculate_rate_metrics(
    data: pd.DataFrame,
    predictions: np.ndarray,
) -> dict[str, float]:
    """Calculate contract-weighted errors for project-month rates."""
    actual = data[TARGET_COLUMN].to_numpy(dtype=float)
    weights = data["expiring_contracts"].to_numpy(dtype=float)
    probabilities = np.clip(np.asarray(predictions), 1e-9, 1 - 1e-9)

    error = probabilities - actual
    weighted_mae = np.average(np.abs(error), weights=weights)
    weighted_rmse = np.sqrt(np.average(error**2, weights=weights))
    successes = data["renewed_contracts"].to_numpy(dtype=float)
    failures = weights - successes
    binomial_log_loss = -np.sum(
        successes * np.log(probabilities)
        + failures * np.log(1 - probabilities)
    ) / weights.sum()

    return {
        "weighted_mae": float(weighted_mae),
        "weighted_rmse": float(weighted_rmse),
        "binomial_log_loss": float(binomial_log_loss),
        "weighted_bias": float(np.average(error, weights=weights)),
        "actual_overall_rate": float(np.average(actual, weights=weights)),
        "predicted_overall_rate": float(
            np.average(probabilities, weights=weights)
        ),
        "project_month_rows": int(len(data)),
        "contracts": int(weights.sum()),
    }


def train_and_evaluate_rate_models(
    data: pd.DataFrame,
    *,
    c_value: float = SELECTED_LOGISTIC_C,
) -> tuple[Pipeline, dict[str, dict[str, float]], pd.DataFrame]:
    """Fit on Train and compare both models on Validation."""
    train = data.loc[data["data_split"].eq("train")].copy()
    validation = data.loc[data["data_split"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise ValueError("Both train and validation rows are required")

    train_rate = (
        train["renewed_contracts"].sum()
        / train["expiring_contracts"].sum()
    )
    baseline_predictions = np.full(len(validation), train_rate)

    x_train, y_train = expand_binomial_rows(train)
    rate_model = build_rate_pipeline(c_value=c_value)
    rate_model.fit(x_train, y_train)
    model_predictions = rate_model.predict_proba(
        validation.loc[:, RATE_MODEL_FEATURES]
    )[:, 1]

    metrics = {
        "historical_rate_baseline": calculate_rate_metrics(
            validation, baseline_predictions
        ),
        "binomial_logistic_regression": calculate_rate_metrics(
            validation, model_predictions
        ),
    }
    predictions = validation[
        [
            "project_id",
            "expiry_month",
            "expiring_contracts",
            "renewed_contracts",
            TARGET_COLUMN,
        ]
    ].copy()
    predictions["baseline_predicted_rate"] = baseline_predictions
    predictions["logistic_predicted_rate"] = model_predictions
    standard_error = np.sqrt(
        model_predictions
        * (1 - model_predictions)
        / validation["expiring_contracts"].to_numpy()
    )
    predictions["logistic_lower_95"] = np.clip(
        model_predictions - 1.96 * standard_error, 0, 1
    )
    predictions["logistic_upper_95"] = np.clip(
        model_predictions + 1.96 * standard_error, 0, 1
    )
    return rate_model, metrics, predictions


def evaluate_final_test(
    data: pd.DataFrame,
    *,
    c_value: float = SELECTED_LOGISTIC_C,
) -> tuple[Pipeline, dict[str, dict[str, float]], pd.DataFrame]:
    """Refit on Train+Validation and evaluate the locked models on Test.

    This function must be used only after model selection and tuning are
    finished. Test data is never included when fitting either model.
    """
    development = data.loc[
        data["data_split"].isin(["train", "validation"])
    ].copy()
    test = data.loc[data["data_split"].eq("test")].copy()
    if development.empty or test.empty:
        raise ValueError("Train+Validation and Test rows are required")

    development_rate = (
        development["renewed_contracts"].sum()
        / development["expiring_contracts"].sum()
    )
    baseline_predictions = np.full(len(test), development_rate)

    x_development, y_development = expand_binomial_rows(development)
    final_model = build_rate_pipeline(c_value=c_value)
    final_model.fit(x_development, y_development)
    logistic_predictions = final_model.predict_proba(
        test.loc[:, RATE_MODEL_FEATURES]
    )[:, 1]

    metrics = {
        "historical_rate_baseline": calculate_rate_metrics(
            test, baseline_predictions
        ),
        "binomial_logistic_regression": calculate_rate_metrics(
            test, logistic_predictions
        ),
    }
    predictions = test[
        [
            "project_id",
            "expiry_month",
            "expiring_contracts",
            "renewed_contracts",
            TARGET_COLUMN,
        ]
    ].copy()
    predictions["baseline_predicted_rate"] = baseline_predictions
    predictions["logistic_predicted_rate"] = logistic_predictions
    return final_model, metrics, predictions


def build_business_predictions(
    test_predictions: pd.DataFrame,
    *,
    selected_model: str = SELECTED_MODEL,
) -> pd.DataFrame:
    """Convert Test predictions into a simple project-month action table.

    Risk represents the expected number of non-renewals relative to the other
    projects in the same month. It is not individual risk.
    """
    prediction_columns = {
        "historical_rate_baseline": "baseline_predicted_rate",
        "binomial_logistic_regression": "logistic_predicted_rate",
    }
    if selected_model not in prediction_columns:
        raise ValueError(f"Unknown selected_model: {selected_model}")

    rate_column = prediction_columns[selected_model]
    required_columns = {
        "project_id",
        "expiry_month",
        "expiring_contracts",
        rate_column,
    }
    missing = sorted(required_columns - set(test_predictions.columns))
    if missing:
        raise ValueError(f"Test predictions are missing columns: {missing}")

    business = test_predictions[
        ["project_id", "expiry_month", "expiring_contracts"]
    ].copy()
    business["selected_model"] = selected_model
    business["predicted_renewal_rate"] = test_predictions[rate_column]
    business["expected_renewed_contracts"] = (
        business["expiring_contracts"]
        * business["predicted_renewal_rate"]
    ).round(1)
    business["expected_not_renewed_contracts"] = (
        business["expiring_contracts"]
        - business["expected_renewed_contracts"]
    ).round(1)

    def assign_monthly_risk(expected_non_renewals: pd.Series) -> np.ndarray:
        """Split projects into low, medium, and high monthly workload."""
        low_cutoff = expected_non_renewals.quantile(0.33)
        high_cutoff = expected_non_renewals.quantile(0.67)
        return np.select(
            [
                expected_non_renewals.le(low_cutoff),
                expected_non_renewals.ge(high_cutoff),
            ],
            ["low", "high"],
            default="medium",
        )

    business["risk_level"] = business.groupby("expiry_month")[
        "expected_not_renewed_contracts"
    ].transform(
        assign_monthly_risk
    )
    return business.sort_values(
        ["expiry_month", "expected_not_renewed_contracts"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def rolling_quarter_backtest(
    data: pd.DataFrame,
    *,
    c_value: float = SELECTED_LOGISTIC_C,
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Evaluate four expanding-window quarters within the Train period."""
    prediction_parts = []
    for fold, (train_end, check_start, check_end) in enumerate(
        BACKTEST_QUARTERS, start=1
    ):
        history = data.loc[data["expiry_month"].le(train_end)].copy()
        check = data.loc[
            data["expiry_month"].between(check_start, check_end)
        ].copy()
        if history.empty or check.empty:
            raise ValueError(f"Backtest fold {fold} has no data")

        historical_rate = (
            history["renewed_contracts"].sum()
            / history["expiring_contracts"].sum()
        )
        x_history, y_history = expand_binomial_rows(history)
        model = build_rate_pipeline(c_value=c_value)
        model.fit(x_history, y_history)

        fold_predictions = check[
            [
                "project_id",
                "expiry_month",
                "expiring_contracts",
                "renewed_contracts",
                TARGET_COLUMN,
            ]
        ].copy()
        fold_predictions["fold"] = fold
        fold_predictions["baseline_predicted_rate"] = historical_rate
        fold_predictions["logistic_predicted_rate"] = model.predict_proba(
            check.loc[:, RATE_MODEL_FEATURES]
        )[:, 1]
        prediction_parts.append(fold_predictions)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = {
        "historical_rate_baseline": calculate_rate_metrics(
            predictions,
            predictions["baseline_predicted_rate"].to_numpy(),
        ),
        "binomial_logistic_regression": calculate_rate_metrics(
            predictions,
            predictions["logistic_predicted_rate"].to_numpy(),
        ),
    }
    return metrics, predictions


def save_rate_outputs(
    validation_model: Pipeline,
    final_model: Pipeline,
    metrics: dict[str, object],
    validation_predictions: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    business_predictions: pd.DataFrame,
) -> dict[str, Path]:
    """Save fitted models, metrics, and prediction tables."""
    validation_model_path = MODEL_DIR / "renewal_rate_logistic.joblib"
    final_model_path = MODEL_DIR / "renewal_rate_logistic_final.joblib"
    metrics_path = REPORT_DIR / "metrics" / "renewal_rate_metrics.json"
    validation_path = (
        REPORT_DIR / "tables" / "renewal_rate_validation_predictions.csv"
    )
    backtest_path = (
        REPORT_DIR / "tables" / "renewal_rate_backtest_predictions.csv"
    )
    test_path = REPORT_DIR / "tables" / "renewal_rate_test_predictions.csv"
    business_path = (
        REPORT_DIR / "tables" / "renewal_rate_business_predictions.csv"
    )
    output_paths = (
        validation_model_path,
        final_model_path,
        metrics_path,
        validation_path,
        backtest_path,
        test_path,
        business_path,
    )
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(validation_model, validation_model_path)
    joblib.dump(final_model, final_model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    validation_predictions.to_csv(
        validation_path, index=False, date_format="%Y-%m-%d"
    )
    backtest_predictions.to_csv(
        backtest_path, index=False, date_format="%Y-%m-%d"
    )
    test_predictions.to_csv(test_path, index=False, date_format="%Y-%m-%d")
    business_predictions.to_csv(
        business_path, index=False, date_format="%Y-%m-%d"
    )
    return {
        "validation_model": validation_model_path.resolve(),
        "final_model": final_model_path.resolve(),
        "metrics": metrics_path.resolve(),
        "validation_predictions": validation_path.resolve(),
        "backtest_predictions": backtest_path.resolve(),
        "test_predictions": test_path.resolve(),
        "business_predictions": business_path.resolve(),
    }


def main() -> None:
    """Run the locked evaluation and save final business outputs."""
    data = load_renewal_rate_data()
    validation_model, validation_metrics, validation_predictions = (
        train_and_evaluate_rate_models(data, c_value=SELECTED_LOGISTIC_C)
    )
    backtest_metrics, backtest_predictions = rolling_quarter_backtest(
        data, c_value=SELECTED_LOGISTIC_C
    )
    final_model, test_metrics, test_predictions = evaluate_final_test(
        data, c_value=SELECTED_LOGISTIC_C
    )
    business_predictions = build_business_predictions(
        test_predictions, selected_model=SELECTED_MODEL
    )
    metrics = {
        "model_settings": {
            "binomial_logistic_C": SELECTED_LOGISTIC_C,
            "selection_metric": "rolling_weighted_mae",
            "selected_model": SELECTED_MODEL,
            "selection_data": "train_and_validation_only",
            "test_used_for_selection": False,
        },
        "rolling_backtest_2025": backtest_metrics,
        "validation_2026_q1": validation_metrics,
        "test_2026_q2": test_metrics,
    }
    output_paths = save_rate_outputs(
        validation_model,
        final_model,
        metrics,
        validation_predictions,
        backtest_predictions,
        test_predictions,
        business_predictions,
    )

    print("Project-month renewal-rate forecasting")
    print(f"Selected Logistic C: {SELECTED_LOGISTIC_C}")
    print(f"Selected model: {SELECTED_MODEL}")
    for evaluation_name in (
        "rolling_backtest_2025",
        "validation_2026_q1",
        "test_2026_q2",
    ):
        evaluation_metrics = metrics[evaluation_name]
        print(evaluation_name)
        for model_name, model_metrics in evaluation_metrics.items():
            print(
                f"  {model_name:30s} "
                f"MAE={model_metrics['weighted_mae']:.3f} "
                f"RMSE={model_metrics['weighted_rmse']:.3f} "
                f"LogLoss={model_metrics['binomial_log_loss']:.3f}"
            )
    print("Test was evaluated once after model selection was locked.")
    for name, path in output_paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
