"""Tune Binomial Logistic Regression with Train-only rolling backtests.

Only the regularization value ``C`` is tuned so the experiment remains easy to
explain. Lower C means stronger regularization; higher C means weaker
regularization. The value with the lowest rolling Weighted MAE is selected.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kaverentai.models.train_renewal_rate import (
    load_renewal_rate_data,
    rolling_quarter_backtest,
)
from kaverentai.paths import REPORT_DIR


CANDIDATE_C_VALUES = (0.001, 0.01, 0.1, 1.0, 10.0)


def tune_logistic_regularization(data: pd.DataFrame) -> pd.DataFrame:
    """Return rolling-backtest metrics for each candidate C value."""
    rows = []
    for c_value in CANDIDATE_C_VALUES:
        metrics, _ = rolling_quarter_backtest(data, c_value=c_value)
        logistic_metrics = metrics["binomial_logistic_regression"]
        rows.append(
            {
                "C": c_value,
                "rolling_weighted_mae": logistic_metrics["weighted_mae"],
                "rolling_weighted_rmse": logistic_metrics["weighted_rmse"],
                "rolling_binomial_log_loss": logistic_metrics[
                    "binomial_log_loss"
                ],
                "rolling_weighted_bias": logistic_metrics["weighted_bias"],
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["rolling_weighted_mae", "C"], ascending=[True, True]
    ).reset_index(drop=True)


def select_best_c(tuning_results: pd.DataFrame) -> float:
    """Read the best C from a sorted tuning table."""
    required = {"C", "rolling_weighted_mae"}
    missing = sorted(required - set(tuning_results.columns))
    if missing:
        raise ValueError(f"Tuning results are missing columns: {missing}")
    if tuning_results.empty:
        raise ValueError("Tuning results must not be empty")
    best_row = tuning_results.sort_values(
        ["rolling_weighted_mae", "C"], ascending=[True, True]
    ).iloc[0]
    return float(best_row["C"])


def save_tuning_results(results: pd.DataFrame) -> Path:
    """Save the report-ready tuning table."""
    path = REPORT_DIR / "tables" / "renewal_rate_logistic_tuning.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    return path.resolve()


def main() -> None:
    """Run tuning and print the selected regularization value."""
    data = load_renewal_rate_data()
    results = tune_logistic_regularization(data)
    best_c = select_best_c(results)
    output_path = save_tuning_results(results)

    print("Binomial Logistic Regression tuning")
    print(f"Candidate C values: {list(CANDIDATE_C_VALUES)}")
    print(f"Selected C: {best_c}")
    print(
        "Best rolling Weighted MAE: "
        f"{results.iloc[0]['rolling_weighted_mae']:.4f}"
    )
    print(f"Saved tuning table: {output_path}")


if __name__ == "__main__":
    main()
