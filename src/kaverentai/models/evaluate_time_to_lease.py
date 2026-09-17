"""ประเมินโมเดล Time-to-Lease ด้วยหน่วยสัปดาห์."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_time_to_lease(
    y_true,
    y_pred,
    model_name: str,
    split_name: str,
) -> dict:
    """คำนวณ MAE, RMSE, R² และสัดส่วนที่คลาดเคลื่อนไม่เกิน 2 สัปดาห์."""

    actual = np.asarray(y_true, dtype=float)
    prediction = np.maximum(np.asarray(y_pred, dtype=float), 0)
    absolute_error = np.abs(actual - prediction)

    return {
        "model": model_name,
        "split": split_name,
        "mae_weeks": float(mean_absolute_error(actual, prediction)),
        "rmse_weeks": float(mean_squared_error(actual, prediction) ** 0.5),
        "r2": float(r2_score(actual, prediction)),
        "within_1_week_pct": float((absolute_error <= 1).mean() * 100),
        "within_2_weeks_pct": float((absolute_error <= 2).mean() * 100),
    }


def build_prediction_table(
    identifiers: pd.DataFrame,
    y_true,
    predictions: dict[str, object],
) -> pd.DataFrame:
    """สร้างตาราง Actual/Predicted สำหรับตรวจสอบ Error รายประกาศ."""

    table = identifiers.reset_index(drop=True).copy()
    table["actual_weeks"] = np.asarray(y_true, dtype=float)

    for model_name, values in predictions.items():
        safe_name = model_name.lower().replace(" ", "_")
        predicted = np.maximum(np.asarray(values, dtype=float), 0)
        table[f"{safe_name}_predicted_weeks"] = predicted
        table[f"{safe_name}_absolute_error"] = np.abs(
            table["actual_weeks"] - predicted
        )

    return table
