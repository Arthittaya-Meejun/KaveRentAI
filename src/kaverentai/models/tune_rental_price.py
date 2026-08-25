"""ปรับ Hyperparameters ของ Gradient Boosting สำหรับราคาค่าเช่า.

ใช้ Rolling Time Validation ภายในชุด Train เพื่อให้ค่าที่เลือกไม่ได้เหมาะ
กับเวลาเพียงช่วงเดียว และไม่แตะ Validation หลักหรือ Test ระหว่างการจูน
"""

from collections.abc import Sequence
from statistics import fmean, pstdev

from joblib import Parallel, delayed
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from kaverentai.data.split_data import chronological_split
from kaverentai.features.rental_price_features import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    build_rental_price_features,
    split_features_and_target,
)
from kaverentai.models.rental_price_models import (
    create_gradient_boosting_pipeline,
)


# แต่ละรอบใช้ข้อมูลอดีตทั้งหมดเป็น Train และใช้ไตรมาสถัดไปตรวจสอบ
# ทุก Fold อยู่ภายในชุด Train เดิมซึ่งสิ้นสุดวันที่ 31 ธันวาคม 2025
ROLLING_VALIDATION_WINDOWS = [
    {
        "name": "2025_Q1",
        "train_end": "2024-12-31",
        "validation_end": "2025-03-31",
    },
    {
        "name": "2025_Q2",
        "train_end": "2025-03-31",
        "validation_end": "2025-06-30",
    },
    {
        "name": "2025_Q3",
        "train_end": "2025-06-30",
        "validation_end": "2025-09-30",
    },
    {
        "name": "2025_Q4",
        "train_end": "2025-09-30",
        "validation_end": "2025-12-31",
    },
]


# ทดลอง 12 ชุด ครอบคลุมโมเดลเล็ก-ใหญ่, Learning Rate, ความลึก,
# จำนวนตัวอย่างขั้นต่ำในใบไม้ และสัดส่วนข้อมูลที่ใช้สร้างต้นไม้แต่ละต้น
# ชุดแรกคือค่าที่ชนะจากการจูนรอบเดิม จึงใช้เป็นจุดอ้างอิงได้
GRADIENT_BOOSTING_PARAMETER_SETS = [
    {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_leaf": 15,
        "subsample": 0.9,
    },
    {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_leaf": 15,
        "subsample": 0.9,
    },
    {
        "n_estimators": 450,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_leaf": 15,
        "subsample": 0.9,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_samples_leaf": 15,
        "subsample": 0.9,
    },
    {
        "n_estimators": 450,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_samples_leaf": 15,
        "subsample": 0.9,
    },
    {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "subsample": 1.0,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "subsample": 0.9,
    },
    {
        "n_estimators": 250,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 15,
        "subsample": 0.9,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 25,
        "subsample": 0.9,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_samples_leaf": 15,
        "subsample": 0.9,
    },
    {
        "n_estimators": 350,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_samples_leaf": 25,
        "subsample": 0.8,
    },
    {
        "n_estimators": 450,
        "learning_rate": 0.02,
        "max_depth": 4,
        "min_samples_leaf": 25,
        "subsample": 0.8,
    },
]


def build_rolling_time_folds(
    train_data: pd.DataFrame,
    windows: Sequence[dict] | None = None,
) -> list[dict]:
    """สร้าง Expanding-window Folds จากข้อมูล Train เท่านั้น."""

    if DATE_COLUMN not in train_data.columns:
        raise KeyError(f"ไม่พบคอลัมน์วันที่: {DATE_COLUMN}")

    windows = list(windows or ROLLING_VALIDATION_WINDOWS)
    event_dates = pd.to_datetime(train_data[DATE_COLUMN], errors="coerce")
    if event_dates.isna().any():
        raise ValueError(f"{DATE_COLUMN} มีวันที่ที่แปลงค่าไม่ได้")

    rolling_folds = []

    for window in windows:
        train_end = pd.Timestamp(window["train_end"])
        validation_end = pd.Timestamp(window["validation_end"])

        fold_train = train_data.loc[event_dates <= train_end].copy()
        fold_validation = train_data.loc[
            (event_dates > train_end) & (event_dates <= validation_end)
        ].copy()

        if fold_train.empty or fold_validation.empty:
            raise ValueError(
                f"Fold {window['name']} ต้องมีทั้ง Train และ Validation"
            )

        X_train, y_train = split_features_and_target(fold_train)
        X_validation, y_validation = split_features_and_target(
            fold_validation
        )

        rolling_folds.append(
            {
                "name": window["name"],
                "train_end": window["train_end"],
                "validation_end": window["validation_end"],
                "X_train": X_train,
                "y_train": y_train,
                "X_validation": X_validation,
                "y_validation": y_validation,
            }
        )

    return rolling_folds


def _evaluate_parameter_set(
    set_number: int,
    params: dict,
    rolling_folds: Sequence[dict],
) -> dict:
    """ฝึกพารามิเตอร์หนึ่งชุดในทุก Rolling Fold."""

    train_mae_values = []
    validation_mae_values = []
    validation_rmse_values = []
    validation_r2_values = []

    for fold in rolling_folds:
        model = create_gradient_boosting_pipeline(
            FEATURE_COLUMNS,
            model_params=params,
        )
        model.fit(fold["X_train"], fold["y_train"])

        train_predictions = model.predict(fold["X_train"])
        validation_predictions = model.predict(fold["X_validation"])

        train_mae_values.append(
            float(
                mean_absolute_error(
                    fold["y_train"],
                    train_predictions,
                )
            )
        )
        validation_mae_values.append(
            float(
                mean_absolute_error(
                    fold["y_validation"],
                    validation_predictions,
                )
            )
        )
        validation_rmse_values.append(
            float(
                mean_squared_error(
                    fold["y_validation"],
                    validation_predictions,
                )
                ** 0.5
            )
        )
        validation_r2_values.append(
            float(
                r2_score(
                    fold["y_validation"],
                    validation_predictions,
                )
            )
        )

    history_row = {
        "set_number": set_number,
        **params,
        "mean_train_mae": fmean(train_mae_values),
        "mean_validation_mae": fmean(validation_mae_values),
        "std_validation_mae": pstdev(validation_mae_values),
        "mean_validation_rmse": fmean(validation_rmse_values),
        "mean_validation_r2": fmean(validation_r2_values),
    }
    history_row["mae_gap"] = (
        history_row["mean_validation_mae"]
        - history_row["mean_train_mae"]
    )

    for fold_number, fold_mae in enumerate(
        validation_mae_values,
        start=1,
    ):
        history_row[f"fold_{fold_number}_mae"] = fold_mae

    return history_row


def tune_gradient_boosting(
    rolling_folds: Sequence[dict],
    parameter_sets: Sequence[dict] | None = None,
    verbose: bool = True,
    n_jobs: int = 1,
) -> tuple[dict, list[dict]]:
    """เลือกชุดค่าจาก Rolling Validation MAE เฉลี่ยต่ำที่สุด.

    ``n_jobs`` กำหนดจำนวนชุดพารามิเตอร์ที่ทดลองพร้อมกัน การรันจากไฟล์
    Train ใช้ 4 งานพร้อมกันเพื่อลดเวลา ส่วน Unit Test ใช้ 1 งาน
    """

    if not rolling_folds:
        raise ValueError("ต้องมี Rolling Validation อย่างน้อย 1 Fold")

    parameter_sets = list(
        parameter_sets or GRADIENT_BOOSTING_PARAMETER_SETS
    )
    if not parameter_sets:
        raise ValueError("ต้องกำหนด Hyperparameter อย่างน้อย 1 ชุด")

    if verbose:
        print(
            f"ทดลอง {len(parameter_sets)} ชุด x "
            f"{len(rolling_folds)} Rolling Folds = "
            f"{len(parameter_sets) * len(rolling_folds)} ครั้ง"
        )

    tuning_history = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_parameter_set)(
            set_number,
            params,
            rolling_folds,
        )
        for set_number, params in enumerate(parameter_sets, start=1)
    )

    tuning_history = sorted(
        tuning_history,
        key=lambda row: row["set_number"],
    )

    if verbose:
        for row in tuning_history:
            print(
                f"ชุดที่ {row['set_number']:>2}: "
                f"MAE เฉลี่ย {row['mean_validation_mae']:.2f} "
                f"(SD {row['std_validation_mae']:.2f})"
            )

    # ถ้า MAE เฉลี่ยเท่ากัน ให้เลือกชุดที่ผลแต่ละ Fold แกว่งน้อยกว่า
    best_row = min(
        tuning_history,
        key=lambda row: (
            row["mean_validation_mae"],
            row["std_validation_mae"],
        ),
    )
    parameter_names = list(parameter_sets[0])
    best_params = {
        key: best_row[key]
        for key in parameter_names
    }

    return best_params, tuning_history


def main() -> None:
    """ทดลอง Hyperparameters และแสดงชุดที่ดีที่สุด."""

    modeling_data = build_rental_price_features()
    train_data, validation_data, test_data = chronological_split(
        modeling_data,
        date_column=DATE_COLUMN,
    )
    rolling_folds = build_rolling_time_folds(train_data)

    for fold in rolling_folds:
        print(
            fold["name"],
            "Train:",
            len(fold["X_train"]),
            "Validation:",
            len(fold["X_validation"]),
        )

    best_params, tuning_history = tune_gradient_boosting(
        rolling_folds,
        n_jobs=4,
    )
    best_row = min(
        tuning_history,
        key=lambda row: row["mean_validation_mae"],
    )

    print("\n" + "=" * 60)
    print("ค่าที่ดีที่สุด:", best_params)
    print(
        "Rolling Validation MAE เฉลี่ย:",
        round(best_row["mean_validation_mae"], 2),
    )
    print("ยังไม่ใช้ Validation หลัก จำนวน", len(validation_data), "แถว")
    print("ยังไม่ใช้ Test จำนวน", len(test_data), "แถว")


if __name__ == "__main__":
    main()
