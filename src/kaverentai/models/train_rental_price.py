"""ฝึกและเปรียบเทียบโมเดลพยากรณ์ราคาค่าเช่า.

โมเดลที่เปรียบเทียบมี 2 ตัว:

1. Forward/Stepwise Multiple Linear Regression สำหรับอธิบายปัจจัยด้านราคา
2. Gradient Boosting Regressor สำหรับจับความสัมพันธ์ที่ไม่เป็นเส้นตรง

เลือกโมเดลจาก Validation MAE แล้วจึงประเมิน Test set เพียงครั้งเดียว
"""

import json

import joblib
import pandas as pd

from kaverentai.data.split_data import chronological_split
from kaverentai.features.rental_price_features import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    build_rental_price_features,
    split_features_and_target,
)
from kaverentai.models.evaluate_regression import evaluate_regression
from kaverentai.models.rental_price_models import (
    create_gradient_boosting_pipeline,
    create_stepwise_linear_pipeline,
    forward_select_features_rolling,
)
from kaverentai.models.tune_rental_price import (
    build_rolling_time_folds,
    tune_gradient_boosting,
)
from kaverentai.paths import MODEL_DIR, REPORT_DIR


RENTAL_PRICE_MODEL_DIR = MODEL_DIR / "rental_price"
METRIC_REPORT_DIR = REPORT_DIR / "metrics"
TABLE_REPORT_DIR = REPORT_DIR / "tables"

STEPWISE_MODEL_PATH = (
    RENTAL_PRICE_MODEL_DIR / "rental_price_stepwise_mlr.joblib"
)
GRADIENT_BOOSTING_MODEL_PATH = (
    RENTAL_PRICE_MODEL_DIR / "rental_price_gradient_boosting.joblib"
)
WINNER_MODEL_PATH = MODEL_DIR / "rental_price_model.joblib"
METADATA_PATH = RENTAL_PRICE_MODEL_DIR / "rental_price_metadata.json"


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    split_name: str,
) -> dict:
    """ประเมินโมเดลและเพิ่มชื่อชุดข้อมูลลงในผลลัพธ์."""

    predictions = model.predict(X)
    results = evaluate_regression(
        y_true=y,
        y_pred=predictions,
        model_name=f"{model_name} - {split_name}",
    )

    return {
        "model": model_name,
        "split": split_name,
        "mae": float(results["mae"]),
        "rmse": float(results["rmse"]),
        "r2": float(results["r2"]),
    }


def create_output_directories() -> None:
    """สร้างโฟลเดอร์ผลลัพธ์ถ้ายังไม่มี."""

    for directory in [
        RENTAL_PRICE_MODEL_DIR,
        METRIC_REPORT_DIR,
        TABLE_REPORT_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def build_stepwise_coefficient_table(stepwise_model) -> pd.DataFrame:
    """สร้างตารางค่าสัมประสิทธิ์เพื่อใช้อธิบาย Stepwise MLR."""

    preprocessor = stepwise_model.named_steps["preprocessor"]
    linear_regression = stepwise_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": linear_regression.coef_,
        }
    )
    coefficients["absolute_coefficient"] = coefficients[
        "coefficient"
    ].abs()
    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    intercept = pd.DataFrame(
        [
            {
                "feature": "intercept",
                "coefficient": float(linear_regression.intercept_),
                "absolute_coefficient": abs(
                    float(linear_regression.intercept_)
                ),
            }
        ]
    )

    return pd.concat([intercept, coefficients], ignore_index=True)


def build_gradient_boosting_importance_table(
    gradient_boosting_model,
) -> pd.DataFrame:
    """สร้างตาราง Feature Importance ของ Gradient Boosting."""

    preprocessor = gradient_boosting_model.named_steps["preprocessor"]
    gradient_boosting = gradient_boosting_model.named_steps["model"]

    importance_table = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": gradient_boosting.feature_importances_,
        }
    )

    return importance_table.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)


def save_training_outputs(
    stepwise_model,
    gradient_boosting_model,
    winner_model,
    winner_name: str,
    selected_features: list[str],
    best_gradient_boosting_params: dict,
    metrics: list[dict],
    selection_history: list[dict],
    tuning_history: list[dict],
    rolling_fold_summary: list[dict],
) -> None:
    """บันทึกโมเดล ตารางผลลัพธ์ และ Metadata ที่จำเป็น."""

    create_output_directories()

    joblib.dump(stepwise_model, STEPWISE_MODEL_PATH)
    joblib.dump(gradient_boosting_model, GRADIENT_BOOSTING_MODEL_PATH)
    joblib.dump(winner_model, WINNER_MODEL_PATH)

    pd.DataFrame(metrics).to_csv(
        METRIC_REPORT_DIR / "rental_price_model_metrics.csv",
        index=False,
    )
    pd.DataFrame(selection_history).to_csv(
        TABLE_REPORT_DIR / "rental_price_stepwise_history.csv",
        index=False,
    )
    pd.DataFrame(tuning_history).to_csv(
        TABLE_REPORT_DIR / "rental_price_gradient_boosting_tuning.csv",
        index=False,
    )
    build_stepwise_coefficient_table(stepwise_model).to_csv(
        TABLE_REPORT_DIR / "rental_price_stepwise_coefficients.csv",
        index=False,
    )
    build_gradient_boosting_importance_table(
        gradient_boosting_model
    ).to_csv(
        TABLE_REPORT_DIR / "rental_price_gradient_boosting_importance.csv",
        index=False,
    )

    metadata = {
        "winner_model": winner_name,
        "model_selection_metric": "official_validation_mae",
        "tuning_metric": "mean_rolling_validation_mae",
        "tuning_method": "4 expanding quarterly time folds",
        "rolling_folds": rolling_fold_summary,
        "all_features": FEATURE_COLUMNS,
        "stepwise_selected_features": selected_features,
        "gradient_boosting_params": best_gradient_boosting_params,
        "metrics": metrics,
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """ฝึก เปรียบเทียบ เลือก และบันทึกโมเดลราคาค่าเช่า."""

    # ------------------------------------------------------------
    # 1. สร้างข้อมูลและแบ่งตามเวลา
    # ------------------------------------------------------------
    modeling_data = build_rental_price_features()
    train_data, validation_data, test_data = chronological_split(
        modeling_data,
        date_column=DATE_COLUMN,
    )

    if train_data.empty or validation_data.empty or test_data.empty:
        raise ValueError("Train, Validation และ Test ต้องมีข้อมูลทุกชุด")

    print("Train:", len(train_data))
    print("Validation:", len(validation_data))
    print("Test:", len(test_data))

    X_train, y_train = split_features_and_target(train_data)
    X_validation, y_validation = split_features_and_target(validation_data)
    X_test, y_test = split_features_and_target(test_data)

    # Rolling Folds ทั้ง 4 รอบสร้างจากชุด Train เท่านั้น จึงยังไม่ใช้
    # Validation หลักและ Test ในขั้นเลือก Feature หรือจูนพารามิเตอร์
    rolling_folds = build_rolling_time_folds(train_data)
    rolling_fold_summary = []
    print("\nRolling Time Validation:")
    for fold in rolling_folds:
        summary = {
            "name": fold["name"],
            "train_end": fold["train_end"],
            "validation_end": fold["validation_end"],
            "train_rows": len(fold["X_train"]),
            "validation_rows": len(fold["X_validation"]),
        }
        rolling_fold_summary.append(summary)
        print(
            summary["name"],
            "Train:",
            summary["train_rows"],
            "Validation:",
            summary["validation_rows"],
        )

    # ------------------------------------------------------------
    # 2. Forward/Stepwise Multiple Linear Regression
    # ------------------------------------------------------------
    print("\nกำลังเลือก Feature สำหรับ Stepwise MLR...")
    selected_features, selection_history = forward_select_features_rolling(
        rolling_folds,
        candidate_features=FEATURE_COLUMNS,
        minimum_improvement=1.0,
    )
    print("Feature ที่เลือก:", selected_features)

    stepwise_validation_model = create_stepwise_linear_pipeline(
        selected_features
    )
    stepwise_validation_model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 3. Gradient Boosting Regressor
    # ------------------------------------------------------------
    print("\nกำลังปรับ Hyperparameters ของ Gradient Boosting...")
    best_gbr_params, tuning_history = tune_gradient_boosting(
        rolling_folds,
        n_jobs=4,
    )
    print("ค่าที่เลือก:", best_gbr_params)

    gradient_boosting_validation_model = (
        create_gradient_boosting_pipeline(
            FEATURE_COLUMNS,
            model_params=best_gbr_params,
        )
    )
    gradient_boosting_validation_model.fit(X_train, y_train)

    # ------------------------------------------------------------
    # 4. เลือกโมเดลจาก Validation MAE เท่านั้น
    # ------------------------------------------------------------
    validation_metrics = [
        evaluate_model(
            stepwise_validation_model,
            X_validation,
            y_validation,
            model_name="Stepwise MLR",
            split_name="validation",
        ),
        evaluate_model(
            gradient_boosting_validation_model,
            X_validation,
            y_validation,
            model_name="Gradient Boosting",
            split_name="validation",
        ),
    ]

    winner_validation_result = min(
        validation_metrics,
        key=lambda row: row["mae"],
    )
    winner_name = winner_validation_result["model"]
    print("\nโมเดลที่เลือกจาก Validation MAE:", winner_name)

    # ------------------------------------------------------------
    # 5. ฝึกโมเดลสุดท้ายด้วย Train + Validation
    # ------------------------------------------------------------
    X_train_validation = pd.concat(
        [X_train, X_validation],
        ignore_index=True,
    )
    y_train_validation = pd.concat(
        [y_train, y_validation],
        ignore_index=True,
    )

    final_stepwise_model = create_stepwise_linear_pipeline(selected_features)
    final_stepwise_model.fit(X_train_validation, y_train_validation)

    final_gradient_boosting_model = create_gradient_boosting_pipeline(
        FEATURE_COLUMNS,
        model_params=best_gbr_params,
    )
    final_gradient_boosting_model.fit(
        X_train_validation,
        y_train_validation,
    )

    # ------------------------------------------------------------
    # 6. ประเมิน Test เพียงครั้งเดียว และไม่เปลี่ยนโมเดลผู้ชนะ
    # ------------------------------------------------------------
    test_metrics = [
        evaluate_model(
            final_stepwise_model,
            X_test,
            y_test,
            model_name="Stepwise MLR",
            split_name="test",
        ),
        evaluate_model(
            final_gradient_boosting_model,
            X_test,
            y_test,
            model_name="Gradient Boosting",
            split_name="test",
        ),
    ]

    winner_model = (
        final_gradient_boosting_model
        if winner_name == "Gradient Boosting"
        else final_stepwise_model
    )

    # ------------------------------------------------------------
    # 7. บันทึกผลลัพธ์ทั้งหมด
    # ------------------------------------------------------------
    save_training_outputs(
        stepwise_model=final_stepwise_model,
        gradient_boosting_model=final_gradient_boosting_model,
        winner_model=winner_model,
        winner_name=winner_name,
        selected_features=selected_features,
        best_gradient_boosting_params=best_gbr_params,
        metrics=[*validation_metrics, *test_metrics],
        selection_history=selection_history,
        tuning_history=tuning_history,
        rolling_fold_summary=rolling_fold_summary,
    )

    print("\nบันทึก Stepwise MLR ที่:", STEPWISE_MODEL_PATH)
    print("บันทึก Gradient Boosting ที่:", GRADIENT_BOOSTING_MODEL_PATH)
    print("บันทึกโมเดลผู้ชนะที่:", WINNER_MODEL_PATH)


if __name__ == "__main__":
    main()
