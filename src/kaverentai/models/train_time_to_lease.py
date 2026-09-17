"""ฝึกและเปรียบเทียบโมเดล Time-to-Lease.

ขั้นตอนการเลือกโมเดล:

1. ฝึก Multiple Linear Regression และ Random Forest ด้วย Train
2. เลือกผู้ชนะจาก Validation MAE
3. ฝึกโมเดลทั้งสองใหม่ด้วย Train + Validation
4. ประเมิน Test เพียงครั้งเดียวโดยไม่เปลี่ยนผู้ชนะ
"""

import json

import joblib
import numpy as np
import pandas as pd

from kaverentai.config import (
    TIME_TO_LEASE_MIN_FOLLOW_UP_WEEKS,
    TIME_TO_LEASE_TEST_END,
    TIME_TO_LEASE_TRAIN_END,
    TIME_TO_LEASE_VALIDATION_END,
)
from kaverentai.features.time_to_lease_features import (
    FEATURE_COLUMNS,
    IDENTIFIER_COLUMNS,
    build_time_to_lease_features,
    split_features_and_target,
)
from kaverentai.models.evaluate_time_to_lease import (
    build_prediction_table,
    evaluate_time_to_lease,
)
from kaverentai.models.time_to_lease_models import (
    DEFAULT_RANDOM_FOREST_PARAMS,
    create_linear_regression_pipeline,
    create_random_forest_pipeline,
)
from kaverentai.paths import MODEL_DIR, REPORT_DIR


TIME_TO_LEASE_MODEL_DIR = MODEL_DIR / "time_to_lease"
LINEAR_MODEL_PATH = TIME_TO_LEASE_MODEL_DIR / "linear_regression.joblib"
RANDOM_FOREST_MODEL_PATH = TIME_TO_LEASE_MODEL_DIR / "random_forest.joblib"
WINNER_MODEL_PATH = MODEL_DIR / "time_to_lease_model.joblib"
METADATA_PATH = TIME_TO_LEASE_MODEL_DIR / "metadata.json"
METRICS_PATH = REPORT_DIR / "metrics" / "time_to_lease_model_metrics.csv"
IMPORTANCE_PATH = REPORT_DIR / "tables" / "time_to_lease_feature_importance.csv"
COEFFICIENT_PATH = REPORT_DIR / "tables" / "time_to_lease_linear_coefficients.csv"
PREDICTION_PATH = REPORT_DIR / "tables" / "time_to_lease_test_predictions.csv"


def _get_split(data: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """เลือกข้อมูลหนึ่งช่วงเวลาและแจ้งเตือนเมื่อไม่มีข้อมูล."""

    split = data.loc[data["data_split"] == split_name].copy()
    if split.empty:
        raise ValueError(f"ไม่พบข้อมูล {split_name}")
    return split


def build_linear_coefficient_table(model) -> pd.DataFrame:
    """สร้างตารางค่าสัมประสิทธิ์ของ Multiple Linear Regression."""

    preprocessor = model.named_steps["preprocessor"]
    regression = model.named_steps["model"]
    table = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "coefficient": regression.coef_,
        }
    )
    table["absolute_coefficient"] = table["coefficient"].abs()
    table = table.sort_values("absolute_coefficient", ascending=False)

    intercept = pd.DataFrame(
        [
            {
                "feature": "intercept",
                "coefficient": float(regression.intercept_),
                "absolute_coefficient": abs(float(regression.intercept_)),
            }
        ]
    )
    return pd.concat([intercept, table], ignore_index=True)


def build_random_forest_importance_table(model) -> pd.DataFrame:
    """รวม Feature Importance ของ Dummy Variables กลับเป็น Feature เดิม."""

    preprocessor = model.named_steps["preprocessor"]
    forest = model.named_steps["model"]
    category_pipeline = preprocessor.named_transformers_["category"]
    encoder = category_pipeline.named_steps["one_hot"]
    categorical_features = list(preprocessor.transformers_[0][2])
    numeric_features = list(preprocessor.transformers_[1][2])

    source_features: list[str] = []
    for feature, categories in zip(categorical_features, encoder.categories_):
        source_features.extend([feature] * len(categories))
    source_features.extend(numeric_features)

    if len(source_features) != len(forest.feature_importances_):
        raise ValueError("จำนวน Feature หลังแปลงไม่ตรงกับ Feature Importance")

    table = pd.DataFrame(
        {
            "feature": source_features,
            "importance": forest.feature_importances_,
        }
    )
    return (
        table.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_and_compare_time_to_lease(
    modeling_data: pd.DataFrame,
    random_forest_params: dict | None = None,
) -> dict:
    """ฝึก เปรียบเทียบ และคืนโมเดลพร้อมผลประเมิน."""

    train_data = _get_split(modeling_data, "train")
    validation_data = _get_split(modeling_data, "validation")
    test_data = _get_split(modeling_data, "test")

    X_train, y_train = split_features_and_target(train_data)
    X_validation, y_validation = split_features_and_target(validation_data)
    X_test, y_test = split_features_and_target(test_data)

    validation_models = {
        "Multiple Linear Regression": create_linear_regression_pipeline(
            FEATURE_COLUMNS
        ),
        "Random Forest": create_random_forest_pipeline(
            FEATURE_COLUMNS,
            model_params=random_forest_params,
        ),
    }
    validation_metrics = []
    for model_name, model in validation_models.items():
        model.fit(X_train, y_train)
        validation_metrics.append(
            evaluate_time_to_lease(
                y_validation,
                model.predict(X_validation),
                model_name=model_name,
                split_name="validation",
            )
        )

    winner_name = min(
        validation_metrics,
        key=lambda row: row["mae_weeks"],
    )["model"]

    X_train_validation = pd.concat(
        [X_train, X_validation], ignore_index=True
    )
    y_train_validation = pd.concat(
        [y_train, y_validation], ignore_index=True
    )
    final_models = {
        "Multiple Linear Regression": create_linear_regression_pipeline(
            FEATURE_COLUMNS
        ),
        "Random Forest": create_random_forest_pipeline(
            FEATURE_COLUMNS,
            model_params=random_forest_params,
        ),
    }

    test_metrics = []
    test_predictions = {}
    for model_name, model in final_models.items():
        model.fit(X_train_validation, y_train_validation)
        prediction = np.maximum(model.predict(X_test), 0)
        test_predictions[model_name] = prediction
        test_metrics.append(
            evaluate_time_to_lease(
                y_test,
                prediction,
                model_name=model_name,
                split_name="test",
            )
        )

    identifiers = test_data[
        [column for column in IDENTIFIER_COLUMNS if column != "data_split"]
    ]
    prediction_table = build_prediction_table(
        identifiers,
        y_test,
        test_predictions,
    )

    # หลังประเมิน Test เสร็จแล้ว จึงฝึกโมเดลสำหรับนำไปใช้งานใหม่ด้วย
    # Matured Cohorts ทั้งสามชุด โดยยังไม่ใช้ post_test ที่ติดตามไม่ครบ
    X_matured = pd.concat(
        [X_train, X_validation, X_test], ignore_index=True
    )
    y_matured = pd.concat(
        [y_train, y_validation, y_test], ignore_index=True
    )
    if winner_name == "Random Forest":
        deployment_model = create_random_forest_pipeline(
            FEATURE_COLUMNS,
            model_params=random_forest_params,
        )
    else:
        deployment_model = create_linear_regression_pipeline(FEATURE_COLUMNS)
    deployment_model.fit(X_matured, y_matured)

    return {
        "winner_name": winner_name,
        "winner_model": deployment_model,
        "linear_model": final_models["Multiple Linear Regression"],
        "random_forest_model": final_models["Random Forest"],
        "metrics": [*validation_metrics, *test_metrics],
        "prediction_table": prediction_table,
        "split_rows": {
            "train": len(train_data),
            "validation": len(validation_data),
            "test": len(test_data),
        },
        "deployment_training_rows": len(X_matured),
        "excluded_post_test_rows": int(
            (modeling_data["data_split"] == "post_test").sum()
        ),
    }


def save_training_outputs(
    results: dict,
    random_forest_params: dict | None = None,
) -> None:
    """บันทึกโมเดล ผลประเมิน และตารางสำหรับรายงาน."""

    for directory in [
        TIME_TO_LEASE_MODEL_DIR,
        METRICS_PATH.parent,
        IMPORTANCE_PATH.parent,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    joblib.dump(results["linear_model"], LINEAR_MODEL_PATH)
    joblib.dump(results["random_forest_model"], RANDOM_FOREST_MODEL_PATH)
    joblib.dump(results["winner_model"], WINNER_MODEL_PATH)

    pd.DataFrame(results["metrics"]).to_csv(METRICS_PATH, index=False)
    results["prediction_table"].to_csv(PREDICTION_PATH, index=False)
    build_linear_coefficient_table(results["linear_model"]).to_csv(
        COEFFICIENT_PATH, index=False
    )
    build_random_forest_importance_table(
        results["random_forest_model"]
    ).to_csv(IMPORTANCE_PATH, index=False)

    forest_params = DEFAULT_RANDOM_FOREST_PARAMS.copy()
    if random_forest_params:
        forest_params.update(random_forest_params)
    metadata = {
        "target": "weeks_on_market",
        "winner_model": results["winner_name"],
        "selection_metric": "validation_mae_weeks",
        "evaluation_method": "matured_time_split",
        "evaluation_cutoffs": {
            "train_end": TIME_TO_LEASE_TRAIN_END,
            "validation_end": TIME_TO_LEASE_VALIDATION_END,
            "test_end": TIME_TO_LEASE_TEST_END,
            "minimum_follow_up_weeks": TIME_TO_LEASE_MIN_FOLLOW_UP_WEEKS,
        },
        "features": FEATURE_COLUMNS,
        "split_rows": results["split_rows"],
        "deployment_training_rows": results["deployment_training_rows"],
        "excluded_post_test_rows": results["excluded_post_test_rows"],
        "random_forest_params": forest_params,
        "metrics": results["metrics"],
        "important_note": (
            "ใช้เฉพาะรายการ leased=True และประเมินเฉพาะ Matured Cohorts "
            f"ที่มีระยะติดตามอย่างน้อย {TIME_TO_LEASE_MIN_FOLLOW_UP_WEEKS} "
            "สัปดาห์; รายการหลัง test_end "
            "เป็น post_test และไม่ใช้วัดผล"
        ),
    }
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    """รันกระบวนการฝึกโมเดล Time-to-Lease ทั้งหมด."""

    modeling_data = build_time_to_lease_features()
    results = train_and_compare_time_to_lease(modeling_data)
    save_training_outputs(results)

    print("จำนวนข้อมูล:", results["split_rows"])
    print("\nผลเปรียบเทียบ:")
    print(pd.DataFrame(results["metrics"]).round(3).to_string(index=False))
    print("\nโมเดลที่เลือกจาก Validation MAE:", results["winner_name"])
    print("บันทึกโมเดลผู้ชนะที่:", WINNER_MODEL_PATH)


if __name__ == "__main__":
    main()
