"""ทดสอบส่วนสำคัญของโมเดลพยากรณ์ราคาค่าเช่า."""

import joblib
import pandas as pd

from kaverentai.features.rental_price_features import (
    FEATURE_COLUMNS,
    prepare_rental_price_input,
)
from kaverentai.models.rental_price_models import (
    create_gradient_boosting_pipeline,
    create_stepwise_linear_pipeline,
    forward_select_features,
    forward_select_features_rolling,
)
from kaverentai.models.train_rental_price import (
    build_gradient_boosting_importance_table,
    build_stepwise_coefficient_table,
)
from kaverentai.models.tune_rental_price import (
    build_rolling_time_folds,
    tune_gradient_boosting,
)
from kaverentai.recommendation.price_engine import recommend_rental_price


def make_small_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """สร้างข้อมูลขนาดเล็กสำหรับ Unit Test โดยไม่อ่าน CSV จริง."""

    rows = []
    targets = []

    for index in range(60):
        size_sqm = 20 + index * 0.2
        floor = index % 10 + 1

        rows.append(
            {
                "floor": floor,
                "size_sqm": size_sqm,
                "room_type": "studio" if index % 2 == 0 else "1br",
                "view": "city" if index % 3 else "pool",
                "furnished": str(index % 2 == 0),
                "lease_months": 12 if index % 2 == 0 else 6,
                "is_renewal": str(index % 4 == 0),
                "project_name": "Project A" if index < 30 else "Project B",
                "university": "University A",
                "distance_to_campus_m": 100 + index,
                "facility_count": 20,
                "start_year": 2025,
                "start_month": str(index % 12 + 1),
            }
        )
        targets.append(4000 + size_sqm * 250 + floor * 5)

    return pd.DataFrame(rows), pd.Series(targets, name="rent")


def make_small_rolling_folds() -> list[dict]:
    """สร้าง 2 Rolling Folds สำหรับทดสอบโดยไม่ใช้ไฟล์ข้อมูลจริง."""

    X, y = make_small_dataset()

    return [
        {
            "name": "fold_1",
            "X_train": X.iloc[:30],
            "y_train": y.iloc[:30],
            "X_validation": X.iloc[30:40],
            "y_validation": y.iloc[30:40],
        },
        {
            "name": "fold_2",
            "X_train": X.iloc[:40],
            "y_train": y.iloc[:40],
            "X_validation": X.iloc[40:50],
            "y_validation": y.iloc[40:50],
        },
    ]


def test_prepare_rental_price_input_keeps_expected_columns():
    X, _ = make_small_dataset()

    prepared = prepare_rental_price_input(X)

    assert list(prepared.columns) == FEATURE_COLUMNS


def test_forward_selection_finds_size_as_important_feature():
    X, y = make_small_dataset()
    X_train, X_validation = X.iloc[:45], X.iloc[45:]
    y_train, y_validation = y.iloc[:45], y.iloc[45:]

    selected, history = forward_select_features(
        X_train,
        y_train,
        X_validation,
        y_validation,
        candidate_features=["floor", "size_sqm", "room_type"],
        minimum_improvement=0.1,
        verbose=False,
    )

    assert "size_sqm" in selected
    assert history


def test_rolling_forward_selection_uses_every_fold():
    selected, history = forward_select_features_rolling(
        make_small_rolling_folds(),
        candidate_features=["floor", "size_sqm", "room_type"],
        minimum_improvement=0.1,
        verbose=False,
    )

    assert "size_sqm" in selected
    assert "fold_1_mae" in history[0]
    assert "fold_2_mae" in history[0]


def test_build_rolling_time_folds_keeps_future_out_of_train():
    X, y = make_small_dataset()
    modeling_data = X.copy()
    modeling_data["rent"] = y
    modeling_data["date_start"] = pd.date_range(
        "2025-01-01",
        periods=len(modeling_data),
        freq="D",
    )
    windows = [
        {
            "name": "test_fold",
            "train_end": "2025-01-30",
            "validation_end": "2025-02-10",
        }
    ]

    folds = build_rolling_time_folds(modeling_data, windows=windows)

    assert len(folds[0]["X_train"]) == 30
    assert len(folds[0]["X_validation"]) == 11


def test_gradient_boosting_tuning_compares_rolling_average():
    parameter_sets = [
        {
            "n_estimators": 5,
            "learning_rate": 0.05,
            "max_depth": 1,
            "min_samples_leaf": 2,
            "subsample": 1.0,
        },
        {
            "n_estimators": 10,
            "learning_rate": 0.05,
            "max_depth": 1,
            "min_samples_leaf": 2,
            "subsample": 1.0,
        },
    ]

    best_params, history = tune_gradient_boosting(
        make_small_rolling_folds(),
        parameter_sets=parameter_sets,
        verbose=False,
        n_jobs=1,
    )

    best_row = min(history, key=lambda row: row["mean_validation_mae"])
    assert best_params["n_estimators"] == best_row["n_estimators"]
    assert len(history) == 2


def test_both_model_pipelines_can_predict():
    X, y = make_small_dataset()

    linear_model = create_stepwise_linear_pipeline(
        ["floor", "size_sqm", "room_type"]
    )
    linear_model.fit(X, y)

    gradient_boosting_model = create_gradient_boosting_pipeline(
        FEATURE_COLUMNS,
        model_params={"n_estimators": 10, "max_depth": 1},
    )
    gradient_boosting_model.fit(X, y)

    assert len(linear_model.predict(X.iloc[:2])) == 2
    assert len(gradient_boosting_model.predict(X.iloc[:2])) == 2

    coefficient_table = build_stepwise_coefficient_table(linear_model)
    importance_table = build_gradient_boosting_importance_table(
        gradient_boosting_model
    )

    assert "intercept" in coefficient_table["feature"].to_list()
    assert importance_table["importance"].sum() > 0


def test_price_engine_loads_saved_pipeline(tmp_path):
    X, y = make_small_dataset()
    model = create_stepwise_linear_pipeline(["floor", "size_sqm"])
    model.fit(X, y)

    model_path = tmp_path / "rental_price_model.joblib"
    joblib.dump(model, model_path)

    room_data = X.iloc[0].to_dict()
    room_data["date_start"] = "2026-08-20"

    result = recommend_rental_price(room_data, model_path=model_path)

    assert result["predicted_rent"] > 0
    assert result["recommended_rent"] > 0
