"""ทดสอบโมเดลและตัวทำนาย Time-to-Lease."""

import numpy as np
import pandas as pd
import pytest

from kaverentai.features.time_to_lease_features import (
    FEATURE_COLUMNS,
    split_features_and_target,
)
from kaverentai.models.evaluate_time_to_lease import evaluate_time_to_lease
from kaverentai.models.time_to_lease_models import (
    create_linear_regression_pipeline,
    create_random_forest_pipeline,
)
from kaverentai.models.time_to_lease_predictor import (
    predict_time_to_lease,
    prepare_prediction_input,
)
from kaverentai.models.train_time_to_lease import (
    build_random_forest_importance_table,
    train_and_compare_time_to_lease,
)


def make_modeling_data(rows: int = 90) -> pd.DataFrame:
    """สร้างข้อมูลจำลองขนาดเล็กที่มี Train/Validation/Test ครบ."""

    records = []
    for index in range(rows):
        size = 22 + index % 12
        asking_rent = 8_000 + size * 180 + (index % 4) * 250
        market_rent = 10_000 + (index % 5) * 100
        split = (
            "train" if index < 60 else "validation" if index < 75 else "test"
        )
        weeks = max(
            0,
            12
            + (asking_rent / market_rent - 1) * 10
            - (index % 10) * 0.45
            + (2 if index % 3 == 0 else 0),
        )
        records.append(
            {
                "listing_id": f"L-{index:03d}",
                "unit_id": f"U-{index:03d}",
                "project_id": index % 3,
                "date_listed": pd.Timestamp("2025-01-01")
                + pd.Timedelta(days=index * 7),
                "data_split": split,
                "floor": index % 12 + 1,
                "size_sqm": size,
                "room_type": "studio" if index % 2 == 0 else "1br",
                "view": "city" if index % 3 else "pool",
                "furnished": str(index % 2 == 0),
                "first_asking_rent": asking_rent,
                "project_name": f"Project {index % 3}",
                "university": "University A",
                "distance_to_campus_m": 100 + (index % 3) * 200,
                "facility_count": 20 + (index % 3) * 10,
                "season_listed": "peak" if index % 4 == 0 else "normal",
                "listing_year": 2025,
                "listing_month": index % 12 + 1,
                "market_occupancy": 0.75 + (index % 5) / 100,
                "market_demand_pressure": 1.0 + (index % 6) / 10,
                "market_median_asking_rent": market_rent,
                "rent_to_market_ratio": asking_rent / market_rent,
                "missing_market_history": str(False),
                "weeks_on_market": weeks,
            }
        )
    return pd.DataFrame(records)


def test_both_time_to_lease_pipelines_can_predict():
    data = make_modeling_data()
    X, y = split_features_and_target(data)

    linear = create_linear_regression_pipeline(FEATURE_COLUMNS)
    forest = create_random_forest_pipeline(
        FEATURE_COLUMNS,
        {"n_estimators": 10, "max_depth": 4, "n_jobs": 1},
    )
    linear.fit(X, y)
    forest.fit(X, y)

    assert len(linear.predict(X.iloc[:3])) == 3
    assert len(forest.predict(X.iloc[:3])) == 3


def test_train_and_compare_uses_validation_to_choose_winner():
    results = train_and_compare_time_to_lease(
        make_modeling_data(),
        random_forest_params={
            "n_estimators": 12,
            "max_depth": 4,
            "n_jobs": 1,
        },
    )

    validation_rows = [
        row for row in results["metrics"] if row["split"] == "validation"
    ]
    expected_winner = min(validation_rows, key=lambda row: row["mae_weeks"])

    assert len(results["metrics"]) == 4
    assert results["winner_name"] == expected_winner["model"]
    assert len(results["prediction_table"]) == 15
    assert results["deployment_training_rows"] == 90


def test_random_forest_importance_is_grouped_by_original_feature():
    data = make_modeling_data()
    X, y = split_features_and_target(data)
    model = create_random_forest_pipeline(
        FEATURE_COLUMNS,
        {"n_estimators": 10, "max_depth": 4, "n_jobs": 1},
    )
    model.fit(X, y)

    importance = build_random_forest_importance_table(model)

    assert set(importance["feature"]) == set(FEATURE_COLUMNS)
    assert importance["importance"].sum() == pytest.approx(1.0)


def test_evaluation_clips_negative_predictions():
    metrics = evaluate_time_to_lease(
        y_true=[0, 2],
        y_pred=[-3, 4],
        model_name="Example",
        split_name="test",
    )

    assert metrics["mae_weeks"] == pytest.approx(1.0)
    assert metrics["rmse_weeks"] == pytest.approx(np.sqrt(2))
    assert metrics["within_2_weeks_pct"] == 100


def test_predictor_builds_date_and_market_features():
    data = make_modeling_data()
    X, y = split_features_and_target(data)
    model = create_random_forest_pipeline(
        FEATURE_COLUMNS,
        {"n_estimators": 10, "max_depth": 4, "n_jobs": 1},
    )
    model.fit(X, y)

    room = {
        "date_listed": "2026-07-07",
        "floor": 7,
        "size_sqm": 25,
        "room_type": "studio",
        "view": "city",
        "furnished": True,
        "first_asking_rent": 12_000,
        "project_name": "Project 1",
        "university": "University A",
        "distance_to_campus_m": 300,
        "facility_count": 30,
        "season_listed": "peak",
        "occupancy": 0.8,
        "median_asking_rent": 10_000,
        "n_searchers": 30,
        "n_listings_open": 20,
    }

    prepared = prepare_prediction_input(room)
    result = predict_time_to_lease(room, model=model)

    assert prepared.loc[0, "listing_year"] == 2026
    assert prepared.loc[0, "listing_month"] == "7"
    assert prepared.loc[0, "market_demand_pressure"] == pytest.approx(1.5)
    assert prepared.loc[0, "rent_to_market_ratio"] == pytest.approx(1.2)
    assert result["predicted_weeks"] >= 0
    assert result["estimated_days"] >= 0
