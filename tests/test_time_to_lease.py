"""ทดสอบการสร้าง Feature สำหรับโมเดล Time-to-Lease."""

import pandas as pd
import pytest

from kaverentai.features.time_to_lease_features import (
    FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    assign_time_split,
    build_time_to_lease_features,
    prepare_time_to_lease_input,
    split_features_and_target,
)


def make_small_datasets() -> dict[str, pd.DataFrame]:
    """สร้างข้อมูลขนาดเล็กเพื่อทดสอบกติกาโดยไม่อ่าน CSV จริง."""

    common = {
        "project_name": "Project A",
        "university": "University A",
        "distance_to_campus_m": 300,
        "facility_count": 20,
        "floor": 5,
        "size_sqm": 25.0,
        "room_type": "studio",
        "view": "city",
        "furnished": True,
        "season_listed": "normal",
        "first_asking_rent": 10_000,
    }
    listings = pd.DataFrame(
        [
            {
                **common,
                "listing_id": "L-TRAIN",
                "unit_id": "U-1",
                "project_id": 1,
                "week_listed": 2,
                "date_listed": "2024-12-31",
                "weeks_on_market": 4,
                "leased": True,
                "week_leased": 6,
            },
            {
                **common,
                "listing_id": "L-VALID",
                "unit_id": "U-2",
                "project_id": 1,
                "week_listed": 3,
                "date_listed": "2025-03-31",
                "weeks_on_market": 2,
                "leased": "True",
                "week_leased": 5,
            },
            {
                **common,
                "listing_id": "L-TEST",
                "unit_id": "U-3",
                "project_id": 1,
                "week_listed": 4,
                "date_listed": "2025-04-01",
                "weeks_on_market": 1,
                "leased": 1,
                "week_leased": 5,
            },
            {
                **common,
                "listing_id": "L-DUP-B",
                "unit_id": "U-4",
                "project_id": 1,
                "week_listed": 4,
                "date_listed": "2025-04-01",
                "weeks_on_market": 1,
                "leased": True,
                "week_leased": 5,
            },
            {
                **common,
                "listing_id": "L-OPEN",
                "unit_id": "U-5",
                "project_id": 1,
                "week_listed": 4,
                "date_listed": "2025-04-01",
                "weeks_on_market": 0,
                "leased": False,
                "week_leased": pd.NA,
            },
            {
                **common,
                "listing_id": "L-NO-HISTORY",
                "unit_id": "U-6",
                "project_id": 2,
                "week_listed": 1,
                "date_listed": "2025-01-07",
                "weeks_on_market": 3,
                "leased": True,
                "week_leased": 4,
            },
        ]
    )

    weekly_market = pd.DataFrame(
        [
            {
                "week": week,
                "date": pd.Timestamp("2026-06-23"),
                "project_id": 1,
                "n_searchers": 20 + week,
                "n_listings_open": 10,
                "occupancy": 0.70 + week / 100,
                "median_asking_rent": 9_000 + week * 100,
            }
            for week in [1, 2, 3, 4]
        ]
        + [
            {
                "week": 1,
                "date": pd.Timestamp("2026-06-23"),
                "project_id": 2,
                "n_searchers": 10,
                "n_listings_open": 5,
                "occupancy": 0.5,
                "median_asking_rent": 8_000,
            }
        ]
    )
    return {"listings": listings, "weekly_market": weekly_market}


def test_builder_keeps_only_completed_primary_listings():
    data = build_time_to_lease_features(make_small_datasets())

    assert set(data["listing_id"]) == {
        "L-TRAIN",
        "L-VALID",
        "L-TEST",
        "L-NO-HISTORY",
    }
    assert (data[TARGET_COLUMN] >= 0).all()


def test_builder_uses_previous_week_market_only():
    data = build_time_to_lease_features(make_small_datasets())
    row = data.loc[data["listing_id"] == "L-TRAIN"].iloc[0]

    # L-TRAIN ลงสัปดาห์ 2 จึงต้องใช้ค่า occupancy ของสัปดาห์ 1 (= 0.71)
    assert row["market_occupancy"] == pytest.approx(0.71)
    assert row["market_median_asking_rent"] == 9_100
    assert row["market_demand_pressure"] == pytest.approx(2.1)


def test_builder_marks_missing_previous_market_history():
    data = build_time_to_lease_features(make_small_datasets())
    row = data.loc[data["listing_id"] == "L-NO-HISTORY"].iloc[0]

    assert bool(row["missing_market_history"])
    assert pd.isna(row["market_occupancy"])


def test_builder_assigns_chronological_split_at_boundaries():
    data = build_time_to_lease_features(make_small_datasets())
    actual = data.set_index("listing_id")["data_split"].to_dict()

    assert actual["L-TRAIN"] == "train"
    assert actual["L-VALID"] == "validation"
    assert actual["L-TEST"] == "test"


def test_features_do_not_contain_post_listing_leakage():
    assert not LEAKAGE_COLUMNS.intersection(FEATURE_COLUMNS)

    data = build_time_to_lease_features(make_small_datasets())
    prepared = prepare_time_to_lease_input(data)

    assert list(prepared.columns) == FEATURE_COLUMNS
    assert not LEAKAGE_COLUMNS.intersection(prepared.columns)


def test_split_features_and_target_returns_matching_rows():
    data = build_time_to_lease_features(make_small_datasets())

    features, target = split_features_and_target(data)

    assert len(features) == len(target) == 4
    assert target.name == TARGET_COLUMN


def test_assign_time_split_rejects_reversed_cutoffs():
    dates = pd.Series(pd.to_datetime(["2025-01-01"]))

    with pytest.raises(ValueError, match="train_end must be earlier"):
        assign_time_split(
            dates,
            train_end="2026-03-31",
            validation_end="2025-12-31",
            test_end="2026-06-30",
        )


def test_assign_time_split_keeps_immature_rows_out_of_test():
    dates = pd.Series(
        pd.to_datetime(
            ["2024-12-31", "2025-03-31", "2025-06-30", "2025-07-01"]
        )
    )

    assert assign_time_split(dates).tolist() == [
        "train",
        "validation",
        "test",
        "post_test",
    ]


def test_builder_rejects_test_without_enough_follow_up():
    datasets = make_small_datasets()
    datasets["weekly_market"]["date"] = pd.Timestamp("2025-12-23")

    with pytest.raises(ValueError, match="ระยะติดตามไม่พอ"):
        build_time_to_lease_features(datasets)
