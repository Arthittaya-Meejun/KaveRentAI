"""Build listing-time features for lease-probability classification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kaverentai.config import (
    DEFAULT_TRAIN_END,
    DEFAULT_VALIDATION_END,
    LEASE_PROBABILITY_HORIZON_WEEKS,
    LEASE_PROBABILITY_TARGET,
)
from kaverentai.data.load_data import load_all_data
from kaverentai.data.split_data import chronological_split
from kaverentai.data.types import coerce_boolean
from kaverentai.paths import MODELING_DATA_DIR


TARGET_COLUMN = LEASE_PROBABILITY_TARGET
ALTERNATIVE_LISTING_SUFFIX = "-B"

NUMERIC_FEATURES = (
    "distance_to_campus_m",
    "facility_count",
    "floor",
    "size_sqm",
    "first_asking_rent",
    "rent_to_market_ratio",
    "market_occupancy",
    "market_demand_pressure",
    "week_listed",
    "listing_month_sin",
    "listing_month_cos",
)

CATEGORICAL_FEATURES = (
    "project_id_category",
    "university",
    "room_type",
    "view",
    "furnished_category",
    "agent_id",
    "season_listed",
)

MODEL_FEATURES = (*NUMERIC_FEATURES, *CATEGORICAL_FEATURES)

OUTPUT_COLUMNS = (
    "listing_id",
    "unit_id",
    "project_id",
    "date_listed",
    "data_split",
    TARGET_COLUMN,
    *MODEL_FEATURES,
)

_LISTING_COLUMNS = {
    "listing_id",
    "unit_id",
    "project_id",
    "university",
    "distance_to_campus_m",
    "facility_count",
    "agent_id",
    "floor",
    "size_sqm",
    "room_type",
    "view",
    "furnished",
    "week_listed",
    "date_listed",
    "season_listed",
    "first_asking_rent",
    "weeks_on_market",
    "leased",
}

_MARKET_COLUMNS = {
    "project_id",
    "week",
    "n_searchers",
    "n_listings_open",
    "occupancy",
    "median_asking_rent",
}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    table_name: str,
) -> None:
    """Raise a clear error when an input table is incomplete."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def build_lease_probability_data(
    listings: pd.DataFrame,
    weekly_market: pd.DataFrame,
    *,
    horizon_weeks: int = LEASE_PROBABILITY_HORIZON_WEEKS,
    train_end: str = DEFAULT_TRAIN_END,
    validation_end: str = DEFAULT_VALIDATION_END,
) -> pd.DataFrame:
    """Create one model row per listing using information known at listing time.

    The target asks whether a listing is leased within a fixed number of weeks.
    Listings too recent to observe the complete target window are excluded. This
    prevents open listings near the dataset end from being treated as failures.

    Rows whose listing ID ends in ``-B`` are alternative versions of an existing
    listing. They share the room, listing date, and outcome with the base row, so
    counting both would give one event twice the weight during model training.
    """
    _require_columns(listings, _LISTING_COLUMNS, "listings")
    _require_columns(weekly_market, _MARKET_COLUMNS, "weekly_market")
    if horizon_weeks <= 0:
        raise ValueError("horizon_weeks must be greater than zero")
    if weekly_market.empty:
        raise ValueError("weekly_market must contain at least one observed week")

    frame = listings.copy()
    frame["leased"] = coerce_boolean(frame["leased"], "listings.leased")
    frame["furnished"] = coerce_boolean(
        frame["furnished"], "listings.furnished"
    )
    frame["date_listed"] = pd.to_datetime(
        frame["date_listed"], errors="coerce"
    )
    if frame["date_listed"].isna().any():
        raise ValueError("listings.date_listed contains missing or invalid values")

    # Keep the source data unchanged, but use one row per observed listing event.
    alternative_listing = frame["listing_id"].astype("string").str.endswith(
        ALTERNATIVE_LISTING_SUFFIX,
        na=False,
    )
    frame = frame.loc[~alternative_listing].copy()

    # The snapshot is taken immediately after the final observed market week.
    observation_end_week = int(weekly_market["week"].max()) + 1
    enough_follow_up = (
        frame["week_listed"] + horizon_weeks <= observation_end_week
    )
    frame = frame.loc[enough_follow_up].copy()
    if frame.empty:
        raise ValueError("No listings have enough follow-up for the target horizon")

    frame[TARGET_COLUMN] = (
        frame["leased"] & frame["weeks_on_market"].le(horizon_weeks)
    )

    market_at_listing = weekly_market[
        [
            "project_id",
            "week",
            "n_searchers",
            "n_listings_open",
            "occupancy",
            "median_asking_rent",
        ]
    ].rename(
        columns={
            "week": "week_listed",
            "n_searchers": "market_n_searchers",
            "n_listings_open": "market_n_listings_open",
            "occupancy": "market_occupancy",
            "median_asking_rent": "market_median_asking_rent",
        }
    )
    frame = frame.merge(
        market_at_listing,
        on=["project_id", "week_listed"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if frame["_merge"].ne("both").any():
        raise ValueError("Some listings have no matching listing-week market data")
    frame = frame.drop(columns="_merge")

    frame["rent_to_market_ratio"] = (
        frame["first_asking_rent"] / frame["market_median_asking_rent"]
    )
    frame["market_demand_pressure"] = (
        frame["market_n_searchers"]
        / frame["market_n_listings_open"].clip(lower=1)
    )
    listing_month = frame["date_listed"].dt.month
    frame["listing_month_sin"] = np.sin(2 * np.pi * listing_month / 12)
    frame["listing_month_cos"] = np.cos(2 * np.pi * listing_month / 12)
    frame["project_id_category"] = frame["project_id"].astype("string")
    frame["furnished_category"] = frame["furnished"].map(
        {True: "furnished", False: "unfurnished"}
    )

    train, validation, test = chronological_split(
        frame,
        "date_listed",
        train_end=train_end,
        validation_end=validation_end,
    )
    frame["data_split"] = ""
    frame.loc[train.index, "data_split"] = "train"
    frame.loc[validation.index, "data_split"] = "validation"
    frame.loc[test.index, "data_split"] = "test"

    result = frame.loc[:, OUTPUT_COLUMNS].copy()
    return result.sort_values(
        ["date_listed", "listing_id"], kind="stable"
    ).reset_index(drop=True)


def save_lease_probability_data(
    data: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """Save the reproducible modeling table."""
    path = Path(output_path) if output_path is not None else (
        MODELING_DATA_DIR / "lease_probability.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path.resolve()


def main() -> None:
    """Build and save the lease-probability modeling table."""
    tables = load_all_data(cleaned=True)
    data = build_lease_probability_data(
        listings=tables["listings"],
        weekly_market=tables["weekly_market"],
    )
    output_path = save_lease_probability_data(data)

    print(f"Saved {len(data):,} listing rows to {output_path}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Positive rate: {data[TARGET_COLUMN].mean():.1%}")
    print(data["data_split"].value_counts().to_string())


if __name__ == "__main__":
    main()
