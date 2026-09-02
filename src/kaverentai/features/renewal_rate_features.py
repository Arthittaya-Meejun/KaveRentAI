"""Aggregate contract outcomes into a project-month renewal-rate table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from kaverentai.config import DEFAULT_TRAIN_END, DEFAULT_VALIDATION_END
from kaverentai.data.load_data import load_all_data
from kaverentai.data.types import coerce_boolean
from kaverentai.paths import MODELING_DATA_DIR


TARGET_COLUMN = "renewal_rate"

RATE_MODEL_FEATURES = (
    "expiring_contracts",
    "share_current_renewal",
    "share_12_month_contract",
    "mean_prior_renewal_rate",
    "project_previous_observed_rate",
    "project_historical_rate",
    "overall_historical_rate",
    "lagged_rent_to_market_ratio",
    "lagged_occupancy",
    "lagged_demand_pressure",
    "month_sin",
    "month_cos",
    "project_id_category",
)

OUTPUT_COLUMNS = (
    "project_id",
    "expiry_month",
    "data_split",
    "expiring_contracts",
    "renewed_contracts",
    TARGET_COLUMN,
    *[feature for feature in RATE_MODEL_FEATURES if feature != "expiring_contracts"],
)


_LEASE_COLUMNS = {
    "lease_id",
    "unit_id",
    "project_id",
    "week_start",
    "week_end",
    "lease_months",
    "rent",
    "is_renewal",
}
_MARKET_COLUMNS = {
    "project_id",
    "week",
    "date",
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
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")


def _build_contract_outcomes(
    leases: pd.DataFrame,
    weekly_market: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str,
) -> pd.DataFrame:
    """Construct expired-contract outcomes internally for aggregation."""
    _require_columns(leases, _LEASE_COLUMNS, "leases")
    _require_columns(weekly_market, _MARKET_COLUMNS, "weekly_market")
    if weekly_market.empty:
        raise ValueError("weekly_market must contain at least one observed week")

    contracts = leases.copy().sort_values(
        ["unit_id", "week_start", "lease_id"], kind="stable"
    ).reset_index(drop=True)
    contracts["is_renewal"] = coerce_boolean(
        contracts["is_renewal"], "leases.is_renewal"
    )

    unit_history = contracts.groupby("unit_id", sort=False)
    contracts["next_week_start"] = unit_history["week_start"].shift(-1)
    contracts["next_is_renewal"] = unit_history["is_renewal"].shift(-1)
    contracts["prior_contract_count"] = unit_history.cumcount()
    prior_renewals = (
        unit_history["is_renewal"].cumsum()
        - contracts["is_renewal"].astype(int)
    )
    contracts["prior_renewal_rate"] = np.where(
        contracts["prior_contract_count"] > 0,
        prior_renewals / contracts["prior_contract_count"],
        np.nan,
    )

    last_observed_week = weekly_market["week"].max()
    contracts = contracts.loc[
        contracts["week_end"].le(last_observed_week)
    ].copy()
    contracts["will_renew"] = (
        contracts["next_week_start"].eq(contracts["week_end"])
        & contracts["next_is_renewal"].eq(True)
    ).astype("int8")

    expiry_market = weekly_market[
        [
            "project_id",
            "week",
            "date",
            "n_searchers",
            "n_listings_open",
            "occupancy",
            "median_asking_rent",
        ]
    ].rename(
        columns={
            "week": "week_end",
            "date": "expiry_date",
            "n_searchers": "expiry_n_searchers",
            "n_listings_open": "expiry_n_listings_open",
            "occupancy": "expiry_occupancy",
            "median_asking_rent": "expiry_median_asking_rent",
        }
    )
    contracts = contracts.merge(
        expiry_market,
        on=["project_id", "week_end"],
        how="left",
        validate="many_to_one",
    )
    contracts["expiry_date"] = pd.to_datetime(
        contracts["expiry_date"], errors="coerce"
    )
    if contracts["expiry_date"].isna().any():
        raise ValueError("Some expired contracts have no matching market date")

    contracts["rent_to_market_ratio"] = (
        contracts["rent"] / contracts["expiry_median_asking_rent"]
    )
    contracts["demand_pressure"] = contracts["expiry_n_searchers"] / contracts[
        "expiry_n_listings_open"
    ].clip(lower=1)

    train_cutoff = pd.Timestamp(train_end).normalize() + pd.Timedelta(days=1)
    validation_cutoff = (
        pd.Timestamp(validation_end).normalize() + pd.Timedelta(days=1)
    )
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must be earlier than validation_end")
    contracts["data_split"] = np.select(
        [
            contracts["expiry_date"].lt(train_cutoff),
            contracts["expiry_date"].lt(validation_cutoff),
        ],
        ["train", "validation"],
        default="test",
    )
    return contracts


def _aggregate_contract_outcomes(contract_data: pd.DataFrame) -> pd.DataFrame:
    """Return one row per project and expiry month.

    All historical-rate and market variables are shifted so that the current
    month's renewal outcomes are never used as predictors for that month.
    """
    required = {
        "project_id",
        "expiry_date",
        "data_split",
        "will_renew",
        "is_renewal",
        "lease_months",
        "prior_renewal_rate",
        "rent_to_market_ratio",
        "expiry_occupancy",
        "demand_pressure",
    }
    missing = sorted(required - set(contract_data.columns))
    if missing:
        raise ValueError(f"Contract data is missing required columns: {missing}")

    contracts = contract_data.copy()
    contracts["expiry_date"] = pd.to_datetime(
        contracts["expiry_date"], errors="coerce"
    )
    if contracts["expiry_date"].isna().any():
        raise ValueError("expiry_date contains missing or invalid values")

    contracts["expiry_month"] = (
        contracts["expiry_date"].dt.to_period("M").dt.to_timestamp()
    )
    contracts["is_12_month_contract"] = contracts["lease_months"].eq(12)

    split_counts = contracts.groupby(
        ["project_id", "expiry_month"]
    )["data_split"].nunique()
    if split_counts.gt(1).any():
        raise ValueError("A project-month contains more than one data split")

    monthly = (
        contracts.groupby(["project_id", "expiry_month"], as_index=False)
        .agg(
            data_split=("data_split", "first"),
            expiring_contracts=("will_renew", "size"),
            renewed_contracts=("will_renew", "sum"),
            share_current_renewal=("is_renewal", "mean"),
            share_12_month_contract=("is_12_month_contract", "mean"),
            mean_prior_renewal_rate=("prior_renewal_rate", "mean"),
            current_rent_to_market_ratio=("rent_to_market_ratio", "mean"),
            current_occupancy=("expiry_occupancy", "mean"),
            current_demand_pressure=("demand_pressure", "mean"),
        )
        .sort_values(["project_id", "expiry_month"], kind="stable")
        .reset_index(drop=True)
    )
    monthly[TARGET_COLUMN] = (
        monthly["renewed_contracts"] / monthly["expiring_contracts"]
    )

    project_history = monthly.groupby("project_id", sort=False)
    monthly["project_previous_observed_rate"] = project_history[
        TARGET_COLUMN
    ].shift(1)
    prior_project_renewals = (
        project_history["renewed_contracts"].cumsum()
        - monthly["renewed_contracts"]
    )
    prior_project_contracts = (
        project_history["expiring_contracts"].cumsum()
        - monthly["expiring_contracts"]
    )
    monthly["project_historical_rate"] = np.where(
        prior_project_contracts > 0,
        prior_project_renewals / prior_project_contracts,
        np.nan,
    )

    # Market conditions are lagged by one observed project-month.
    monthly["lagged_rent_to_market_ratio"] = project_history[
        "current_rent_to_market_ratio"
    ].shift(1)
    monthly["lagged_occupancy"] = project_history[
        "current_occupancy"
    ].shift(1)
    monthly["lagged_demand_pressure"] = project_history[
        "current_demand_pressure"
    ].shift(1)

    overall_month = (
        monthly.groupby("expiry_month", as_index=False)
        .agg(
            month_renewed_contracts=("renewed_contracts", "sum"),
            month_expiring_contracts=("expiring_contracts", "sum"),
        )
        .sort_values("expiry_month", kind="stable")
    )
    overall_month["overall_historical_rate"] = (
        overall_month["month_renewed_contracts"].cumsum()
        - overall_month["month_renewed_contracts"]
    ) / (
        overall_month["month_expiring_contracts"].cumsum()
        - overall_month["month_expiring_contracts"]
    ).replace(0, np.nan)
    monthly = monthly.merge(
        overall_month[["expiry_month", "overall_historical_rate"]],
        on="expiry_month",
        how="left",
        validate="many_to_one",
    )

    month_number = monthly["expiry_month"].dt.month
    monthly["month_sin"] = np.sin(2 * np.pi * month_number / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * month_number / 12)
    monthly["project_id_category"] = monthly["project_id"].astype("string")

    result = monthly.loc[:, OUTPUT_COLUMNS].copy()
    result = result.sort_values(
        ["expiry_month", "project_id"], kind="stable"
    ).reset_index(drop=True)
    return result


def build_renewal_rate_data(
    leases: pd.DataFrame,
    weekly_market: pd.DataFrame,
    *,
    train_end: str = DEFAULT_TRAIN_END,
    validation_end: str = DEFAULT_VALIDATION_END,
) -> pd.DataFrame:
    """Build the complete project-month dataset from cleaned source tables."""
    contract_outcomes = _build_contract_outcomes(
        leases,
        weekly_market,
        train_end=train_end,
        validation_end=validation_end,
    )
    return _aggregate_contract_outcomes(contract_outcomes)


def save_renewal_rate_data(
    data: pd.DataFrame,
    output_path: str | Path | None = None,
) -> Path:
    """Save the aggregate modeling table."""
    path = Path(output_path) if output_path is not None else (
        MODELING_DATA_DIR / "renewal_rate_monthly.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path.resolve()


def main() -> None:
    """Build the project-month table directly from cleaned source data."""
    tables = load_all_data(cleaned=True)
    monthly = build_renewal_rate_data(
        leases=tables["leases"],
        weekly_market=tables["weekly_market"],
    )
    output_path = save_renewal_rate_data(monthly)

    print(f"Saved {len(monthly):,} project-month rows to {output_path}")
    print(
        f"Period: {monthly['expiry_month'].min():%Y-%m} to "
        f"{monthly['expiry_month'].max():%Y-%m}"
    )
    print(
        "Overall renewal rate: "
        f"{monthly['renewed_contracts'].sum() / monthly['expiring_contracts'].sum():.1%}"
    )


if __name__ == "__main__":
    main()
