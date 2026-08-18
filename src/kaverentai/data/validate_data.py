"""Lightweight schema checks for the five cleaned datasets."""

from collections.abc import Mapping

import pandas as pd

from kaverentai.data.load_data import load_all_data


REQUIRED_COLUMNS = {
    "projects": {"project_id", "project_name", "university"},
    "units": {"unit_id", "project_id", "floor", "size_sqm"},
    "listings": {
        "listing_id",
        "unit_id",
        "project_id",
        "date_listed",
        "first_asking_rent",
        "leased",
        "weeks_on_market",
    },
    "leases": {
        "lease_id",
        "unit_id",
        "project_id",
        "date_start",
        "rent",
        "is_renewal",
    },
    "weekly_market": {
        "week",
        "date",
        "project_id",
        "occupancy",
        "n_searchers",
    },
}


def validate_tables(tables: Mapping[str, pd.DataFrame]) -> list[str]:
    """Return human-readable validation errors; an empty list means valid."""
    errors: list[str] = []

    for table_name, required in REQUIRED_COLUMNS.items():
        if table_name not in tables:
            errors.append(f"Missing table: {table_name}")
            continue

        missing = sorted(required - set(tables[table_name].columns))
        if missing:
            errors.append(f"{table_name}: missing columns {missing}")

        if tables[table_name].empty:
            errors.append(f"{table_name}: table is empty")

    return errors


def main() -> None:
    """Validate cleaned data and exit with a useful message."""
    errors = validate_tables(load_all_data(cleaned=True))
    if errors:
        raise SystemExit("Data validation failed:\n- " + "\n- ".join(errors))
    print("All cleaned datasets passed the basic schema checks.")


if __name__ == "__main__":
    main()
