"""Load the five source tables from a consistent location."""

from pathlib import Path

import pandas as pd

from kaverentai.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR


TABLES = ("projects", "units", "listings", "leases", "weekly_market")


def load_all_data(
    cleaned: bool = True,
    data_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load every project table and return a dictionary keyed by table name."""
    base_dir = Path(data_dir) if data_dir is not None else (
        PROCESSED_DATA_DIR if cleaned else RAW_DATA_DIR
    )
    suffix = "_cleaned" if cleaned else ""

    datasets: dict[str, pd.DataFrame] = {}
    for table in TABLES:
        path = base_dir / f"{table}{suffix}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Required dataset was not found: {path}")
        datasets[table] = pd.read_csv(path)

    return datasets


if __name__ == "__main__":
    for name, frame in load_all_data().items():
        print(f"{name:14s} rows={len(frame):,} columns={len(frame.columns)}")
