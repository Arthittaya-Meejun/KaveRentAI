"""Reusable chronological Train/Validation/Test split."""

import pandas as pd


def chronological_split(
    data: pd.DataFrame,
    date_column: str,
    train_end: str = "2025-12-31",
    validation_end: str = "2026-03-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by event date without shuffling future data into the past."""
    if date_column not in data.columns:
        raise KeyError(f"Date column was not found: {date_column}")

    frame = data.copy()
    event_dates = pd.to_datetime(frame[date_column], errors="coerce")
    if event_dates.isna().any():
        bad_rows = int(event_dates.isna().sum())
        raise ValueError(f"{date_column} contains {bad_rows} unparseable values")

    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)

    train = frame.loc[event_dates <= train_cutoff].copy()
    validation = frame.loc[
        (event_dates > train_cutoff) & (event_dates <= validation_cutoff)
    ].copy()
    test = frame.loc[event_dates > validation_cutoff].copy()

    return train, validation, test
