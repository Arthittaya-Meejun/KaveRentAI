"""Reusable chronological Train/Validation/Test split."""

import pandas as pd

from kaverentai.config import DEFAULT_TRAIN_END, DEFAULT_VALIDATION_END


def chronological_split(
    data: pd.DataFrame,
    date_column: str,
    train_end: str = DEFAULT_TRAIN_END,
    validation_end: str = DEFAULT_VALIDATION_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split rows by event date without shuffling future data into the past."""
    if date_column not in data.columns:
        raise KeyError(f"Date column was not found: {date_column}")

    frame = data.copy()
    event_dates = pd.to_datetime(frame[date_column], errors="coerce")
    if event_dates.isna().any():
        bad_rows = int(event_dates.isna().sum())
        raise ValueError(f"{date_column} contains {bad_rows} unparseable values")

    train_cutoff = pd.Timestamp(train_end).normalize() + pd.Timedelta(days=1)
    validation_cutoff = (
        pd.Timestamp(validation_end).normalize() + pd.Timedelta(days=1)
    )
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must be earlier than validation_end")

    train = frame.loc[event_dates < train_cutoff].copy()
    validation = frame.loc[
        (event_dates >= train_cutoff) & (event_dates < validation_cutoff)
    ].copy()
    test = frame.loc[event_dates >= validation_cutoff].copy()

    return train, validation, test
