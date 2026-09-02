"""Small, reusable helpers for cleaning column data types."""

from __future__ import annotations

import numpy as np
import pandas as pd


def coerce_boolean(series: pd.Series, column_name: str) -> pd.Series:
    """Convert common boolean values and reject ambiguous values.

    Using ``astype(bool)`` is unsafe for text because both ``"True"`` and
    ``"False"`` are non-empty strings, so both would become ``True``.
    """
    text_values = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    def normalize(value: object) -> bool | None:
        if pd.isna(value):
            return None
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, np.integer)) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            return text_values.get(value.strip().lower())
        return None

    converted = series.map(normalize)
    if converted.isna().any():
        invalid_values = series.loc[converted.isna()].drop_duplicates().tolist()
        raise ValueError(
            f"{column_name} contains unsupported boolean values: "
            f"{invalid_values[:5]}"
        )
    return converted.astype(bool)
