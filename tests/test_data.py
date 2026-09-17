import pandas as pd
import pytest

from kaverentai.config import (
    DEFAULT_TRAIN_END,
    DEFAULT_VALIDATION_END,
    RANDOM_SEED,
    load_model_config,
)
from kaverentai.data.load_data import TABLES, load_all_data
from kaverentai.data.split_data import chronological_split
from kaverentai.data.validate_data import validate_tables


def test_cleaned_datasets_load() -> None:
    datasets = load_all_data(cleaned=True)
    assert set(datasets) == set(TABLES)
    assert all(not frame.empty for frame in datasets.values())


def test_cleaned_datasets_have_required_columns() -> None:
    assert validate_tables(load_all_data(cleaned=True)) == []


def test_chronological_split_includes_entire_cutoff_days() -> None:
    data = pd.DataFrame(
        {
            "event_time": [
                "2025-12-31 23:59:59",
                "2026-01-01 00:00:00",
                "2026-03-31 23:59:59",
                "2026-04-01 00:00:00",
            ]
        }
    )

    train, validation, test = chronological_split(data, "event_time")

    assert train.index.tolist() == [0]
    assert validation.index.tolist() == [1, 2]
    assert test.index.tolist() == [3]


def test_chronological_split_rejects_reversed_cutoffs() -> None:
    data = pd.DataFrame({"event_time": ["2026-01-01"]})

    with pytest.raises(ValueError, match="train_end must be earlier"):
        chronological_split(
            data,
            "event_time",
            train_end="2026-03-31",
            validation_end="2025-12-31",
        )


def test_runtime_settings_come_from_model_config() -> None:
    config = load_model_config()

    assert DEFAULT_TRAIN_END == config["time_split"]["train_end"]
    assert DEFAULT_VALIDATION_END == config["time_split"]["validation_end"]
    assert RANDOM_SEED == config["random_seed"]
