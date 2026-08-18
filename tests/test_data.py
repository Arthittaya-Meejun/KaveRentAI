from kaverentai.data.load_data import TABLES, load_all_data
from kaverentai.data.validate_data import validate_tables


def test_cleaned_datasets_load() -> None:
    datasets = load_all_data(cleaned=True)
    assert set(datasets) == set(TABLES)
    assert all(not frame.empty for frame in datasets.values())


def test_cleaned_datasets_have_required_columns() -> None:
    assert validate_tables(load_all_data(cleaned=True)) == []
