import numpy as np
import pandas as pd
import pytest

from kaverentai.data.load_data import load_all_data
from kaverentai.features.lease_probability_features import (
    ALTERNATIVE_LISTING_SUFFIX,
    MODEL_FEATURES,
    TARGET_COLUMN,
    build_lease_probability_data,
)
from kaverentai.models.evaluate_classification import (
    build_threshold_analysis,
    calculate_classification_metrics,
)
from kaverentai.models.diagnose_lease_probability import FEATURE_SETS
from kaverentai.models.train_lease_probability import (
    build_catboost_pipeline,
    build_logistic_pipeline,
)


def _build_modeling_data() -> pd.DataFrame:
    tables = load_all_data(cleaned=True)
    return build_lease_probability_data(
        listings=tables["listings"],
        weekly_market=tables["weekly_market"],
    )


def test_lease_probability_data_has_complete_target_follow_up() -> None:
    data = _build_modeling_data()

    assert len(data) == 30_097
    assert data[TARGET_COLUMN].sum() == 17_459
    assert data["listing_id"].is_unique
    assert not data["listing_id"].str.endswith(
        ALTERNATIVE_LISTING_SUFFIX
    ).any()
    assert set(data["data_split"]) == {"train", "validation", "test"}
    assert set(MODEL_FEATURES).issubset(data.columns)


def test_model_features_exclude_post_listing_leakage() -> None:
    post_listing_columns = {
        "asking_rent",
        "weeks_on_market",
        "n_viewings",
        "week_leased",
        "tenant_segment",
        "lease_months",
        "leased",
    }

    assert post_listing_columns.isdisjoint(MODEL_FEATURES)


def test_four_week_target_excludes_right_censored_listings() -> None:
    tables = load_all_data(cleaned=True)
    data = build_lease_probability_data(
        listings=tables["listings"],
        weekly_market=tables["weekly_market"],
    )

    assert data["week_listed"].max() == 178


def test_alternative_listing_rows_share_the_base_event() -> None:
    """Document why -B rows are excluded rather than treated independently."""
    listings = load_all_data(cleaned=True)["listings"]
    alternatives = listings.loc[
        listings["listing_id"].str.endswith(ALTERNATIVE_LISTING_SUFFIX)
    ].copy()
    alternatives["base_listing_id"] = alternatives["listing_id"].str.removesuffix(
        ALTERNATIVE_LISTING_SUFFIX
    )

    paired = alternatives.merge(
        listings.add_suffix("_base"),
        left_on="base_listing_id",
        right_on="listing_id_base",
        how="left",
        validate="one_to_one",
    )

    assert len(alternatives) == 1_066
    assert paired["listing_id_base"].notna().all()
    for column in [
        "unit_id",
        "project_id",
        "week_listed",
        "date_listed",
        "first_asking_rent",
        "weeks_on_market",
        "leased",
        "week_leased",
    ]:
        assert paired[column].equals(paired[f"{column}_base"])


def test_logistic_pipeline_returns_valid_probabilities() -> None:
    data = _build_modeling_data()
    train = data.loc[data["data_split"].eq("train")]
    validation = data.loc[data["data_split"].eq("validation")]
    model = build_logistic_pipeline()
    model.fit(train.loc[:, MODEL_FEATURES], train[TARGET_COLUMN])

    probabilities = model.predict_proba(
        validation.loc[:, MODEL_FEATURES]
    )[:, 1]

    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_classification_metrics_are_perfect_for_perfect_predictions() -> None:
    actual = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.01, 0.10, 0.90, 0.99])

    metrics = calculate_classification_metrics(actual, probabilities)

    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["brier_score"] < 0.01


def test_classification_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="finite values"):
        calculate_classification_metrics([0, 1], [0.2, np.nan])


def test_threshold_analysis_compares_requested_thresholds() -> None:
    analysis = build_threshold_analysis(
        [0, 0, 1, 1],
        [0.10, 0.40, 0.60, 0.90],
        thresholds=np.array([0.30, 0.50, 0.70]),
    )

    assert analysis["threshold"].tolist() == [0.30, 0.50, 0.70]
    assert analysis.loc[1, "balanced_accuracy"] == 1.0


def test_ablation_feature_sets_are_valid_model_subsets() -> None:
    for features in FEATURE_SETS.values():
        assert set(features).issubset(MODEL_FEATURES)
        assert len(features) == len(set(features))

    reduced_features = FEATURE_SETS["without_all_time"]
    pipeline = build_catboost_pipeline(reduced_features)
    cat_features = pipeline.named_steps["model"].get_param("cat_features")

    assert "season_listed" not in reduced_features
    assert set(cat_features).issubset(reduced_features)
