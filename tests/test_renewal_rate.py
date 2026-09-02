import numpy as np
import pandas as pd
import pytest

from kaverentai.data.load_data import load_all_data
from kaverentai.features.renewal_rate_features import (
    RATE_MODEL_FEATURES,
    build_renewal_rate_data,
)
from kaverentai.models.train_renewal_rate import (
    SELECTED_MODEL,
    build_business_predictions,
    build_rate_pipeline,
    calculate_rate_metrics,
    evaluate_final_test,
    expand_binomial_rows,
)


def _build_rate_data():
    tables = load_all_data(cleaned=True)
    return build_renewal_rate_data(
        leases=tables["leases"],
        weekly_market=tables["weekly_market"],
    )


def test_project_month_data_preserves_contract_totals() -> None:
    data = _build_rate_data()

    assert len(data) == 438
    assert not data.duplicated(["project_id", "expiry_month"]).any()
    assert data["expiring_contracts"].sum() == 36_595
    assert data["renewed_contracts"].sum() == 17_644
    assert data["renewal_rate"].between(0, 1).all()
    assert set(RATE_MODEL_FEATURES).issubset(data.columns)


def test_historical_features_do_not_use_current_outcome() -> None:
    data = _build_rate_data().sort_values(
        ["project_id", "expiry_month"]
    )
    first_project_month = data.groupby("project_id").head(1)

    assert first_project_month["project_historical_rate"].isna().all()
    assert first_project_month["project_previous_observed_rate"].isna().all()
    assert first_project_month["lagged_occupancy"].isna().all()


def test_string_boolean_values_preserve_renewal_outcomes() -> None:
    tables = load_all_data(cleaned=True)
    expected = build_renewal_rate_data(
        leases=tables["leases"],
        weekly_market=tables["weekly_market"],
    )
    string_leases = tables["leases"].copy()
    string_leases["is_renewal"] = string_leases["is_renewal"].map(
        {True: "True", False: "False"}
    )

    actual = build_renewal_rate_data(
        leases=string_leases,
        weekly_market=tables["weekly_market"],
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_invalid_boolean_values_are_rejected() -> None:
    tables = load_all_data(cleaned=True)
    invalid_leases = tables["leases"].copy()
    invalid_leases["is_renewal"] = invalid_leases["is_renewal"].astype(
        "object"
    )
    invalid_leases.loc[invalid_leases.index[0], "is_renewal"] = "unknown"

    with pytest.raises(ValueError, match="unsupported boolean values"):
        build_renewal_rate_data(
            leases=invalid_leases,
            weekly_market=tables["weekly_market"],
        )


def test_binomial_expansion_and_pipeline_probabilities() -> None:
    data = _build_rate_data()
    train = data.loc[data["data_split"].eq("train")]
    validation = data.loc[data["data_split"].eq("validation")]
    x_train, y_train = expand_binomial_rows(train)

    assert len(x_train) == train["expiring_contracts"].sum()
    assert y_train.sum() == train["renewed_contracts"].sum()

    model = build_rate_pipeline(c_value=0.01)
    assert model.named_steps["model"].C == 0.01
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(
        validation.loc[:, RATE_MODEL_FEATURES]
    )[:, 1]
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_rate_metrics_are_zero_for_perfect_predictions() -> None:
    data = _build_rate_data().head(10)
    metrics = calculate_rate_metrics(data, data["renewal_rate"].to_numpy())

    assert metrics["weighted_mae"] == 0.0
    assert metrics["weighted_rmse"] == 0.0
    assert metrics["weighted_bias"] == 0.0


def test_final_test_and_business_outputs_are_valid() -> None:
    data = _build_rate_data()
    test = data.loc[data["data_split"].eq("test")]

    _, metrics, test_predictions = evaluate_final_test(data)
    business = build_business_predictions(test_predictions)

    assert len(test_predictions) == len(test) == 48
    assert set(metrics) == {
        "historical_rate_baseline",
        "binomial_logistic_regression",
    }
    assert business["selected_model"].eq(SELECTED_MODEL).all()
    assert business["predicted_renewal_rate"].between(0, 1).all()
    assert set(business["risk_level"]).issubset({"low", "medium", "high"})
    expected_total = (
        business["expected_renewed_contracts"]
        + business["expected_not_renewed_contracts"]
    )
    assert np.allclose(expected_total, business["expiring_contracts"])
