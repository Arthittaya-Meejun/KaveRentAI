"""Train and compare models for four-week lease probability.

The code intentionally keeps the workflow explicit:
1. Fit models on Train.
2. Compare them on Validation using log loss.
3. Lock the selected model in ``configs/model_config.yaml``.
4. Refit that model on Train + Validation and evaluate Test once.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from kaverentai.config import (
    LEASE_CATBOOST_DEPTH,
    LEASE_CATBOOST_ITERATIONS,
    LEASE_CATBOOST_LEARNING_RATE,
    LEASE_PROBABILITY_SELECTED_MODEL,
    RANDOM_SEED,
)
from kaverentai.features.lease_probability_features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)
from kaverentai.models.evaluate_classification import (
    build_prediction_table,
    build_threshold_analysis,
    calculate_classification_metrics,
)
from kaverentai.paths import MODELING_DATA_DIR, MODEL_DIR, REPORT_DIR


LOGISTIC_MODEL = "logistic_regression"
CATBOOST_MODEL = "catboost_classifier"
SUPPORTED_MODELS = (LOGISTIC_MODEL, CATBOOST_MODEL)
SELECTED_MODEL = LEASE_PROBABILITY_SELECTED_MODEL


def load_lease_probability_data(
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the reproducible listing-level modeling table."""
    data_path = Path(path) if path is not None else (
        MODELING_DATA_DIR / "lease_probability.csv"
    )
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Lease-probability data was not found: {data_path}. "
            "Run `python -m kaverentai.features.lease_probability_features` "
            "first."
        )
    return pd.read_csv(data_path, parse_dates=["date_listed"])


def _feature_groups(
    model_features: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Separate a requested feature set into numeric and categorical names."""
    unknown = sorted(set(model_features) - set(MODEL_FEATURES))
    if unknown:
        raise ValueError(f"Unknown lease-probability features: {unknown}")
    numeric = tuple(
        feature for feature in NUMERIC_FEATURES if feature in model_features
    )
    categorical = tuple(
        feature for feature in CATEGORICAL_FEATURES if feature in model_features
    )
    return numeric, categorical


def _logistic_preprocessor(
    model_features: tuple[str, ...] = MODEL_FEATURES,
) -> ColumnTransformer:
    """Prepare numeric and categorical columns for Logistic Regression."""
    numeric_features, categorical_features = _feature_groups(model_features)
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("fill_missing", SimpleImputer(strategy="median")),
                        ("standardize", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "fill_missing",
                            SimpleImputer(strategy="most_frequent"),
                        ),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )


def build_logistic_pipeline(
    model_features: tuple[str, ...] = MODEL_FEATURES,
) -> Pipeline:
    """Build the interpretable baseline model."""
    return Pipeline(
        steps=[
            ("prepare_data", _logistic_preprocessor(model_features)),
            (
                "model",
                LogisticRegression(max_iter=1_000, random_state=RANDOM_SEED),
            ),
        ]
    )


def _tree_preprocessor(
    model_features: tuple[str, ...] = MODEL_FEATURES,
) -> ColumnTransformer:
    """Fill missing values while keeping category names for CatBoost."""
    numeric_features, categorical_features = _feature_groups(model_features)
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_features,
            ),
            (
                "categorical",
                SimpleImputer(
                    strategy="constant",
                    fill_value="missing",
                ),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )
    return preprocessor.set_output(transform="pandas")


def build_catboost_pipeline(
    model_features: tuple[str, ...] = MODEL_FEATURES,
) -> Pipeline:
    """Build the non-linear comparison model."""
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise ImportError(
            "CatBoost is required. Install project dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from error

    _, categorical_features = _feature_groups(model_features)
    model = CatBoostClassifier(
        iterations=LEASE_CATBOOST_ITERATIONS,
        depth=LEASE_CATBOOST_DEPTH,
        learning_rate=LEASE_CATBOOST_LEARNING_RATE,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_SEED,
        cat_features=list(categorical_features),
        verbose=False,
        allow_writing_files=False,
    )
    return Pipeline(
        steps=[
            ("prepare_data", _tree_preprocessor(model_features)),
            ("model", model),
        ]
    )


def build_model(
    model_name: str,
    model_features: tuple[str, ...] = MODEL_FEATURES,
) -> Pipeline:
    """Create a model from its stable configuration name."""
    if model_name == LOGISTIC_MODEL:
        return build_logistic_pipeline(model_features)
    if model_name == CATBOOST_MODEL:
        return build_catboost_pipeline(model_features)
    raise ValueError(f"Unknown lease-probability model: {model_name}")


def train_and_evaluate_validation(
    data: pd.DataFrame,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, float | int]], pd.DataFrame]:
    """Fit both candidates on Train and compare them on Validation."""
    train = data.loc[data["data_split"].eq("train")].copy()
    validation = data.loc[data["data_split"].eq("validation")].copy()
    if train.empty or validation.empty:
        raise ValueError("Both train and validation rows are required")

    x_train = train.loc[:, MODEL_FEATURES]
    y_train = train[TARGET_COLUMN]
    x_validation = validation.loc[:, MODEL_FEATURES]
    y_validation = validation[TARGET_COLUMN]

    fitted_models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, float | int]] = {}
    predictions = validation[
        ["listing_id", "project_id", "date_listed", TARGET_COLUMN]
    ].copy()

    for model_name in SUPPORTED_MODELS:
        model = build_model(model_name)
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_validation)[:, 1]
        fitted_models[model_name] = model
        metrics[model_name] = calculate_classification_metrics(
            y_validation, probabilities
        )
        predictions[f"{model_name}_probability"] = probabilities

    return fitted_models, metrics, predictions


def evaluate_locked_test(
    data: pd.DataFrame,
    *,
    selected_model: str = SELECTED_MODEL,
) -> tuple[Pipeline, dict[str, float | int], pd.DataFrame]:
    """Refit the locked model on Train + Validation and evaluate Test."""
    if selected_model not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown selected model: {selected_model}")

    development = data.loc[
        data["data_split"].isin(["train", "validation"])
    ].copy()
    test = data.loc[data["data_split"].eq("test")].copy()
    if development.empty or test.empty:
        raise ValueError("Train+Validation and Test rows are required")

    model = build_model(selected_model)
    model.fit(
        development.loc[:, MODEL_FEATURES],
        development[TARGET_COLUMN],
    )
    probabilities = model.predict_proba(test.loc[:, MODEL_FEATURES])[:, 1]
    metrics = calculate_classification_metrics(
        test[TARGET_COLUMN], probabilities
    )
    predictions = build_prediction_table(
        test,
        probabilities,
        target_column=TARGET_COLUMN,
    )
    predictions["selected_model"] = selected_model
    return model, metrics, predictions


def build_feature_importance_table(model: Pipeline) -> pd.DataFrame:
    """Return readable feature importance from a fitted model pipeline."""
    preprocessor = model.named_steps["prepare_data"]
    estimator = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importance = abs(estimator.coef_[0])
    else:
        raise ValueError("The selected model does not expose feature importance")

    return (
        pd.DataFrame(
            {"feature": feature_names, "importance": importance}
        )
        .sort_values("importance", ascending=False, kind="stable")
        .reset_index(drop=True)
    )


def save_outputs(
    final_model: Pipeline,
    metrics: dict[str, object],
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    threshold_analysis: pd.DataFrame,
) -> dict[str, Path]:
    """Save the final model and report-ready outputs."""
    model_path = MODEL_DIR / "lease_probability" / "final_model.joblib"
    metrics_path = REPORT_DIR / "metrics" / "lease_probability_metrics.json"
    validation_path = (
        REPORT_DIR / "tables" / "lease_probability_validation_predictions.csv"
    )
    test_path = (
        REPORT_DIR / "tables" / "lease_probability_test_predictions.csv"
    )
    importance_path = (
        REPORT_DIR / "tables" / "lease_probability_feature_importance.csv"
    )
    threshold_path = (
        REPORT_DIR / "tables" / "lease_probability_threshold_analysis.csv"
    )
    for path in (
        model_path,
        metrics_path,
        validation_path,
        test_path,
        importance_path,
        threshold_path,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    validation_predictions.to_csv(validation_path, index=False)
    test_predictions.to_csv(test_path, index=False)
    feature_importance.to_csv(importance_path, index=False)
    threshold_analysis.to_csv(threshold_path, index=False)
    return {
        "final_model": model_path.resolve(),
        "metrics": metrics_path.resolve(),
        "validation_predictions": validation_path.resolve(),
        "test_predictions": test_path.resolve(),
        "feature_importance": importance_path.resolve(),
        "threshold_analysis": threshold_path.resolve(),
    }


def main() -> None:
    """Run validation comparison and the locked final test."""
    data = load_lease_probability_data()
    _, validation_metrics, validation_predictions = (
        train_and_evaluate_validation(data)
    )
    final_model, test_metrics, test_predictions = evaluate_locked_test(data)
    feature_importance = build_feature_importance_table(final_model)
    selected_probability_column = f"{SELECTED_MODEL}_probability"
    threshold_analysis = build_threshold_analysis(
        validation_predictions[TARGET_COLUMN],
        validation_predictions[selected_probability_column],
    )
    metrics = {
        "model_settings": {
            "target": TARGET_COLUMN,
            "selection_metric": "validation_log_loss",
            "selected_model": SELECTED_MODEL,
            "test_used_for_selection": False,
        },
        "validation": validation_metrics,
        "test": {SELECTED_MODEL: test_metrics},
    }
    output_paths = save_outputs(
        final_model,
        metrics,
        validation_predictions,
        test_predictions,
        feature_importance,
        threshold_analysis,
    )

    print("Lease probability within four weeks")
    for model_name, model_metrics in validation_metrics.items():
        print(
            f"Validation {model_name:24s} "
            f"AUC={model_metrics['roc_auc']:.3f} "
            f"LogLoss={model_metrics['log_loss']:.3f} "
            f"F1={model_metrics['f1']:.3f}"
        )
    print(f"Locked model: {SELECTED_MODEL}")
    print(
        f"Test AUC={test_metrics['roc_auc']:.3f} "
        f"LogLoss={test_metrics['log_loss']:.3f} "
        f"F1={test_metrics['f1']:.3f}"
    )
    for name, path in output_paths.items():
        print(f"Saved {name}: {path}")


if __name__ == "__main__":
    main()
