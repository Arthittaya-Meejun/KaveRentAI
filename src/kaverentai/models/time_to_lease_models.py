"""Pipeline ของโมเดล Time-to-Lease ที่ใช้ในโครงงาน.

เปรียบเทียบโมเดลที่นักศึกษา Data Science ใช้อธิบายได้ง่าย 2 แบบ:

1. Multiple Linear Regression เป็นโมเดลพื้นฐาน
2. Random Forest Regressor เป็นโมเดลหลักสำหรับความสัมพันธ์ไม่เป็นเส้นตรง
"""

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from kaverentai.config import RANDOM_SEED
from kaverentai.features.time_to_lease_features import CATEGORICAL_COLUMNS


DEFAULT_RANDOM_FOREST_PARAMS = {
    "n_estimators": 150,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "max_features": 0.8,
    "random_state": RANDOM_SEED,
    # ใช้หนึ่ง Worker เพื่อให้รันได้เหมือนกันทั้ง Windows, Notebook และ CI
    "n_jobs": 1,
}


def build_time_to_lease_preprocessor(
    feature_columns: Sequence[str],
    drop_first_category: bool,
) -> ColumnTransformer:
    """สร้างขั้นตอนเติม Missing และ One-hot Encoding."""

    categorical_features = [
        column for column in feature_columns if column in CATEGORICAL_COLUMNS
    ]
    numeric_features = [
        column for column in feature_columns if column not in CATEGORICAL_COLUMNS
    ]

    category_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first" if drop_first_category else None,
                    sparse_output=False,
                ),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    return ColumnTransformer(
        transformers=[
            ("category", category_pipeline, categorical_features),
            ("number", numeric_pipeline, numeric_features),
        ],
        remainder="drop",
    )


def create_linear_regression_pipeline(
    feature_columns: Sequence[str],
) -> Pipeline:
    """สร้าง Pipeline ของ Multiple Linear Regression."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_time_to_lease_preprocessor(
                    feature_columns,
                    drop_first_category=True,
                ),
            ),
            ("model", LinearRegression()),
        ]
    )


def create_random_forest_pipeline(
    feature_columns: Sequence[str],
    model_params: dict | None = None,
) -> Pipeline:
    """สร้าง Pipeline ของ Random Forest Regressor."""

    params = DEFAULT_RANDOM_FOREST_PARAMS.copy()
    if model_params:
        params.update(model_params)

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_time_to_lease_preprocessor(
                    feature_columns,
                    drop_first_category=False,
                ),
            ),
            ("model", RandomForestRegressor(**params)),
        ]
    )
