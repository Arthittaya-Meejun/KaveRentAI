"""เครื่องมือสร้างโมเดลพยากรณ์ราคาค่าเช่า.

แยกฟังก์ชันสร้าง Pipeline และ Forward Selection ออกจากไฟล์ฝึกโมเดล
เพื่อให้โค้ดใน ``train_rental_price.py`` อ่านเป็นลำดับขั้นตอนได้ง่าย
"""

from collections.abc import Sequence
from statistics import fmean, pstdev

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from kaverentai.features.rental_price_features import CATEGORICAL_COLUMNS


RANDOM_SEED = 42


def build_preprocessor(
    feature_columns: Sequence[str],
    drop_first_category: bool,
) -> ColumnTransformer:
    """สร้างขั้นตอนเตรียมข้อมูลตัวเลขและข้อมูลประเภทกลุ่ม.

    Linear Regression ใช้ ``drop='first'`` เพื่อลดปัญหาตัวแปร Dummy
    ซ้ำซ้อน ส่วน Gradient Boosting สามารถใช้ทุกกลุ่มได้
    """

    categorical_features = [
        column
        for column in feature_columns
        if column in CATEGORICAL_COLUMNS
    ]
    numeric_features = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_COLUMNS
    ]

    transformers = []

    if categorical_features:
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
        transformers.append(
            ("category", category_pipeline, categorical_features)
        )

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("number", numeric_pipeline, numeric_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def create_stepwise_linear_pipeline(
    selected_features: Sequence[str],
) -> Pipeline:
    """สร้าง Pipeline ของ Multiple Linear Regression."""

    if not selected_features:
        raise ValueError("Stepwise MLR ต้องมี Feature อย่างน้อย 1 คอลัมน์")

    preprocessor = build_preprocessor(
        selected_features,
        drop_first_category=True,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def create_gradient_boosting_pipeline(
    feature_columns: Sequence[str],
    model_params: dict | None = None,
) -> Pipeline:
    """สร้าง Pipeline ของ Gradient Boosting Regressor."""

    params = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "subsample": 1.0,
        "random_state": RANDOM_SEED,
    }
    if model_params is not None:
        params.update(model_params)

    preprocessor = build_preprocessor(
        feature_columns,
        drop_first_category=False,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", GradientBoostingRegressor(**params)),
        ]
    )


def forward_select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    candidate_features: Sequence[str],
    minimum_improvement: float = 1.0,
    verbose: bool = True,
) -> tuple[list[str], list[dict]]:
    """เลือก Feature แบบ Forward Selection ด้วย Validation MAE.

    ในแต่ละรอบจะทดลองเพิ่ม Feature ที่ยังไม่ได้เลือกทีละคอลัมน์ แล้วเก็บ
    Feature ที่ทำให้ Validation MAE ต่ำที่สุด การเลือกหยุดเมื่อ MAE ดีขึ้น
    น้อยกว่า ``minimum_improvement`` บาท
    """

    validation_fold = {
        "name": "validation",
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
    }

    return forward_select_features_rolling(
        rolling_folds=[validation_fold],
        candidate_features=candidate_features,
        minimum_improvement=minimum_improvement,
        verbose=verbose,
    )


def forward_select_features_rolling(
    rolling_folds: Sequence[dict],
    candidate_features: Sequence[str],
    minimum_improvement: float = 1.0,
    verbose: bool = True,
) -> tuple[list[str], list[dict]]:
    """เลือก Feature จาก MAE เฉลี่ยของ Rolling Validation หลายช่วงเวลา.

    แต่ละ Fold ต้องมี ``X_train``, ``y_train``, ``X_validation`` และ
    ``y_validation`` การใช้หลายช่วงเวลาช่วยลดโอกาสเลือก Feature ที่เหมาะ
    กับ Validation เพียงช่วงเดียวโดยบังเอิญ
    """

    if not rolling_folds:
        raise ValueError("ต้องมี Rolling Validation อย่างน้อย 1 Fold")

    prepared_folds = []

    # เตรียม One-hot Encoding เพียงครั้งเดียวต่อ Fold เพื่อให้การทดลอง
    # เพิ่ม Feature หลายรอบไม่ต้องแปลงข้อมูลซ้ำโดยไม่จำเป็น
    for fold_number, fold in enumerate(rolling_folds, start=1):
        selection_preprocessor = build_preprocessor(
            candidate_features,
            drop_first_category=True,
        )
        transformed_train = selection_preprocessor.fit_transform(
            fold["X_train"]
        )
        transformed_validation = selection_preprocessor.transform(
            fold["X_validation"]
        )
        feature_indices = _get_transformed_feature_indices(
            selection_preprocessor,
            candidate_features,
        )

        prepared_folds.append(
            {
                "name": fold.get("name", f"fold_{fold_number}"),
                "X_train": transformed_train,
                "y_train": fold["y_train"],
                "X_validation": transformed_validation,
                "y_validation": fold["y_validation"],
                "feature_indices": feature_indices,
            }
        )

    selected_features: list[str] = []
    remaining_features = list(candidate_features)
    selection_history: list[dict] = []
    best_mae = float("inf")

    while remaining_features:
        candidate_scores = []

        for candidate in remaining_features:
            trial_features = [*selected_features, candidate]
            fold_mae_values = []

            for fold in prepared_folds:
                # Feature ประเภทข้อความหนึ่งคอลัมน์อาจกลายเป็น Dummy
                # หลายคอลัมน์ จึงต้องเพิ่มตำแหน่งทั้งหมดพร้อมกัน
                trial_indices = []
                for feature in trial_features:
                    trial_indices.extend(fold["feature_indices"][feature])

                trial_model = LinearRegression()
                trial_model.fit(
                    fold["X_train"][:, trial_indices],
                    fold["y_train"],
                )

                predictions = trial_model.predict(
                    fold["X_validation"][:, trial_indices]
                )
                fold_mae_values.append(
                    float(
                        mean_absolute_error(
                            fold["y_validation"],
                            predictions,
                        )
                    )
                )

            candidate_scores.append(
                {
                    "candidate": candidate,
                    "fold_mae_values": fold_mae_values,
                    "mean_mae": fmean(fold_mae_values),
                }
            )

        best_candidate_result = min(
            candidate_scores,
            key=lambda item: item["mean_mae"],
        )
        best_candidate = best_candidate_result["candidate"]
        candidate_mae = best_candidate_result["mean_mae"]
        fold_mae_values = best_candidate_result["fold_mae_values"]

        improvement = best_mae - candidate_mae

        # รอบแรกต้องเลือกอย่างน้อยหนึ่ง Feature ส่วนรอบถัดไปจะเลือกต่อ
        # เฉพาะเมื่อ Validation MAE ดีขึ้นตามเกณฑ์ที่กำหนด
        if selected_features and improvement < minimum_improvement:
            if verbose:
                print(
                    "หยุด Forward Selection เพราะ MAE ดีขึ้นเพียง",
                    round(improvement, 2),
                    "บาท",
                )
            break

        selected_features.append(best_candidate)
        remaining_features.remove(best_candidate)
        best_mae = candidate_mae

        history_row = {
            "step": len(selected_features),
            "added_feature": best_candidate,
            "mean_rolling_mae": float(candidate_mae),
            "std_rolling_mae": float(pstdev(fold_mae_values)),
            "mae_improvement": (
                None if len(selected_features) == 1 else float(improvement)
            ),
        }
        for fold_number, fold_mae in enumerate(fold_mae_values, start=1):
            history_row[f"fold_{fold_number}_mae"] = fold_mae
        selection_history.append(history_row)

        if verbose:
            print(
                f"รอบที่ {len(selected_features)}:",
                f"เพิ่ม {best_candidate},",
                f"Rolling MAE เฉลี่ย = {candidate_mae:.2f}",
            )

    return selected_features, selection_history


def _get_transformed_feature_indices(
    preprocessor: ColumnTransformer,
    feature_columns: Sequence[str],
) -> dict[str, list[int]]:
    """จับคู่ Feature เดิมกับตำแหน่งคอลัมน์หลัง One-hot Encoding.

    ตัวอย่างเช่น ``room_type`` หนึ่งคอลัมน์อาจกลายเป็น Dummy Variables
    หลายคอลัมน์ ฟังก์ชันนี้ทำให้ Forward Selection เพิ่มทั้งกลุ่มพร้อมกัน
    จึงยังอธิบายผลเป็นชื่อ Feature เดิมได้ง่าย
    """

    feature_indices: dict[str, list[int]] = {}
    current_index = 0

    categorical_features = [
        column
        for column in feature_columns
        if column in CATEGORICAL_COLUMNS
    ]
    numeric_features = [
        column
        for column in feature_columns
        if column not in CATEGORICAL_COLUMNS
    ]

    if categorical_features:
        category_pipeline = preprocessor.named_transformers_["category"]
        encoder = category_pipeline.named_steps["one_hot"]

        for feature, categories, dropped_index in zip(
            categorical_features,
            encoder.categories_,
            encoder.drop_idx_,
        ):
            number_of_columns = len(categories)
            if dropped_index is not None:
                number_of_columns -= 1

            feature_indices[feature] = list(
                range(current_index, current_index + number_of_columns)
            )
            current_index += number_of_columns

    for feature in numeric_features:
        feature_indices[feature] = [current_index]
        current_index += 1

    return feature_indices
