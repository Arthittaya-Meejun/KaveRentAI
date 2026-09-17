"""ฟังก์ชันทำนายระยะเวลาในการปล่อยเช่าสำหรับ Web Application."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from kaverentai.features.time_to_lease_features import (
    FEATURE_COLUMNS,
    prepare_time_to_lease_input,
)
from kaverentai.paths import MODEL_DIR


MODEL_PATH = MODEL_DIR / "time_to_lease_model.joblib"
MARKET_FEATURES = [
    "market_occupancy",
    "market_demand_pressure",
    "market_median_asking_rent",
]


def prepare_prediction_input(room_data: dict) -> pd.DataFrame:
    """สร้าง Feature ของห้องหนึ่งห้องให้เหมือนขั้นฝึกโมเดล."""

    prepared = room_data.copy()
    if "date_listed" not in prepared:
        raise KeyError("กรุณาระบุ date_listed")

    listing_date = pd.to_datetime(prepared["date_listed"], errors="coerce")
    if pd.isna(listing_date):
        raise ValueError("date_listed ไม่ใช่วันที่ที่ถูกต้อง")
    prepared["listing_year"] = listing_date.year
    prepared["listing_month"] = listing_date.month

    if "market_occupancy" not in prepared and "occupancy" in prepared:
        prepared["market_occupancy"] = prepared["occupancy"]
    if (
        "market_median_asking_rent" not in prepared
        and "median_asking_rent" in prepared
    ):
        prepared["market_median_asking_rent"] = prepared[
            "median_asking_rent"
        ]

    if "market_demand_pressure" not in prepared:
        searchers = prepared.get("n_searchers")
        open_listings = prepared.get("n_listings_open")
        prepared["market_demand_pressure"] = (
            searchers / open_listings
            if searchers is not None and open_listings not in (None, 0)
            else np.nan
        )

    for column in MARKET_FEATURES:
        prepared.setdefault(column, np.nan)

    median_rent = prepared["market_median_asking_rent"]
    prepared["rent_to_market_ratio"] = (
        prepared.get("first_asking_rent", np.nan) / median_rent
        if pd.notna(median_rent) and median_rent != 0
        else np.nan
    )
    prepared["missing_market_history"] = any(
        pd.isna(prepared[column]) for column in MARKET_FEATURES
    )

    input_data = pd.DataFrame([prepared])
    return prepare_time_to_lease_input(input_data)


def predict_time_to_lease(
    room_data: dict,
    model=None,
    model_path: str | Path = MODEL_PATH,
) -> dict:
    """ทำนายจำนวนสัปดาห์และวันโดยไม่คืนค่าติดลบ."""

    if model is None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ไม่พบโมเดล Time-to-Lease: {path}")
        model = joblib.load(path)

    input_data = prepare_prediction_input(room_data)
    raw_prediction = float(model.predict(input_data)[0])
    predicted_weeks = max(0.0, raw_prediction)

    return {
        "predicted_weeks": predicted_weeks,
        "estimated_weeks": max(0, round(predicted_weeks)),
        "estimated_days": max(0, round(predicted_weeks * 7)),
    }
