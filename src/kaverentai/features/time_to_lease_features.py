"""สร้าง Modeling Dataset สำหรับโมเดลพยากรณ์ Time-to-Lease.

Target คือ ``weeks_on_market`` หรือจำนวนสัปดาห์ตั้งแต่เริ่มลงประกาศจน
ปล่อยเช่าสำเร็จ โดยใช้เฉพาะข้อมูลที่ทราบได้ในวันเริ่มลงประกาศเท่านั้น
เพื่อป้องกัน Data Leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from kaverentai.config import (
    TIME_TO_LEASE_MIN_FOLLOW_UP_WEEKS,
    TIME_TO_LEASE_TEST_END,
    TIME_TO_LEASE_TRAIN_END,
    TIME_TO_LEASE_VALIDATION_END,
)
from kaverentai.data.load_data import load_all_data
from kaverentai.paths import MODELING_DATA_DIR


DATE_COLUMN = "date_listed"
TARGET_COLUMN = "weeks_on_market"
DEFAULT_OUTPUT_PATH = MODELING_DATA_DIR / "time_to_lease.csv"

IDENTIFIER_COLUMNS = [
    "listing_id",
    "unit_id",
    "project_id",
    DATE_COLUMN,
    "data_split",
]

FEATURE_COLUMNS = [
    "floor",
    "size_sqm",
    "room_type",
    "view",
    "furnished",
    "first_asking_rent",
    "project_name",
    "university",
    "distance_to_campus_m",
    "facility_count",
    "season_listed",
    "listing_year",
    "listing_month",
    "market_occupancy",
    "market_demand_pressure",
    "market_median_asking_rent",
    "rent_to_market_ratio",
    "missing_market_history",
]

CATEGORICAL_COLUMNS = [
    "room_type",
    "view",
    "furnished",
    "project_name",
    "university",
    "season_listed",
    "listing_month",
    "missing_market_history",
]

# คอลัมน์เหล่านี้เกิดขึ้นหลังลงประกาศ จึงห้ามเป็น Feature ของโมเดล
LEAKAGE_COLUMNS = {
    "asking_rent",
    "leased",
    "week_leased",
    "n_viewings",
    "tenant_segment",
    "lease_months",
}


def _to_boolean(series: pd.Series) -> pd.Series:
    """แปลงค่าบูลีนทั้งแบบ bool และข้อความให้เป็น Boolean dtype."""

    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean")

    normalized = series.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )
    return mapped.astype("boolean")


def _require_columns(
    data: pd.DataFrame,
    columns: list[str],
    table_name: str,
) -> None:
    """ตรวจว่าตารางมีคอลัมน์ที่จำเป็นครบก่อนเริ่มสร้าง Feature."""

    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(f"{table_name} ขาดคอลัมน์ที่จำเป็น: {missing}")


def assign_time_split(
    dates: pd.Series,
    train_end: str = TIME_TO_LEASE_TRAIN_END,
    validation_end: str = TIME_TO_LEASE_VALIDATION_END,
    test_end: str = TIME_TO_LEASE_TEST_END,
) -> pd.Series:
    """กำหนด Matured Train/Validation/Test จากวันเริ่มลงประกาศ.

    แถวหลัง ``test_end`` จะเป็น ``post_test`` และไม่นำไปประเมินโมเดล
    เพราะยังมีระยะติดตามไม่ครบตามที่กำหนด
    """

    event_dates = pd.to_datetime(dates, errors="coerce")
    if event_dates.isna().any():
        raise ValueError(f"{DATE_COLUMN} มีวันที่ที่แปลงไม่ได้")

    train_cutoff = pd.Timestamp(train_end).normalize()
    validation_cutoff = pd.Timestamp(validation_end).normalize()
    test_cutoff = pd.Timestamp(test_end).normalize()
    if not train_cutoff < validation_cutoff < test_cutoff:
        raise ValueError(
            "train_end must be earlier than validation_end and test_end"
        )

    split = np.select(
        [
            event_dates <= train_cutoff,
            event_dates <= validation_cutoff,
            event_dates <= test_cutoff,
        ],
        ["train", "validation", "test"],
        default="post_test",
    )
    return pd.Series(split, index=dates.index, dtype="string")


def build_time_to_lease_features(
    datasets: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """รวมข้อมูลและสร้าง Feature สำหรับทำนายจำนวนสัปดาห์ก่อนปล่อยเช่า.

    ใช้เฉพาะประกาศหลักที่ ``leased=True`` เพราะรายการที่ยังไม่สำเร็จยังไม่รู้
    Target ที่แท้จริง และตัดรายการจำลองซ้ำที่ลงท้ายด้วย ``-B`` ออก
    ข้อมูลตลาดจะเลื่อนย้อนหลังหนึ่งสัปดาห์ก่อน Merge เพื่อให้เป็นข้อมูลที่มีอยู่
    แล้ว ณ วันเริ่มลงประกาศ
    """

    data = load_all_data(cleaned=True) if datasets is None else datasets
    if "listings" not in data or "weekly_market" not in data:
        raise KeyError("ต้องมีตาราง listings และ weekly_market")

    listings = data["listings"].copy()
    market = data["weekly_market"].copy()

    listing_columns = [
        "listing_id",
        "unit_id",
        "project_id",
        "project_name",
        "university",
        "distance_to_campus_m",
        "facility_count",
        "floor",
        "size_sqm",
        "room_type",
        "view",
        "furnished",
        "week_listed",
        DATE_COLUMN,
        "season_listed",
        "first_asking_rent",
        TARGET_COLUMN,
        "leased",
        "week_leased",
    ]
    market_columns = [
        "week",
        "date",
        "project_id",
        "n_searchers",
        "n_listings_open",
        "occupancy",
        "median_asking_rent",
    ]
    _require_columns(listings, listing_columns, "listings")
    _require_columns(market, market_columns, "weekly_market")

    # จุดสิ้นสุดการสังเกตข้อมูล คือหนึ่งสัปดาห์หลังวันเริ่มสัปดาห์สุดท้าย
    market_dates = pd.to_datetime(market["date"], errors="coerce")
    if market_dates.isna().any():
        raise ValueError("weekly_market.date มีวันที่ที่แปลงไม่ได้")
    observation_end = market_dates.max() + pd.Timedelta(days=7)
    available_follow_up = (
        observation_end - pd.Timestamp(TIME_TO_LEASE_TEST_END)
    ).days // 7
    if available_follow_up < TIME_TO_LEASE_MIN_FOLLOW_UP_WEEKS:
        raise ValueError(
            "Test set มีระยะติดตามไม่พอ: "
            f"ต้องการ {TIME_TO_LEASE_MIN_FOLLOW_UP_WEEKS} สัปดาห์ "
            f"แต่มี {available_follow_up} สัปดาห์"
        )

    # 1. ตัด Scenario -B และเก็บเฉพาะประกาศที่ทราบ Target แล้ว
    is_primary_listing = ~listings["listing_id"].astype("string").str.endswith(
        "-B", na=False
    )
    is_leased = _to_boolean(listings["leased"]).fillna(False)
    modeling_data = listings.loc[
        is_primary_listing & is_leased, listing_columns
    ].copy()

    # 2. ตรวจและเตรียม Target/วันที่
    modeling_data[DATE_COLUMN] = pd.to_datetime(
        modeling_data[DATE_COLUMN], errors="coerce"
    )
    modeling_data[TARGET_COLUMN] = pd.to_numeric(
        modeling_data[TARGET_COLUMN], errors="coerce"
    )
    if modeling_data[DATE_COLUMN].isna().any():
        raise ValueError(f"{DATE_COLUMN} มี Missing Value หรือรูปแบบวันที่ไม่ถูกต้อง")
    if modeling_data[TARGET_COLUMN].isna().any():
        raise ValueError(f"{TARGET_COLUMN} มี Missing Value หรือไม่ใช่ตัวเลข")
    if (modeling_data[TARGET_COLUMN] < 0).any():
        raise ValueError(f"{TARGET_COLUMN} ต้องไม่ติดลบ")
    expected_duration = (
        pd.to_numeric(modeling_data["week_leased"], errors="coerce")
        - pd.to_numeric(modeling_data["week_listed"], errors="coerce")
    )
    if expected_duration.isna().any() or not expected_duration.eq(
        modeling_data[TARGET_COLUMN]
    ).all():
        raise ValueError(
            f"{TARGET_COLUMN} ไม่ตรงกับ week_leased - week_listed"
        )
    if modeling_data["listing_id"].duplicated().any():
        raise ValueError("พบ listing_id ซ้ำหลังตัด Scenario -B")

    # 3. สร้าง Feature จากวันที่ ซึ่งทราบได้ตั้งแต่วันเริ่มลงประกาศ
    modeling_data["listing_year"] = modeling_data[DATE_COLUMN].dt.year
    modeling_data["listing_month"] = modeling_data[DATE_COLUMN].dt.month

    # 4. ใช้ข้อมูลตลาดสัปดาห์ก่อนหน้าเท่านั้น ป้องกันการเห็นผลของสัปดาห์ปัจจุบัน
    if market.duplicated(["project_id", "week"]).any():
        raise ValueError("weekly_market มี project_id และ week ซ้ำ")

    market_history = market[market_columns].drop(columns="date").copy()
    market_history["week_listed"] = market_history["week"] + 1
    market_history = market_history.rename(
        columns={
            "week": "market_history_week",
            "occupancy": "market_occupancy",
            "median_asking_rent": "market_median_asking_rent",
        }
    )

    modeling_data = modeling_data.merge(
        market_history,
        on=["project_id", "week_listed"],
        how="left",
        validate="many_to_one",
    )

    modeling_data["market_demand_pressure"] = (
        modeling_data["n_searchers"]
        / modeling_data["n_listings_open"].replace(0, np.nan)
    )
    modeling_data["rent_to_market_ratio"] = (
        modeling_data["first_asking_rent"]
        / modeling_data["market_median_asking_rent"].replace(0, np.nan)
    )
    required_market_features = [
        "market_occupancy",
        "market_demand_pressure",
        "market_median_asking_rent",
    ]
    modeling_data["missing_market_history"] = modeling_data[
        required_market_features
    ].isna().any(axis=1)

    # 5. แบ่งข้อมูลตามเวลาโดยไม่สุ่มข้อมูลอนาคตย้อนกลับไปใน Train
    modeling_data["data_split"] = assign_time_split(
        modeling_data[DATE_COLUMN]
    )

    # 6. เก็บเฉพาะรหัสสำหรับตรวจสอบ, Feature และ Target
    selected_columns = [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, TARGET_COLUMN]
    result = modeling_data[selected_columns].copy()
    result = result.sort_values([DATE_COLUMN, "listing_id"]).reset_index(drop=True)

    if LEAKAGE_COLUMNS.intersection(FEATURE_COLUMNS):
        raise AssertionError("FEATURE_COLUMNS มีคอลัมน์ที่ทำให้เกิด Data Leakage")

    return result


def prepare_time_to_lease_input(data: pd.DataFrame) -> pd.DataFrame:
    """เลือกและจัดชนิด Feature ให้พร้อมส่งเข้า Pipeline."""

    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise KeyError(f"ข้อมูล Feature ไม่ครบ: {missing}")

    prepared = data[FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype("string")
    return prepared


def split_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """แยก Feature (X) และจำนวนสัปดาห์เป้าหมาย (y)."""

    if TARGET_COLUMN not in data.columns:
        raise KeyError(f"ไม่พบ Target: {TARGET_COLUMN}")
    features = prepare_time_to_lease_input(data)
    target = pd.to_numeric(data[TARGET_COLUMN], errors="raise")
    return features, target


def save_time_to_lease_features(
    data: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> Path:
    """บันทึก Modeling Dataset เป็น CSV และคืนตำแหน่งไฟล์."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    return path


def main() -> None:
    """สร้างและบันทึก Modeling Dataset พร้อมสรุปจำนวนข้อมูล."""

    modeling_data = build_time_to_lease_features()
    output_path = save_time_to_lease_features(modeling_data)

    print("บันทึกไฟล์:", output_path)
    print("จำนวนแถวและคอลัมน์:", modeling_data.shape)
    print("จำนวนข้อมูลแต่ละ Split:")
    print(modeling_data["data_split"].value_counts().to_string())
    print("จำนวนรายการที่ไม่มีข้อมูลตลาดย้อนหลัง:", end=" ")
    print(int(modeling_data["missing_market_history"].sum()))


if __name__ == "__main__":
    main()
