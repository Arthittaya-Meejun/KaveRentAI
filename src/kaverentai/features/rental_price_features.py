"""สร้าง Feature สำหรับโมเดลพยากรณ์ราคาค่าเช่า.

ไฟล์นี้เก็บรายชื่อ Feature ไว้ที่เดียว เพื่อให้ขั้นฝึกโมเดลและขั้นนำ
โมเดลไปใช้งานเลือกคอลัมน์ตรงกันเสมอ
"""

import pandas as pd

from kaverentai.data.load_data import load_all_data


DATE_COLUMN = "date_start"
TARGET_COLUMN = "rent"

FEATURE_COLUMNS = [
    "floor",
    "size_sqm",
    "room_type",
    "view",
    "furnished",
    "lease_months",
    "is_renewal",
    "project_name",
    "university",
    "distance_to_campus_m",
    "facility_count",
    "start_year",
    "start_month",
]

CATEGORICAL_COLUMNS = [
    "room_type",
    "view",
    "furnished",
    "is_renewal",
    "project_name",
    "university",
    "start_month",
]


def build_rental_price_features() -> pd.DataFrame:
    """รวมข้อมูลและสร้าง Feature สำหรับทำนายราคาค่าเช่า."""

    # 1. โหลดข้อมูลที่ผ่านการทำความสะอาดแล้ว
    data = load_all_data(cleaned=True)

    leases = data["leases"]
    units = data["units"]
    projects = data["projects"]

    # 2. เลือกข้อมูลห้องที่ต้องการใช้
    unit_columns = [
        "unit_id",
        "floor",
        "size_sqm",
        "room_type",
        "view",
        "furnished",
    ]

    # 3. รวมข้อมูลสัญญาเช่ากับข้อมูลห้อง
    rental_data = leases.merge(
        units[unit_columns],
        on="unit_id",
        how="left",
    )

    # 4. เลือกข้อมูลโครงการที่ต้องการใช้
    project_columns = [
        "project_id",
        "project_name",
        "university",
        "distance_to_campus_m",
        "facility_count",
    ]

    # 5. รวมข้อมูลโครงการ
    rental_data = rental_data.merge(
        projects[project_columns],
        on="project_id",
        how="left",
    )

    # 6. แปลงวันที่ให้อยู่ในรูปแบบ datetime
    rental_data["date_start"] = pd.to_datetime(rental_data["date_start"])

    # 7. สร้าง Feature ปีและเดือนจากวันที่เริ่มสัญญา
    rental_data["start_year"] = rental_data["date_start"].dt.year
    rental_data["start_month"] = rental_data["date_start"].dt.month

    # 8. เลือกคอลัมน์สำหรับนำไปสร้างโมเดล
    selected_columns = [DATE_COLUMN, *FEATURE_COLUMNS, TARGET_COLUMN]

    modeling_data = rental_data[selected_columns].copy()

    return modeling_data


def prepare_rental_price_input(data: pd.DataFrame) -> pd.DataFrame:
    """เลือกและจัดชนิดข้อมูล Feature ให้พร้อมส่งเข้า Pipeline.

    Parameters
    ----------
    data:
        ตารางที่มี Feature สำหรับพยากรณ์ราคา อาจมีคอลัมน์อื่นรวมอยู่ด้วย

    Returns
    -------
    pandas.DataFrame
        ตารางที่มีเฉพาะ Feature และเรียงคอลัมน์ตรงกับตอนฝึกโมเดล
    """

    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in data.columns
    ]
    if missing_columns:
        raise KeyError(f"ข้อมูล Feature ไม่ครบ: {missing_columns}")

    prepared_data = data[FEATURE_COLUMNS].copy()

    # แปลงคอลัมน์ประเภทกลุ่มเป็นข้อความ เพื่อให้ค่าระหว่าง Train และ Predict
    # มีรูปแบบตรงกัน เช่น True จะเป็น "True" ทั้งสองขั้นตอน
    for column in CATEGORICAL_COLUMNS:
        prepared_data[column] = prepared_data[column].astype(str)

    return prepared_data


def split_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """แยก Feature (X) และราคาค่าเช่าเป้าหมาย (y)."""

    if TARGET_COLUMN not in data.columns:
        raise KeyError(f"ไม่พบ Target: {TARGET_COLUMN}")

    features = prepare_rental_price_input(data)
    target = pd.to_numeric(data[TARGET_COLUMN], errors="raise")

    return features, target


def main():
    """สร้าง Feature และแสดงผลการตรวจสอบเบื้องต้น."""

    modeling_data = build_rental_price_features()

    print("จำนวนแถวและคอลัมน์:", modeling_data.shape)
    print("จำนวน Missing Value:", modeling_data.isnull().sum().sum())
    print()
    print(modeling_data.head())


if __name__ == "__main__":
    main()
