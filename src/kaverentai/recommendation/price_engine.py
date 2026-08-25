"""ระบบแนะนำราคาค่าเช่าจากโมเดลที่ฝึกไว้."""

from pathlib import Path

import joblib
import pandas as pd

from kaverentai.features.rental_price_features import (
    prepare_rental_price_input,
)
from kaverentai.paths import MODEL_DIR


MODEL_PATH = MODEL_DIR / "rental_price_model.joblib"

def recommend_rental_price(room_data, model_path=MODEL_PATH):
    """ทำนายและแนะนำราคาค่าเช่าจากรายละเอียดห้องหนึ่งห้อง."""

    # 1. ตรวจสอบว่ามีไฟล์โมเดลอยู่จริง
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"ไม่พบโมเดลราคาเช่า: {model_path}"
        )

    # 2. คัดลอกข้อมูลเพื่อไม่แก้ไข Dictionary ต้นฉบับ
    prepared_data = room_data.copy()

    # 3. ตรวจสอบและแปลงวันที่เริ่มสัญญา
    if "date_start" not in prepared_data:
        raise KeyError("กรุณาระบุ date_start")

    start_date = pd.to_datetime(
        prepared_data["date_start"],
        errors="coerce",
    )
    if pd.isna(start_date):
        raise ValueError("date_start ไม่ใช่วันที่ที่ถูกต้อง")

    # 4. สร้าง Feature ปีและเดือนให้เหมือนตอนฝึกโมเดล
    prepared_data["start_year"] = start_date.year
    prepared_data["start_month"] = start_date.month

    # 5. แปลงข้อมูลหนึ่งห้องให้เป็นตารางหนึ่งแถว
    input_data = pd.DataFrame([prepared_data])
    input_data = prepare_rental_price_input(input_data)

    # 6. โหลด Pipeline ซึ่งรวมทั้งขั้นเตรียมข้อมูลและโมเดลไว้แล้ว
    model = joblib.load(model_path)
    predicted_rent = float(model.predict(input_data)[0])

    # ราคาค่าเช่าไม่ควรติดลบ และปัดเป็นจำนวนเต็มเพื่อให้อ่านง่าย
    recommended_rent = max(0, round(predicted_rent))

    return {
        "predicted_rent": predicted_rent,
        "recommended_rent": recommended_rent,
    }


def main():
    """ทดลองแนะนำราคาด้วยข้อมูลห้องตัวอย่าง."""

    sample_room = {
        "floor": 7,
        "size_sqm": 25.8,
        "room_type": "studio",
        "view": "city",
        "furnished": True,
        "lease_months": 12,
        "is_renewal": False,
        "project_name": "Kave Condo",
        "university": "ม.กรุงเทพ",
        "distance_to_campus_m": 100,
        "facility_count": 20,
        "date_start": "2026-08-20",
    }

    result = recommend_rental_price(sample_room)

    print("ราคาที่โมเดลทำนาย:", round(result["predicted_rent"], 2), "บาท")
    print("ราคาเช่าที่แนะนำ:", result["recommended_rent"], "บาท/เดือน")


if __name__ == "__main__":
    main()
