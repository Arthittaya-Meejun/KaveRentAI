# KaveRentAI Tasks

ไฟล์นี้ใช้ดูว่าใครต้องทำอะไร และงานแต่ละส่วนเสร็จถึงไหนแล้ว

## ความหมายของสถานะ

- `Todo` — ยังไม่เริ่ม
- `Doing` — กำลังทำ
- `Review` — ทำเสร็จแล้ว รออีกคนตรวจ
- `Done` — ตรวจแล้วและใช้งานได้

## ทอฝัน: Data และ Classification

| งาน | ไฟล์หลัก | สถานะ |
|---|---|---|
| ตรวจคุณภาพข้อมูลทั้ง 5 ตาราง | `notebooks/01_common_data_quality.ipynb` | Todo |
| ตรวจโค้ดโหลดข้อมูลที่เตรียมไว้ | `src/kaverentai/data/load_data.py` | Review |
| ตรวจโค้ดตรวจสอบข้อมูลที่เตรียมไว้ | `src/kaverentai/data/validate_data.py` | Review |
| EDA การต่อสัญญา | `notebooks/02_renewal_eda.ipynb` | Todo |
| สร้าง Feature การต่อสัญญา | `src/kaverentai/features/renewal_features.py` | Todo |
| สร้างโมเดลการต่อสัญญา | `src/kaverentai/models/train_renewal.py` | Todo |
| EDA โอกาสปล่อยเช่า | `notebooks/03_lease_probability_eda.ipynb` | Todo |
| สร้าง Feature โอกาสปล่อยเช่า | `src/kaverentai/features/lease_probability_features.py` | Todo |
| สร้างโมเดลโอกาสปล่อยเช่า | `src/kaverentai/models/train_lease_probability.py` | Todo |
| สร้างตัวประเมินโมเดล Classification | `src/kaverentai/models/evaluate_classification.py` | Todo |

## อาทิตยา: Regression และ Decision System

| งาน | ไฟล์หลัก | สถานะ |
|---|---|---|
| ตรวจโค้ดแบ่ง Train/Validation/Test ที่เตรียมไว้ | `src/kaverentai/data/split_data.py` | Review |
| EDA ราคาค่าเช่า | `notebooks/04_rental_price_eda.ipynb` | Todo |
| สร้าง Feature ราคาค่าเช่า | `src/kaverentai/features/rental_price_features.py` | Todo |
| สร้างโมเดลราคาค่าเช่า | `src/kaverentai/models/train_rental_price.py` | Todo |
| EDA ระยะเวลาปล่อยเช่า | `notebooks/05_weeks_on_market_eda.ipynb` | Todo |
| สร้าง Feature ระยะเวลาปล่อยเช่า | `src/kaverentai/features/weeks_on_market_features.py` | Todo |
| สร้างโมเดลระยะเวลาปล่อยเช่า | `src/kaverentai/models/train_weeks_on_market.py` | Todo |
| สร้างตัวประเมินโมเดล Regression | `src/kaverentai/models/evaluate_regression.py` | Todo |
| สร้างระบบแนะนำราคา | `src/kaverentai/recommendation/price_engine.py` | Todo |
| สร้างระบบโปรโมชั่นด้านราคา | `src/kaverentai/recommendation/promotion_engine.py` | Todo |
| สร้าง What-if Simulation | `src/kaverentai/simulation/what_if.py` | Todo |
| สร้างตัวคำนวณรายได้ประมาณการ | `src/kaverentai/simulation/revenue_calculator.py` | Todo |
| สร้าง Web Application | `app/` | Todo |
| รวมผลลัพธ์สำหรับรายงาน | `notebooks/06_final_results.ipynb` | Todo |

## ไฟล์ที่ต้องตกลงกันก่อนแก้

ไฟล์ต่อไปนี้เป็นไฟล์ส่วนกลาง หากต้องแก้ให้แจ้งอีกคนก่อน

- `README.md`
- `TASKS.md`
- `requirements.txt`
- `configs/model_config.yaml`
- `docs/project_scope.md`

## วิธีเริ่มงานแต่ละครั้ง

1. เปิด `TASKS.md`
2. หางานที่มีชื่อตัวเอง
3. เปลี่ยนสถานะของงานเป็น `Doing`
4. สร้างหรือแก้เฉพาะไฟล์ที่ระบุ
5. ทำเสร็จแล้วเปลี่ยนสถานะเป็น `Review`
6. ให้อีกคนตรวจ เมื่อผ่านแล้วจึงเปลี่ยนเป็น `Done`

## กติกาง่าย ๆ

- ไม่แก้ Notebook ไฟล์เดียวกันพร้อมกัน
- ดึงงานล่าสุดด้วย `git pull` ก่อนเริ่มทำ
- หนึ่งงานใช้หนึ่ง branch
- ทำงานเสร็จให้เปิด Pull Request เพื่อให้อีกคนตรวจ
- ห้ามแก้ CSV ใน `data/raw/` ด้วยมือ
