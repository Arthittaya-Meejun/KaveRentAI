# โมเดลพยากรณ์ราคาค่าเช่า

เอกสารนี้อธิบายส่วนงานพยากรณ์ราคาค่าเช่าของอาทิตยา โดยใช้ข้อมูลเดิม
จากตาราง `leases`, `units` และ `projects` และไม่สร้างข้อมูลจำลองเพิ่ม

## โมเดลที่เปรียบเทียบ

1. **Forward/Stepwise Multiple Linear Regression**
   - เพิ่ม Feature ทีละกลุ่ม
   - เลือก Feature ที่ทำให้ Rolling Validation MAE เฉลี่ยลดลงมากที่สุด
   - เหมาะสำหรับอธิบายว่าปัจจัยใดสัมพันธ์กับราคาค่าเช่า
2. **Gradient Boosting Regressor**
   - รวม Decision Tree ขนาดเล็กหลายต้นตามลำดับ
   - เหมาะสำหรับความสัมพันธ์ที่ไม่เป็นเส้นตรง

การเลือก Feature และการจูนพารามิเตอร์ใช้ **Rolling Time Validation 4 Fold**
ภายในชุด Train จากนั้นเลือกโมเดลผู้ชนะจาก **Validation MAE หลัก** เท่านั้น
หลังเลือกเสร็จจึงประเมิน Test set หนึ่งครั้ง

## ลำดับการทำงาน

```text
สร้าง Feature
    ↓
แบ่ง Train / Validation / Test ตามเวลา
    ↓
สร้าง Rolling Validation 4 Fold ภายใน Train
    ↓
Forward Selection และจูน Gradient Boosting จาก MAE เฉลี่ย 4 Fold
    ↓
เลือกโมเดลจาก Validation MAE
    ↓
ฝึกใหม่ด้วย Train + Validation
    ↓
ประเมิน Test และบันทึก Pipeline
```

## Feature ที่ Stepwise เลือก

- `room_type`
- `project_name`
- `start_month`
- `view`
- `is_renewal`
- `furnished`
- `floor`

Forward Selection หยุดเมื่อการเพิ่ม Feature ถัดไปช่วยลด Rolling
Validation MAE เฉลี่ยน้อยกว่า 1 บาท

## Rolling Time Validation

| Fold | Train ถึง | Validation ถึง | Train rows | Validation rows |
|---|---|---|---:|---:|
| 2025 Q1 | 31 ธ.ค. 2024 | 31 มี.ค. 2025 | 23,898 | 2,546 |
| 2025 Q2 | 31 มี.ค. 2025 | 30 มิ.ย. 2025 | 26,444 | 6,614 |
| 2025 Q3 | 30 มิ.ย. 2025 | 30 ก.ย. 2025 | 33,058 | 3,001 |
| 2025 Q4 | 30 ก.ย. 2025 | 31 ธ.ค. 2025 | 36,059 | 4,521 |

Gradient Boosting ทดลอง 12 ชุดใน 4 Fold รวม 48 การทดลอง ชุดที่ชนะมี
Rolling Validation MAE เฉลี่ย 860.59 บาท และส่วนเบี่ยงเบนมาตรฐาน
124.13 บาท

## Hyperparameters ของ Gradient Boosting ที่เลือก

```text
n_estimators = 350
learning_rate = 0.03
max_depth = 4
min_samples_leaf = 15
subsample = 0.9
random_state = 42
```

## ผลการเปรียบเทียบ

| Model | Split | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Stepwise MLR | Validation | 770.84 | 1,051.46 | 0.7186 |
| Gradient Boosting | Validation | 759.60 | 1,021.25 | 0.7345 |
| Stepwise MLR | Test | 1,113.72 | 1,395.80 | 0.5854 |
| Gradient Boosting | Test | 1,032.37 | 1,314.12 | 0.6325 |

Gradient Boosting เป็นผู้ชนะเพราะมี Validation MAE ต่ำกว่า และผล Test
ยังสอดคล้องกัน โดย Gradient Boosting มี MAE และ RMSE ต่ำกว่า Stepwise
MLR ทั้งสองชุดข้อมูล

ผล Test ที่ลดลงจาก Validation แสดงว่ารูปแบบราคาในช่วงเวลาล่าสุดเปลี่ยนไป
ควรระบุประเด็นนี้เป็นข้อจำกัดในรายงาน

## ไฟล์สำคัญ

- `src/kaverentai/features/rental_price_features.py` — สร้าง Feature
- `src/kaverentai/models/rental_price_models.py` — สร้าง Pipeline และเลือก Feature
- `src/kaverentai/models/tune_rental_price.py` — ปรับ Gradient Boosting
- `src/kaverentai/models/train_rental_price.py` — ฝึกและเปรียบเทียบโมเดล
- `src/kaverentai/recommendation/price_engine.py` — โหลดโมเดลผู้ชนะไปใช้งาน
- `tests/test_rental_price.py` — Unit Tests

## วิธีรัน

```powershell
.venv\Scripts\python.exe -m kaverentai.models.train_rental_price
.venv\Scripts\python.exe -m kaverentai.recommendation.price_engine
.venv\Scripts\python.exe -m pytest -q
```

## เอกสารอ้างอิงหลัก

- แบบจำลองพยากรณ์อัตราค่าเช่าที่อยู่อาศัยประเภทหอพักหรืออพาร์ตเมนต์
  ในพื้นที่โดยรอบสถาบันการศึกษา กรณีศึกษา มหาวิทยาลัยธรรมศาสตร์
  ศูนย์รังสิต — ใช้อ้างอิงแนวทาง Stepwise Multiple Linear Regression
- Hernes, M., Tutak, M., & Siewiera, A. (2024). *Prediction of residential
  real estate price on primary market using machine learning*.
  Procedia Computer Science, 246, 3142–3147.
  DOI: 10.1016/j.procs.2024.09.358 — ใช้อ้างอิง Gradient Boosting
