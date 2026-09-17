# KaveRentAI

ระบบสนับสนุนการตัดสินใจสำหรับธุรกิจเช่าคอนโด โดยใช้ข้อมูลจำลองของโครงการ Kave เพื่อพัฒนาโมเดลทำนายและระบบทดลองสถานการณ์ด้านราคา

> ข้อมูลทั้งหมดเป็นข้อมูลจำลอง ห้ามอ้างอิงตัวเลขในชุดข้อมูลว่าเป็นผลประกอบการจริงของบริษัท

## ขอบเขตของระบบ

โปรเจกต์ประกอบด้วยโมเดลหลัก 2 งาน

| งาน | Target | Baseline | โมเดลเปรียบเทียบ | โมเดลที่เลือก |
|---|---|---|---|---|
| ราคาค่าเช่า | `rent` | Linear Regression | CatBoostRegressor | รอสรุปผล |
| ระยะเวลาปล่อยเช่า | `weeks_on_market` | Multiple Linear Regression | Random Forest Regressor | Random Forest Regressor |

ผลจากโมเดลจะนำไปใช้ใน Price Recommendation Engine, What-if Simulation, Dashboard และ Web Application

งาน Time-to-Lease ใช้ข้อมูลที่ทราบ ณ วันเริ่มลงประกาศ เพื่อประมาณจำนวนสัปดาห์
ที่คาดว่าจะใช้ในการปล่อยเช่า และใช้เฉพาะประกาศที่สังเกตการปล่อยเช่าสำเร็จแล้ว
สำหรับการฝึกและประเมินโมเดล Regression

## โครงสร้างโปรเจกต์

```text
KaveRentAI/
|-- data/
|   |-- raw/              # ข้อมูลต้นฉบับ ห้ามแก้โดยตรง
|   |-- processed/        # ข้อมูลที่ทำความสะอาดแล้ว
|   `-- modeling/         # ตารางที่สร้างเพื่อฝึกโมเดล
|-- notebooks/            # EDA และการทดลอง
|-- src/kaverentai/       # โค้ดที่เรียกใช้ซ้ำ
|-- configs/              # ค่าตั้งต้นของโมเดลและการแบ่งข้อมูล
|-- models/               # โมเดลที่ฝึกแล้ว (ไม่เก็บใน Git)
|-- reports/              # กราฟ คะแนน และตารางที่สร้างจากโค้ด
|-- app/                  # Streamlit Web Application
|-- tests/                # การทดสอบโค้ด
`-- docs/                 # คู่มือข้อมูลและเอกสารโครงการ
```

## การติดตั้ง

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## เริ่มใช้งาน

ก่อนเริ่มพัฒนา ให้เปิด [TASKS.md](TASKS.md) เพื่อดูงาน ผู้รับผิดชอบ และสถานะล่าสุด

โหลดข้อมูลที่ทำความสะอาดแล้ว:

```python
from kaverentai.data.load_data import load_all_data

data = load_all_data(cleaned=True)
listings = data["listings"]
leases = data["leases"]
```

ตรวจสอบโครงสร้างข้อมูล:

```powershell
python -m kaverentai.data.validate_data
```

สร้าง Modeling Dataset และฝึกโมเดล Time-to-Lease:

```powershell
python -m kaverentai.features.time_to_lease_features
python -m kaverentai.models.train_time_to_lease
```

เปิด Web Application:

```powershell
streamlit run app/Home.py
```

รันทดสอบ:

```powershell
pytest
```

## การแบ่งข้อมูลตามเวลา

งานราคาค่าเช่าใช้ช่วงเวลาหลักของโครงการ:

- Train: ปี 2023-2025
- Validation: มกราคม-มีนาคม 2026
- Test: เมษายน-มิถุนายน 2026

งาน Time-to-Lease ใช้ **Matured Time Split** เพื่อให้ Test มีระยะติดตาม
อย่างน้อย 52 สัปดาห์ ลดปัญหา right-censoring:

- Train: ปี 2023-2024
- Validation: มกราคม-มีนาคม 2025
- Test: เมษายน-มิถุนายน 2025
- หลังมิถุนายน 2025: `post_test` ไม่ใช้เลือกหรือประเมินโมเดล

ห้ามใช้ Test ระหว่างเลือกโมเดลหรือปรับพารามิเตอร์

## การแบ่งงาน

- ทอฝัน: Data Quality และ Time-to-Lease
- อาทิตยา: Time Split, Rental Price, Price Engine, What-if และ Web Application

ทั้งสองคนทำ EDA ใน Notebook แยกกัน และนำโค้ดที่ต้องใช้ร่วมกันมาไว้ใน `src/kaverentai/`

## เอกสารเพิ่มเติม

- [รายการงานและผู้รับผิดชอบ](TASKS.md)
- [วิธีทำงานร่วมกันผ่าน Git](CONTRIBUTING.md)
- [คำอธิบายโฟลเดอร์ข้อมูล](data/README.md)
- [คำอธิบาย Notebook และผู้รับผิดชอบ](notebooks/README.md)
- [ขอบเขตและกติกาการทำงาน](docs/project_scope.md)
- [รายการเอกสารทั้งหมด](docs/README.md)
