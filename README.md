# KaveRentAI

ระบบสนับสนุนการตัดสินใจสำหรับธุรกิจเช่าคอนโด โดยใช้ข้อมูลจำลองของโครงการ Kave เพื่อพัฒนาโมเดลทำนายและระบบทดลองสถานการณ์ด้านราคา

> ข้อมูลทั้งหมดเป็นข้อมูลจำลอง ห้ามอ้างอิงตัวเลขในชุดข้อมูลว่าเป็นผลประกอบการจริงของบริษัท

## ขอบเขตของระบบ

โปรเจกต์ประกอบด้วยโมเดลหลัก 4 งาน

| งาน | Target | Baseline | โมเดลเปรียบเทียบ | โมเดลที่เลือก |
|---|---|---|---|---|
| อัตราการต่อสัญญารายโครงการ–เดือน | `renewal_rate` | Historical-rate Baseline | Binomial Logistic Regression | Historical-rate Baseline |
| ราคาค่าเช่า | `rent` | Linear Regression | CatBoostRegressor | รอสรุปผล |
| โอกาสปล่อยเช่าภายใน 4 สัปดาห์ | `leased_within_4_weeks` | Logistic Regression | CatBoostClassifier | CatBoostClassifier |
| ระยะเวลาปล่อยเช่า | `weeks_on_market` | Linear Regression | CatBoostRegressor | รอสรุปผล |

ผลจากโมเดลจะนำไปใช้ใน Price Recommendation Engine, What-if Simulation, Dashboard และ Web Application

งาน Renewal ใช้หน่วยวิเคราะห์ระดับ **โครงการ–เดือน** เพื่อประมาณอัตราและจำนวน
สัญญาที่คาดว่าจะต่อ ไม่ได้ใช้เพื่อระบุผู้เช่ารายบุคคล โดยผล Validation ปัจจุบัน
เลือก Historical-rate Baseline เนื่องจากให้ความคลาดเคลื่อนต่ำกว่าโมเดลเปรียบเทียบ

งาน Lease Probability ใช้ข้อมูลที่ทราบในวันลงประกาศเพื่อประมาณโอกาสปล่อยเช่า
ภายใน 4 สัปดาห์ และตัดประกาศช่วงท้ายที่มีเวลาติดตามไม่ครบออก ผล Validation
เลือก CatBoostClassifier เนื่องจากให้ Log Loss ต่ำกว่า Logistic Regression

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

สร้างข้อมูลและฝึกโมเดล Lease Probability:

```powershell
python -m kaverentai.features.lease_probability_features
python -m kaverentai.models.train_lease_probability
python -m kaverentai.models.diagnose_lease_probability
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

- Train: ปี 2023-2025
- Validation: มกราคม-มีนาคม 2026
- Test: เมษายน-มิถุนายน 2026

ห้ามใช้ Test ระหว่างเลือกโมเดลหรือปรับพารามิเตอร์

## การแบ่งงาน

- ทอฝัน: Data Quality, Common Features, Renewal และ Lease Probability
- อาทิตยา: Time Split, Rental Price, Weeks-on-market, Price Engine, What-if และ Web Application

ทั้งสองคนทำ EDA ใน Notebook แยกกัน และนำโค้ดที่ต้องใช้ร่วมกันมาไว้ใน `src/kaverentai/`

## เอกสารเพิ่มเติม

- [รายการงานและผู้รับผิดชอบ](TASKS.md)
- [วิธีทำงานร่วมกันผ่าน Git](CONTRIBUTING.md)
- [คำอธิบายโฟลเดอร์ข้อมูล](data/README.md)
- [คำอธิบาย Notebook และผู้รับผิดชอบ](notebooks/README.md)
- [ขอบเขตและกติกาการทำงาน](docs/project_scope.md)
- [รายการเอกสารทั้งหมด](docs/README.md)
