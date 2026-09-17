# Notebooks

Notebook ใช้สำหรับ EDA การทดลอง และสรุปผล แต่โค้ดที่ต้องเรียกซ้ำควรย้ายไปไว้ใน `src/kaverentai/`

| ลำดับ | ชื่อไฟล์ | ผู้รับผิดชอบ |
|---|---|---|
| 1 | `01_common_data_quality.ipynb` | ทอฝัน |
| 2 | `02_time_to_lease_eda.ipynb` | ทอฝัน |
| 4 | `04_rental_price_eda.ipynb` | อาทิตยา |
| 6 | `06_final_results.ipynb` | อาทิตยา และทอฝันช่วยตรวจ |

แต่ละคนสร้างและแก้เฉพาะ Notebook ของตัวเอง เพื่อลดปัญหา Git conflict

`02_time_to_lease_eda.ipynb` ใช้ `weeks_on_market` เป็น Target และใช้เฉพาะ
ประกาศที่ `leased = True` สำหรับงาน Regression โดย Feature ต้องเป็นข้อมูลที่
ทราบแล้ว ณ วันเริ่มลงประกาศ การประเมินใช้ Matured Time Split โดยเว้นระยะ
ติดตามหลัง Test อย่างน้อย 52 สัปดาห์ และไม่นำ `post_test` ไปวัดผล
