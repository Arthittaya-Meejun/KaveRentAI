# วิธีทำงานร่วมกัน

## ก่อนเริ่มงาน

1. เปิด `TASKS.md` และเลือกงานที่มีชื่อตัวเอง
2. เปลี่ยนสถานะงานจาก `Todo` เป็น `Doing`
3. ดึงงานล่าสุดจาก GitHub

```powershell
git switch main
git pull origin main
```

4. สร้าง branch ใหม่สำหรับงานนั้น

```powershell
git switch -c feature/ชื่อสั้นของงาน
```

ตัวอย่าง:

```powershell
git switch -c feature/renewal-eda
```

## เมื่อทำงานเสร็จ

```powershell
git add .
git commit -m "Add renewal EDA"
git push -u origin feature/renewal-eda
```

จากนั้นเปิด Pull Request บน GitHub และให้อีกคนตรวจก่อนรวมเข้า `main`

## กติกา

- หนึ่งงานใช้หนึ่ง branch
- ไม่แก้ Notebook ไฟล์เดียวกันพร้อมกัน
- ไม่แก้ไฟล์ใน `data/raw/` ด้วยมือ
- ไม่อัปโหลดรหัสผ่าน Token หรือไฟล์ `.env`
- หากต้องแก้ไฟล์ส่วนกลาง ให้แจ้งอีกคนก่อน
- เมื่อ Pull Request ผ่านแล้ว ให้เปลี่ยนสถานะใน `TASKS.md` เป็น `Done`
