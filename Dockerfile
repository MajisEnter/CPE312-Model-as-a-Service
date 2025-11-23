# Base image ที่ใช้ Python 3.11 แบบขนาดเล็ก
FROM python:3.11-slim

# สร้างโฟลเดอร์ทำงานภายใน container
WORKDIR /app

# คัดลอก requirements.txt และติดตั้ง dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกไฟล์โปรเจกต์ทั้งหมดเข้า container
COPY . .

# สร้างโมเดล (train model) ตอน build image
RUN python train_model.py

# เปิด port 5000 สำหรับ Flask
EXPOSE 5000

# สั่งให้รัน Flask app ผ่าน Gunicorn (production server)
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]