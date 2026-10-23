@echo off
chcp 65001 > nul
title Image Pro - ขยายและเพิ่มความคมชัดรูปภาพ

echo ===================================================
echo     Image Pro - โปรแกรมขยายและเพิ่มความคมชัดรูปภาพ
echo ===================================================
echo กำลังเริ่มต้นโปรแกรม...
echo.

python main.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo เกิดข้อผิดพลาดในการรันโปรแกรม กรุณาตรวจสอบว่าได้ติดตั้งไลบรารีครบถ้วนแล้ว:
    echo pip install -r requirements.txt
    echo.
    pause
)
