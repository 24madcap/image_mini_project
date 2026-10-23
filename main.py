"""Image Pro - จุดเริ่มต้นของแอปพลิเคชัน

แอปพลิเคชันเดสก์ท็อปสำหรับขยายและเพิ่มความคมชัดรูปภาพ
"""

import sys
import os

# เพิ่ม project root เข้า path เพื่อให้ import ได้จากทุกที่
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

from app.main_window import MainWindow


def main() -> None:
    """ฟังก์ชันหลัก - สร้างและเรียกใช้แอปพลิเคชัน"""
    # เปิดใช้ High DPI scaling
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("Image Pro")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("ImagePro")

    # ตั้งไอคอนแอป (ถ้ามี)
    icon_path = os.path.join(os.path.dirname(__file__), "assets", "icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    # ตั้ง style ทั่วไป
    app.setStyle("Fusion")

    # สร้างและแสดงหน้าต่างหลัก
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
