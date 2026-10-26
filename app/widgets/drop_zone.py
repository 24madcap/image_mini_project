"""วิดเจ็ตพื้นที่ลากและวางไฟล์ (Drag & Drop Zone)

แสดงพื้นที่สำหรับลากไฟล์รูปภาพเข้ามา พร้อมปุ่มเลือกไฟล์
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QPainter, QColor, QPen, QFont
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog

from app.utils.file_utils import is_supported_image, FILE_FILTER


class DropZone(QWidget):
    """พื้นที่ลากและวางไฟล์รูปภาพ
    
    Signals:
        file_dropped: ส่งเส้นทางไฟล์เมื่อมีการวางไฟล์
        files_dropped: ส่งรายการเส้นทางไฟล์เมื่อวางหลายไฟล์
    """
    file_dropped = pyqtSignal(str)
    files_dropped = pyqtSignal(list)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(300, 200)
        self._is_hovering = False
        self._setup_ui()

    def _setup_ui(self) -> None:
        """สร้าง UI ของ Drop Zone"""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        # ไอคอน
        icon_label = QLabel("🖼️")
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 72px; background: transparent; border: none;")
        layout.addWidget(icon_label)

        # ข้อความหลัก
        main_label = QLabel("ลากและวางรูปภาพที่นี่")
        main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_label.setStyleSheet(
            "font-size: 22px; font-weight: bold; color: #cccccc; "
            "background: transparent; border: none;"
        )
        layout.addWidget(main_label)

        # ข้อความรอง
        sub_label = QLabel("รองรับ PNG, JPG, BMP, TIFF, WebP")
        sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub_label.setStyleSheet(
            "font-size: 13px; color: #888888; background: transparent; border: none;"
        )
        layout.addWidget(sub_label)

        # ปุ่มเลือกไฟล์
        browse_btn = QPushButton("📂 เลือกไฟล์")
        browse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_btn.setFixedSize(190, 48)
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a9eff;
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 16px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #3a8eef;
            }
            QPushButton:pressed {
                background-color: #2a7edf;
            }
        """)
        browse_btn.clicked.connect(self._browse_file)
        layout.addWidget(browse_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setStyleSheet("""
            DropZone {
                background-color: #2a2a2a;
                border: 2px dashed #555555;
                border-radius: 12px;
            }
        """)

    def _browse_file(self) -> None:
        """เปิด dialog เลือกไฟล์"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "เลือกรูปภาพ",
            "",
            FILE_FILTER,
        )
        if file_paths:
            # กรองเฉพาะไฟล์ที่รองรับ
            valid = [f for f in file_paths if is_supported_image(f)]
            if len(valid) == 1:
                self.file_dropped.emit(valid[0])
            elif len(valid) > 1:
                self.files_dropped.emit(valid)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """ตรวจสอบเมื่อลากไฟล์เข้ามา"""
        if event.mimeData().hasUrls():
            # ตรวจสอบว่ามีไฟล์รูปภาพหรือไม่
            urls = event.mimeData().urls()
            has_image = any(
                is_supported_image(url.toLocalFile())
                for url in urls
                if url.isLocalFile()
            )
            if has_image:
                event.acceptProposedAction()
                self._is_hovering = True
                self.setStyleSheet("""
                    DropZone {
                        background-color: #1a3a5a;
                        border: 2px dashed #4a9eff;
                        border-radius: 12px;
                    }
                """)
                self.update()

    def dragLeaveEvent(self, event) -> None:
        """เมื่อลากไฟล์ออกจากพื้นที่"""
        self._is_hovering = False
        self.setStyleSheet("""
            DropZone {
                background-color: #2a2a2a;
                border: 2px dashed #555555;
                border-radius: 12px;
            }
        """)
        self.update()

    def dropEvent(self, event: QDropEvent) -> None:
        """เมื่อวางไฟล์"""
        self._is_hovering = False
        self.setStyleSheet("""
            DropZone {
                background-color: #2a2a2a;
                border: 2px dashed #555555;
                border-radius: 12px;
            }
        """)

        urls = event.mimeData().urls()
        valid_files: list[str] = []
        for url in urls:
            if url.isLocalFile():
                path = url.toLocalFile()
                if is_supported_image(path):
                    valid_files.append(path)

        if len(valid_files) == 1:
            self.file_dropped.emit(valid_files[0])
        elif len(valid_files) > 1:
            self.files_dropped.emit(valid_files)

        event.acceptProposedAction()
        self.update()
