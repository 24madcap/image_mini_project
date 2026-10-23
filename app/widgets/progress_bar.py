"""วิดเจ็ตแถบความคืบหน้า (Progress Bar)

แสดงความคืบหน้าการประมวลผลพร้อมข้อความสถานะ
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QProgressBar, QLabel


class ProcessingProgressBar(QWidget):
    """แถบแสดงความคืบหน้าการประมวลผล"""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """สร้าง UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        # แถวบนสุด: ข้อความสถานะ + เปอร์เซ็นต์
        top_row = QHBoxLayout()

        self._status_label = QLabel("พร้อมใช้งาน")
        self._status_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        top_row.addWidget(self._status_label)

        top_row.addStretch()

        self._percent_label = QLabel("")
        self._percent_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        top_row.addWidget(self._percent_label)

        layout.addLayout(top_row)

        # แถบ progress
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #3a3a3a;
                border: none;
                border-radius: 4px;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4a9eff, stop:1 #7c4dff
                );
                border-radius: 4px;
            }
        """)
        layout.addWidget(self._progress_bar)

        # ซ่อนแถบเริ่มต้น
        self._progress_bar.setVisible(False)
        self._percent_label.setVisible(False)

    def set_progress(self, value: int) -> None:
        """ตั้งค่าความคืบหน้า (0-100)"""
        value = max(0, min(100, value))
        self._progress_bar.setValue(value)
        self._percent_label.setText(f"{value}%")

        if not self._progress_bar.isVisible():
            self._progress_bar.setVisible(True)
            self._percent_label.setVisible(True)

    def set_status(self, text: str) -> None:
        """ตั้งข้อความสถานะ"""
        self._status_label.setText(text)

    def set_processing(self, is_processing: bool) -> None:
        """สลับสถานะกำลังประมวลผล"""
        self._progress_bar.setVisible(is_processing)
        self._percent_label.setVisible(is_processing)
        if is_processing:
            self._status_label.setStyleSheet("color: #4a9eff; font-size: 11px;")
        else:
            self._status_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")

    def reset(self) -> None:
        """รีเซ็ตแถบความคืบหน้า"""
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._percent_label.setVisible(False)
        self._percent_label.setText("")
        self._status_label.setText("พร้อมใช้งาน")
        self._status_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")

    def set_complete(self) -> None:
        """แสดงสถานะเสร็จสิ้น"""
        self._progress_bar.setValue(100)
        self._percent_label.setText("100%")
        self._status_label.setText("✅ ประมวลผลเสร็จสิ้น")
        self._status_label.setStyleSheet("color: #4caf50; font-size: 11px;")

    def set_error(self, message: str = "เกิดข้อผิดพลาด") -> None:
        """แสดงสถานะข้อผิดพลาด"""
        self._status_label.setText(f"❌ {message}")
        self._status_label.setStyleSheet("color: #f44336; font-size: 11px;")
        self._progress_bar.setVisible(False)
        self._percent_label.setVisible(False)
