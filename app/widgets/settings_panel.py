"""แผงควบคุมการตั้งค่า (Settings Panel)

ประกอบด้วย:
- พรีเซ็ตแนะนำ (Quick Presets) เช่น ขยายรูปถ่าย, ขยายภาพการ์ตูน, ขยายด่วน
- ตัวเลือกโหมดขยายภาพ (AI Real-ESRGAN / Traditional Interpolation)
- ตัวเลือกโมเดล AI (x4plus, Anime 6B, x2plus)
- ตัวเลือกการเพิ่มความคมชัด (Unsharp Mask, Laplacian, Detail Enhancement)
- โหมดปรับเฉพาะความสว่าง (Luminance Only) เพื่อป้องกันสัญญาณรบกวนสี
- ตัวเลือกคุณภาพการบันทึกภาพ
- ปุ่มประมวลผล, ปุ่มบันทึก, และปุ่มประมวลผลแบตช์
"""

from typing import Optional, Union

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QRadioButton,
    QComboBox, QLabel, QSlider, QCheckBox, QPushButton, QSpinBox,
    QButtonGroup, QScrollArea, QFileDialog, QMessageBox,
)

from app.processing.upscaler import (
    UpscaleMode, UpscaleParams, InterpolationMethod, check_ai_available,
)
from app.processing.ai_engine import AVAILABLE_MODELS
from app.processing.sharpener import (
    SharpenMethod, UnsharpMaskParams, LaplacianParams, DetailEnhanceParams,
)
from app.utils.file_utils import SAVE_FILTER


class SettingsPanel(QWidget):
    """แผงควบคุมการตั้งค่าทางด้านขวาของแอปพลิเคชัน"""

    process_clicked = pyqtSignal()
    save_clicked = pyqtSignal(str, int)
    batch_clicked = pyqtSignal(str)
    settings_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedWidth(330)
        self._block_signals = False
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """สร้างส่วนต่อประสานทั้งหมดของแผงการตั้งค่า"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                background: #202020;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #505050;
                border-radius: 4px;
                min-height: 24px;
            }
            QScrollBar::handle:vertical:hover {
                background: #686868;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """)

        container = QWidget()
        self._main_layout = QVBoxLayout(container)
        self._main_layout.setContentsMargins(10, 10, 10, 10)
        self._main_layout.setSpacing(12)

        # หัวข้อแผงควบคุม
        title = QLabel("⚙ การตั้งค่า (Settings)")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff; padding: 2px 0;")
        self._main_layout.addWidget(title)

        # ── กลุ่มที่ 0: พรีเซ็ตแนะนำ (Quick Presets) ──
        self._create_presets_group()

        # ── กลุ่มที่ 1: การขยายขนาดภาพ (Upscaling) ──
        self._create_upscale_group()

        # ── กลุ่มที่ 2: การเพิ่มความคมชัด (Sharpening) ──
        self._create_sharpen_group()

        # ── กลุ่มที่ 3: คุณภาพการบันทึก (Save Quality) ──
        self._create_save_quality_group()

        # ── กลุ่มที่ 4: ปุ่มดำเนินการหลัก (Actions) ──
        self._create_action_buttons()

        self._main_layout.addStretch()
        scroll.setWidget(container)

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    # ──────────────────────────────────────
    # 0. พรีเซ็ต
    # ──────────────────────────────────────
    def _create_presets_group(self) -> None:
        """สร้างส่วนเลือกพรีเซ็ตสำเร็จรูป"""
        group = QGroupBox("🎯 พรีเซ็ตแนะนำ (Presets)")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(6)

        self._preset_combo = QComboBox()
        self._preset_combo.addItem("กำหนดค่าเอง (Custom)", "custom")
        self._preset_combo.addItem("📸 ขยายภาพถ่าย (AI 4x + คมชัดสมดุล)", "photo_4x")
        self._preset_combo.addItem("🎨 ขยายภาพการ์ตูน/อนิเมะ (AI Anime 4x)", "anime_4x")
        self._preset_combo.addItem("⚡ ขยายด่วน (Lanczos 2x + Unsharp Mask)", "fast_2x")
        self._preset_combo.addItem("✨ เพิ่มความคมชัดเท่านั้น (ไม่ขยายภาพ)", "sharpen_only")
        self._preset_combo.setStyleSheet(self._combo_style())
        self._preset_combo.currentIndexChanged.connect(self._on_preset_selected)
        layout.addWidget(self._preset_combo)

        self._main_layout.addWidget(group)

    # ──────────────────────────────────────
    # 1. การขยายภาพ
    # ──────────────────────────────────────
    def _create_upscale_group(self) -> None:
        """สร้างกลุ่มตัวเลือกการขยายขนาดภาพ"""
        group = QGroupBox("📐 ขยายขนาดภาพ (Upscaling)")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        # สวิตช์เปิด/ปิด
        self._upscale_enabled = QCheckBox("เปิดใช้การขยายขนาดภาพ")
        self._upscale_enabled.setChecked(True)
        self._upscale_enabled.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 13px;")
        layout.addWidget(self._upscale_enabled)

        # โหมดการขยาย
        mode_label = QLabel("เทคโนโลยีการขยาย:")
        mode_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(mode_label)

        self._mode_group = QButtonGroup(self)

        self._ai_radio = QRadioButton("🤖 AI Super-Resolution (Real-ESRGAN)")
        self._ai_radio.setChecked(True)
        self._ai_radio.setStyleSheet("color: #ffffff; font-weight: 500;")
        self._mode_group.addButton(self._ai_radio)
        layout.addWidget(self._ai_radio)

        self._traditional_radio = QRadioButton("🔧 Traditional (OpenCV Interpolation)")
        self._traditional_radio.setStyleSheet("color: #cccccc;")
        self._mode_group.addButton(self._traditional_radio)
        layout.addWidget(self._traditional_radio)

        # ตัวเลือกโมเดล AI
        self._ai_options = QWidget()
        ai_layout = QVBoxLayout(self._ai_options)
        ai_layout.setContentsMargins(16, 2, 0, 2)
        ai_layout.setSpacing(4)

        ai_model_label = QLabel("โมเดล AI:")
        ai_model_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        ai_layout.addWidget(ai_model_label)

        self._ai_model_combo = QComboBox()
        for key, model_info in AVAILABLE_MODELS.items():
            self._ai_model_combo.addItem(model_info.display_name, key)
        self._ai_model_combo.setStyleSheet(self._combo_style())
        ai_layout.addWidget(self._ai_model_combo)

        layout.addWidget(self._ai_options)

        # ตัวเลือก Traditional
        self._traditional_options = QWidget()
        trad_layout = QVBoxLayout(self._traditional_options)
        trad_layout.setContentsMargins(16, 2, 0, 2)
        trad_layout.setSpacing(4)

        interp_label = QLabel("วิธี Interpolation:")
        interp_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        trad_layout.addWidget(interp_label)

        self._interp_combo = QComboBox()
        for method in InterpolationMethod:
            self._interp_combo.addItem(method.display_name, method)
        self._interp_combo.setCurrentIndex(3)  # Lanczos ค่าเริ่มต้น
        self._interp_combo.setStyleSheet(self._combo_style())
        trad_layout.addWidget(self._interp_combo)

        self._traditional_options.setVisible(False)
        layout.addWidget(self._traditional_options)

        # สเกลการขยาย
        scale_row = QHBoxLayout()
        scale_label = QLabel("อัตราขยาย:")
        scale_label.setStyleSheet("color: #cccccc;")
        scale_row.addWidget(scale_label)

        self._scale_combo = QComboBox()
        self._scale_combo.addItem("1.5 เท่า (1.5x)", 1.5)
        self._scale_combo.addItem("2 เท่า (2x)", 2.0)
        self._scale_combo.addItem("3 เท่า (3x)", 3.0)
        self._scale_combo.addItem("4 เท่า (4x แนะนำสำหรับ AI)", 4.0)
        self._scale_combo.addItem("6 เท่า (6x)", 6.0)
        self._scale_combo.addItem("8 เท่า (8x)", 8.0)
        self._scale_combo.setCurrentIndex(3)  # 4x
        self._scale_combo.setStyleSheet(self._combo_style())
        scale_row.addWidget(self._scale_combo)
        layout.addLayout(scale_row)

        # กำหนดขนาด pixel เอง
        self._custom_size_check = QCheckBox("กำหนดขนาดเอง (Custom Dimensions)")
        self._custom_size_check.setStyleSheet("color: #cccccc; font-size: 11px;")
        layout.addWidget(self._custom_size_check)

        self._custom_size_widget = QWidget()
        custom_layout = QHBoxLayout(self._custom_size_widget)
        custom_layout.setContentsMargins(12, 0, 0, 0)
        custom_layout.setSpacing(4)

        self._custom_w = QSpinBox()
        self._custom_w.setRange(1, 65536)
        self._custom_w.setValue(1920)
        self._custom_w.setSuffix(" px")
        self._custom_w.setStyleSheet(self._spinbox_style())
        custom_layout.addWidget(QLabel("กว้าง:"))
        custom_layout.addWidget(self._custom_w)

        x_lbl = QLabel("×")
        x_lbl.setStyleSheet("color: #888888; font-weight: bold;")
        custom_layout.addWidget(x_lbl)

        self._custom_h = QSpinBox()
        self._custom_h.setRange(1, 65536)
        self._custom_h.setValue(1080)
        self._custom_h.setSuffix(" px")
        self._custom_h.setStyleSheet(self._spinbox_style())
        custom_layout.addWidget(QLabel("สูง:"))
        custom_layout.addWidget(self._custom_h)

        self._custom_size_widget.setVisible(False)
        layout.addWidget(self._custom_size_widget)

        self._main_layout.addWidget(group)

    # ──────────────────────────────────────
    # 2. ความคมชัด
    # ──────────────────────────────────────
    def _create_sharpen_group(self) -> None:
        """สร้างกลุ่มตัวเลือกการเพิ่มความคมชัด"""
        group = QGroupBox("✨ เพิ่มความคมชัด (Sharpening)")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self._sharpen_enabled = QCheckBox("เปิดใช้การเพิ่มความคมชัด")
        self._sharpen_enabled.setChecked(True)
        self._sharpen_enabled.setStyleSheet("color: #ffffff; font-weight: bold; font-size: 13px;")
        layout.addWidget(self._sharpen_enabled)

        self._sharpen_content = QWidget()
        content_layout = QVBoxLayout(self._sharpen_content)
        content_layout.setContentsMargins(0, 2, 0, 0)
        content_layout.setSpacing(8)

        # วิธีเพิ่มความคมชัด
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("วิธี:"))
        self._sharpen_combo = QComboBox()
        self._sharpen_combo.addItem("Unsharp Mask (มาตรฐาน ยอดนิยม)", SharpenMethod.UNSHARP_MASK.value)
        self._sharpen_combo.addItem("Laplacian Filter (เน้นขอบ)", SharpenMethod.LAPLACIAN.value)
        self._sharpen_combo.addItem("Detail Enhancement (คงรูปขอบ)", SharpenMethod.DETAIL_ENHANCE.value)
        self._sharpen_combo.setStyleSheet(self._combo_style())
        method_row.addWidget(self._sharpen_combo)
        content_layout.addLayout(method_row)

        # Checkbox: ปรับเฉพาะ Luminance (LAB L-channel)
        self._luminance_only_check = QCheckBox("ปรับเฉพาะความสว่าง (Luminance Only)")
        self._luminance_only_check.setChecked(True)
        self._luminance_only_check.setToolTip("ป้องกันสัญญาณรบกวนของสี (Color Noise) โดยเพิ่มความคมชัดเฉพาะช่องแสง")
        self._luminance_only_check.setStyleSheet("color: #81c784; font-size: 11px;")
        content_layout.addWidget(self._luminance_only_check)

        # ── 1. Unsharp Mask Parameters ──
        self._unsharp_widget = QWidget()
        unsharp_lay = QVBoxLayout(self._unsharp_widget)
        unsharp_lay.setContentsMargins(0, 0, 0, 0)
        unsharp_lay.setSpacing(4)

        self._unsharp_radius_label = QLabel("รัศมี (Radius): 2.0")
        self._unsharp_radius_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        unsharp_lay.addWidget(self._unsharp_radius_label)
        self._unsharp_radius = QSlider(Qt.Orientation.Horizontal)
        self._unsharp_radius.setRange(2, 80)  # 0.2 ถึง 8.0
        self._unsharp_radius.setValue(20)
        self._unsharp_radius.setStyleSheet(self._slider_style())
        unsharp_lay.addWidget(self._unsharp_radius)

        self._unsharp_amount_label = QLabel("ความแรง (Amount): 1.5")
        self._unsharp_amount_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        unsharp_lay.addWidget(self._unsharp_amount_label)
        self._unsharp_amount = QSlider(Qt.Orientation.Horizontal)
        self._unsharp_amount.setRange(1, 40)  # 0.1 ถึง 4.0
        self._unsharp_amount.setValue(15)
        self._unsharp_amount.setStyleSheet(self._slider_style())
        unsharp_lay.addWidget(self._unsharp_amount)

        self._unsharp_threshold_label = QLabel("ขีดจำกัด (Threshold): 0")
        self._unsharp_threshold_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        unsharp_lay.addWidget(self._unsharp_threshold_label)
        self._unsharp_threshold = QSlider(Qt.Orientation.Horizontal)
        self._unsharp_threshold.setRange(0, 100)
        self._unsharp_threshold.setValue(0)
        self._unsharp_threshold.setStyleSheet(self._slider_style())
        unsharp_lay.addWidget(self._unsharp_threshold)

        content_layout.addWidget(self._unsharp_widget)

        # ── 2. Laplacian Parameters ──
        self._laplacian_widget = QWidget()
        lap_lay = QVBoxLayout(self._laplacian_widget)
        lap_lay.setContentsMargins(0, 0, 0, 0)
        lap_lay.setSpacing(4)

        lap_lay.addWidget(QLabel("ขนาด Kernel:"))
        self._lap_ksize = QComboBox()
        self._lap_ksize.addItems(["1", "3", "5", "7"])
        self._lap_ksize.setCurrentIndex(1)
        self._lap_ksize.setStyleSheet(self._combo_style())
        lap_lay.addWidget(self._lap_ksize)

        self._lap_strength_label = QLabel("ความแรง: 0.50")
        self._lap_strength_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        lap_lay.addWidget(self._lap_strength_label)
        self._lap_strength = QSlider(Qt.Orientation.Horizontal)
        self._lap_strength.setRange(2, 40)  # 0.05 ถึง 1.00
        self._lap_strength.setValue(10)
        self._lap_strength.setStyleSheet(self._slider_style())
        lap_lay.addWidget(self._lap_strength)

        self._laplacian_widget.setVisible(False)
        content_layout.addWidget(self._laplacian_widget)

        # ── 3. Detail Enhance Parameters ──
        self._detail_widget = QWidget()
        detail_lay = QVBoxLayout(self._detail_widget)
        detail_lay.setContentsMargins(0, 0, 0, 0)
        detail_lay.setSpacing(4)

        self._detail_sigma_s_label = QLabel("Spatial Sigma: 10")
        self._detail_sigma_s_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        detail_lay.addWidget(self._detail_sigma_s_label)
        self._detail_sigma_s = QSlider(Qt.Orientation.Horizontal)
        self._detail_sigma_s.setRange(1, 200)
        self._detail_sigma_s.setValue(10)
        self._detail_sigma_s.setStyleSheet(self._slider_style())
        detail_lay.addWidget(self._detail_sigma_s)

        self._detail_sigma_r_label = QLabel("Range Sigma: 0.15")
        self._detail_sigma_r_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        detail_lay.addWidget(self._detail_sigma_r_label)
        self._detail_sigma_r = QSlider(Qt.Orientation.Horizontal)
        self._detail_sigma_r.setRange(1, 100)  # 0.01 ถึง 1.00
        self._detail_sigma_r.setValue(15)
        self._detail_sigma_r.setStyleSheet(self._slider_style())
        detail_lay.addWidget(self._detail_sigma_r)

        self._detail_widget.setVisible(False)
        content_layout.addWidget(self._detail_widget)

        # ความเข้มการผสม (Intensity / Blend)
        self._intensity_label = QLabel("ความเข้มการผสมผลลัพธ์: 100%")
        self._intensity_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        content_layout.addWidget(self._intensity_label)
        self._intensity_slider = QSlider(Qt.Orientation.Horizontal)
        self._intensity_slider.setRange(0, 100)
        self._intensity_slider.setValue(100)
        self._intensity_slider.setStyleSheet(self._slider_style())
        content_layout.addWidget(self._intensity_slider)

        layout.addWidget(self._sharpen_content)
        self._main_layout.addWidget(group)

    # ──────────────────────────────────────
    # 3. คุณภาพบันทึก
    # ──────────────────────────────────────
    def _create_save_quality_group(self) -> None:
        """สร้างกลุ่มตัวเลือกคุณภาพการบันทึกภาพ"""
        group = QGroupBox("💾 คุณภาพการบันทึกภาพ")
        group.setStyleSheet(self._group_style())
        layout = QVBoxLayout(group)
        layout.setSpacing(4)

        self._quality_label = QLabel("คุณภาพ (JPEG/WebP): 95%")
        self._quality_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(self._quality_label)

        self._quality_slider = QSlider(Qt.Orientation.Horizontal)
        self._quality_slider.setRange(1, 100)
        self._quality_slider.setValue(95)
        self._quality_slider.setStyleSheet(self._slider_style())
        layout.addWidget(self._quality_slider)

        self._main_layout.addWidget(group)

    # ──────────────────────────────────────
    # 4. ปุ่มการทำงาน
    # ──────────────────────────────────────
    def _create_action_buttons(self) -> None:
        """สร้างปุ่มประมวลผล บันทึก และประมวลผลแบตช์"""
        btn_layout = QVBoxLayout()
        btn_layout.setSpacing(8)

        # ปุ่มประมวลผล
        self._process_btn = QPushButton("▶ ประมวลผลภาพ (Process)")
        self._process_btn.setFixedHeight(46)
        self._process_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._process_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1976d2, stop:1 #2196f3);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e88e5, stop:1 #42a5f5);
            }
            QPushButton:pressed {
                background: #1565c0;
            }
            QPushButton:disabled {
                background: #424242;
                color: #888888;
            }
        """)
        btn_layout.addWidget(self._process_btn)

        # ปุ่มบันทึกผลลัพธ์
        self._save_btn = QPushButton("💾 บันทึกรูปภาพผลลัพธ์ (Save)")
        self._save_btn.setFixedHeight(40)
        self._save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_btn.setEnabled(False)
        self._save_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #2e7d32, stop:1 #43a047);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #388e3c, stop:1 #4caf50);
            }
            QPushButton:pressed {
                background: #1b5e20;
            }
            QPushButton:disabled {
                background: #424242;
                color: #888888;
            }
        """)
        btn_layout.addWidget(self._save_btn)

        # ปุ่มประมวลผลแบตช์
        self._batch_btn = QPushButton("📁 ประมวลผลแบตช์ทั้งโฟลเดอร์ (Batch)")
        self._batch_btn.setFixedHeight(36)
        self._batch_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._batch_btn.setStyleSheet("""
            QPushButton {
                background: #3949ab;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background: #3f51b5; }
            QPushButton:pressed { background: #283593; }
            QPushButton:disabled { background: #424242; color: #888888; }
        """)
        btn_layout.addWidget(self._batch_btn)

        self._main_layout.addLayout(btn_layout)

    # ──────────────────────────────────────
    # สัญญาณและการทำงาน
    # ──────────────────────────────────────
    def _connect_signals(self) -> None:
        """เชื่อมต่อสัญญาณทั้งหมด"""
        self._ai_radio.toggled.connect(self._on_mode_changed)
        self._traditional_radio.toggled.connect(self._on_mode_changed)
        self._ai_model_combo.currentIndexChanged.connect(self._on_ai_model_changed)
        self._custom_size_check.toggled.connect(self._custom_size_widget.setVisible)
        self._custom_size_check.toggled.connect(self._notify_change)
        self._upscale_enabled.toggled.connect(self._notify_change)
        self._scale_combo.currentIndexChanged.connect(self._notify_change)
        self._interp_combo.currentIndexChanged.connect(self._notify_change)

        self._sharpen_enabled.toggled.connect(self._sharpen_content.setVisible)
        self._sharpen_enabled.toggled.connect(self._notify_change)
        self._sharpen_combo.currentIndexChanged.connect(self._on_sharpen_method_changed)
        self._luminance_only_check.toggled.connect(self._notify_change)

        # สไลเดอร์อัพเดทป้ายกำกับ
        self._unsharp_radius.valueChanged.connect(
            lambda v: self._unsharp_radius_label.setText(f"รัศมี (Radius): {v / 10.0:.1f}")
        )
        self._unsharp_amount.valueChanged.connect(
            lambda v: self._unsharp_amount_label.setText(f"ความแรง (Amount): {v / 10.0:.1f}")
        )
        self._unsharp_threshold.valueChanged.connect(
            lambda v: self._unsharp_threshold_label.setText(f"ขีดจำกัด (Threshold): {v}")
        )
        self._lap_strength.valueChanged.connect(
            lambda v: self._lap_strength_label.setText(f"ความแรง: {v / 20.0:.2f}")
        )
        self._detail_sigma_s.valueChanged.connect(
            lambda v: self._detail_sigma_s_label.setText(f"Spatial Sigma: {v}")
        )
        self._detail_sigma_r.valueChanged.connect(
            lambda v: self._detail_sigma_r_label.setText(f"Range Sigma: {v / 100.0:.2f}")
        )
        self._intensity_slider.valueChanged.connect(
            lambda v: self._intensity_label.setText(f"ความเข้มการผสมผลลัพธ์: {v}%")
        )
        self._quality_slider.valueChanged.connect(
            lambda v: self._quality_label.setText(f"คุณภาพ (JPEG/WebP): {v}%")
        )

        for s in [
            self._unsharp_radius, self._unsharp_amount, self._unsharp_threshold,
            self._lap_strength, self._detail_sigma_s, self._detail_sigma_r,
            self._intensity_slider,
        ]:
            s.valueChanged.connect(self._notify_change)

        self._process_btn.clicked.connect(self.process_clicked.emit)
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._batch_btn.clicked.connect(self._on_batch_clicked)

    def _notify_change(self) -> None:
        """แจ้งเตือนเมื่อมีการเปลี่ยนการตั้งค่า"""
        if not self._block_signals:
            self.settings_changed.emit()

    def _on_mode_changed(self) -> None:
        """เมื่อสลับโหมด AI / Traditional"""
        is_ai = self._ai_radio.isChecked()
        self._ai_options.setVisible(is_ai)
        self._traditional_options.setVisible(not is_ai)
        self._notify_change()

    def _on_ai_model_changed(self) -> None:
        """เมื่อเปลี่ยนโมเดล AI"""
        model_name = self._ai_model_combo.currentData()
        info = AVAILABLE_MODELS.get(model_name)
        if info:
            # ปรับสเกลเริ่มต้นให้ตรงกับโมเดล
            target_scale = float(info.scale)
            for i in range(self._scale_combo.count()):
                if self._scale_combo.itemData(i) == target_scale:
                    self._scale_combo.setCurrentIndex(i)
                    break
        self._notify_change()

    def _on_sharpen_method_changed(self, index: int) -> None:
        """เมื่อเปลี่ยนวิธีเพิ่มความคมชัด"""
        method = self._sharpen_combo.currentData()
        self._unsharp_widget.setVisible(method == SharpenMethod.UNSHARP_MASK.value)
        self._laplacian_widget.setVisible(method == SharpenMethod.LAPLACIAN.value)
        self._detail_widget.setVisible(method == SharpenMethod.DETAIL_ENHANCE.value)
        self._notify_change()

    def _on_preset_selected(self, index: int) -> None:
        """เมื่อผู้ใช้เลือกพรีเซ็ตสำเร็จรูป"""
        preset = self._preset_combo.currentData()
        if preset == "custom":
            return

        self._block_signals = True

        if preset == "photo_4x":
            self._upscale_enabled.setChecked(True)
            self._ai_radio.setChecked(True)
            # หา index ของ RealESRGAN_x4plus
            idx = self._ai_model_combo.findData("RealESRGAN_x4plus")
            if idx >= 0:
                self._ai_model_combo.setCurrentIndex(idx)
            self._scale_combo.setCurrentIndex(self._scale_combo.findData(4.0))
            self._custom_size_check.setChecked(False)

            self._sharpen_enabled.setChecked(True)
            self._sharpen_combo.setCurrentIndex(self._sharpen_combo.findData(SharpenMethod.UNSHARP_MASK.value))
            self._luminance_only_check.setChecked(True)
            self._unsharp_radius.setValue(18)
            self._unsharp_amount.setValue(14)
            self._unsharp_threshold.setValue(0)
            self._intensity_slider.setValue(100)

        elif preset == "anime_4x":
            self._upscale_enabled.setChecked(True)
            self._ai_radio.setChecked(True)
            idx = self._ai_model_combo.findData("RealESRGAN_x4plus_anime_6B")
            if idx >= 0:
                self._ai_model_combo.setCurrentIndex(idx)
            self._scale_combo.setCurrentIndex(self._scale_combo.findData(4.0))
            self._custom_size_check.setChecked(False)

            self._sharpen_enabled.setChecked(True)
            self._sharpen_combo.setCurrentIndex(self._sharpen_combo.findData(SharpenMethod.UNSHARP_MASK.value))
            self._luminance_only_check.setChecked(True)
            self._unsharp_radius.setValue(15)
            self._unsharp_amount.setValue(12)
            self._unsharp_threshold.setValue(2)
            self._intensity_slider.setValue(90)

        elif preset == "fast_2x":
            self._upscale_enabled.setChecked(True)
            self._traditional_radio.setChecked(True)
            self._interp_combo.setCurrentIndex(3)  # Lanczos
            self._scale_combo.setCurrentIndex(self._scale_combo.findData(2.0))
            self._custom_size_check.setChecked(False)

            self._sharpen_enabled.setChecked(True)
            self._sharpen_combo.setCurrentIndex(self._sharpen_combo.findData(SharpenMethod.UNSHARP_MASK.value))
            self._luminance_only_check.setChecked(True)
            self._unsharp_radius.setValue(20)
            self._unsharp_amount.setValue(16)
            self._unsharp_threshold.setValue(0)
            self._intensity_slider.setValue(100)

        elif preset == "sharpen_only":
            self._upscale_enabled.setChecked(False)
            self._sharpen_enabled.setChecked(True)
            self._sharpen_combo.setCurrentIndex(self._sharpen_combo.findData(SharpenMethod.UNSHARP_MASK.value))
            self._luminance_only_check.setChecked(True)
            self._unsharp_radius.setValue(22)
            self._unsharp_amount.setValue(18)
            self._unsharp_threshold.setValue(0)
            self._intensity_slider.setValue(100)

        self._on_mode_changed()
        self._on_sharpen_method_changed(self._sharpen_combo.currentIndex())
        self._block_signals = False
        self.settings_changed.emit()

    def _on_save_clicked(self) -> None:
        """เปิด Dialog บันทึกไฟล์ภาพ"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "บันทึกรูปภาพ", "", SAVE_FILTER,
        )
        if file_path:
            quality = self._quality_slider.value()
            self.save_clicked.emit(file_path, quality)

    def _on_batch_clicked(self) -> None:
        """เปิด Dialog เลือกโฟลเดอร์สำหรับแบตช์"""
        folder = QFileDialog.getExistingDirectory(
            self, "เลือกโฟลเดอร์ที่เก็บไฟล์รูปภาพ", "",
        )
        if folder:
            self.batch_clicked.emit(folder)

    # ──────────────────────────────────────
    # Public Getters / Setters
    # ──────────────────────────────────────
    def get_upscale_params(self) -> Optional[UpscaleParams]:
        """ดึงพารามิเตอร์การขยายภาพ"""
        if not self._upscale_enabled.isChecked():
            return None

        params = UpscaleParams()
        if self._ai_radio.isChecked():
            params.mode = UpscaleMode.AI_REALESRGAN
            params.ai_model_name = self._ai_model_combo.currentData()
        else:
            params.mode = UpscaleMode.TRADITIONAL
            params.interpolation = self._interp_combo.currentData()

        params.scale = self._scale_combo.currentData() or 2.0
        params.use_custom_size = self._custom_size_check.isChecked()
        params.custom_width = self._custom_w.value()
        params.custom_height = self._custom_h.value()
        return params

    def get_sharpen_enabled(self) -> bool:
        return self._sharpen_enabled.isChecked()

    def get_sharpen_method(self) -> Optional[str]:
        if not self._sharpen_enabled.isChecked():
            return None
        return self._sharpen_combo.currentData()

    def get_sharpen_params(self) -> Union[UnsharpMaskParams, LaplacianParams, DetailEnhanceParams, None]:
        if not self._sharpen_enabled.isChecked():
            return None

        method = self._sharpen_combo.currentData()
        lum = self._luminance_only_check.isChecked()

        if method == SharpenMethod.UNSHARP_MASK.value:
            return UnsharpMaskParams(
                radius=self._unsharp_radius.value() / 10.0,
                amount=self._unsharp_amount.value() / 10.0,
                threshold=self._unsharp_threshold.value(),
                luminance_only=lum,
            )
        elif method == SharpenMethod.LAPLACIAN.value:
            return LaplacianParams(
                kernel_size=int(self._lap_ksize.currentText()),
                strength=self._lap_strength.value() / 20.0,
                luminance_only=lum,
            )
        elif method == SharpenMethod.DETAIL_ENHANCE.value:
            return DetailEnhanceParams(
                sigma_s=float(self._detail_sigma_s.value()),
                sigma_r=self._detail_sigma_r.value() / 100.0,
            )
        return None

    def get_sharpen_intensity(self) -> float:
        return self._intensity_slider.value() / 100.0

    def get_save_quality(self) -> int:
        return self._quality_slider.value()

    def set_save_enabled(self, enabled: bool) -> None:
        self._save_btn.setEnabled(enabled)

    def set_processing(self, is_processing: bool) -> None:
        self._process_btn.setEnabled(not is_processing)
        self._batch_btn.setEnabled(not is_processing)
        if is_processing:
            self._process_btn.setText("⏳ กำลังประมวลผล...")
        else:
            self._process_btn.setText("▶ ประมวลผลภาพ (Process)")

    def update_custom_size_from_image(self, width: int, height: int) -> None:
        scale = self._scale_combo.currentData() or 2.0
        self._custom_w.setValue(int(width * scale))
        self._custom_h.setValue(int(height * scale))

    # ──────────────────────────────────────
    # Styles
    # ──────────────────────────────────────
    @staticmethod
    def _group_style() -> str:
        return """
            QGroupBox {
                font-size: 13px;
                font-weight: bold;
                color: #ffffff;
                border: 1px solid #3c3c3c;
                border-radius: 8px;
                margin-top: 8px;
                padding: 12px 8px 8px 8px;
                background-color: #242424;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
                color: #4a9eff;
            }
        """

    @staticmethod
    def _combo_style() -> str:
        return """
            QComboBox {
                background-color: #2f2f2f;
                color: #ffffff;
                border: 1px solid #4a4a4a;
                border-radius: 5px;
                padding: 5px 8px;
                font-size: 12px;
            }
            QComboBox:hover { border-color: #4a9eff; }
            QComboBox::drop-down { border: none; width: 22px; }
            QComboBox::down-arrow {
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #cccccc;
                margin-right: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #ffffff;
                selection-background-color: #1976d2;
                border: 1px solid #444;
            }
        """

    @staticmethod
    def _slider_style() -> str:
        return """
            QSlider::groove:horizontal {
                background: #383838;
                height: 6px;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #2196f3;
                width: 16px;
                height: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #64b5f6;
            }
            QSlider::sub-page:horizontal {
                background: #1976d2;
                border-radius: 3px;
            }
        """

    @staticmethod
    def _spinbox_style() -> str:
        return """
            QSpinBox {
                background-color: #2f2f2f;
                color: #ffffff;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
            }
            QSpinBox:hover { border-color: #4a9eff; }
        """
