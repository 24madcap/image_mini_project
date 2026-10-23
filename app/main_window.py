"""หน้าต่างหลักของแอปพลิเคชัน Image Pro (MainWindow)

ประสานงานระหว่างส่วนประกอบทั้งหมด:
- แถบเมนูและแถบเครื่องมือ (Menu Bar & Tool Bar)
- วิดเจ็ตแสดงภาพและเปรียบเทียบ (ImageViewer)
- แผงควบคุมการตั้งค่า (SettingsPanel)
- แถบความคืบหน้า (ProcessingProgressBar)
- Background Workers สำหรับประมวลผลภาพเดี่ยวและแบตช์
"""

import os
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QKeySequence
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QStatusBar, QMenuBar, QMenu, QMessageBox, QFileDialog, QLabel,
    QToolBar, QApplication, QComboBox, QProgressDialog, QPushButton,
)

from app.image_viewer import ImageViewer
from app.widgets.drop_zone import DropZone
from app.widgets.settings_panel import SettingsPanel
from app.widgets.progress_bar import ProcessingProgressBar
from app.processing.upscaler import (
    UpscaleWorker, UpscaleParams, check_ai_available, upscale_image,
)
from app.processing.sharpener import sharpen_image, SharpenMethod
from app.utils.file_utils import (
    load_image_cv2, save_image_cv2, get_image_info,
    format_file_size, get_images_in_folder, FILE_FILTER, SAVE_FILTER,
)


class BatchWorker(QThread):
    """Background Thread สำหรับประมวลผลแบตช์หลายไฟล์โดยไม่ทำให้ UI ค้าง"""

    file_started = pyqtSignal(int, int, str)       # index, total, filename
    batch_progress = pyqtSignal(int)               # 0-100 overall
    file_finished = pyqtSignal(int, int, str, bool) # index, total, filename, success
    batch_finished = pyqtSignal(int, int, str)     # success_count, fail_count, output_dir
    error = pyqtSignal(str)

    def __init__(
        self,
        files: list[str],
        output_dir: str,
        upscale_params: Optional[UpscaleParams] = None,
        sharpen_enabled: bool = False,
        sharpen_method: Optional[str] = None,
        sharpen_params: Optional[object] = None,
        sharpen_intensity: float = 1.0,
        quality: int = 95,
    ) -> None:
        super().__init__()
        self.files = files
        self.output_dir = output_dir
        self.upscale_params = upscale_params
        self.sharpen_enabled = sharpen_enabled
        self.sharpen_method = sharpen_method
        self.sharpen_params = sharpen_params
        self.sharpen_intensity = sharpen_intensity
        self.quality = quality
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        total = len(self.files)
        success_count = 0
        fail_count = 0

        for i, file_path in enumerate(self.files):
            if self._cancelled:
                break

            fname = os.path.basename(file_path)
            self.file_started.emit(i + 1, total, fname)
            overall_pct = int((i / total) * 100)
            self.batch_progress.emit(overall_pct)

            try:
                img = load_image_cv2(file_path)
                if img is None:
                    fail_count += 1
                    self.file_finished.emit(i + 1, total, fname, False)
                    continue

                result = img

                # 1. ขยายภาพ
                if self.upscale_params is not None:
                    result = upscale_image(
                        result,
                        self.upscale_params,
                        is_cancelled=lambda: self._cancelled,
                    )

                if self._cancelled:
                    break

                # 2. เพิ่มความคมชัด
                if self.sharpen_enabled and self.sharpen_method is not None:
                    method = SharpenMethod(self.sharpen_method)
                    result = sharpen_image(
                        result,
                        method,
                        self.sharpen_params,
                        self.sharpen_intensity,
                    )

                if self._cancelled:
                    break

                # 3. บันทึกผลลัพธ์
                base_name, ext = os.path.splitext(fname)
                out_path = os.path.join(self.output_dir, f"{base_name}_pro{ext}")
                saved = save_image_cv2(result, out_path, self.quality)
                if saved:
                    success_count += 1
                    self.file_finished.emit(i + 1, total, fname, True)
                else:
                    fail_count += 1
                    self.file_finished.emit(i + 1, total, fname, False)

            except Exception as e:
                fail_count += 1
                self.file_finished.emit(i + 1, total, fname, False)

        self.batch_progress.emit(100)
        self.batch_finished.emit(success_count, fail_count, self.output_dir)


class MainWindow(QMainWindow):
    """หน้าต่างหลักของแอปพลิเคชัน Image Pro"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Pro - ขยายและเพิ่มความคมชัดรูปภาพ")
        self.setMinimumSize(1150, 720)
        self.resize(1440, 880)

        # ข้อมูลสถานะ
        self._current_file: Optional[str] = None
        self._original_img: Optional[np.ndarray] = None
        self._result_img: Optional[np.ndarray] = None
        self._worker: Optional[UpscaleWorker] = None
        self._batch_worker: Optional[BatchWorker] = None
        self._process_start_time: float = 0.0

        # สร้างและจัดวาง UI
        self._setup_menu_bar()
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_status_bar()
        self._connect_signals()
        self._apply_dark_theme()

    # ──────────────────────────────────────
    # เมนูบาร์
    # ──────────────────────────────────────
    def _setup_menu_bar(self) -> None:
        menubar = self.menuBar()
        menubar.setStyleSheet("""
            QMenuBar {
                background-color: #242424;
                color: #dddddd;
                border-bottom: 1px solid #3c3c3c;
                padding: 2px 4px;
                font-size: 12px;
            }
            QMenuBar::item {
                padding: 4px 10px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background-color: #1976d2;
                color: #ffffff;
            }
            QMenu {
                background-color: #282828;
                color: #dddddd;
                border: 1px solid #3c3c3c;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 24px 6px 12px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background-color: #1976d2;
                color: #ffffff;
            }
            QMenu::separator {
                height: 1px;
                background: #3c3c3c;
                margin: 4px 8px;
            }
        """)

        # เมนู: ไฟล์
        file_menu = menubar.addMenu("ไฟล์ (File)")

        open_act = QAction("📂 เปิดรูปภาพ... (Open Image)", self)
        open_act.setShortcut(QKeySequence("Ctrl+O"))
        open_act.triggered.connect(self._open_file)
        file_menu.addAction(open_act)

        open_folder_act = QAction("📁 เปิดโฟลเดอร์แบตช์... (Open Folder)", self)
        open_folder_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_folder_act.triggered.connect(self._open_folder_batch)
        file_menu.addAction(open_folder_act)

        file_menu.addSeparator()

        self._save_act = QAction("💾 บันทึกรูปภาพ... (Save Result)", self)
        self._save_act.setShortcut(QKeySequence("Ctrl+S"))
        self._save_act.setEnabled(False)
        self._save_act.triggered.connect(self._save_file)
        file_menu.addAction(self._save_act)

        file_menu.addSeparator()

        exit_act = QAction("❌ ออกจากโปรแกรม (Exit)", self)
        exit_act.setShortcut(QKeySequence("Alt+F4"))
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # เมนู: การประมวลผล
        process_menu = menubar.addMenu("การประมวลผล (Process)")

        process_act = QAction("▶ ประมวลผลภาพ (Start Process)", self)
        process_act.setShortcut(QKeySequence("F5"))
        process_act.triggered.connect(self._process_image)
        process_menu.addAction(process_act)

        cancel_act = QAction("⏹ ยกเลิกการประมวลผล (Cancel)", self)
        cancel_act.setShortcut(QKeySequence("Esc"))
        cancel_act.triggered.connect(self._cancel_processing)
        process_menu.addAction(cancel_act)

        process_menu.addSeparator()

        reset_act = QAction("🔄 รีเซ็ตผลลัพธ์ (Reset Result)", self)
        reset_act.setShortcut(QKeySequence("Ctrl+R"))
        reset_act.triggered.connect(self._reset_result)
        process_menu.addAction(reset_act)

        # เมนู: มุมมอง
        view_menu = menubar.addMenu("มุมมอง (View)")

        view_slider_act = QAction("◀▶ สไลเดอร์เปรียบเทียบ (Split Slider)", self)
        view_slider_act.triggered.connect(lambda: self._set_viewer_mode(ImageViewer.MODE_COMPARE_SLIDER))
        view_menu.addAction(view_slider_act)

        view_side_act = QAction("⏹⏹ แสดงคู่กัน (Side-by-Side)", self)
        view_side_act.triggered.connect(lambda: self._set_viewer_mode(ImageViewer.MODE_SIDE_BY_SIDE))
        view_menu.addAction(view_side_act)

        view_single_act = QAction("🖼 แสดงภาพเดี่ยว (Single Image)", self)
        view_single_act.triggered.connect(lambda: self._set_viewer_mode(ImageViewer.MODE_SINGLE))
        view_menu.addAction(view_single_act)

        view_menu.addSeparator()

        zoom_in_act = QAction("🔍+ ซูมเข้า (Zoom In)", self)
        zoom_in_act.setShortcut(QKeySequence("Ctrl+="))
        zoom_in_act.triggered.connect(lambda: self._viewer.zoom_in())
        view_menu.addAction(zoom_in_act)

        zoom_out_act = QAction("🔍- ซูมออก (Zoom Out)", self)
        zoom_out_act.setShortcut(QKeySequence("Ctrl+-"))
        zoom_out_act.triggered.connect(lambda: self._viewer.zoom_out())
        view_menu.addAction(zoom_out_act)

        zoom_fit_act = QAction("📐 ปรับพอดีหน้าต่าง (Fit to Screen)", self)
        zoom_fit_act.setShortcut(QKeySequence("Ctrl+0"))
        zoom_fit_act.triggered.connect(lambda: self._viewer.zoom_fit())
        view_menu.addAction(zoom_fit_act)

        zoom_actual_act = QAction("🔲 ซูมขนาดจริง 100% (Actual Size)", self)
        zoom_actual_act.setShortcut(QKeySequence("Ctrl+1"))
        zoom_actual_act.triggered.connect(lambda: self._viewer.zoom_actual())
        view_menu.addAction(zoom_actual_act)

        # เมนู: ช่วยเหลือ
        help_menu = menubar.addMenu("ช่วยเหลือ (Help)")

        ai_status_act = QAction("🤖 ตรวจสอบสถานะ AI", self)
        ai_status_act.triggered.connect(self._check_ai_status)
        help_menu.addAction(ai_status_act)

        about_act = QAction("ℹ เกี่ยวกับ Image Pro", self)
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    # ──────────────────────────────────────
    # แถบเครื่องมือ
    # ──────────────────────────────────────
    def _setup_toolbar(self) -> None:
        toolbar = QToolBar("เครื่องมือหลัก")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #242424;
                border-bottom: 1px solid #3c3c3c;
                spacing: 6px;
                padding: 4px 8px;
            }
            QToolButton {
                background-color: transparent;
                color: #cccccc;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: 500;
            }
            QToolButton:hover {
                background-color: #333333;
                border-color: #555555;
                color: #ffffff;
            }
            QToolButton:pressed {
                background-color: #1976d2;
                color: #ffffff;
            }
        """)

        toolbar.addAction("📂 เปิดรูปภาพ", self._open_file)
        self._tb_save_act = toolbar.addAction("💾 บันทึกผลลัพธ์", self._save_file)
        self._tb_save_act.setEnabled(False)

        toolbar.addSeparator()

        # กลุ่มเลือกโหมดการแสดงผล
        toolbar.addWidget(QLabel(" มุมมอง: "))
        self._view_mode_combo = QComboBox()
        self._view_mode_combo.addItem("◀▶ สไลเดอร์เปรียบเทียบ", ImageViewer.MODE_COMPARE_SLIDER)
        self._view_mode_combo.addItem("⏹⏹ คู่กันซ้าย-ขวา", ImageViewer.MODE_SIDE_BY_SIDE)
        self._view_mode_combo.addItem("🖼 ภาพเดี่ยว", ImageViewer.MODE_SINGLE)
        self._view_mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #2f2f2f;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
            }
            QComboBox::drop-down { border: none; width: 16px; }
            QComboBox::down-arrow {
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 5px solid #cccccc;
                margin-right: 4px;
            }
        """)
        self._view_mode_combo.currentIndexChanged.connect(
            lambda: self._set_viewer_mode(self._view_mode_combo.currentData())
        )
        toolbar.addWidget(self._view_mode_combo)

        toolbar.addSeparator()

        toolbar.addAction("🔍+", lambda: self._viewer.zoom_in())
        toolbar.addAction("🔍-", lambda: self._viewer.zoom_out())
        toolbar.addAction("📐 พอดี", lambda: self._viewer.zoom_fit())
        toolbar.addAction("🔲 100%", lambda: self._viewer.zoom_actual())

        self._zoom_label = QLabel(" 100% ")
        self._zoom_label.setStyleSheet("color: #4a9eff; font-weight: bold; font-size: 11px;")
        toolbar.addWidget(self._zoom_label)

        self.addToolBar(toolbar)

    # ──────────────────────────────────────
    # วิดเจ็ตหลัก
    # ──────────────────────────────────────
    def _setup_central_widget(self) -> None:
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # พื้นที่แสดงภาพ (ซ้าย)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 6, 2, 6)
        left_layout.setSpacing(6)

        self._drop_zone = DropZone()
        self._viewer = ImageViewer()
        self._viewer.setVisible(False)

        left_layout.addWidget(self._drop_zone)
        left_layout.addWidget(self._viewer)

        # แถบแสดงความคืบหน้า
        self._progress = ProcessingProgressBar()
        left_layout.addWidget(self._progress)

        # แผงตั้งค่า (ขวา)
        self._settings = SettingsPanel()

        # Splitter แบ่งพื้นที่ซ้าย-ขวา
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_widget)
        splitter.addWidget(self._settings)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setHandleWidth(3)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #383838;
            }
        """)

        main_layout.addWidget(splitter)
        self.setCentralWidget(central)

    # ──────────────────────────────────────
    # แถบสถานะ (Status Bar)
    # ──────────────────────────────────────
    def _setup_status_bar(self) -> None:
        statusbar = self.statusBar()
        statusbar.setStyleSheet("""
            QStatusBar {
                background-color: #1e1e1e;
                color: #aaaaaa;
                border-top: 1px solid #333333;
                font-size: 11px;
                padding: 3px 8px;
            }
        """)

        self._status_info = QLabel("สถานะ: พร้อมใช้งาน")
        self._status_orig_size = QLabel("")
        self._status_result_size = QLabel("")
        self._status_file = QLabel("")

        statusbar.addWidget(self._status_info, 1)
        statusbar.addPermanentWidget(self._status_orig_size)
        statusbar.addPermanentWidget(self._status_result_size)
        statusbar.addPermanentWidget(self._status_file)

    # ──────────────────────────────────────
    # สัญญาณและการตอบสนอง
    # ──────────────────────────────────────
    def _connect_signals(self) -> None:
        # Drop Zone
        self._drop_zone.file_dropped.connect(self._load_image)
        self._drop_zone.files_dropped.connect(self._on_files_dropped)

        # Settings Panel
        self._settings.process_clicked.connect(self._process_image)
        self._settings.save_clicked.connect(self._on_save)
        self._settings.batch_clicked.connect(self._process_batch)

        # Viewer
        self._viewer.zoom_changed.connect(
            lambda z: self._zoom_label.setText(f" {z}% ")
        )

    def _set_viewer_mode(self, mode: str) -> None:
        self._viewer.set_view_mode(mode)
        idx = self._view_mode_combo.findData(mode)
        if idx >= 0 and self._view_mode_combo.currentIndex() != idx:
            self._view_mode_combo.setCurrentIndex(idx)

    # ──────────────────────────────────────
    # การโหลดภาพ
    # ──────────────────────────────────────
    def _open_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self, "เลือกไฟล์รูปภาพ", "", FILE_FILTER,
        )
        if file_path:
            self._load_image(file_path)

    def _load_image(self, file_path: str) -> None:
        img = load_image_cv2(file_path)
        if img is None:
            QMessageBox.critical(
                self,
                "ข้อผิดพลาด",
                f"ไม่สามารถเปิดไฟล์รูปภาพได้:\n{file_path}\n\nกรุณาตรวจสอบว่าไฟล์ไม่เสียหายและเป็นรูปแบบที่รองรับ",
            )
            return

        self._current_file = file_path
        self._original_img = img
        self._result_img = None

        # สลับหน้าจอเป็น ImageViewer
        self._drop_zone.setVisible(False)
        self._viewer.setVisible(True)
        self._viewer.set_before_image(img)
        self._viewer.set_after_image(None)

        # อัพเดทข้อมูลใน UI
        h, w = img.shape[:2]
        size_bytes = os.path.getsize(file_path)
        size_str = format_file_size(size_bytes)

        self._status_file.setText(f"ไฟล์: {os.path.basename(file_path)}")
        self._status_orig_size.setText(f"ต้นฉบับ: {w}×{h} ({size_str})")
        self._status_result_size.setText("")
        self._status_info.setText("สถานะ: โหลดรูปภาพสำเร็จ")

        self._settings.set_save_enabled(False)
        self._save_act.setEnabled(False)
        self._tb_save_act.setEnabled(False)

        self._settings.update_custom_size_from_image(w, h)
        self._progress.reset()

        self.setWindowTitle(
            f"Image Pro — {os.path.basename(file_path)} ({w}×{h})"
        )

    def _on_files_dropped(self, files: list[str]) -> None:
        if len(files) == 1:
            self._load_image(files[0])
        elif len(files) > 1:
            reply = QMessageBox.question(
                self,
                "วางหลายไฟล์",
                f"คุณได้ลากรูปภาพเข้ามาทั้งหมด {len(files)} ไฟล์\n\n"
                f"ต้องการประมวลผลแบบแบตช์ทั้ง {len(files)} ไฟล์นี้เลยหรือไม่?\n"
                f"(กด 'No' หากต้องการเปิดเฉพาะภาพแรก)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._process_batch_files(files)
            else:
                self._load_image(files[0])

    # ──────────────────────────────────────
    # การประมวลผล
    # ──────────────────────────────────────
    def _process_image(self) -> None:
        if self._original_img is None:
            QMessageBox.warning(self, "ยังไม่มีภาพ", "กรุณาเปิดรูปภาพก่อนเริ่มประมวลผล")
            return

        upscale_params = self._settings.get_upscale_params()
        sharpen_enabled = self._settings.get_sharpen_enabled()

        if upscale_params is None and not sharpen_enabled:
            QMessageBox.information(
                self,
                "ไม่มีการตั้งค่า",
                "กรุณาเปิดใช้งานอย่างน้อย 1 อย่างระหว่าง:\n- การขยายขนาดภาพ\n- การเพิ่มความคมชัด",
            )
            return

        # เริ่มประมวลผลใน background
        self._process_start_time = time.time()
        self._settings.set_processing(True)
        self._progress.set_processing(True)
        self._progress.set_status("⏳ กำลังเริ่มต้นประมวลผล...")
        self._progress.set_progress(0)

        self._worker = UpscaleWorker(
            img=self._original_img,
            upscale_params=upscale_params,
            sharpen_enabled=sharpen_enabled,
            sharpen_method=self._settings.get_sharpen_method(),
            sharpen_params=self._settings.get_sharpen_params(),
            sharpen_intensity=self._settings.get_sharpen_intensity(),
        )
        self._worker.progress.connect(self._progress.set_progress)
        self._worker.status_updated.connect(self._progress.set_status)
        self._worker.finished.connect(self._on_process_finished)
        self._worker.error.connect(self._on_process_error)
        self._worker.start()

    def _cancel_processing(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._progress.set_status("⏹ กำลังยกเลิก...")
            self._settings.set_processing(False)

    def _on_process_finished(self, result: np.ndarray) -> None:
        elapsed = time.time() - self._process_start_time
        self._result_img = result
        self._viewer.set_after_image(result)

        self._settings.set_processing(False)
        self._settings.set_save_enabled(True)
        self._save_act.setEnabled(True)
        self._tb_save_act.setEnabled(True)

        self._progress.set_complete()

        # รายงานขนาด
        oh, ow = self._original_img.shape[:2]
        rh, rw = result.shape[:2]
        scale_ratio = rw / ow

        self._status_result_size.setText(f"ผลลัพธ์: {rw}×{rh} ({scale_ratio:.1f}x)")
        self._status_info.setText(f"สถานะ: ประมวลผลเสร็จใน {elapsed:.1f} วินาที ({ow}×{oh} → {rw}×{rh})")

        self._worker = None

    def _on_process_error(self, error_msg: str) -> None:
        self._settings.set_processing(False)
        self._progress.set_error("เกิดข้อผิดพลาดในการประมวลผล")

        QMessageBox.critical(
            self,
            "ข้อผิดพลาดระหว่างประมวลผล",
            f"เกิดข้อผิดพลาด:\n\n{error_msg}",
        )
        self._worker = None

    def _reset_result(self) -> None:
        self._result_img = None
        self._viewer.set_after_image(None)
        self._settings.set_save_enabled(False)
        self._save_act.setEnabled(False)
        self._tb_save_act.setEnabled(False)
        self._status_result_size.setText("")
        self._status_info.setText("สถานะ: พร้อมใช้งาน")
        self._progress.reset()

    # ──────────────────────────────────────
    # การบันทึก
    # ──────────────────────────────────────
    def _save_file(self) -> None:
        if self._result_img is None:
            QMessageBox.warning(self, "ไม่มีผลลัพธ์", "กรุณาประมวลผลภาพก่อนทำการบันทึก")
            return

        # ตั้งชื่อไฟล์แนะนำ
        init_name = ""
        if self._current_file:
            base, ext = os.path.splitext(self._current_file)
            init_name = f"{base}_pro{ext}"

        file_path, _ = QFileDialog.getSaveFileName(
            self, "บันทึกภาพผลลัพธ์", init_name, SAVE_FILTER,
        )
        if file_path:
            quality = self._settings.get_save_quality()
            self._on_save(file_path, quality)

    def _on_save(self, file_path: str, quality: int) -> None:
        if self._result_img is None:
            QMessageBox.warning(self, "ไม่มีผลลัพธ์", "กรุณาประมวลผลภาพก่อนทำการบันทึก")
            return

        success = save_image_cv2(self._result_img, file_path, quality)
        if success:
            fname = os.path.basename(file_path)
            self._status_info.setText(f"สถานะ: บันทึกไฟล์สำเร็จ → {fname}")
            self._progress.set_status(f"✅ บันทึกสำเร็จ: {fname}")
        else:
            QMessageBox.critical(
                self, "ข้อผิดพลาด", f"ไม่สามารถบันทึกไฟล์ได้:\n{file_path}"
            )

    # ──────────────────────────────────────
    # ประมวลผลแบตช์
    # ──────────────────────────────────────
    def _open_folder_batch(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "เลือกโฟลเดอร์ที่มีไฟล์รูปภาพ", ""
        )
        if folder:
            self._process_batch(folder)

    def _process_batch(self, folder_path: str) -> None:
        images = get_images_in_folder(folder_path)
        if not images:
            QMessageBox.warning(
                self, "ไม่พบรูปภาพ", f"ไม่พบไฟล์รูปภาพที่รองรับในโฟลเดอร์:\n{folder_path}"
            )
            return
        self._process_batch_files(images)

    def _process_batch_files(self, files: list[str]) -> None:
        parent_dir = os.path.dirname(files[0])
        output_dir = os.path.join(parent_dir, "image_pro_output")
        os.makedirs(output_dir, exist_ok=True)

        upscale_params = self._settings.get_upscale_params()
        sharpen_enabled = self._settings.get_sharpen_enabled()

        if upscale_params is None and not sharpen_enabled:
            QMessageBox.information(
                self, "ไม่มีการตั้งค่า", "กรุณาเปิดการขยายภาพหรือความคมชัดก่อนประมวลผลแบตช์"
            )
            return

        # เริ่มต้น Background Batch Worker
        self._settings.set_processing(True)
        self._progress.set_processing(True)
        self._progress.set_status(f"กำลังประมวลผลแบตช์ {len(files)} ไฟล์...")

        self._batch_worker = BatchWorker(
            files=files,
            output_dir=output_dir,
            upscale_params=upscale_params,
            sharpen_enabled=sharpen_enabled,
            sharpen_method=self._settings.get_sharpen_method(),
            sharpen_params=self._settings.get_sharpen_params(),
            sharpen_intensity=self._settings.get_sharpen_intensity(),
            quality=self._settings.get_save_quality(),
        )

        self._batch_worker.file_started.connect(
            lambda cur, tot, name: self._progress.set_status(f"กำลังประมวลผล {cur}/{tot}: {name}")
        )
        self._batch_worker.batch_progress.connect(self._progress.set_progress)
        self._batch_worker.batch_finished.connect(self._on_batch_finished)
        self._batch_worker.start()

    def _on_batch_finished(self, success_cnt: int, fail_cnt: int, out_dir: str) -> None:
        self._settings.set_processing(False)
        self._progress.set_progress(100)
        self._progress.set_status(f"✅ แบตช์เสร็จสิ้น: สำเร็จ {success_cnt} ไฟล์")
        self._status_info.setText(f"สถานะ: แบตช์เสร็จสิ้น ({success_cnt} สำเร็จ, {fail_cnt} ล้มเหลว)")

        QMessageBox.information(
            self,
            "ประมวลผลแบตช์เสร็จสิ้น",
            f"ประมวลผลเสร็จสิ้นเรียบร้อยแล้ว!\n\n"
            f"✔ ประมวลผลสำเร็จ: {success_cnt} ไฟล์\n"
            f"✖ ล้มเหลว: {fail_cnt} ไฟล์\n\n"
            f"บันทึกผลลัพธ์ไว้ที่โฟลเดอร์:\n{out_dir}",
        )
        self._batch_worker = None

    # ──────────────────────────────────────
    # เมนูช่วยเหลือ & ข้อมูล
    # ──────────────────────────────────────
    def _check_ai_status(self) -> None:
        ok, msg = check_ai_available()
        if ok:
            QMessageBox.information(self, "สถานะระบบ AI", f"✅ {msg}")
        else:
            QMessageBox.warning(self, "สถานะระบบ AI", f"⚠️ {msg}")

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "เกี่ยวกับ Image Pro",
            "<h2>Image Pro v1.0.0</h2>"
            "<p><b>โปรแกรมขยายขนาดและเพิ่มความคมชัดรูปภาพคุณภาพสูง</b></p>"
            "<p>พัฒนาด้วย Python, PyQt6, PyTorch, และ OpenCV</p>"
            "<hr>"
            "<p><b>คุณสมบัติเด่น:</b></p>"
            "<ul>"
            "<li><b>AI Super-Resolution:</b> โมเดล Real-ESRGAN (x4plus, Anime 6B, x2plus) พร้อม Tiled Inference</li>"
            "<li><b>Traditional Interpolation:</b> Lanczos, Bicubic, Bilinear, Nearest</li>"
            "<li><b>High-Precision Sharpening:</b> Unsharp Mask, Laplacian, Detail Enhancement พร้อมโหมด Luminance Only</li>"
            "<li><b>Smart Comparison:</b> Split Slider เปรียบเทียบซ้อนทับ และ Side-by-Side ดูคู่กัน</li>"
            "<li><b>Batch Processing:</b> ประมวลผลภาพทั้งโฟลเดอร์ในเบื้องหลัง</li>"
            "</ul>"
        )

    # ──────────────────────────────────────
    # การลากวางไฟล์ที่หน้าต่างหลัก
    # ──────────────────────────────────────
    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            from app.utils.file_utils import is_supported_image
            urls = event.mimeData().urls()
            if any(url.isLocalFile() and is_supported_image(url.toLocalFile()) for url in urls):
                event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        from app.utils.file_utils import is_supported_image
        urls = event.mimeData().urls()
        valid_files = [
            url.toLocalFile() for url in urls
            if url.isLocalFile() and is_supported_image(url.toLocalFile())
        ]
        if len(valid_files) == 1:
            self._load_image(valid_files[0])
        elif len(valid_files) > 1:
            self._on_files_dropped(valid_files)
        event.acceptProposedAction()

    # ──────────────────────────────────────
    # Dark Theme
    # ──────────────────────────────────────
    def _apply_dark_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                color: #dddddd;
                font-family: "Segoe UI", "Tahoma", "Noto Sans Thai", sans-serif;
            }
            QLabel {
                color: #dddddd;
            }
            QToolTip {
                background-color: #2e2e2e;
                color: #ffffff;
                border: 1px solid #4a4a4a;
                padding: 4px 8px;
                border-radius: 4px;
            }
            QMessageBox {
                background-color: #242424;
            }
            QMessageBox QLabel {
                color: #dddddd;
                font-size: 13px;
            }
            QMessageBox QPushButton {
                background-color: #333333;
                color: #ffffff;
                border: 1px solid #4c4c4c;
                border-radius: 4px;
                padding: 6px 18px;
                min-width: 75px;
            }
            QMessageBox QPushButton:hover {
                background-color: #1976d2;
                border-color: #1976d2;
            }
        """)
        self.setAcceptDrops(True)
