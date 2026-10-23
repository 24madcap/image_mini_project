"""ชุดทดสอบ Smoke Test สำหรับ GUI Components (PyQt6)"""

import os
import sys
import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# ใช้ offscreen platform สำหรับ headless testing
os.environ["QT_QPA_PLATFORM"] = "offscreen"


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_drop_zone_widget(qapp: QApplication) -> None:
    from app.widgets.drop_zone import DropZone
    dz = DropZone()
    assert dz is not None
    assert dz.acceptDrops() is True


def test_progress_bar_widget(qapp: QApplication) -> None:
    from app.widgets.progress_bar import ProcessingProgressBar
    pb = ProcessingProgressBar()
    pb.set_status("กำลังทดสอบ")
    pb.set_progress(42)
    assert pb._progress_bar.value() == 42
    pb.set_complete()
    assert pb._progress_bar.value() == 100
    pb.reset()
    assert pb._progress_bar.value() == 0


def test_settings_panel_widget(qapp: QApplication) -> None:
    from app.widgets.settings_panel import SettingsPanel
    from app.processing.upscaler import UpscaleMode
    from app.processing.sharpener import SharpenMethod

    sp = SettingsPanel()
    assert sp is not None

    # ทดสอบค่าเริ่มต้น
    upscale_params = sp.get_upscale_params()
    assert upscale_params is not None
    assert upscale_params.mode == UpscaleMode.AI_REALESRGAN

    sharpen_enabled = sp.get_sharpen_enabled()
    assert sharpen_enabled is True
    method = sp.get_sharpen_method()
    assert method == SharpenMethod.UNSHARP_MASK.value

    # ทดสอบการเลือกพรีเซ็ต
    idx = sp._preset_combo.findData("fast_2x")
    sp._preset_combo.setCurrentIndex(idx)
    params_fast = sp.get_upscale_params()
    assert params_fast.mode == UpscaleMode.TRADITIONAL
    assert params_fast.scale == 2.0


def test_image_viewer_widget(qapp: QApplication) -> None:
    from app.image_viewer import ImageViewer
    viewer = ImageViewer()
    assert viewer is not None

    # ทดสอบการใส่ภาพ
    img_before = np.zeros((100, 100, 3), dtype=np.uint8)
    viewer.set_before_image(img_before)
    assert viewer._before_qimg is not None

    img_after = np.ones((200, 200, 3), dtype=np.uint8) * 255
    viewer.set_after_image(img_after)
    assert viewer._after_qimg is not None
    assert viewer._mode == ImageViewer.MODE_COMPARE_SLIDER

    # ทดสอบการเปลี่ยนโหมด
    viewer.set_view_mode(ImageViewer.MODE_SIDE_BY_SIDE)
    assert viewer._mode == ImageViewer.MODE_SIDE_BY_SIDE

    # ทดสอบการซูม
    viewer.zoom_in()
    assert viewer._zoom > 0
    viewer.zoom_fit()
    viewer.zoom_actual()
    assert viewer._zoom == 1.0


def test_main_window_widget(qapp: QApplication) -> None:
    from app.main_window import MainWindow
    win = MainWindow()
    assert win is not None
    assert "Image Pro" in win.windowTitle()

    # ทดสอบจำลองการโหลดภาพ
    test_img = np.random.randint(0, 256, (80, 80, 3), dtype=np.uint8)
    win._original_img = test_img
    win._viewer.set_before_image(test_img)
    assert win._viewer._before_qimg is not None

    # ทดสอบการรีเซ็ต
    win._reset_result()
    assert win._result_img is None
