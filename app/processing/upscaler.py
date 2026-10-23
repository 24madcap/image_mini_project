"""โมดูลขยายภาพ (Upscaling)

ประสานงานระหว่าง:
1. AI Super Resolution - ใช้ Real-ESRGAN พร้อม Tiled Inference
2. Traditional Upscaling - ใช้ OpenCV Interpolation (Nearest, Bilinear, Bicubic, Lanczos)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from app.processing.ai_engine import (
    AVAILABLE_MODELS,
    check_ai_status,
    run_tiled_ai_upscale,
)
from app.processing.sharpener import (
    SharpenMethod,
    sharpen_image,
    UnsharpMaskParams,
    LaplacianParams,
    DetailEnhanceParams,
)


class UpscaleMode(Enum):
    """โหมดการขยายภาพ"""
    AI_REALESRGAN = "ai_realesrgan"
    TRADITIONAL = "traditional"


class InterpolationMethod(Enum):
    """วิธี interpolation สำหรับการขยายแบบดั้งเดิม"""
    NEAREST = ("Nearest Neighbor (เร็วสุด)", cv2.INTER_NEAREST)
    BILINEAR = ("Bilinear (ภาพนุ่มนวล)", cv2.INTER_LINEAR)
    BICUBIC = ("Bicubic (คมชัดปานกลาง)", cv2.INTER_CUBIC)
    LANCZOS = ("Lanczos (คมชัดสูงสุด แนะนำ)", cv2.INTER_LANCZOS4)

    def __init__(self, display_name: str, cv2_flag: int) -> None:
        self.display_name = display_name
        self.cv2_flag = cv2_flag


@dataclass
class UpscaleParams:
    """พารามิเตอร์การขยายภาพ"""
    mode: UpscaleMode = UpscaleMode.TRADITIONAL
    scale: float = 2.0
    use_custom_size: bool = False
    custom_width: int = 0
    custom_height: int = 0
    interpolation: InterpolationMethod = InterpolationMethod.LANCZOS
    ai_model_name: str = "RealESRGAN_x4plus"


def check_ai_available() -> tuple[bool, str]:
    """ตรวจสอบว่าระบบ AI Super-Resolution พร้อมใช้งานหรือไม่"""
    return check_ai_status()


def upscale_traditional(
    img: np.ndarray,
    params: UpscaleParams,
) -> np.ndarray:
    """ขยายภาพด้วย OpenCV interpolation

    Args:
        img: ภาพต้นฉบับ (BGR หรือ BGRA)
        params: พารามิเตอร์การขยาย
    """
    h, w = img.shape[:2]

    if params.use_custom_size and params.custom_width > 0 and params.custom_height > 0:
        new_w = params.custom_width
        new_h = params.custom_height
    else:
        new_w = max(1, int(round(w * params.scale)))
        new_h = max(1, int(round(h * params.scale)))

    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=params.interpolation.cv2_flag,
    )


def upscale_ai(
    img: np.ndarray,
    params: UpscaleParams,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """ขยายภาพด้วยโมเดล AI Real-ESRGAN

    หากผู้ใช้ต้องการขนาดที่ไม่ตรงกับ native scale ของโมเดล (เช่น ต้องการ 3x หรือขนาดกำหนดเอง)
    จะใช้ AI ขยายก่อน จากนั้นปรับขนาดให้ตรงตามเป้าหมายด้วย Lanczos อย่างแม่นยำ
    """
    model_name = params.ai_model_name
    if model_name not in AVAILABLE_MODELS:
        model_name = "RealESRGAN_x4plus"

    # คำนวณขนาดเป้าหมาย
    h, w = img.shape[:2]
    if params.use_custom_size and params.custom_width > 0 and params.custom_height > 0:
        target_w = params.custom_width
        target_h = params.custom_height
    else:
        target_w = max(1, int(round(w * params.scale)))
        target_h = max(1, int(round(h * params.scale)))

    # ขยายด้วย AI Engine
    ai_result = run_tiled_ai_upscale(
        img,
        model_name=model_name,
        tile_size=400,
        tile_pad=12,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )

    if is_cancelled and is_cancelled():
        return img

    # ปรับขนาดเพิ่มเติมถ้าขนาดเป้าหมายต่างจากผลลัพธ์ของโมเดล AI
    cur_h, cur_w = ai_result.shape[:2]
    if cur_w != target_w or cur_h != target_h:
        if progress_callback:
            progress_callback(97, "กำลังปรับขนาดขั้นสุดท้ายให้ตรงตามเป้าหมาย...")
        ai_result = cv2.resize(
            ai_result,
            (target_w, target_h),
            interpolation=cv2.INTER_LANCZOS4,
        )

    return ai_result


def upscale_image(
    img: np.ndarray,
    params: UpscaleParams,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    """ฟังก์ชันหลักสำหรับขยายภาพตามโหมดที่เลือก"""
    if params.mode == UpscaleMode.AI_REALESRGAN:
        return upscale_ai(img, params, progress_callback=progress_callback, is_cancelled=is_cancelled)
    else:
        if progress_callback:
            progress_callback(50, f"กำลังขยายภาพด้วย {params.interpolation.display_name}...")
        result = upscale_traditional(img, params)
        if progress_callback:
            progress_callback(100, "ขยายภาพเสร็จสิ้น")
        return result


class UpscaleWorker(QThread):
    """Thread สำหรับประมวลผลการขยายภาพและเพิ่มความคมชัดในเบื้องหลัง"""

    progress = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished = pyqtSignal(np.ndarray)
    error = pyqtSignal(str)

    def __init__(
        self,
        img: np.ndarray,
        upscale_params: Optional[UpscaleParams] = None,
        sharpen_enabled: bool = False,
        sharpen_method: Optional[str] = None,
        sharpen_params: Optional[object] = None,
        sharpen_intensity: float = 1.0,
    ) -> None:
        super().__init__()
        self._img = img.copy()
        self._upscale_params = upscale_params
        self._sharpen_enabled = sharpen_enabled
        self._sharpen_method = sharpen_method
        self._sharpen_params = sharpen_params
        self._sharpen_intensity = sharpen_intensity
        self._cancelled = False

    def cancel(self) -> None:
        """ยกเลิกการทำงาน"""
        self._cancelled = True

    def _is_cancelled(self) -> bool:
        return self._cancelled

    def run(self) -> None:
        """ประมวลผลบน background thread"""
        try:
            result = self._img.copy()

            # ขั้นตอนที่ 1: การขยายขนาดภาพ
            if self._upscale_params is not None:
                self.progress.emit(5)
                self.status_updated.emit("กำลังเริ่มต้นขยายขนาดภาพ...")

                def on_progress(pct: int, msg: str = "") -> None:
                    # แมพเปอร์เซ็นต์ไปยังช่วง 5 - 75%
                    mapped = 5 + int(pct * 0.70)
                    self.progress.emit(mapped)
                    if msg:
                        self.status_updated.emit(msg)

                result = upscale_image(
                    result,
                    self._upscale_params,
                    progress_callback=on_progress,
                    is_cancelled=self._is_cancelled,
                )

                if self._cancelled:
                    self.status_updated.emit("ยกเลิกการทำงานแล้ว")
                    return

            self.progress.emit(75)

            # ขั้นตอนที่ 2: การเพิ่มความคมชัด
            if self._sharpen_enabled and self._sharpen_method is not None:
                self.status_updated.emit("กำลังเพิ่มความคมชัดให้กับภาพ...")
                method = SharpenMethod(self._sharpen_method)

                result = sharpen_image(
                    result,
                    method,
                    self._sharpen_params,
                    self._sharpen_intensity,
                )

                if self._cancelled:
                    self.status_updated.emit("ยกเลิกการทำงานแล้ว")
                    return

            self.progress.emit(100)
            self.status_updated.emit("ประมวลผลเสร็จสมบูรณ์")
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))
